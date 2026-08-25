#!/usr/bin/env python3
"""Download, hash, and validate score-blind S3 evaluation candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

CANDIDATE_FILES = {
    "negative": {
        "filename": "candidate_negative_keys.parquet",
        "rows": 209_157,
    },
    "positive": {
        "filename": "candidate_positive_keys.parquet",
        "rows": 371,
    },
}
CANDIDATE_COLUMNS = {"galaxy_id", "s3_key", "bytes", "etag", "stage1_reason"}
PAYLOAD_COLUMNS = [
    "s3_key",
    "bucket",
    "galaxy_id",
    "expected_bytes",
    "etag",
    "local_relpath",
    "payload_bytes",
    "payload_md5",
    "payload_sha256",
    "valid",
    "error",
    "ra",
    "dec",
    "brickname",
    "pixel_scale",
    "layer",
    "bands",
    "pipeline_version",
    "quality_ok",
    "has_g",
    "has_r",
    "has_z",
    "nobs_z",
    "tractor_type",
    "pool",
    "stored_split",
    "psfsize_r",
    "psfdepth_r",
    "galdepth_r",
    "ebv",
    "tier",
    "match_type",
    "nan_fraction",
    "central_nan_fraction",
]
MAX_COMPRESSED_BYTES = 1_000_000
MAX_MEMBER_BYTES = 2_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10_000_000
MAX_ARCHIVE_MEMBERS = 128
DATABASE_SCHEMA_VERSION = "2"
SAFE_GALAXY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PAYLOAD_TABLE_SQL = """
CREATE TABLE payloads (
    s3_key TEXT PRIMARY KEY,
    bucket TEXT NOT NULL,
    galaxy_id TEXT NOT NULL,
    expected_bytes INTEGER NOT NULL,
    etag TEXT NOT NULL,
    local_relpath TEXT,
    payload_bytes INTEGER,
    payload_md5 TEXT,
    payload_sha256 TEXT,
    valid INTEGER NOT NULL,
    error TEXT,
    ra REAL,
    dec REAL,
    brickname TEXT,
    pixel_scale REAL,
    layer TEXT,
    bands TEXT,
    pipeline_version TEXT,
    quality_ok INTEGER,
    has_g INTEGER,
    has_r INTEGER,
    has_z INTEGER,
    nobs_z INTEGER,
    tractor_type TEXT,
    pool TEXT,
    stored_split TEXT,
    psfsize_r REAL,
    psfdepth_r REAL,
    galdepth_r REAL,
    ebv REAL,
    tier TEXT,
    match_type TEXT,
    nan_fraction REAL,
    central_nan_fraction REAL
)
"""
METADATA_TABLE_SQL = """
CREATE TABLE run_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sql(value: str) -> str:
    return " ".join(value.replace("IF NOT EXISTS", "").split()).casefold()


def scalar(data: Any, key: str) -> Any:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.ndim != 0:
        return None
    item = value.item()
    return item.item() if hasattr(item, "item") else item


def required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be Boolean")
    return bool(value)


def required_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def required_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def validate_payload(body: bytes, bucket: str, galaxy_id: str) -> dict[str, Any]:
    if len(body) > MAX_COMPRESSED_BYTES:
        raise ValueError(f"compressed payload exceeds {MAX_COMPRESSED_BYTES} bytes")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("NPZ archive has an invalid member count")
        member_names = [member.filename for member in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("NPZ archive contains duplicate member names")
        if any(
            member.file_size > MAX_MEMBER_BYTES
            or member.compress_size > MAX_COMPRESSED_BYTES
            for member in members
        ):
            raise ValueError("NPZ member exceeds the size limit")
        if sum(member.file_size for member in members) > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("NPZ archive exceeds the uncompressed size limit")
        if any(
            Path(member.filename).is_absolute()
            or ".." in Path(member.filename).parts
            or not member.filename.endswith(".npy")
            for member in members
        ):
            raise ValueError("NPZ archive contains an unsafe member name")
        for member in members:
            with archive.open(member) as stream:
                while stream.read(64 * 1024):
                    pass

    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        if "cutout" not in data.files:
            raise ValueError("missing cutout array")
        cutout = np.asarray(data["cutout"])
        if cutout.shape != (101, 101, 3):
            raise ValueError(f"cutout shape is {cutout.shape}, expected (101,101,3)")
        if cutout.dtype != np.float32:
            raise ValueError(f"cutout dtype is {cutout.dtype}, expected float32")
        if not np.isfinite(cutout).all():
            raise ValueError("cutout contains nonfinite values")

        metadata_id = required_string(
            scalar(data, "meta_galaxy_id"), "meta_galaxy_id"
        )
        if metadata_id != galaxy_id:
            raise ValueError(
                f"metadata galaxy ID {metadata_id!r} differs from key {galaxy_id!r}"
            )
        expected_type = "negative" if bucket == "negative" else "positive"
        cutout_type = required_string(
            scalar(data, "meta_cutout_type"), "meta_cutout_type"
        )
        if cutout_type != expected_type:
            raise ValueError(
                f"metadata cutout type {cutout_type!r} differs from {expected_type!r}"
            )

        ra = required_float(scalar(data, "meta_ra"), "meta_ra")
        dec = required_float(scalar(data, "meta_dec"), "meta_dec")
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            raise ValueError("coordinates outside celestial ranges")

        metadata = {
            "ra": ra,
            "dec": dec,
            "brickname": scalar(data, "meta_brickname"),
            "size": scalar(data, "meta_size"),
            "pixel_scale": scalar(data, "meta_pixel_scale"),
            "layer": scalar(data, "meta_layer"),
            "bands": scalar(data, "meta_bands_requested"),
            "pipeline_version": scalar(data, "meta_pipeline_version"),
            "quality_ok": scalar(data, "meta_quality_ok"),
            "has_g": scalar(data, "meta_has_g"),
            "has_r": scalar(data, "meta_has_r"),
            "has_z": scalar(data, "meta_has_z"),
            "nobs_z": scalar(data, "meta_nobs_z"),
            "tractor_type": scalar(data, "meta_type"),
            "pool": scalar(data, "meta_pool"),
            "stored_split": scalar(data, "meta_split"),
            "psfsize_r": scalar(data, "meta_psfsize_r"),
            "psfdepth_r": scalar(data, "meta_psfdepth_r"),
            "galdepth_r": scalar(data, "meta_galdepth_r"),
            "ebv": scalar(data, "meta_ebv"),
            "tier": scalar(data, "meta_tier"),
            "match_type": scalar(data, "meta_match_type"),
            "nan_fraction": scalar(data, "meta_nan_frac"),
            "central_nan_fraction": scalar(data, "meta_central_nan_frac"),
        }
        metadata["brickname"] = required_string(
            metadata["brickname"], "meta_brickname"
        )
        metadata["size"] = required_int(metadata["size"], "meta_size", minimum=1)
        metadata["pixel_scale"] = required_float(
            metadata["pixel_scale"], "meta_pixel_scale", minimum=0.0
        )
        metadata["layer"] = required_string(metadata["layer"], "meta_layer")
        metadata["bands"] = required_string(metadata["bands"], "meta_bands_requested")
        metadata["pipeline_version"] = required_string(
            metadata["pipeline_version"], "meta_pipeline_version"
        )
        for field in ("quality_ok", "has_g", "has_r", "has_z"):
            metadata[field] = required_bool(metadata[field], f"meta_{field}")
        metadata["nobs_z"] = required_int(
            metadata["nobs_z"], "meta_nobs_z", minimum=0
        )
        metadata["nan_fraction"] = required_float(
            metadata["nan_fraction"], "meta_nan_frac", minimum=0.0
        )
        metadata["central_nan_fraction"] = required_float(
            metadata["central_nan_fraction"],
            "meta_central_nan_frac",
            minimum=0.0,
        )
        if not np.isclose(metadata["pixel_scale"], 0.262):
            raise ValueError("unexpected pixel scale")
        if metadata["size"] != 101:
            raise ValueError("metadata cutout size is not 101")
        if metadata["layer"] != "ls-dr10" or metadata["bands"] != "grz":
            raise ValueError("unexpected survey layer or band set")
        if not all(metadata[field] for field in ("quality_ok", "has_g", "has_r", "has_z")):
            raise ValueError("cutout quality or required-band flag failed")
        if metadata["nobs_z"] < 3:
            raise ValueError("fewer than three z-band observations")
        if not (
            metadata["nan_fraction"] == 0.0
            and metadata["central_nan_fraction"] == 0.0
        ):
            raise ValueError("cutout metadata reports nonzero NaN fraction")
        if bucket == "negative":
            metadata["tractor_type"] = required_string(
                metadata["tractor_type"], "meta_type"
            )
            if metadata["tractor_type"] not in {"SER", "DEV", "REX"}:
                raise ValueError("negative candidate has an ineligible Tractor type")
            for field in ("psfsize_r", "psfdepth_r", "galdepth_r"):
                metadata[field] = required_float(
                    metadata[field], f"meta_{field}", minimum=0.0
                )
            metadata["ebv"] = required_float(metadata["ebv"], "meta_ebv")
        else:
            metadata["tier"] = required_string(metadata["tier"], "meta_tier")
            metadata["match_type"] = required_string(
                metadata["match_type"], "meta_match_type"
            )
        return metadata


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(PAYLOAD_TABLE_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1))
    connection.execute(
        METADATA_TABLE_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
    )
    for table, expected_sql in (
        ("payloads", PAYLOAD_TABLE_SQL),
        ("run_metadata", METADATA_TABLE_SQL),
    ):
        observed = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if observed is None or normalized_sql(observed[0]) != normalized_sql(expected_sql):
            raise ValueError(f"existing {table} database schema differs")
    existing_version = connection.execute(
        "SELECT value FROM run_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if existing_version is None:
        existing_rows = connection.execute("SELECT COUNT(*) FROM payloads").fetchone()[0]
        if existing_rows:
            raise ValueError("nonempty versionless payload database cannot be resumed")
        connection.execute(
            "INSERT INTO run_metadata (key, value) VALUES (?, ?)",
            ("schema_version", DATABASE_SCHEMA_VERSION),
        )
    elif existing_version[0] != DATABASE_SCHEMA_VERSION:
        raise ValueError("existing payload database has the wrong schema version")
    connection.commit()
    return connection


def existing_valid_keys(connection: sqlite3.Connection, root: Path) -> set[str]:
    valid: set[str] = set()
    for key, bucket, galaxy_id, relative, size, md5, sha256 in connection.execute(
        """
        SELECT s3_key, bucket, galaxy_id, local_relpath, payload_bytes,
               payload_md5, payload_sha256
        FROM payloads WHERE valid = 1
        """
    ):
        path = root / str(relative)
        if path.is_file() and path.stat().st_size == int(size):
            body = path.read_bytes()
            digest = hashlib.md5(body).hexdigest()  # noqa: S324
            strong_digest = hashlib.sha256(body).hexdigest()
            if digest == md5 and strong_digest == sha256:
                try:
                    validate_payload(body, str(bucket), str(galaxy_id))
                except Exception:  # noqa: BLE001
                    continue
                valid.add(str(key))
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--stage1",
        type=Path,
        default=Path("evaluation_dataset_prep/stage1_antijoin"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/candidate_payloads"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = arguments.output_dir / "COMPLETE.json"
    completion_marker.unlink(missing_ok=True)
    (arguments.output_dir / "FAILED.json").unlink(missing_ok=True)

    candidates: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for bucket, specification in CANDIDATE_FILES.items():
        filename = str(specification["filename"])
        path = arguments.stage1 / filename
        input_hashes[filename] = sha256_file(path)
        frame = pd.read_parquet(path)
        missing = sorted(CANDIDATE_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} lacks required columns: {missing}")
        if len(frame) != specification["rows"]:
            raise ValueError(
                f"{filename} has {len(frame)} rows, expected {specification['rows']}"
            )
        if frame[list(CANDIDATE_COLUMNS)].isna().any().any():
            raise ValueError(f"{filename} contains null required values")
        if not frame["stage1_reason"].eq("candidate").all():
            raise ValueError(f"{filename} contains non-candidate rows")
        if frame["s3_key"].duplicated().any() or frame["galaxy_id"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate keys or identifiers")
        if frame["galaxy_id"].astype(str).str.casefold().duplicated().any():
            raise ValueError(f"{filename} contains case-fold identifier collisions")
        if not frame["etag"].astype(str).str.fullmatch(r"[0-9a-fA-F]{32}").all():
            raise ValueError(f"{filename} contains a malformed ETag")
        expected_segment = f"/cutouts/{bucket}s/"
        if not frame["s3_key"].astype(str).str.contains(
            expected_segment, regex=False
        ).all():
            raise ValueError(f"{filename} contains a key from the wrong bucket")
        for row in frame.sort_values("galaxy_id", kind="stable").itertuples():
            galaxy_id = str(row.galaxy_id)
            key = str(row.s3_key)
            expected_bytes = int(row.bytes)
            if not SAFE_GALAXY_ID.fullmatch(galaxy_id):
                raise ValueError(f"{filename} contains unsafe galaxy ID: {galaxy_id}")
            if Path(key).name != f"{galaxy_id}.npz":
                raise ValueError(f"{filename} key basename differs from galaxy ID")
            if expected_bytes <= 0 or expected_bytes > MAX_COMPRESSED_BYTES:
                raise ValueError(f"{filename} contains an invalid byte count")
            candidates.append(
                {
                    "bucket": bucket,
                    "galaxy_id": galaxy_id,
                    "s3_key": key,
                    "bytes": expected_bytes,
                    "etag": str(row.etag).lower(),
                }
            )
    candidate_keys = {record["s3_key"] for record in candidates}
    if len(candidate_keys) != len(candidates):
        raise ValueError("candidate files overlap in S3 key")
    candidate_ids = [record["galaxy_id"].casefold() for record in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("negative and positive candidates share an identifier")
    destinations = [
        str(Path(record["bucket"]) / record["galaxy_id"][:4] / f"{record['galaxy_id']}.npz").casefold()
        for record in candidates
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("candidate rows collide on a local destination")
    snapshot_payload = json.dumps(
        {
            "input_hashes": input_hashes,
            "records": [
                [
                    record["bucket"],
                    record["galaxy_id"],
                    record["s3_key"],
                    record["bytes"],
                    record["etag"],
                ]
                for record in candidates
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    candidate_snapshot_sha256 = hashlib.sha256(snapshot_payload).hexdigest()

    database_path = arguments.output_dir / "payloads.sqlite"
    connection = initialize_database(database_path)
    previous_snapshot = connection.execute(
        "SELECT value FROM run_metadata WHERE key = 'candidate_snapshot_sha256'"
    ).fetchone()
    if previous_snapshot is None:
        connection.execute(
            "INSERT INTO run_metadata (key, value) VALUES (?, ?)",
            ("candidate_snapshot_sha256", candidate_snapshot_sha256),
        )
        connection.execute(
            "INSERT INTO run_metadata (key, value) VALUES (?, ?)",
            ("candidate_input_hashes", json.dumps(input_hashes, sort_keys=True)),
        )
        connection.commit()
    elif previous_snapshot[0] != candidate_snapshot_sha256:
        raise ValueError(
            "candidate snapshot changed; use a new output directory rather than "
            "resuming incompatible state"
        )
    expected_by_key = {
        record["s3_key"]: (
            record["bucket"],
            record["galaxy_id"],
            record["bytes"],
            record["etag"],
        )
        for record in candidates
    }
    for key, bucket, galaxy_id, expected_bytes, etag in connection.execute(
        "SELECT s3_key, bucket, galaxy_id, expected_bytes, etag FROM payloads"
    ):
        expected = expected_by_key.get(str(key))
        if expected is None:
            raise ValueError(f"database contains stale candidate key: {key}")
        if (bucket, galaxy_id, expected_bytes, etag) != expected:
            raise ValueError(f"database candidate tuple changed for key: {key}")
    done = existing_valid_keys(connection, arguments.output_dir)
    pending = [record for record in candidates if record["s3_key"] not in done]
    print(
        f"candidates={len(candidates)} resumed={len(done)} pending={len(pending)}",
        flush=True,
    )

    session = boto3.Session(profile_name=arguments.profile)
    client = session.client(
        "s3",
        config=Config(
            max_pool_connections=max(64, arguments.workers * 2),
            retries={"max_attempts": 12, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=60,
        ),
    )

    def fetch_body(record: dict[str, Any]) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            body_stream = None
            try:
                response = client.get_object(
                    Bucket="darkhaloscope",
                    Key=record["s3_key"],
                    IfMatch=f'"{record["etag"]}"',
                )
                body_stream = response["Body"]
                if int(response["ContentLength"]) != record["bytes"]:
                    raise ValueError("S3 ContentLength differs from frozen listing")
                if int(response["ContentLength"]) > MAX_COMPRESSED_BYTES:
                    raise ValueError("S3 object exceeds the compressed size limit")
                chunks: list[bytes] = []
                received = 0
                while True:
                    chunk = body_stream.read(64 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > record["bytes"]:
                        raise ValueError("S3 body exceeds the frozen byte count")
                    chunks.append(chunk)
                body = b"".join(chunks)
                if len(body) != record["bytes"]:
                    raise ValueError(
                        f"byte count {len(body)} differs from listing "
                        f"{record['bytes']}"
                    )
                return body
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < 3:
                    time.sleep(0.25 * (2 ** (attempt - 1)))
            finally:
                if body_stream is not None:
                    body_stream.close()
        assert last_error is not None
        raise last_error

    def process(record: dict[str, Any]) -> tuple[Any, ...]:
        bucket = record["bucket"]
        galaxy_id = record["galaxy_id"]
        key = record["s3_key"]
        relative = Path(bucket) / galaxy_id[:4] / f"{galaxy_id}.npz"
        destination = arguments.output_dir / relative
        try:
            destination.resolve().relative_to(arguments.output_dir.resolve())
        except ValueError as error:
            raise ValueError("candidate destination escapes output directory") from error
        try:
            body = fetch_body(record)
            md5 = hashlib.md5(body).hexdigest()  # noqa: S324
            if md5 != record["etag"]:
                raise ValueError(f"MD5 {md5} differs from S3 ETag {record['etag']}")
            sha256 = hashlib.sha256(body).hexdigest()
            metadata = validate_payload(body, bucket, galaxy_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.partial")
            temporary.write_bytes(body)
            temporary.replace(destination)
            return (
                key,
                bucket,
                galaxy_id,
                record["bytes"],
                record["etag"],
                relative.as_posix(),
                len(body),
                md5,
                sha256,
                1,
                None,
                metadata["ra"],
                metadata["dec"],
                metadata["brickname"],
                metadata["pixel_scale"],
                metadata["layer"],
                metadata["bands"],
                metadata["pipeline_version"],
                metadata["quality_ok"],
                metadata["has_g"],
                metadata["has_r"],
                metadata["has_z"],
                metadata["nobs_z"],
                metadata["tractor_type"],
                metadata["pool"],
                metadata["stored_split"],
                metadata["psfsize_r"],
                metadata["psfdepth_r"],
                metadata["galdepth_r"],
                metadata["ebv"],
                metadata["tier"],
                metadata["match_type"],
                metadata["nan_fraction"],
                metadata["central_nan_fraction"],
            )
        except Exception as error:  # noqa: BLE001
            return (
                key,
                bucket,
                galaxy_id,
                record["bytes"],
                record["etag"],
                None,
                None,
                None,
                None,
                0,
                f"{type(error).__name__}: {error}",
                *([None] * 23),
            )

    placeholders = ",".join("?" for _ in PAYLOAD_COLUMNS)
    insert = (
        f"INSERT OR REPLACE INTO payloads ({','.join(PAYLOAD_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    processed = 0
    failed = 0
    last_commit = 0
    executor = ThreadPoolExecutor(max_workers=arguments.workers)
    pending_iterator = iter(pending)
    in_flight = {
        executor.submit(process, record)
        for record in (
            next(pending_iterator, None)
            for _ in range(max(arguments.workers * 4, 1))
        )
        if record is not None
    }
    try:
        while in_flight:
            completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                connection.execute(insert, result)
                processed += 1
                failed += int(result[9] == 0)
                replacement = next(pending_iterator, None)
                if replacement is not None:
                    in_flight.add(executor.submit(process, replacement))
            if processed - last_commit >= 500:
                connection.commit()
                last_commit = processed
            if processed and processed % 5000 == 0:
                print(
                    f"processed={processed}/{len(pending)} failed={failed}",
                    flush=True,
                )
            if processed >= 100 and failed / processed > 0.20:
                raise RuntimeError(
                    "more than 20% of processed candidates failed; stopping for review"
                )
    except BaseException:
        connection.commit()
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    connection.commit()

    manifest = pd.read_sql_query(
        "SELECT * FROM payloads ORDER BY bucket, galaxy_id", connection
    )
    if set(manifest["s3_key"].astype(str)) != candidate_keys:
        raise ValueError("payload database key set differs from candidate snapshot")
    manifest_path = arguments.output_dir / "payload_manifest.parquet"
    temporary_manifest = arguments.output_dir / "payload_manifest.partial.parquet"
    manifest.to_parquet(temporary_manifest, index=False)
    temporary_manifest.replace(manifest_path)
    summary = {
        "schema_version": 1,
        "status": "payload_validation_complete",
        "candidates": len(candidates),
        "database_rows": len(manifest),
        "valid": int(manifest["valid"].sum()),
        "invalid": int((manifest["valid"] == 0).sum()),
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "candidate_input_hashes": input_hashes,
        "by_bucket": manifest.groupby("bucket")["valid"]
        .agg(["count", "sum"])
        .to_dict(orient="index"),
        "payload_bytes": int(manifest["payload_bytes"].fillna(0).sum()),
        "detector_scores_used": False,
    }
    summary_path = arguments.output_dir / "summary.json"
    temporary_summary = arguments.output_dir / "summary.partial.json"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    completion = {
        "schema_version": 1,
        "status": "complete" if summary["invalid"] == 0 else "complete_with_failures",
        "invalid": summary["invalid"],
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "payload_manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if summary["invalid"] == 0:
        temporary_completion = arguments.output_dir / "COMPLETE.partial.json"
        temporary_completion.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_completion.replace(completion_marker)
    else:
        failed_marker = arguments.output_dir / "FAILED.json"
        temporary_failure = arguments.output_dir / "FAILED.partial.json"
        temporary_failure.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_failure.replace(failed_marker)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
