#!/usr/bin/env python3
"""Fetch score-blind DR10 Tractor rows for proposed experiment hosts."""

from __future__ import annotations

import argparse
import email.utils
import fcntl
import hashlib
import io
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERVICE = "https://datalab.noirlab.edu/query/query"
ANONYMOUS_TOKEN = "anonymous.0.0.anon_access"
BRICK = re.compile(r"^[0-9]{4}[mp][0-9]{3}$")
MAX_RESPONSE_BYTES = 1_000_000
MAX_WORKERS = 16
MAX_COORDINATE_SEPARATION_ARCSEC = 0.05
COLUMNS = [
    "brickname",
    "objid",
    "brick_primary",
    "maskbits",
    "type",
    "ra",
    "dec",
    "flux_g",
    "flux_r",
    "flux_z",
    "nobs_g",
    "nobs_r",
    "nobs_z",
    "psfsize_g",
    "psfsize_r",
    "psfsize_z",
    "psfdepth_g",
    "psfdepth_r",
    "psfdepth_z",
    "galdepth_r",
    "sersic",
    "shape_r",
    "shape_e1",
    "shape_e2",
]
QUERY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        {"service": SERVICE, "columns": COLUMNS, "table": "ls_dr10.tractor"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_brick(
    brickname: str,
    objids: list[int],
    limiter: RateLimiter,
) -> pd.DataFrame:
    if not BRICK.fullmatch(brickname) or not objids:
        raise ValueError("invalid brick query")
    ids = ",".join(str(value) for value in sorted(set(objids)))
    sql = (
        f"SELECT {','.join(COLUMNS)} FROM ls_dr10.tractor "
        f"WHERE brickname='{brickname}' AND objid IN ({ids})"
    )
    url = SERVICE + "?" + urllib.parse.urlencode(
        {
            "sql": sql,
            "ofmt": "csv",
            "out": "",
            "async": "False",
            "drop": "False",
        }
    )
    request = urllib.request.Request(
        url,
        headers={"X-DL-AuthToken": ANONYMOUS_TOKEN},
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            limiter.wait()
            with urllib.request.urlopen(request, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > MAX_RESPONSE_BYTES:
                    raise ValueError("catalogue response exceeds the size limit")
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("catalogue response exceeds the size limit")
            frame = pd.read_csv(io.BytesIO(body))
            if list(frame.columns) != COLUMNS:
                raise ValueError("catalogue response has unexpected columns")
            return frame
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    retry_time = email.utils.parsedate_to_datetime(retry_after)
                    delay = max(0.0, retry_time.timestamp() - time.time())
            else:
                delay = float(2 ** (attempt - 1))
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(float(2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def initialize(
    path: Path,
    input_sha256: str,
    implementation_sha256: str,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            brickname TEXT PRIMARY KEY,
            requested_ids_json TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            rows_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    expected_metadata = {
        "input_sha256": input_sha256,
        "implementation_sha256": implementation_sha256,
        "query_contract_sha256": QUERY_CONTRACT_SHA256,
        "service": SERVICE,
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata").fetchall())
    if existing and existing != expected_metadata:
        raise ValueError("query database provenance differs from this invocation")
    if not existing:
        connection.executemany(
            "INSERT INTO metadata (key,value) VALUES (?,?)",
            sorted(expected_metadata.items()),
        )
    connection.commit()
    return connection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        type=Path,
        default=Path(
            "evaluation_dataset_prep/role_proposal/proposed_grec_hosts.parquet"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/tractor_hosts"),
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (arguments.output_dir / ".query.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another Tractor query process is already running") from error
    implementation_sha256 = sha256_file(Path(__file__).resolve())

    proposal_completion = arguments.hosts.parent / "ROLE_PROPOSAL_COMPLETE.json"
    proposal_summary_path = arguments.hosts.parent / "summary.json"
    if not proposal_completion.is_file() or not proposal_summary_path.is_file():
        raise ValueError("role-proposal provenance is missing")
    proposal_receipt = json.loads(proposal_completion.read_text())
    proposal_summary = json.loads(proposal_summary_path.read_text())
    input_sha256 = sha256_file(arguments.hosts)
    if (
        proposal_receipt.get("status")
        != "score_blind_role_proposal_complete_not_final"
        or
        proposal_receipt.get("proposed_grec_hosts_sha256") != input_sha256
        or proposal_summary.get("proposed_grec_hosts_sha256") != input_sha256
        or proposal_receipt.get("summary_sha256")
        != sha256_file(proposal_summary_path)
        or proposal_summary.get("status")
        != "score_blind_role_proposal_not_final_assignment"
    ):
        raise ValueError("host manifest does not match the role-proposal evidence")
    hosts = pd.read_parquet(arguments.hosts)
    required = {"galaxy_id", "brickname", "ra", "dec"}
    missing = sorted(required - set(hosts.columns))
    if missing or len(hosts) != 4_500 or hosts["galaxy_id"].duplicated().any():
        raise ValueError(f"invalid proposed host manifest; missing={missing}")
    score_columns = [
        column
        for column in hosts.columns
        if any(token in column.lower() for token in ("score", "logit", "probability"))
    ]
    if score_columns:
        raise ValueError(f"score-bearing host columns are forbidden: {score_columns}")
    if hosts[list(required)].isna().any().any():
        raise ValueError("proposed host manifest contains null identity fields")
    hosts = hosts.copy()
    hosts["objid"] = hosts["galaxy_id"].astype(str).str.rsplit("_", n=1).str[1]
    if not hosts["objid"].str.fullmatch(r"[0-9]+").all():
        raise ValueError("host galaxy ID does not end in a numeric objid")
    hosts["objid"] = hosts["objid"].astype(int)
    expected_ids = hosts.apply(
        lambda row: f"{row['brickname']}_{row['objid']}", axis=1
    )
    if not expected_ids.eq(hosts["galaxy_id"].astype(str)).all():
        raise ValueError("galaxy_id does not equal brickname_objid")
    if not hosts["brickname"].astype(str).str.fullmatch(BRICK).all():
        raise ValueError("host manifest contains an invalid brickname")
    if not (
        np.isfinite(hosts["ra"].to_numpy(float)).all()
        and np.isfinite(hosts["dec"].to_numpy(float)).all()
        and hosts["ra"].between(0.0, 360.0, inclusive="left").all()
        and hosts["dec"].between(-90.0, 90.0, inclusive="both").all()
    ):
        raise ValueError("host manifest contains invalid coordinates")
    expected_pairs = set(zip(hosts["brickname"], hosts["objid"], strict=True))
    groups = {
        str(brick): sorted(group["objid"].astype(int).tolist())
        for brick, group in hosts.groupby("brickname")
    }
    (arguments.output_dir / "COMPLETE.json").unlink(missing_ok=True)
    (arguments.output_dir / "FAILED.json").unlink(missing_ok=True)

    connection = initialize(
        arguments.output_dir / "tractor_queries.sqlite",
        input_sha256,
        implementation_sha256,
    )
    completed: set[str] = set()
    for brick, requested_json, rows_json in connection.execute(
        """
        SELECT brickname,requested_ids_json,rows_json
        FROM queries WHERE status='complete'
        """
    ):
        if brick not in groups or json.loads(requested_json) != groups[brick]:
            raise ValueError(f"resumed request differs for brick {brick}")
        frame = pd.read_json(io.StringIO(rows_json), orient="records")
        if list(frame.columns) != COLUMNS:
            raise ValueError(f"resumed response schema differs for brick {brick}")
        returned = set(zip(frame["brickname"], frame["objid"], strict=True))
        if returned != {(brick, value) for value in groups[brick]}:
            raise ValueError(f"resumed response membership differs for brick {brick}")
        if len(frame) != len(groups[brick]):
            raise ValueError(f"resumed response has duplicate rows for brick {brick}")
        completed.add(str(brick))
    pending = [(brick, ids) for brick, ids in groups.items() if brick not in completed]
    print(
        f"bricks={len(groups)} resumed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    limiter = RateLimiter(requests_per_second=4.0)

    def work(item: tuple[str, list[int]]) -> tuple[str, str, str | None, str | None]:
        brick, ids = item
        try:
            frame = query_brick(brick, ids, limiter)
            returned = set(zip(frame["brickname"], frame["objid"], strict=True))
            expected = {(brick, value) for value in ids}
            if returned != expected or len(frame) != len(ids):
                raise ValueError("catalogue response did not return the exact requested rows")
            return brick, "complete", None, frame.to_json(orient="records")
        except Exception as error:  # noqa: BLE001
            return brick, "failed", f"{type(error).__name__}: {error}", None

    executor = ThreadPoolExecutor(max_workers=arguments.workers)
    iterator = iter(pending)
    in_flight = {
        executor.submit(work, item)
        for item in (
            next(iterator, None) for _ in range(max(arguments.workers * 2, 1))
        )
        if item is not None
    }
    processed = 0
    failures = 0
    try:
        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                brick, status, error, rows_json = future.result()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO queries
                    (brickname, requested_ids_json, status, error, rows_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        brick,
                        json.dumps(groups[brick]),
                        status,
                        error,
                        rows_json,
                    ),
                )
                processed += 1
                failures += int(status == "failed")
                replacement = next(iterator, None)
                if replacement is not None:
                    in_flight.add(executor.submit(work, replacement))
            if processed % 50 == 0:
                connection.commit()
                print(
                    f"processed={processed}/{len(pending)} failures={failures}",
                    flush=True,
                )
            if processed >= 20 and failures / processed > 0.25:
                raise RuntimeError("catalogue query failure rate exceeds 25%")
    except BaseException:
        connection.commit()
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    connection.commit()

    records: list[dict[str, Any]] = []
    failed_queries: list[dict[str, str]] = []
    for brick, status, error, rows_json in connection.execute(
        "SELECT brickname,status,error,rows_json FROM queries ORDER BY brickname"
    ):
        if status != "complete":
            failed_queries.append({"brickname": brick, "error": error})
            continue
        records.extend(json.loads(rows_json))
    catalogue = pd.DataFrame(records)
    if not catalogue.empty:
        returned_pairs = set(
            zip(catalogue["brickname"], catalogue["objid"].astype(int), strict=True)
        )
        if returned_pairs - expected_pairs:
            raise ValueError("stored catalogue contains unrequested host rows")
        if len(catalogue) != len(returned_pairs):
            raise ValueError("stored catalogue contains duplicate host rows")
        complete_membership = returned_pairs == expected_pairs
        catalogue["_tractor_found"] = True
        catalogue = hosts.merge(
            catalogue,
            on=["brickname", "objid"],
            how="left",
            validate="one_to_one",
            suffixes=("_payload", "_tractor"),
        )
        ra_payload = np.deg2rad(catalogue["ra_payload"].to_numpy(float))
        ra_tractor = np.deg2rad(catalogue["ra_tractor"].to_numpy(float))
        dec_payload = np.deg2rad(catalogue["dec_payload"].to_numpy(float))
        dec_tractor = np.deg2rad(catalogue["dec_tractor"].to_numpy(float))
        delta_ra = np.arctan2(
            np.sin(ra_payload - ra_tractor),
            np.cos(ra_payload - ra_tractor),
        )
        delta_dec = dec_payload - dec_tractor
        haversine = np.sin(delta_dec / 2.0) ** 2 + (
            np.cos(dec_payload)
            * np.cos(dec_tractor)
            * np.sin(delta_ra / 2.0) ** 2
        )
        separation = (
            2.0
            * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
            * 180.0
            / np.pi
            * 3600.0
        )
        catalogue["payload_tractor_separation_arcsec"] = separation
        if (
            catalogue.loc[catalogue["_tractor_found"].fillna(False), "_tractor_found"]
            .eq(True)
            .all()
            and (
                not np.isfinite(
                    catalogue.loc[
                        catalogue["_tractor_found"].fillna(False),
                        "payload_tractor_separation_arcsec",
                    ]
                ).all()
                or np.nanmax(separation) > MAX_COORDINATE_SEPARATION_ARCSEC
            )
        ):
            raise ValueError("Tractor rows are missing or coordinates do not match")
    else:
        complete_membership = False

    complete = (
        not failed_queries
        and complete_membership
        and len(catalogue) == len(hosts)
        and catalogue["_tractor_found"].fillna(False).all()
    )
    output_name = "tractor_hosts.parquet" if complete else "tractor_hosts_incomplete.parquet"
    output = arguments.output_dir / output_name
    temporary = arguments.output_dir / output_name.replace(
        ".parquet", ".partial.parquet"
    )
    catalogue.to_parquet(temporary, index=False)
    temporary.replace(output)
    summary = {
        "schema_version": 1,
        "status": "complete" if complete else "complete_with_failures",
        "requested_hosts": len(hosts),
        "returned_hosts": int(catalogue["_tractor_found"].fillna(False).sum())
        if "_tractor_found" in catalogue
        else 0,
        "failed_queries": failed_queries,
        "score_columns_requested": False,
        "input_sha256": input_sha256,
        "query_contract_sha256": QUERY_CONTRACT_SHA256,
        "implementation_sha256": implementation_sha256,
        "maximum_coordinate_separation_arcsec": (
            float(np.max(catalogue["payload_tractor_separation_arcsec"]))
            if complete
            else None
        ),
        "output_sha256": sha256_file(output),
    }
    summary_path = arguments.output_dir / "summary.json"
    temporary_summary = arguments.output_dir / "summary.partial.json"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    if complete:
        temporary_completion = arguments.output_dir / "COMPLETE.partial.json"
        temporary_completion.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "summary_sha256": sha256_file(summary_path),
                    "output_sha256": sha256_file(output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_completion.replace(arguments.output_dir / "COMPLETE.json")
        return 0
    failed_marker = arguments.output_dir / "FAILED.json"
    temporary_failed = arguments.output_dir / "FAILED.partial.json"
    temporary_failed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete_with_failures",
                "summary_sha256": sha256_file(summary_path),
                "output_sha256": sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_failed.replace(failed_marker)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
