#!/usr/bin/env python3
"""Resume score-blind DR10 Tractor retrieval from authoritative brick FITS files.

The Data Lab query database is opened read-only.  Direct FITS files are
downloaded only for bricks without a complete Data Lab query, selected rows are
committed to a separate SQLite database, and full FITS files are always removed
after extraction.
"""

from __future__ import annotations

import argparse
import email.utils
import fcntl
import hashlib
import http.client
import io
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from astropy.io import fits

EXPECTED_PROPOSAL_SHA256 = (
    "d5f8f72f7695598523642d858edee84713e48f4bd29504bcdc7bebd1b7a54255"
)
EXPECTED_DATALAB_QUERY_CONTRACT_SHA256 = (
    "3ce53ae2344066e0ffe2bc763154a67ed992e0043d559f3bd7415e61cb32a458"
)
EXPECTED_DATALAB_SERVICE = "https://datalab.noirlab.edu/query/query"
FITS_BASE_URL = (
    "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/tractor"
)
BRICK_PATTERN = re.compile(r"^[0-9]{4}[mp][0-9]{3}$")
MAX_WORKERS = 8
MAX_COORDINATE_SEPARATION_ARCSEC = 0.05
DEFAULT_MAX_FITS_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
ALLOWED_MORPHOLOGIES = ("DEV", "REX", "SER")
PROHIBITED_MASK_BIT_POSITIONS = (1, 5, 6, 7, 11, 12, 13)
PROHIBITED_MASK = sum(1 << position for position in PROHIBITED_MASK_BIT_POSITIONS)
MINIMUM_NOBS = 3
OBSERVED_Z_MAG_LIMIT = 20.0
DECLINATION_MIN_DEGREES = -18.0
DECLINATION_MAX_DEGREES = 32.0
USER_AGENT = (
    "stronglens-realism-gap-tractor-fallback/1.0 "
    "(score-blind research catalog retrieval; bounded concurrency)"
)

COLUMNS = (
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
)
INTEGER_COLUMNS = {
    "objid",
    "brick_primary",
    "maskbits",
    "nobs_g",
    "nobs_r",
    "nobs_z",
}
TEXT_COLUMNS = {"brickname", "type"}
FLOAT_COLUMNS = set(COLUMNS) - INTEGER_COLUMNS - TEXT_COLUMNS
REQUIRED_FINITE_FLUX_COLUMNS = ("flux_g", "flux_r", "flux_z")
REQUIRED_FINITE_MODEL_COLUMNS = (
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
)
FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"(^|_)(score|scores|logit|logits|prob|probability|prediction)(_|$)",
    re.IGNORECASE,
)
ELIGIBILITY_CONTRACT = {
    "allowed_morphologies": ALLOWED_MORPHOLOGIES,
    "brick_primary_required": True,
    "coordinate_tolerance_arcsec": MAX_COORDINATE_SEPARATION_ARCSEC,
    "declination_open_interval_degrees": (
        DECLINATION_MIN_DEGREES,
        DECLINATION_MAX_DEGREES,
    ),
    "finite_flux_columns": REQUIRED_FINITE_FLUX_COLUMNS,
    "finite_model_columns": REQUIRED_FINITE_MODEL_COLUMNS,
    "minimum_exact_integer_nobs": MINIMUM_NOBS,
    "nonpositive_flux_z_exempt_from_observed_zmag": True,
    "observed_zmag_formula": "22.5-2.5*log10(flux_z)",
    "observed_zmag_strict_upper_limit": OBSERVED_Z_MAG_LIMIT,
    "prohibited_mask_bit_positions_zero_indexed": PROHIBITED_MASK_BIT_POSITIONS,
}
ELIGIBILITY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        ELIGIBILITY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
FALLBACK_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "base_url": FITS_BASE_URL,
            "columns": COLUMNS,
            "coordinate_tolerance_arcsec": MAX_COORDINATE_SEPARATION_ARCSEC,
            "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
            "expected_proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "receipt_hash_required": True,
            "resume_revalidates_complete_receipts": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class ContractError(ValueError):
    """A provenance, schema, membership, or science contract was violated."""


class SizeLimitError(ValueError):
    """A remote FITS file exceeded the configured byte cap."""


class ContentLengthMismatchError(IOError):
    """A response body did not match its declared Content-Length."""


class FetchFailure(RuntimeError):
    """A download failed after its permitted attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]],
        started_at: str,
        completed_at: str,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.started_at = started_at
        self.completed_at = completed_at


@dataclass(frozen=True)
class Proposal:
    frame: pd.DataFrame
    groups: dict[str, list[int]]
    coordinates: dict[tuple[str, int], tuple[float, float]]
    sha256: str


@dataclass(frozen=True)
class DataLabSnapshot:
    rows: dict[tuple[str, int], dict[str, Any]]
    complete_bricks: set[str]
    query_status: dict[str, tuple[str, str | None]]
    logical_sha256: str
    dispositions: dict[tuple[str, int], dict[str, Any]]
    selected_rows: int


@dataclass(frozen=True)
class DownloadedFile:
    path: Path
    requested_url: str
    response_url: str
    http_status: int
    etag: str | None
    last_modified: str | None
    content_length: int | None
    downloaded_bytes: int
    sha256: str
    attempts: list[dict[str, Any]]
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class BrickResult:
    brickname: str
    expected_objids: list[int]
    status: str
    rows: list[dict[str, Any]]
    dispositions: list[dict[str, Any]]
    receipt: dict[str, Any]


class StartRateLimiter:
    """Bound aggregate request starts across downloader threads."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(requests_per_second) or requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive and finite")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._interval
        if delay:
            self._sleeper(delay)


def utc_now() -> str:
    return (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def forbidden_columns(columns: Iterable[str]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if FORBIDDEN_COLUMN_PATTERN.search(str(column))
    )


def normalize_scalar(value: Any) -> Any:
    if np.ma.is_masked(value) or value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("ascii", errors="strict").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def is_exact_integer(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer()
    return False


def is_finite_number(value: Any) -> bool:
    return (
        value is not None
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def normalize_tractor_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if set(row) != set(COLUMNS):
        missing = sorted(set(COLUMNS) - set(row))
        extra = sorted(set(row) - set(COLUMNS))
        raise ContractError(
            f"Tractor row schema differs; missing={missing}, extra={extra}"
        )
    normalized: dict[str, Any] = {}
    for column in COLUMNS:
        value = normalize_scalar(row[column])
        if column in TEXT_COLUMNS:
            normalized[column] = None if value is None else str(value).strip()
        elif column in INTEGER_COLUMNS:
            if column == "brick_primary" and isinstance(value, bool):
                normalized[column] = int(value)
            else:
                normalized[column] = int(value) if is_exact_integer(value) else value
        else:
            try:
                normalized[column] = None if value is None else float(value)
            except (TypeError, ValueError):
                normalized[column] = value
    return normalized


def angular_separation_arcsec(
    ra_1_degrees: float,
    dec_1_degrees: float,
    ra_2_degrees: float,
    dec_2_degrees: float,
) -> float:
    ra_1 = math.radians(ra_1_degrees)
    ra_2 = math.radians(ra_2_degrees)
    dec_1 = math.radians(dec_1_degrees)
    dec_2 = math.radians(dec_2_degrees)
    delta_ra = math.atan2(math.sin(ra_1 - ra_2), math.cos(ra_1 - ra_2))
    delta_dec = dec_1 - dec_2
    haversine = math.sin(delta_dec / 2.0) ** 2 + (
        math.cos(dec_1) * math.cos(dec_2) * math.sin(delta_ra / 2.0) ** 2
    )
    return (
        2.0
        * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))
        * 180.0
        / math.pi
        * 3600.0
    )


def validate_selected_rows(
    brickname: str,
    expected_objids: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    coordinates: Mapping[tuple[str, int], tuple[float, float]],
    *,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    expected = {(brickname, int(objid)) for objid in expected_objids}
    normalized = [normalize_tractor_row(row) for row in rows]
    returned: set[tuple[str, int]] = set()
    for row in normalized:
        if row["brickname"] is None or not is_exact_integer(row["objid"]):
            raise ContractError(f"{brickname}: selected row identity is malformed")
        returned.add((str(row["brickname"]), int(row["objid"])))
    if len(normalized) != len(expected) or len(returned) != len(normalized):
        raise ContractError(
            f"{brickname}: selected row cardinality/uniqueness differs "
            f"(expected {len(expected)}, got {len(normalized)})"
        )
    if returned != expected:
        missing = sorted(expected - returned)
        extra = sorted(returned - expected)
        raise ContractError(
            f"{brickname}: selected membership differs; missing={missing}, extra={extra}"
        )

    normalized.sort(key=lambda row: int(row["objid"]))
    dispositions = [
        evaluate_eligibility(row, coordinates, source=source) for row in normalized
    ]
    separations = [
        float(disposition["payload_tractor_separation_arcsec"])
        for disposition in dispositions
        if disposition["payload_tractor_separation_arcsec"] is not None
    ]
    return normalized, dispositions, max(separations, default=0.0)


def tractor_row_sha256(row: Mapping[str, Any]) -> str:
    normalized = normalize_tractor_row(row)
    return hashlib.sha256(
        canonical_json({column: normalized[column] for column in COLUMNS}).encode(
            "utf-8"
        )
    ).hexdigest()


def evaluate_eligibility(
    row: Mapping[str, Any],
    coordinates: Mapping[tuple[str, int], tuple[float, float]],
    *,
    source: str,
) -> dict[str, Any]:
    normalized = normalize_tractor_row(row)
    brick = str(normalized["brickname"])
    objid = int(normalized["objid"])
    pair = (brick, objid)

    morphology = (
        str(normalized["type"]).upper() if normalized["type"] is not None else None
    )
    passes_morphology = morphology in ALLOWED_MORPHOLOGIES
    passes_brick_primary = normalized["brick_primary"] is True or (
        is_exact_integer(normalized["brick_primary"])
        and int(normalized["brick_primary"]) == 1
    )

    nobs_passes = {
        column: (
            is_exact_integer(normalized[column])
            and int(normalized[column]) >= MINIMUM_NOBS
        )
        for column in ("nobs_g", "nobs_r", "nobs_z")
    }
    flux_passes = {
        column: is_finite_number(normalized[column])
        for column in REQUIRED_FINITE_FLUX_COLUMNS
    }
    model_passes = {
        column: is_finite_number(normalized[column])
        for column in REQUIRED_FINITE_MODEL_COLUMNS
    }

    maskbits = normalized["maskbits"]
    maskbits_exact = is_exact_integer(maskbits) and int(maskbits) >= 0
    present_mask_positions = (
        [
            position
            for position in PROHIBITED_MASK_BIT_POSITIONS
            if int(maskbits) & (1 << position)
        ]
        if maskbits_exact
        else []
    )
    passes_maskbits = maskbits_exact and not present_mask_positions

    flux_z = normalized["flux_z"]
    flux_z_finite = is_finite_number(flux_z)
    observed_zmag: float | None = None
    observed_zmag_exempt = False
    if flux_z_finite and float(flux_z) > 0.0:
        observed_zmag = 22.5 - 2.5 * math.log10(float(flux_z))
        passes_observed_zmag = observed_zmag < OBSERVED_Z_MAG_LIMIT
    elif flux_z_finite:
        observed_zmag_exempt = True
        passes_observed_zmag = True
    else:
        passes_observed_zmag = False

    ra = normalized["ra"]
    dec = normalized["dec"]
    passes_finite_coordinates = (
        is_finite_number(ra)
        and is_finite_number(dec)
        and 0.0 <= float(ra) < 360.0
        and -90.0 <= float(dec) <= 90.0
    )
    passes_declination_window = (
        passes_finite_coordinates
        and DECLINATION_MIN_DEGREES < float(dec) < DECLINATION_MAX_DEGREES
    )
    separation: float | None = None
    if passes_finite_coordinates and pair in coordinates:
        payload_ra, payload_dec = coordinates[pair]
        separation = angular_separation_arcsec(
            payload_ra, payload_dec, float(ra), float(dec)
        )
    passes_coordinate_match = (
        separation is not None
        and math.isfinite(separation)
        and separation <= MAX_COORDINATE_SEPARATION_ARCSEC
    )

    predicates = {
        "brick_primary": passes_brick_primary,
        "coordinate_match": passes_coordinate_match,
        "declination_window": passes_declination_window,
        "finite_coordinates": passes_finite_coordinates,
        "finite_flux_g": flux_passes["flux_g"],
        "finite_flux_r": flux_passes["flux_r"],
        "finite_flux_z": flux_passes["flux_z"],
        "finite_required_model_fields": all(model_passes.values()),
        "maskbits": passes_maskbits,
        "morphology": passes_morphology,
        "nobs_g": nobs_passes["nobs_g"],
        "nobs_r": nobs_passes["nobs_r"],
        "nobs_z": nobs_passes["nobs_z"],
        "observed_zmag": passes_observed_zmag,
    }
    reasons = [f"failed_{name}" for name, passed in predicates.items() if not passed]
    disposition: dict[str, Any] = {
        "brickname": brick,
        "disposition_reasons": reasons,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "eligible": all(predicates.values()),
        "flux_z_nonpositive_magnitude_exemption": observed_zmag_exempt,
        "morphology": morphology,
        "nonfinite_required_model_fields": [
            column for column, passed in model_passes.items() if not passed
        ],
        "objid": objid,
        "observed_zmag": observed_zmag,
        "passes": predicates,
        "payload_tractor_separation_arcsec": separation,
        "prohibited_mask_bit_positions_present": present_mask_positions,
        "row_sha256": tractor_row_sha256(normalized),
        "source": source,
    }
    disposition["disposition_sha256"] = hashlib.sha256(
        canonical_json(disposition).encode("utf-8")
    ).hexdigest()
    return disposition


def selected_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {column: normalize_tractor_row(row)[column] for column in COLUMNS}
        for row in rows
    ]
    normalized.sort(key=lambda row: (str(row["brickname"]), int(row["objid"])))
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def dispositions_sha256(dispositions: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        dispositions,
        key=lambda item: (str(item["brickname"]), int(item["objid"])),
    )
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def load_proposal(
    hosts_path: Path,
    *,
    expected_sha256: str = EXPECTED_PROPOSAL_SHA256,
) -> Proposal:
    if not hosts_path.is_file():
        raise ContractError(f"proposal parquet does not exist: {hosts_path}")
    proposal_sha256 = sha256_file(hosts_path)
    if proposal_sha256 != expected_sha256:
        raise ContractError(
            "role proposal SHA-256 differs from the exact authorized input: "
            f"{proposal_sha256}"
        )

    completion_path = hosts_path.parent / "ROLE_PROPOSAL_COMPLETE.json"
    summary_path = hosts_path.parent / "summary.json"
    if not completion_path.is_file() or not summary_path.is_file():
        raise ContractError("role-proposal completion evidence is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "score_blind_role_proposal_complete_not_final"
        or completion.get("proposed_grec_hosts_sha256") != proposal_sha256
        or completion.get("summary_sha256") != sha256_file(summary_path)
        or summary.get("status") != "score_blind_role_proposal_not_final_assignment"
        or summary.get("proposed_grec_hosts_sha256") != proposal_sha256
        or summary.get("detector_scores_used") is not False
    ):
        raise ContractError("role-proposal evidence does not authorize this parquet")

    frame = pd.read_parquet(hosts_path)
    required = {"galaxy_id", "brickname", "ra", "dec"}
    missing = sorted(required - set(frame.columns))
    score_columns = forbidden_columns(frame.columns)
    if missing or score_columns:
        raise ContractError(
            f"invalid proposal columns; missing={missing}, "
            f"score-bearing={score_columns}"
        )
    if len(frame) != 4_500:
        raise ContractError(f"proposal must contain 4,500 hosts, got {len(frame)}")
    if frame["galaxy_id"].duplicated().any():
        raise ContractError("proposal galaxy_id values are not unique")
    if frame[list(required)].isna().any().any():
        raise ContractError("proposal contains null host identity fields")
    if not frame["brickname"].astype(str).str.fullmatch(BRICK_PATTERN).all():
        raise ContractError("proposal contains an invalid brickname")

    objid_text = frame["galaxy_id"].astype(str).str.rsplit("_", n=1).str[-1]
    if not objid_text.str.fullmatch(r"[0-9]+").all():
        raise ContractError("proposal galaxy_id does not end in a numeric objid")
    objids = objid_text.astype("int64")
    expected_ids = frame["brickname"].astype(str) + "_" + objids.astype(str)
    if not expected_ids.eq(frame["galaxy_id"].astype(str)).all():
        raise ContractError("proposal galaxy_id is not exactly brickname_objid")
    if "objid" in frame.columns:
        if not frame["objid"].astype("int64").eq(objids).all():
            raise ContractError("proposal objid column conflicts with galaxy_id")
    else:
        frame = frame.copy()
        frame["objid"] = objids

    ra = frame["ra"].to_numpy(dtype=float)
    dec = frame["dec"].to_numpy(dtype=float)
    if not (
        np.isfinite(ra).all()
        and np.isfinite(dec).all()
        and ((ra >= 0.0) & (ra < 360.0)).all()
        and ((dec >= -90.0) & (dec <= 90.0)).all()
    ):
        raise ContractError("proposal coordinates are invalid")
    pairs = list(
        zip(
            frame["brickname"].astype(str),
            frame["objid"].astype(int),
            strict=True,
        )
    )
    if len(set(pairs)) != len(pairs):
        raise ContractError("proposal (brickname, objid) pairs are not unique")

    groups = {
        str(brick): sorted(group["objid"].astype(int).tolist())
        for brick, group in frame.groupby("brickname", sort=True)
    }
    coordinates = {
        pair: (float(payload_ra), float(payload_dec))
        for pair, payload_ra, payload_dec in zip(pairs, ra, dec, strict=True)
    }
    return Proposal(frame, groups, coordinates, proposal_sha256)


def open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ContractError(f"Data Lab SQLite database does not exist: {path}")
    quoted = urllib.parse.quote(str(path.resolve()), safe="/")
    connection = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_datalab_snapshot(
    database_path: Path,
    proposal: Proposal,
) -> DataLabSnapshot:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    dispositions: dict[tuple[str, int], dict[str, Any]] = {}
    complete_bricks: set[str] = set()
    query_status: dict[str, tuple[str, str | None]] = {}
    digest = hashlib.sha256()
    selected_rows = 0
    with open_sqlite_read_only(database_path) as connection:
        connection.execute("BEGIN")
        metadata = dict(
            connection.execute("SELECT key,value FROM metadata ORDER BY key").fetchall()
        )
        expected_metadata = {
            "input_sha256": proposal.sha256,
            "query_contract_sha256": EXPECTED_DATALAB_QUERY_CONTRACT_SHA256,
            "service": EXPECTED_DATALAB_SERVICE,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ContractError(
                    f"Data Lab metadata {key!r} differs from {expected!r}"
                )

        records = connection.execute(
            """
            SELECT brickname,requested_ids_json,status,error,rows_json
            FROM queries ORDER BY brickname
            """
        ).fetchall()
        for brick_raw, requested_json, status_raw, error, rows_json in records:
            brick = str(brick_raw)
            status = str(status_raw)
            if brick not in proposal.groups:
                raise ContractError(f"Data Lab database has unrequested brick {brick}")
            try:
                requested = [int(value) for value in json.loads(requested_json)]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"Data Lab requested IDs are invalid for {brick}"
                ) from exc
            if requested != proposal.groups[brick]:
                raise ContractError(
                    f"Data Lab requested IDs differ from proposal for {brick}"
                )
            query_status[brick] = (status, None if error is None else str(error))
            if status != "complete":
                continue
            if rows_json is None:
                raise ContractError(f"complete Data Lab query lacks rows for {brick}")
            try:
                decoded = json.loads(rows_json)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"complete Data Lab rows JSON is invalid for {brick}"
                ) from exc
            if not isinstance(decoded, list):
                raise ContractError(f"Data Lab rows are not a list for {brick}")
            validated, brick_dispositions, _ = validate_selected_rows(
                brick,
                requested,
                decoded,
                proposal.coordinates,
                source="datalab",
            )
            for row, disposition in zip(validated, brick_dispositions, strict=True):
                pair = (str(row["brickname"]), int(row["objid"]))
                if pair in dispositions:
                    raise ContractError(f"duplicate Data Lab row {pair}")
                dispositions[pair] = disposition
                if disposition["eligible"]:
                    rows[pair] = row
                selected_rows += 1
            complete_bricks.add(brick)
            digest.update(
                canonical_json(
                    {
                        "brickname": brick,
                        "dispositions": brick_dispositions,
                        "requested_objids": requested,
                        "rows": validated,
                    }
                ).encode("utf-8")
            )
    return DataLabSnapshot(
        rows,
        complete_bricks,
        query_status,
        digest.hexdigest(),
        dispositions,
        selected_rows,
    )


def tractor_url(brickname: str) -> str:
    if not BRICK_PATTERN.fullmatch(brickname):
        raise ContractError(f"invalid brickname: {brickname}")
    return f"{FITS_BASE_URL}/{brickname[:3]}/tractor-{brickname}.fits"


def retry_after_seconds(headers: Any, default: float) -> float:
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return default
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            retry_time = email.utils.parsedate_to_datetime(str(value))
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=UTC)
            delay = retry_time.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return default
    return min(120.0, max(0.0, delay))


def download_with_retries(
    url: str,
    spool_directory: Path,
    *,
    max_bytes: int,
    timeout_seconds: float,
    retries: int,
    limiter: Any,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> DownloadedFile:
    spool_directory.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    attempts: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    for attempt_number in range(1, retries + 1):
        temporary_path: Path | None = None
        attempt_started = utc_now()
        headers: Any = None
        status: int | None = None
        try:
            limiter.wait()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/fits,application/octet-stream;q=0.9",
                    "Accept-Encoding": "identity",
                    "User-Agent": USER_AGENT,
                },
            )
            with opener(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                headers = response.headers
                encoding = headers.get("Content-Encoding")
                if encoding and str(encoding).lower() != "identity":
                    raise ContractError(f"unexpected Content-Encoding {encoding!r}")
                content_length_text = headers.get("Content-Length")
                content_length: int | None = None
                if content_length_text is not None:
                    try:
                        content_length = int(content_length_text)
                    except (TypeError, ValueError) as exc:
                        raise ContractError(
                            "remote Content-Length is not an integer"
                        ) from exc
                    if content_length < 0:
                        raise ContractError("remote Content-Length is negative")
                    if content_length > max_bytes:
                        raise SizeLimitError(
                            f"declared FITS size {content_length} exceeds "
                            f"{max_bytes}-byte cap"
                        )

                descriptor, name = tempfile.mkstemp(
                    prefix="tractor-",
                    suffix=".fits.partial",
                    dir=spool_directory,
                )
                temporary_path = Path(name)
                digest = hashlib.sha256()
                downloaded = 0
                with os.fdopen(descriptor, "wb") as output:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise SizeLimitError(
                                f"FITS body exceeds {max_bytes}-byte cap"
                            )
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if content_length is not None and downloaded != content_length:
                    raise ContentLengthMismatchError(
                        f"declared {content_length} bytes but read {downloaded}"
                    )
                completed_at = utc_now()
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "completed_at": completed_at,
                        "downloaded_bytes": downloaded,
                        "http_status": status,
                        "started_at": attempt_started,
                        "status": "complete",
                    }
                )
                return DownloadedFile(
                    path=temporary_path,
                    requested_url=url,
                    response_url=str(
                        response.geturl() if hasattr(response, "geturl") else url
                    ),
                    http_status=status,
                    etag=headers.get("ETag"),
                    last_modified=headers.get("Last-Modified"),
                    content_length=content_length,
                    downloaded_bytes=downloaded,
                    sha256=digest.hexdigest(),
                    attempts=attempts,
                    started_at=started_at,
                    completed_at=completed_at,
                )
        except urllib.error.HTTPError as exc:
            last_error = exc
            status = exc.code
            headers = exc.headers
            retryable = exc.code in RETRYABLE_HTTP_CODES
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionResetError,
            http.client.IncompleteRead,
            ContentLengthMismatchError,
        ) as exc:
            last_error = exc
            retryable = True
        except (SizeLimitError, ContractError) as exc:
            last_error = exc
            retryable = False
        except Exception as exc:  # noqa: BLE001 - converted to a download receipt
            last_error = exc
            retryable = False

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        error_text = f"{type(last_error).__name__}: {last_error}"
        attempt_record: dict[str, Any] = {
            "attempt": attempt_number,
            "completed_at": utc_now(),
            "error": error_text,
            "http_status": status,
            "started_at": attempt_started,
            "status": "failed",
        }
        should_retry = retryable and attempt_number < retries
        if should_retry:
            default_delay = min(30.0, 2.0 ** (attempt_number - 1))
            delay = retry_after_seconds(headers, default_delay)
            delay += min(1.0, max(0.0, float(jitter())))
            attempt_record["retry_delay_seconds"] = delay
        attempts.append(attempt_record)
        if not should_retry:
            completed_at = utc_now()
            raise FetchFailure(
                error_text,
                attempts=attempts,
                started_at=started_at,
                completed_at=completed_at,
            ) from last_error
        sleeper(delay)

    raise AssertionError("download retry loop exited unexpectedly")


def extract_required_rows(
    fits_path: Path,
    brickname: str,
    expected_objids: Sequence[int],
    coordinates: Mapping[tuple[str, int], tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    required_upper = {column.upper() for column in COLUMNS}
    with fits.open(
        fits_path,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as hdus:
        candidates: list[tuple[Any, dict[str, str]]] = []
        for hdu in hdus:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            names = list(hdu.columns.names or [])
            mapping = {str(name).upper(): str(name) for name in names}
            if len(mapping) != len(names):
                raise ContractError("FITS table has case-insensitive duplicate columns")
            if required_upper.issubset(mapping):
                candidates.append((hdu, mapping))
        if len(candidates) != 1:
            raise ContractError(
                f"{brickname}: expected exactly one FITS table with required "
                f"columns, found {len(candidates)}"
            )
        table, column_mapping = candidates[0]
        data = table.data
        file_objids = np.asarray(data[column_mapping["OBJID"]], dtype=np.int64)
        positions = np.flatnonzero(
            np.isin(file_objids, np.asarray(expected_objids, dtype=np.int64))
        )
        selected: list[dict[str, Any]] = []
        for position in positions:
            selected.append(
                {
                    column: normalize_scalar(
                        data[column_mapping[column.upper()]][position]
                    )
                    for column in COLUMNS
                }
            )
    return validate_selected_rows(
        brickname,
        expected_objids,
        selected,
        coordinates,
        source="nersc_fits",
    )


def process_brick(
    brickname: str,
    expected_objids: list[int],
    proposal: Proposal,
    *,
    spool_directory: Path,
    max_bytes: int,
    timeout_seconds: float,
    retries: int,
    limiter: StartRateLimiter,
) -> BrickResult:
    url = tractor_url(brickname)
    downloaded: DownloadedFile | None = None
    try:
        downloaded = download_with_retries(
            url,
            spool_directory,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            retries=retries,
            limiter=limiter,
        )
        rows, dispositions, maximum_separation = extract_required_rows(
            downloaded.path,
            brickname,
            expected_objids,
            proposal.coordinates,
        )
        ineligible = [
            disposition for disposition in dispositions if not disposition["eligible"]
        ]
        status = "failed" if ineligible else "complete"
        error = (
            "eligibility_failed_replacement_required: "
            + "; ".join(
                f"{item['brickname']}_{item['objid']}="
                + ",".join(item["disposition_reasons"])
                for item in ineligible
            )
            if ineligible
            else None
        )
        receipt = {
            "attempt_count": len(downloaded.attempts),
            "attempts_json": canonical_json(downloaded.attempts),
            "completed_at": downloaded.completed_at,
            "content_length": downloaded.content_length,
            "downloaded_bytes": downloaded.downloaded_bytes,
            "error": error,
            "etag": downloaded.etag,
            "fits_sha256": downloaded.sha256,
            "full_fits_retained": 0,
            "http_status": downloaded.http_status,
            "last_modified": downloaded.last_modified,
            "maximum_coordinate_separation_arcsec": maximum_separation,
            "response_url": downloaded.response_url,
            "row_count": len(rows),
            "started_at": downloaded.started_at,
            "status": status,
            "url": downloaded.requested_url,
        }
        return BrickResult(
            brickname,
            expected_objids,
            status,
            rows,
            dispositions,
            receipt,
        )
    except FetchFailure as exc:
        receipt = {
            "attempt_count": len(exc.attempts),
            "attempts_json": canonical_json(exc.attempts),
            "completed_at": exc.completed_at,
            "content_length": None,
            "downloaded_bytes": None,
            "error": f"{type(exc.__cause__).__name__}: {exc.__cause__}",
            "etag": None,
            "fits_sha256": None,
            "full_fits_retained": 0,
            "http_status": next(
                (
                    attempt["http_status"]
                    for attempt in reversed(exc.attempts)
                    if attempt.get("http_status") is not None
                ),
                None,
            ),
            "last_modified": None,
            "maximum_coordinate_separation_arcsec": None,
            "response_url": None,
            "row_count": 0,
            "started_at": exc.started_at,
            "status": "failed",
            "url": url,
        }
        return BrickResult(brickname, expected_objids, "failed", [], [], receipt)
    except Exception as exc:  # noqa: BLE001 - scientific/FITS failure receipt
        now = utc_now()
        receipt = {
            "attempt_count": len(downloaded.attempts) if downloaded else 0,
            "attempts_json": canonical_json(downloaded.attempts if downloaded else []),
            "completed_at": now,
            "content_length": downloaded.content_length if downloaded else None,
            "downloaded_bytes": (downloaded.downloaded_bytes if downloaded else None),
            "error": f"{type(exc).__name__}: {exc}",
            "etag": downloaded.etag if downloaded else None,
            "fits_sha256": downloaded.sha256 if downloaded else None,
            "full_fits_retained": 0,
            "http_status": downloaded.http_status if downloaded else None,
            "last_modified": downloaded.last_modified if downloaded else None,
            "maximum_coordinate_separation_arcsec": None,
            "response_url": downloaded.response_url if downloaded else None,
            "row_count": 0,
            "started_at": downloaded.started_at if downloaded else now,
            "status": "failed",
            "url": url,
        }
        return BrickResult(brickname, expected_objids, "failed", [], [], receipt)
    finally:
        if downloaded is not None:
            downloaded.path.unlink(missing_ok=True)


def initialize_fallback_database(
    path: Path,
    *,
    proposal: Proposal,
    datalab: DataLabSnapshot,
    implementation_sha256: str,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    expected_metadata = {
        "datalab_complete_logical_sha256": datalab.logical_sha256,
        "datalab_query_contract_sha256": EXPECTED_DATALAB_QUERY_CONTRACT_SHA256,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "fallback_contract_sha256": FALLBACK_CONTRACT_SHA256,
        "fits_base_url": FITS_BASE_URL,
        "implementation_sha256": implementation_sha256,
        "proposal_sha256": proposal.sha256,
        "schema_version": "2",
    }
    existing = dict(
        connection.execute("SELECT key,value FROM metadata ORDER BY key").fetchall()
    )
    if existing and existing != expected_metadata:
        differing = sorted(
            key
            for key in set(existing) | set(expected_metadata)
            if existing.get(key) != expected_metadata.get(key)
        )
        connection.close()
        raise ContractError(
            f"fallback database provenance differs for keys: {differing}"
        )
    if not existing:
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(expected_metadata.items()),
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS invocations (
            invocation_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            arguments_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invocation_id TEXT NOT NULL REFERENCES invocations(invocation_id),
            brickname TEXT NOT NULL,
            expected_objids_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('complete','failed')),
            url TEXT NOT NULL,
            response_url TEXT,
            http_status INTEGER,
            etag TEXT,
            last_modified TEXT,
            content_length INTEGER,
            downloaded_bytes INTEGER,
            fits_sha256 TEXT,
            row_count INTEGER NOT NULL,
            eligibility_passed_count INTEGER NOT NULL,
            maximum_coordinate_separation_arcsec REAL,
            attempt_count INTEGER NOT NULL,
            attempts_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            error TEXT,
            full_fits_retained INTEGER NOT NULL CHECK(full_fits_retained = 0),
            source_contract_sha256 TEXT NOT NULL,
            selected_rows_sha256 TEXT NOT NULL,
            eligibility_dispositions_sha256 TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL,
            resume_state TEXT NOT NULL DEFAULT 'active'
                CHECK(resume_state IN ('active','stale')),
            resume_validation_error TEXT,
            last_resume_validation_at TEXT,
            UNIQUE(invocation_id, brickname)
        )
        """
    )
    row_columns = """
        brickname TEXT NOT NULL,
        objid INTEGER NOT NULL,
        brick_primary INTEGER NOT NULL CHECK(brick_primary = 1),
        maskbits INTEGER,
        type TEXT,
        ra REAL NOT NULL,
        dec REAL NOT NULL,
        flux_g REAL,
        flux_r REAL,
        flux_z REAL,
        nobs_g INTEGER,
        nobs_r INTEGER,
        nobs_z INTEGER,
        psfsize_g REAL,
        psfsize_r REAL,
        psfsize_z REAL,
        psfdepth_g REAL,
        psfdepth_r REAL,
        psfdepth_z REAL,
        galdepth_r REAL,
        sersic REAL,
        shape_r REAL,
        shape_e1 REAL,
        shape_e2 REAL
    """.strip()
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS fallback_rows (
            receipt_id INTEGER NOT NULL REFERENCES receipts(receipt_id),
            {row_columns},
            payload_tractor_separation_arcsec REAL NOT NULL,
            source_fits_sha256 TEXT NOT NULL,
            source_etag TEXT NOT NULL,
            source_downloaded_bytes INTEGER NOT NULL,
            source_receipt_sha256 TEXT NOT NULL,
            PRIMARY KEY(brickname, objid)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS eligibility_dispositions (
            source_record_key TEXT PRIMARY KEY,
            receipt_id INTEGER REFERENCES receipts(receipt_id),
            source TEXT NOT NULL CHECK(source IN ('datalab','nersc_fits')),
            brickname TEXT NOT NULL,
            objid INTEGER NOT NULL,
            eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
            row_sha256 TEXT NOT NULL,
            eligibility_contract_sha256 TEXT NOT NULL,
            disposition_sha256 TEXT NOT NULL,
            disposition_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS one_complete_receipt_per_brick
        ON receipts(brickname)
        WHERE status='complete' AND resume_state='active'
        """
    )
    store_datalab_dispositions(connection, datalab)
    connection.commit()
    return connection


def disposition_record_values(
    source_record_key: str,
    disposition: Mapping[str, Any],
    *,
    receipt_id: int | None,
    recorded_at: str,
) -> tuple[Any, ...]:
    disposition_json = canonical_json(disposition)
    return (
        source_record_key,
        receipt_id,
        str(disposition["source"]),
        str(disposition["brickname"]),
        int(disposition["objid"]),
        int(bool(disposition["eligible"])),
        str(disposition["row_sha256"]),
        str(disposition["eligibility_contract_sha256"]),
        str(disposition["disposition_sha256"]),
        disposition_json,
        recorded_at,
    )


def store_datalab_dispositions(
    connection: sqlite3.Connection,
    datalab: DataLabSnapshot,
) -> None:
    recorded_at = utc_now()
    for pair in sorted(datalab.dispositions):
        disposition = datalab.dispositions[pair]
        source_record_key = f"datalab:{datalab.logical_sha256}:{pair[0]}:{pair[1]}"
        expected = disposition_record_values(
            source_record_key,
            disposition,
            receipt_id=None,
            recorded_at=recorded_at,
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO eligibility_dispositions(
                source_record_key,receipt_id,source,brickname,objid,eligible,
                row_sha256,eligibility_contract_sha256,disposition_sha256,
                disposition_json,recorded_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            expected,
        )
        observed = connection.execute(
            """
            SELECT source_record_key,receipt_id,source,brickname,objid,eligible,
                   row_sha256,eligibility_contract_sha256,disposition_sha256,
                   disposition_json
            FROM eligibility_dispositions WHERE source_record_key=?
            """,
            (source_record_key,),
        ).fetchone()
        if observed is None or tuple(observed) != expected[:-1]:
            raise ContractError(
                f"stored Data Lab eligibility disposition differs for {pair}"
            )


RECEIPT_HASH_COLUMNS = (
    "invocation_id",
    "brickname",
    "expected_objids_json",
    "status",
    "url",
    "response_url",
    "http_status",
    "etag",
    "last_modified",
    "content_length",
    "downloaded_bytes",
    "fits_sha256",
    "row_count",
    "eligibility_passed_count",
    "maximum_coordinate_separation_arcsec",
    "attempt_count",
    "attempts_json",
    "started_at",
    "completed_at",
    "error",
    "full_fits_retained",
    "source_contract_sha256",
    "selected_rows_sha256",
    "eligibility_dispositions_sha256",
)
RECEIPT_INSERT_COLUMNS = RECEIPT_HASH_COLUMNS + ("receipt_sha256",)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = {column: receipt[column] for column in RECEIPT_HASH_COLUMNS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_complete_receipt(
    connection: sqlite3.Connection,
    receipt: Mapping[str, Any],
    proposal: Proposal,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    brick = str(receipt["brickname"])
    expected_url = tractor_url(brick)
    try:
        expected_ids_value = json.loads(str(receipt["expected_objids_json"]))
        expected_ids = [int(value) for value in expected_ids_value]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"{brick}: malformed receipt expected IDs") from exc
    if canonical_json(expected_ids) != receipt["expected_objids_json"]:
        raise ContractError(f"{brick}: receipt expected IDs are not canonical")
    if brick not in proposal.groups or expected_ids != proposal.groups[brick]:
        raise ContractError(f"{brick}: receipt expected IDs differ from proposal")
    if receipt["status"] != "complete" or receipt["resume_state"] != "active":
        raise ContractError(f"{brick}: receipt is not active complete")
    if receipt["url"] != expected_url or receipt["response_url"] != expected_url:
        raise ContractError(f"{brick}: receipt URL differs from authoritative URL")
    if receipt["source_contract_sha256"] != FALLBACK_CONTRACT_SHA256:
        raise ContractError(f"{brick}: receipt source contract differs")
    if not isinstance(receipt["fits_sha256"], str) or not SHA256_PATTERN.fullmatch(
        receipt["fits_sha256"]
    ):
        raise ContractError(f"{brick}: receipt source FITS SHA-256 is malformed")
    if (
        not isinstance(receipt["etag"], str)
        or not receipt["etag"].strip()
        or not isinstance(receipt["last_modified"], str)
        or not receipt["last_modified"].strip()
    ):
        raise ContractError(f"{brick}: receipt ETag/Last-Modified is missing")
    if (
        not is_exact_integer(receipt["content_length"])
        or not is_exact_integer(receipt["downloaded_bytes"])
        or int(receipt["content_length"]) <= 0
        or int(receipt["content_length"]) != int(receipt["downloaded_bytes"])
    ):
        raise ContractError(f"{brick}: receipt byte counts are inconsistent")
    if (
        not is_exact_integer(receipt["http_status"])
        or int(receipt["http_status"]) != 200
    ):
        raise ContractError(f"{brick}: receipt HTTP status is not 200")
    try:
        attempts = json.loads(str(receipt["attempts_json"]))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{brick}: receipt attempts JSON is malformed") from exc
    if canonical_json(attempts) != receipt["attempts_json"]:
        raise ContractError(f"{brick}: receipt attempts JSON is not canonical")
    if (
        not isinstance(attempts, list)
        or not is_exact_integer(receipt["attempt_count"])
        or int(receipt["attempt_count"]) != len(attempts)
        or not attempts
        or attempts[-1].get("status") != "complete"
        or attempts[-1].get("downloaded_bytes") != receipt["downloaded_bytes"]
    ):
        raise ContractError(f"{brick}: receipt attempts do not prove completion")
    if (
        not isinstance(receipt["receipt_sha256"], str)
        or not SHA256_PATTERN.fullmatch(receipt["receipt_sha256"])
        or receipt_sha256(receipt) != receipt["receipt_sha256"]
    ):
        raise ContractError(f"{brick}: receipt hash validation failed")

    selected_columns = ",".join(COLUMNS)
    stored_values = connection.execute(
        f"""
        SELECT {selected_columns} FROM fallback_rows
        WHERE receipt_id=? ORDER BY brickname,objid
        """,
        (int(receipt["receipt_id"]),),
    ).fetchall()
    stored_rows = [dict(zip(COLUMNS, values, strict=True)) for values in stored_values]
    source_anchors = connection.execute(
        """
        SELECT DISTINCT source_fits_sha256,source_etag,
                        source_downloaded_bytes,source_receipt_sha256
        FROM fallback_rows WHERE receipt_id=?
        """,
        (int(receipt["receipt_id"]),),
    ).fetchall()
    expected_anchor = (
        receipt["fits_sha256"],
        receipt["etag"],
        receipt["downloaded_bytes"],
        receipt["receipt_sha256"],
    )
    if source_anchors != [expected_anchor]:
        raise ContractError(f"{brick}: selected-row source anchors differ")
    rows, recomputed_dispositions, maximum_separation = validate_selected_rows(
        brick,
        expected_ids,
        stored_rows,
        proposal.coordinates,
        source="nersc_fits",
    )
    if (
        not is_exact_integer(receipt["row_count"])
        or int(receipt["row_count"]) != len(rows)
        or len(rows) != len(expected_ids)
        or selected_rows_sha256(rows) != receipt["selected_rows_sha256"]
    ):
        raise ContractError(f"{brick}: selected row count/content validation failed")
    if (
        not all(item["eligible"] for item in recomputed_dispositions)
        or not is_exact_integer(receipt["eligibility_passed_count"])
        or int(receipt["eligibility_passed_count"]) != len(expected_ids)
    ):
        raise ContractError(f"{brick}: resumed rows are not all eligible")
    if (
        receipt["maximum_coordinate_separation_arcsec"] is None
        or abs(
            float(receipt["maximum_coordinate_separation_arcsec"]) - maximum_separation
        )
        > 1e-12
    ):
        raise ContractError(f"{brick}: receipt coordinate maximum differs")

    stored_disposition_rows = connection.execute(
        """
        SELECT disposition_json FROM eligibility_dispositions
        WHERE receipt_id=? ORDER BY brickname,objid
        """,
        (int(receipt["receipt_id"]),),
    ).fetchall()
    try:
        stored_dispositions = [
            json.loads(value[0]) for value in stored_disposition_rows
        ]
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"{brick}: stored eligibility disposition is malformed"
        ) from exc
    if (
        stored_dispositions != recomputed_dispositions
        or dispositions_sha256(stored_dispositions)
        != receipt["eligibility_dispositions_sha256"]
    ):
        raise ContractError(f"{brick}: eligibility disposition content/hash differs")
    return rows, recomputed_dispositions


def rows_from_fallback_database(
    connection: sqlite3.Connection,
    proposal: Proposal,
    missing_datalab_bricks: set[str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], set[str]]:
    receipt_cursor = connection.execute(
        """
        SELECT * FROM receipts
        WHERE status='complete' AND resume_state='active'
        ORDER BY brickname
        """
    )
    receipt_columns = [description[0] for description in receipt_cursor.description]
    active_receipts = [
        dict(zip(receipt_columns, values, strict=True))
        for values in receipt_cursor.fetchall()
    ]
    output: dict[tuple[str, int], dict[str, Any]] = {}
    complete_bricks: set[str] = set()
    stale: list[tuple[int, str, str]] = []
    for receipt in active_receipts:
        brick = str(receipt["brickname"])
        if brick not in missing_datalab_bricks:
            stale.append(
                (
                    int(receipt["receipt_id"]),
                    brick,
                    "brick is no longer missing from Data Lab",
                )
            )
            continue
        try:
            validated, _ = validate_complete_receipt(connection, receipt, proposal)
        except (ContractError, TypeError, ValueError) as exc:
            stale.append((int(receipt["receipt_id"]), brick, str(exc)))
            continue
        for row in validated:
            pair = (brick, int(row["objid"]))
            output[pair] = row
        complete_bricks.add(brick)
    if stale:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for receipt_id, brick, error in stale:
                connection.execute(
                    """
                    UPDATE receipts
                    SET resume_state='stale',resume_validation_error=?,
                        last_resume_validation_at=?
                    WHERE receipt_id=?
                    """,
                    (error, utc_now(), receipt_id),
                )
                connection.execute(
                    "DELETE FROM fallback_rows WHERE receipt_id=?",
                    (receipt_id,),
                )
                print(
                    f"stale_receipt={receipt_id} brick={brick} reason={error}",
                    flush=True,
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return output, complete_bricks


def store_brick_result(
    connection: sqlite3.Connection,
    invocation_id: str,
    result: BrickResult,
    coordinates: Mapping[tuple[str, int], tuple[float, float]],
) -> None:
    receipt = {
        **result.receipt,
        "invocation_id": invocation_id,
        "brickname": result.brickname,
        "expected_objids_json": canonical_json(result.expected_objids),
        "eligibility_passed_count": sum(
            int(bool(item["eligible"])) for item in result.dispositions
        ),
        "eligibility_dispositions_sha256": dispositions_sha256(result.dispositions),
        "selected_rows_sha256": selected_rows_sha256(result.rows),
        "source_contract_sha256": FALLBACK_CONTRACT_SHA256,
    }
    if int(receipt["row_count"]) != len(result.rows):
        raise ContractError(
            f"{result.brickname}: receipt row count differs before storage"
        )
    if result.rows and len(result.dispositions) != len(result.rows):
        raise ContractError(
            f"{result.brickname}: not every selected row has a disposition"
        )
    if result.status == "complete" and (
        len(result.rows) != len(result.expected_objids)
        or not all(item["eligible"] for item in result.dispositions)
    ):
        raise ContractError(
            f"{result.brickname}: ineligible/incomplete result marked complete"
        )
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    placeholders = ",".join("?" for _ in RECEIPT_INSERT_COLUMNS)
    connection.execute("BEGIN IMMEDIATE")
    try:
        cursor = connection.execute(
            f"""
            INSERT INTO receipts({",".join(RECEIPT_INSERT_COLUMNS)})
            VALUES ({placeholders})
            """,
            tuple(receipt[column] for column in RECEIPT_INSERT_COLUMNS),
        )
        receipt_id = int(cursor.lastrowid)
        recorded_at = utc_now()
        for disposition in result.dispositions:
            pair = (
                str(disposition["brickname"]),
                int(disposition["objid"]),
            )
            source_record_key = f"receipt:{receipt_id}:{pair[0]}:{pair[1]}"
            connection.execute(
                """
                INSERT INTO eligibility_dispositions(
                    source_record_key,receipt_id,source,brickname,objid,eligible,
                    row_sha256,eligibility_contract_sha256,disposition_sha256,
                    disposition_json,recorded_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                disposition_record_values(
                    source_record_key,
                    disposition,
                    receipt_id=receipt_id,
                    recorded_at=recorded_at,
                ),
            )
        if result.status == "complete":
            row_columns = (
                ("receipt_id",)
                + COLUMNS
                + (
                    "payload_tractor_separation_arcsec",
                    "source_fits_sha256",
                    "source_etag",
                    "source_downloaded_bytes",
                    "source_receipt_sha256",
                )
            )
            row_placeholders = ",".join("?" for _ in row_columns)
            for row in result.rows:
                pair = (str(row["brickname"]), int(row["objid"]))
                payload_ra, payload_dec = coordinates[pair]
                separation = angular_separation_arcsec(
                    payload_ra,
                    payload_dec,
                    float(row["ra"]),
                    float(row["dec"]),
                )
                connection.execute(
                    f"""
                    INSERT INTO fallback_rows({",".join(row_columns)})
                    VALUES ({row_placeholders})
                    """,
                    (receipt_id,)
                    + tuple(row[column] for column in COLUMNS)
                    + (
                        separation,
                        receipt["fits_sha256"],
                        receipt["etag"],
                        receipt["downloaded_bytes"],
                        receipt["receipt_sha256"],
                    ),
                )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def make_manifest(
    proposal: Proposal,
    datalab_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    fallback_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[pd.DataFrame, bool]:
    overlap = set(datalab_rows) & set(fallback_rows)
    if overlap:
        raise ContractError(f"Data Lab/fallback overlap exists: {sorted(overlap)[:3]}")
    expected_pairs = set(proposal.coordinates)
    combined = {**datalab_rows, **fallback_rows}
    if set(combined) - expected_pairs:
        raise ContractError("combined Tractor rows include unrequested hosts")

    records: list[dict[str, Any]] = []
    for pair in sorted(combined):
        source = "datalab" if pair in datalab_rows else "nersc_fits"
        records.append({**combined[pair], "_tractor_source": source})
    tractor = pd.DataFrame(records, columns=[*COLUMNS, "_tractor_source"])
    manifest = proposal.frame.merge(
        tractor,
        on=["brickname", "objid"],
        how="left",
        validate="one_to_one",
        suffixes=("_payload", "_tractor"),
    )
    manifest["_tractor_found"] = manifest["_tractor_source"].notna()
    separation = np.full(len(manifest), np.nan, dtype=float)
    found = manifest["_tractor_found"].to_numpy(dtype=bool)
    if found.any():
        payload_ra = manifest.loc[found, "ra_payload"].to_numpy(dtype=float)
        payload_dec = manifest.loc[found, "dec_payload"].to_numpy(dtype=float)
        tractor_ra = manifest.loc[found, "ra_tractor"].to_numpy(dtype=float)
        tractor_dec = manifest.loc[found, "dec_tractor"].to_numpy(dtype=float)
        values = [
            angular_separation_arcsec(a, b, c, d)
            for a, b, c, d in zip(
                payload_ra, payload_dec, tractor_ra, tractor_dec, strict=True
            )
        ]
        separation[found] = np.asarray(values, dtype=float)
    manifest["payload_tractor_separation_arcsec"] = separation
    if len(manifest) != 4_500 or manifest["galaxy_id"].duplicated().any():
        raise ContractError("combined manifest is not exactly 4,500 unique hosts")
    scores = forbidden_columns(manifest.columns)
    if scores:
        raise ContractError(f"combined manifest has score-bearing columns: {scores}")
    complete = set(combined) == expected_pairs and bool(
        manifest["_tractor_found"].all()
    )
    if complete:
        max_separation = float(manifest["payload_tractor_separation_arcsec"].max())
        if max_separation > MAX_COORDINATE_SEPARATION_ARCSEC:
            raise ContractError("combined manifest coordinate validation failed")
    return manifest, complete


def export_outputs(
    output_directory: Path,
    connection: sqlite3.Connection,
    proposal: Proposal,
    datalab: DataLabSnapshot,
    missing_datalab_bricks: set[str],
    *,
    invocation_id: str,
    canary: bool,
    implementation_sha256: str,
    spool_empty_proven: bool,
) -> tuple[dict[str, Any], bool]:
    if not spool_empty_proven:
        raise ContractError(
            "cannot export retention claim without proving FITS spool empty"
        )
    fallback_rows, fallback_complete_bricks = rows_from_fallback_database(
        connection, proposal, missing_datalab_bricks
    )
    fallback_frame = pd.DataFrame(
        [fallback_rows[pair] for pair in sorted(fallback_rows)],
        columns=COLUMNS,
    )
    fallback_rows_path = output_directory / "fallback_rows.parquet"
    atomic_write_parquet(fallback_frame, fallback_rows_path)

    receipts = pd.read_sql_query(
        "SELECT * FROM receipts ORDER BY receipt_id", connection
    )
    receipts_path = output_directory / "fallback_receipts.parquet"
    atomic_write_parquet(receipts, receipts_path)

    disposition_cursor = connection.execute(
        """
        SELECT source_record_key,receipt_id,recorded_at,disposition_json
        FROM eligibility_dispositions
        ORDER BY source,brickname,objid,source_record_key
        """
    )
    disposition_exports: list[dict[str, Any]] = []
    for (
        source_record_key,
        receipt_id,
        recorded_at,
        disposition_json,
    ) in disposition_cursor.fetchall():
        disposition = json.loads(disposition_json)
        disposition_exports.append(
            {
                "source_record_key": source_record_key,
                "receipt_id": receipt_id,
                "recorded_at": recorded_at,
                "source": disposition["source"],
                "brickname": disposition["brickname"],
                "objid": disposition["objid"],
                "eligible": disposition["eligible"],
                "disposition_reasons_json": canonical_json(
                    disposition["disposition_reasons"]
                ),
                "row_sha256": disposition["row_sha256"],
                "eligibility_contract_sha256": disposition[
                    "eligibility_contract_sha256"
                ],
                "disposition_sha256": disposition["disposition_sha256"],
                "morphology": disposition["morphology"],
                "observed_zmag": disposition["observed_zmag"],
                "flux_z_nonpositive_magnitude_exemption": disposition[
                    "flux_z_nonpositive_magnitude_exemption"
                ],
                "payload_tractor_separation_arcsec": disposition[
                    "payload_tractor_separation_arcsec"
                ],
                "prohibited_mask_bit_positions_present_json": canonical_json(
                    disposition["prohibited_mask_bit_positions_present"]
                ),
                "nonfinite_required_model_fields_json": canonical_json(
                    disposition["nonfinite_required_model_fields"]
                ),
                **{
                    f"passes_{name}": passed
                    for name, passed in disposition["passes"].items()
                },
            }
        )
    dispositions_frame = pd.DataFrame(disposition_exports)
    dispositions_path = output_directory / "eligibility_dispositions.parquet"
    atomic_write_parquet(dispositions_frame, dispositions_path)

    latest_cursor = connection.execute(
        """
        SELECT r.*
        FROM receipts r
        JOIN (
            SELECT brickname,MAX(receipt_id) AS receipt_id
            FROM receipts GROUP BY brickname
        ) latest ON latest.receipt_id=r.receipt_id
        ORDER BY r.brickname
        """
    )
    receipt_columns = [description[0] for description in latest_cursor.description]
    latest_receipts = {
        str(values[receipt_columns.index("brickname")]): dict(
            zip(receipt_columns, values, strict=True)
        )
        for values in latest_cursor.fetchall()
    }
    latest_dispositions: dict[tuple[str, int], dict[str, Any]] = {}
    for receipt in latest_receipts.values():
        for (disposition_json,) in connection.execute(
            """
            SELECT disposition_json FROM eligibility_dispositions
            WHERE receipt_id=? ORDER BY brickname,objid
            """,
            (int(receipt["receipt_id"]),),
        ):
            disposition = json.loads(disposition_json)
            pair = (
                str(disposition["brickname"]),
                int(disposition["objid"]),
            )
            latest_dispositions[pair] = disposition

    failures: list[dict[str, Any]] = []
    resolved_pairs = set(datalab.rows) | set(fallback_rows)
    unresolved_pairs = set(proposal.coordinates) - resolved_pairs
    unresolved_by_brick: dict[str, list[int]] = {}
    for brick, objid in sorted(unresolved_pairs):
        unresolved_by_brick.setdefault(brick, []).append(objid)
    for brick, unresolved_objids in unresolved_by_brick.items():
        datalab_status, datalab_error = datalab.query_status.get(
            brick, ("not_recorded", None)
        )
        receipt = latest_receipts.get(brick)
        eligibility_failures = []
        for objid in unresolved_objids:
            pair = (brick, objid)
            disposition = datalab.dispositions.get(pair)
            if disposition is None or disposition["eligible"]:
                disposition = latest_dispositions.get(pair)
            if disposition is not None and not disposition["eligible"]:
                eligibility_failures.append(
                    {
                        "objid": objid,
                        "reasons": disposition["disposition_reasons"],
                        "source": disposition["source"],
                    }
                )
        fallback_status = "not_attempted"
        if receipt is not None:
            fallback_status = (
                "stale"
                if receipt["resume_state"] == "stale"
                else str(receipt["status"])
            )
        failures.append(
            {
                "brickname": brick,
                "datalab_error": datalab_error,
                "datalab_status": datalab_status,
                "eligibility_failures_json": canonical_json(eligibility_failures),
                "replacement_required": bool(eligibility_failures),
                "unresolved_objids_json": canonical_json(unresolved_objids),
                "fallback_error": (receipt["error"] if receipt is not None else None),
                "fallback_status": fallback_status,
                "latest_receipt_id": (
                    int(receipt["receipt_id"]) if receipt is not None else None
                ),
            }
        )
    failed_frame = pd.DataFrame(
        failures,
        columns=[
            "brickname",
            "datalab_status",
            "datalab_error",
            "unresolved_objids_json",
            "fallback_status",
            "fallback_error",
            "latest_receipt_id",
            "replacement_required",
            "eligibility_failures_json",
        ],
    )
    failed_path = output_directory / "failed_audit.parquet"
    atomic_write_parquet(failed_frame, failed_path)
    atomic_write_json(
        output_directory / "failed_audit.json",
        {
            "schema_version": 1,
            "unresolved_bricks": len(failures),
            "unresolved_hosts": len(unresolved_pairs),
            "failures": failures,
        },
    )

    manifest, complete = make_manifest(proposal, datalab.rows, fallback_rows)
    manifest_name = (
        "tractor_hosts.parquet" if complete else "tractor_hosts_incomplete.parquet"
    )
    manifest_path = output_directory / manifest_name
    atomic_write_parquet(manifest, manifest_path)

    complete_receipts = receipts.loc[receipts["status"].eq("complete")]
    successful_sizes = complete_receipts["downloaded_bytes"].dropna()
    mean_brick_bytes = float(successful_sizes.mean()) if len(successful_sizes) else None
    projected_full_bytes = (
        round(mean_brick_bytes * len(missing_datalab_bricks))
        if mean_brick_bytes is not None
        else None
    )
    invocation_receipts = receipts.loc[receipts["invocation_id"].eq(invocation_id)]
    invocation_bytes = int(invocation_receipts["downloaded_bytes"].fillna(0).sum())
    maximum_separation = (
        float(
            manifest.loc[
                manifest["_tractor_found"],
                "payload_tractor_separation_arcsec",
            ].max()
        )
        if manifest["_tractor_found"].any()
        else None
    )
    status = (
        "complete"
        if complete
        else ("canary_complete_manifest_incomplete" if canary else "incomplete")
    )
    summary: dict[str, Any] = {
        "datalab_complete_bricks": len(datalab.complete_bricks),
        "datalab_eligible_hosts": len(datalab.rows),
        "datalab_selected_hosts": datalab.selected_rows,
        "datalab_complete_logical_sha256": datalab.logical_sha256,
        "failed_audit_sha256": sha256_file(failed_path),
        "fallback_complete_bricks": len(fallback_complete_bricks),
        "fallback_complete_hosts": len(fallback_rows),
        "fallback_contract_sha256": FALLBACK_CONTRACT_SHA256,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "eligibility_dispositions": len(dispositions_frame),
        "eligibility_dispositions_sha256": sha256_file(dispositions_path),
        "fallback_receipts_sha256": sha256_file(receipts_path),
        "fallback_rows_sha256": sha256_file(fallback_rows_path),
        "full_fits_retained": False,
        "fits_spool_empty_proven": True,
        "implementation_sha256": implementation_sha256,
        "invocation_downloaded_bytes": invocation_bytes,
        "manifest_hosts": len(manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_tractor_rows_found": int(manifest["_tractor_found"].sum()),
        "maximum_coordinate_separation_arcsec": maximum_separation,
        "mean_successful_fits_bytes": mean_brick_bytes,
        "missing_datalab_bricks": len(missing_datalab_bricks),
        "projected_full_fallback_bytes_from_successful_mean": projected_full_bytes,
        "proposal_sha256": proposal.sha256,
        "schema_version": 1,
        "score_columns_included": False,
        "status": status,
        "unresolved_bricks": len(failures),
        "unresolved_hosts": len(unresolved_pairs),
    }
    summary_path = output_directory / "summary.json"
    atomic_write_json(summary_path, summary)
    marker_payload = {
        "manifest_sha256": summary["manifest_sha256"],
        "schema_version": 1,
        "status": status,
        "summary_sha256": sha256_file(summary_path),
    }
    if complete:
        atomic_write_json(output_directory / "COMPLETE.json", marker_payload)
        (output_directory / "INCOMPLETE.json").unlink(missing_ok=True)
        (output_directory / "tractor_hosts_incomplete.parquet").unlink(missing_ok=True)
    else:
        atomic_write_json(output_directory / "INCOMPLETE.json", marker_payload)
        (output_directory / "COMPLETE.json").unlink(missing_ok=True)
    return summary, complete


def assert_no_spool_orphans(output_directory: Path) -> None:
    spool = output_directory / ".fits-spool"
    if not spool.exists():
        return
    orphans = sorted(str(path) for path in spool.iterdir())
    if orphans:
        raise ContractError(
            "orphaned files exist in .fits-spool; refusing startup: "
            + ", ".join(orphans)
        )
    spool.rmdir()


def prove_spool_empty(output_directory: Path) -> bool:
    spool = output_directory / ".fits-spool"
    if not spool.exists():
        return True
    retained = sorted(str(path) for path in spool.iterdir())
    if retained:
        raise ContractError(
            "FITS spool is not empty after download completion: " + ", ".join(retained)
        )
    spool.rmdir()
    return True


def run_downloads(
    selected: list[tuple[str, list[int]]],
    *,
    proposal: Proposal,
    connection: sqlite3.Connection,
    invocation_id: str,
    output_directory: Path,
    workers: int,
    max_bytes: int,
    timeout_seconds: float,
    retries: int,
    requests_per_second: float,
    canary: bool,
) -> tuple[int, int, bool, bool]:
    limiter = StartRateLimiter(requests_per_second)
    spool = output_directory / ".fits-spool"
    spool.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    circuit_broken = False

    def submit(
        executor: ThreadPoolExecutor,
        item: tuple[str, list[int]],
    ) -> Future[BrickResult]:
        brick, ids = item
        return executor.submit(
            process_brick,
            brick,
            ids,
            proposal,
            spool_directory=spool,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            retries=retries,
            limiter=limiter,
        )

    iterator = iter(selected)
    executor = ThreadPoolExecutor(max_workers=workers)
    in_flight = {
        submit(executor, item)
        for item in (next(iterator, None) for _ in range(workers))
        if item is not None
    }
    try:
        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                store_brick_result(
                    connection,
                    invocation_id,
                    result,
                    proposal.coordinates,
                )
                completed += int(result.status == "complete")
                failed += int(result.status == "failed")
                processed = completed + failed
                print(
                    f"processed={processed}/{len(selected)} "
                    f"complete={completed} failed={failed} "
                    f"brick={result.brickname}",
                    flush=True,
                )
            processed = completed + failed
            if (
                not canary
                and processed >= max(8, workers * 2)
                and failed / processed > 0.25
            ):
                circuit_broken = True
                for future in in_flight:
                    future.cancel()
                break
            for _ in range(len(finished)):
                replacement = next(iterator, None)
                if replacement is None:
                    break
                in_flight.add(submit(executor, replacement))
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=circuit_broken)
    spool_empty_proven = prove_spool_empty(output_directory)
    return completed, failed, circuit_broken, spool_empty_proven


def invocation_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "bricks": arguments.brick,
        "canary": arguments.canary,
        "datalab_db": str(arguments.datalab_db.resolve()),
        "hosts": str(arguments.hosts.resolve()),
        "max_fits_mib": arguments.max_fits_mib,
        "request_rate": arguments.request_rate,
        "retries": arguments.retries,
        "timeout_seconds": arguments.timeout_seconds,
        "workers": arguments.workers,
    }


def run(arguments: argparse.Namespace) -> int:
    if not 1 <= arguments.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if not 1 <= arguments.retries <= 6:
        raise ValueError("retries must be between 1 and 6")
    if not 10.0 <= arguments.timeout_seconds <= 600.0:
        raise ValueError("timeout-seconds must be between 10 and 600")
    if not 0.1 <= arguments.request_rate <= 4.0:
        raise ValueError("request-rate must be between 0.1 and 4 per second")
    if not 1.0 <= arguments.max_fits_mib <= 1024.0:
        raise ValueError("max-fits-mib must be between 1 and 1024")
    if arguments.canary and arguments.brick:
        raise ValueError("--canary and --brick are mutually exclusive")

    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_directory / ".fallback.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another fallback process holds the output lock") from exc

    assert_no_spool_orphans(output_directory)
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    proposal = load_proposal(
        arguments.hosts.resolve(),
        expected_sha256=arguments.authorized_proposal_sha256,
    )
    datalab = load_datalab_snapshot(arguments.datalab_db.resolve(), proposal)
    missing_datalab = set(proposal.groups) - datalab.complete_bricks
    connection = initialize_fallback_database(
        output_directory / "tractor_fallback.sqlite",
        proposal=proposal,
        datalab=datalab,
        implementation_sha256=implementation_sha256,
    )
    invocation_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    mode = (
        "canary"
        if arguments.canary
        else ("selected_bricks" if arguments.brick else "full")
    )
    connection.execute(
        """
        INSERT INTO invocations(
            invocation_id,mode,started_at,status,arguments_json
        ) VALUES (?,?,?,?,?)
        """,
        (
            invocation_id,
            mode,
            utc_now(),
            "running",
            canonical_json(invocation_arguments(arguments)),
        ),
    )
    connection.commit()
    try:
        _, resumed_complete = rows_from_fallback_database(
            connection, proposal, missing_datalab
        )
        pending = [
            (brick, proposal.groups[brick])
            for brick in sorted(missing_datalab - resumed_complete)
        ]
        if arguments.brick:
            requested_bricks = list(dict.fromkeys(arguments.brick))
            invalid = [
                brick
                for brick in requested_bricks
                if not BRICK_PATTERN.fullmatch(brick)
            ]
            not_missing = [
                brick for brick in requested_bricks if brick not in missing_datalab
            ]
            already_complete = [
                brick for brick in requested_bricks if brick in resumed_complete
            ]
            if invalid or not_missing or already_complete:
                raise ContractError(
                    "selected bricks violate fallback scope; "
                    f"invalid={invalid}, not_missing={not_missing}, "
                    f"already_complete={already_complete}"
                )
            selected_set = set(requested_bricks)
            pending = [item for item in pending if item[0] in selected_set]
            if len(pending) != len(requested_bricks):
                raise ContractError("one or more selected bricks were not found")
        elif arguments.canary:
            pending = pending[:2]

        print(
            f"proposal_hosts={len(proposal.frame)} "
            f"datalab_complete_bricks={len(datalab.complete_bricks)} "
            f"datalab_complete_hosts={len(datalab.rows)} "
            f"fallback_resumed_bricks={len(resumed_complete)} "
            f"missing_bricks={len(missing_datalab)} selected={len(pending)}",
            flush=True,
        )
        completed, failed, circuit_broken, spool_empty_proven = run_downloads(
            pending,
            proposal=proposal,
            connection=connection,
            invocation_id=invocation_id,
            output_directory=output_directory,
            workers=min(arguments.workers, max(1, len(pending))),
            max_bytes=int(arguments.max_fits_mib * 1024 * 1024),
            timeout_seconds=arguments.timeout_seconds,
            retries=arguments.retries,
            requests_per_second=arguments.request_rate,
            canary=arguments.canary,
        )
        invocation_status = (
            "circuit_broken"
            if circuit_broken
            else ("complete_with_failures" if failed else "complete")
        )
        connection.execute(
            """
            UPDATE invocations SET completed_at=?,status=?
            WHERE invocation_id=?
            """,
            (utc_now(), invocation_status, invocation_id),
        )
        connection.commit()
        summary, manifest_complete = export_outputs(
            output_directory,
            connection,
            proposal,
            datalab,
            missing_datalab,
            invocation_id=invocation_id,
            canary=arguments.canary,
            implementation_sha256=implementation_sha256,
            spool_empty_proven=spool_empty_proven,
        )
        projected = summary["projected_full_fallback_bytes_from_successful_mean"]
        print(
            f"manifest_found={summary['manifest_tractor_rows_found']}/4500 "
            f"unresolved_bricks={summary['unresolved_bricks']} "
            f"invocation_bytes={summary['invocation_downloaded_bytes']} "
            f"projected_full_bytes={projected}",
            flush=True,
        )
        if arguments.canary:
            return 0 if failed == 0 and completed == len(pending) else 1
        if arguments.brick:
            return 0 if failed == 0 and completed == len(pending) else 1
        return 0 if manifest_complete else 1
    except BaseException:
        connection.execute(
            """
            UPDATE invocations SET completed_at=?,status='aborted'
            WHERE invocation_id=?
            """,
            (utc_now(), invocation_id),
        )
        connection.commit()
        raise
    finally:
        connection.close()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def synthetic_fits_bytes(
    *,
    brickname: str = "1234p567",
    primary: Sequence[bool] = (True, True, True),
) -> bytes:
    count = len(primary)
    objids = np.arange(10, 10 + count, dtype=np.int64)
    arrays: dict[str, Any] = {
        "brickname": np.asarray([brickname] * count),
        "objid": objids,
        "brick_primary": np.asarray(primary, dtype=bool),
        "maskbits": np.zeros(count, dtype=np.int64),
        "type": np.asarray(["SER"] * count),
        "ra": np.asarray([100.0 + i * 0.001 for i in range(count)]),
        "dec": np.asarray([20.0 + i * 0.001 for i in range(count)]),
        "nobs_g": np.full(count, 3, dtype=np.int64),
        "nobs_r": np.full(count, 3, dtype=np.int64),
        "nobs_z": np.full(count, 3, dtype=np.int64),
    }
    for column in FLOAT_COLUMNS - {"ra", "dec"}:
        arrays[column] = np.full(count, 1.25, dtype=float)
    arrays["flux_g"] = np.full(count, 20.0, dtype=float)
    arrays["flux_r"] = np.full(count, 20.0, dtype=float)
    arrays["flux_z"] = np.full(count, 20.0, dtype=float)
    fits_columns = []
    for column in COLUMNS:
        if column == "brickname":
            format_string = "8A"
        elif column == "type":
            format_string = "4A"
        elif column == "brick_primary":
            format_string = "L"
        elif column in INTEGER_COLUMNS:
            format_string = "K"
        else:
            format_string = "D"
        fits_columns.append(
            fits.Column(
                name=column.upper(),
                format=format_string,
                array=arrays[column],
            )
        )
    fits_columns.append(
        fits.Column(
            name="DETECTOR_SCORE",
            format="D",
            array=np.full(count, 0.99),
        )
    )
    output = io.BytesIO()
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits_columns)]
    ).writeto(output)
    return output.getvalue()


def valid_synthetic_row(
    *,
    brickname: str = "1234p567",
    objid: int = 10,
    ra: float = 100.0,
    dec: float = 20.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "brickname": brickname,
        "objid": objid,
        "brick_primary": 1,
        "maskbits": 0,
        "type": "SER",
        "ra": ra,
        "dec": dec,
        "flux_g": 20.0,
        "flux_r": 20.0,
        "flux_z": 20.0,
        "nobs_g": 3,
        "nobs_r": 3,
        "nobs_z": 3,
    }
    for column in REQUIRED_FINITE_MODEL_COLUMNS:
        row[column] = 1.25
    return row


class _FakeResponse:
    def __init__(self, body: bytes, headers: Mapping[str, str]) -> None:
        self._body = io.BytesIO(body)
        self.headers = dict(headers)
        self.status = 200

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return "https://example.test/final.fits"


class _NoWaitLimiter:
    def wait(self) -> None:
        return None


def make_resume_test_fixture(
    directory: Path,
) -> tuple[sqlite3.Connection, Proposal, dict[str, Any]]:
    row = valid_synthetic_row()
    proposal = Proposal(
        pd.DataFrame([{"galaxy_id": "1234p567_10"}]),
        {"1234p567": [10]},
        {("1234p567", 10): (100.0, 20.0)},
        EXPECTED_PROPOSAL_SHA256,
    )
    datalab = DataLabSnapshot({}, set(), {}, "d" * 64, {}, 0)
    connection = initialize_fallback_database(
        directory / "resume.sqlite",
        proposal=proposal,
        datalab=datalab,
        implementation_sha256="e" * 64,
    )
    invocation_id = "test-invocation"
    connection.execute(
        """
        INSERT INTO invocations(
            invocation_id,mode,started_at,status,arguments_json
        ) VALUES (?,?,?,?,?)
        """,
        (invocation_id, "test", "2026-09-16T00:00:00Z", "running", "{}"),
    )
    connection.commit()
    normalized, dispositions, maximum_separation = validate_selected_rows(
        "1234p567",
        [10],
        [row],
        proposal.coordinates,
        source="nersc_fits",
    )
    attempts = [
        {
            "attempt": 1,
            "completed_at": "2026-09-16T00:00:01Z",
            "downloaded_bytes": 100,
            "http_status": 200,
            "started_at": "2026-09-16T00:00:00Z",
            "status": "complete",
        }
    ]
    result = BrickResult(
        "1234p567",
        [10],
        "complete",
        normalized,
        dispositions,
        {
            "attempt_count": 1,
            "attempts_json": canonical_json(attempts),
            "completed_at": "2026-09-16T00:00:01Z",
            "content_length": 100,
            "downloaded_bytes": 100,
            "error": None,
            "etag": '"test-etag"',
            "fits_sha256": "a" * 64,
            "full_fits_retained": 0,
            "http_status": 200,
            "last_modified": "Wed, 16 Sep 2026 00:00:00 GMT",
            "maximum_coordinate_separation_arcsec": maximum_separation,
            "response_url": tractor_url("1234p567"),
            "row_count": 1,
            "started_at": "2026-09-16T00:00:00Z",
            "status": "complete",
            "url": tractor_url("1234p567"),
        },
    )
    store_brick_result(
        connection,
        invocation_id,
        result,
        proposal.coordinates,
    )
    cursor = connection.execute(
        """
        SELECT * FROM receipts
        WHERE status='complete' AND resume_state='active'
        """
    )
    columns = [description[0] for description in cursor.description]
    receipt = dict(zip(columns, cursor.fetchone(), strict=True))
    return connection, proposal, receipt


class FallbackSelfTests(unittest.TestCase):
    def test_extracts_only_required_rows_and_columns(self) -> None:
        body = synthetic_fits_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.fits"
            path.write_bytes(body)
            coordinates = {
                ("1234p567", 10): (100.0, 20.0),
                ("1234p567", 12): (100.002, 20.002),
            }
            rows, dispositions, separation = extract_required_rows(
                path, "1234p567", [10, 12], coordinates
            )
        self.assertEqual([row["objid"] for row in rows], [10, 12])
        self.assertEqual(list(rows[0]), list(COLUMNS))
        self.assertNotIn("detector_score", rows[0])
        self.assertTrue(all(item["eligible"] for item in dispositions))
        self.assertLess(separation, 1e-8)

    def test_rejects_non_primary_selected_row(self) -> None:
        body = synthetic_fits_bytes(primary=(False,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.fits"
            path.write_bytes(body)
            _, dispositions, _ = extract_required_rows(
                path,
                "1234p567",
                [10],
                {("1234p567", 10): (100.0, 20.0)},
            )
        self.assertFalse(dispositions[0]["eligible"])
        self.assertFalse(dispositions[0]["passes"]["brick_primary"])

    def test_adversarial_eligibility_predicates(self) -> None:
        cases = [
            ("brick_primary", {"brick_primary": 0}, "brick_primary"),
            ("morphology", {"type": "EXP"}, "morphology"),
            ("nobs_below", {"nobs_g": 2}, "nobs_g"),
            ("nobs_fractional", {"nobs_r": 3.5}, "nobs_r"),
            ("nobs_string", {"nobs_z": "3"}, "nobs_z"),
            ("nonfinite_flux_g", {"flux_g": math.nan}, "finite_flux_g"),
            ("nonfinite_flux_r", {"flux_r": math.inf}, "finite_flux_r"),
            ("strict_z20", {"flux_z": 10.0}, "observed_zmag"),
            ("nonfinite_model", {"shape_e1": math.nan}, "finite_required_model_fields"),
            (
                "coordinate_mismatch",
                {"ra": 100.0 + 1.0 / 3600.0},
                "coordinate_match",
            ),
            ("declination_boundary", {"dec": -18.0}, "declination_window"),
        ]
        for name, mutation, predicate in cases:
            with self.subTest(name=name):
                row = valid_synthetic_row()
                row.update(mutation)
                _, dispositions, _ = validate_selected_rows(
                    "1234p567",
                    [10],
                    [row],
                    {("1234p567", 10): (100.0, 20.0)},
                    source="nersc_fits",
                )
                self.assertFalse(dispositions[0]["eligible"])
                self.assertFalse(dispositions[0]["passes"][predicate])
        for position in PROHIBITED_MASK_BIT_POSITIONS:
            with self.subTest(mask_position=position):
                row = valid_synthetic_row()
                row["maskbits"] = 1 << position
                _, dispositions, _ = validate_selected_rows(
                    "1234p567",
                    [10],
                    [row],
                    {("1234p567", 10): (100.0, 20.0)},
                    source="nersc_fits",
                )
                self.assertFalse(dispositions[0]["passes"]["maskbits"])
                self.assertEqual(
                    dispositions[0]["prohibited_mask_bit_positions_present"],
                    [position],
                )

    def test_nonpositive_finite_flux_z_is_exact_exemption(self) -> None:
        for flux_z in (0.0, -1.0):
            with self.subTest(flux_z=flux_z):
                row = valid_synthetic_row()
                row["flux_z"] = flux_z
                _, dispositions, _ = validate_selected_rows(
                    "1234p567",
                    [10],
                    [row],
                    {("1234p567", 10): (100.0, 20.0)},
                    source="nersc_fits",
                )
                disposition = dispositions[0]
                self.assertTrue(disposition["eligible"])
                self.assertTrue(disposition["flux_z_nonpositive_magnitude_exemption"])
                self.assertIsNone(disposition["observed_zmag"])
                self.assertTrue(disposition["passes"]["finite_flux_z"])
                self.assertTrue(disposition["passes"]["observed_zmag"])

    def test_download_retries_and_hashes_without_retention(self) -> None:
        body = synthetic_fits_bytes()
        calls = 0

        def opener(*_: Any, **__: Any) -> _FakeResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.URLError("transient mock")
            return _FakeResponse(
                body,
                {
                    "Content-Length": str(len(body)),
                    "ETag": '"mock-etag"',
                    "Last-Modified": "Wed, 16 Sep 2026 00:00:00 GMT",
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            downloaded = download_with_retries(
                "https://example.test/mock.fits",
                Path(directory),
                max_bytes=len(body) + 1,
                timeout_seconds=10,
                retries=2,
                limiter=_NoWaitLimiter(),
                opener=opener,
                sleeper=lambda _: None,
                jitter=lambda: 0.0,
            )
            self.assertEqual(calls, 2)
            self.assertEqual(downloaded.downloaded_bytes, len(body))
            self.assertEqual(downloaded.sha256, hashlib.sha256(body).hexdigest())
            self.assertEqual(downloaded.etag, '"mock-etag"')
            downloaded.path.unlink()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_declared_size_cap_is_definitive(self) -> None:
        def opener(*_: Any, **__: Any) -> _FakeResponse:
            return _FakeResponse(b"x", {"Content-Length": "100", "ETag": '"too-big"'})

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FetchFailure) as context:
                download_with_retries(
                    "https://example.test/mock.fits",
                    Path(directory),
                    max_bytes=10,
                    timeout_seconds=10,
                    retries=3,
                    limiter=_NoWaitLimiter(),
                    opener=opener,
                    sleeper=lambda _: None,
                    jitter=lambda: 0.0,
                )
            self.assertEqual(len(context.exception.attempts), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_resume_revalidates_all_receipt_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            connection, proposal, receipt = make_resume_test_fixture(
                Path(directory_text)
            )
            try:
                rows, dispositions = validate_complete_receipt(
                    connection, receipt, proposal
                )
                self.assertEqual(len(rows), 1)
                self.assertTrue(dispositions[0]["eligible"])

                mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
                    (
                        "url",
                        lambda value: value.update(
                            {
                                "url": "https://example.test/evil.fits",
                                "response_url": "https://example.test/evil.fits",
                            }
                        ),
                    ),
                    (
                        "source_contract",
                        lambda value: value.update(
                            {"source_contract_sha256": "f" * 64}
                        ),
                    ),
                    (
                        "expected_ids",
                        lambda value: value.update({"expected_objids_json": "[11]"}),
                    ),
                    (
                        "row_count",
                        lambda value: value.update({"row_count": 2}),
                    ),
                    (
                        "source_fits_sha",
                        lambda value: value.update({"fits_sha256": "b" * 64}),
                    ),
                    (
                        "etag",
                        lambda value: value.update({"etag": '"changed"'}),
                    ),
                    (
                        "bytes",
                        lambda value: value.update(
                            {
                                "content_length": 101,
                                "downloaded_bytes": 101,
                                "attempts_json": canonical_json(
                                    [
                                        {
                                            "attempt": 1,
                                            "completed_at": "2026-09-16T00:00:01Z",
                                            "downloaded_bytes": 101,
                                            "http_status": 200,
                                            "started_at": "2026-09-16T00:00:00Z",
                                            "status": "complete",
                                        }
                                    ]
                                ),
                            }
                        ),
                    ),
                    (
                        "selected_content_hash",
                        lambda value: value.update({"selected_rows_sha256": "c" * 64}),
                    ),
                ]
                for name, mutate in mutations:
                    with self.subTest(name=name):
                        changed = dict(receipt)
                        mutate(changed)
                        changed["receipt_sha256"] = receipt_sha256(changed)
                        with self.assertRaises(ContractError):
                            validate_complete_receipt(connection, changed, proposal)
                changed = dict(receipt)
                changed["receipt_sha256"] = "0" * 64
                with self.assertRaisesRegex(ContractError, "receipt hash"):
                    validate_complete_receipt(connection, changed, proposal)
            finally:
                connection.close()

    def test_stale_row_is_removed_for_requery(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            connection, proposal, _ = make_resume_test_fixture(Path(directory_text))
            try:
                connection.execute("UPDATE fallback_rows SET flux_g=999.0")
                connection.commit()
                rows, complete = rows_from_fallback_database(
                    connection, proposal, {"1234p567"}
                )
                self.assertEqual(rows, {})
                self.assertEqual(complete, set())
                state = connection.execute(
                    "SELECT resume_state FROM receipts"
                ).fetchone()[0]
                self.assertEqual(state, "stale")
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM fallback_rows").fetchone()[
                        0
                    ],
                    0,
                )
            finally:
                connection.close()

    def test_orphaned_spool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            output = Path(directory_text)
            spool = output / ".fits-spool"
            spool.mkdir()
            (spool / "orphan.fits.partial").write_bytes(b"orphan")
            with self.assertRaisesRegex(ContractError, "orphaned"):
                assert_no_spool_orphans(output)


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FallbackSelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        type=Path,
        default=Path(
            "evaluation_dataset_prep/role_proposal/proposed_grec_hosts.parquet"
        ),
    )
    parser.add_argument(
        "--datalab-db",
        type=Path,
        default=Path("evaluation_dataset_prep/tractor_hosts_v2/tractor_queries.sqlite"),
    )
    parser.add_argument(
        "--authorized-proposal-sha256",
        default=EXPECTED_PROPOSAL_SHA256,
        help="exact SHA-256 of the independently authorized host proposal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/stronglens_overnight/tractor_fallback_output"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="aggregate request starts per second (maximum 4)",
    )
    parser.add_argument("--max-fits-mib", type=float, default=256.0)
    parser.add_argument(
        "--canary",
        action="store_true",
        help="retrieve only the first two still-missing bricks",
    )
    parser.add_argument(
        "--brick",
        action="append",
        default=[],
        help="retrieve one explicitly named missing brick (repeatable)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run local FITS and mocked-network tests, then exit",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_tests()
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
