#!/usr/bin/env python3
"""Build and finalize an additive canonical-test-only G_rec host repair."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from fetch_tractor_rows import (
    ANONYMOUS_TOKEN,
    SERVICE,
    RateLimiter,
    initialize,
    query_brick,
)
from fetch_tractor_rows import COLUMNS as FETCH_COLUMNS
from grec_host_selection import (
    MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL,
    ROLE_SEED,
    select_grec_hosts,
)
from tractor_fallback import COLUMNS as TRACTOR_COLUMNS
from tractor_fallback import evaluate_eligibility

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_v2/canonical_test_host_repair_v1/selection"
EXPECTED_SHA256 = {
    "eligible": "91147024bd10b046024abee3012ced583a128b33cb8cbda"
    "2eb901f0f1290e43e",
    "test": "3a7a2a616bca2b31d0c0ec00bf86d15e87356d298087e14f"
    "a7ca6021481b6e2b",
    "threshold_calibration": "435119ef16722aac70eeaa8fa4e4c601011b3c120a"
    "5fa79c96a562fb1d42c46e",
    "achieved_fpr": "2dd559dbde377c53916b2a52925a8e10fc0f16b7e079162a"
    "02195e91a616b96f",
    "prior_selection": "60429cc8f00de5ffcb05b44a93a61ae30d7109ee99842b56"
    "ac192d73bd94b023",
}
EXPECTED_ALLOWED_ROWS = 27_950
EXPECTED_ALLOWED_GROUPS = 2_243
EXPECTED_ALLOWED_ID_SHA256 = (
    "ffd336ec7240a79fa5d0b4e1e04abc57b26bdcc3599d200ff3ec26e8b72a8f16"
)
EXPECTED_SELECTED_ID_SHA256 = (
    "9d882cb84e0d12ad13748ca3dcc8a6a9827420108bfc39c2303f28357136c065"
)
EXPECTED_RETAINED = 3_364
EXPECTED_REPLACED = 1_136
HISTORICAL_SAMPLED_DIR = Path(
    os.environ.get(
        "STRONGLENS_HISTORICAL_SAMPLED_DIR",
        ROOT / "data/downloads/historical_sampled_negatives",
    )
)
HISTORICAL_SAMPLED_AGGREGATE_SHA256 = (
    "da298a4142a321a856469229537193a92f3aa97503a2fce7fbf48a6d4f979e16"
)


class RepairError(RuntimeError):
    """The bounded canonical-host repair failed a frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_manifest_sha256(values: pd.Series) -> str:
    material = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(material.encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RepairError(f"required input is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RepairError(f"input hash differs for {path}: {observed} != {expected}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def input_paths() -> dict[str, Path]:
    return {
        "eligible": ROOT
        / "evaluation_dataset_prep/spatial_selection/"
        "eligible_negative_candidates.parquet",
        "test": ROOT / "data/manifests/test.parquet",
        "threshold_calibration": ROOT
        / "results_v2/evaluation_v2/threshold_calibration.parquet",
        "achieved_fpr": ROOT / "results_v2/evaluation_v2/achieved_fpr.parquet",
        "prior_selection": ROOT
        / "evaluation_dataset_prep/final_selection/final_grec_selection.parquet",
    }


def load_frozen_inputs() -> dict[str, pd.DataFrame]:
    paths = input_paths()
    for name, path in paths.items():
        require_hash(path, EXPECTED_SHA256[name])
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def build_allowed_pool(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    test = inputs["test"]
    test_negative = test[
        test["label"].eq(0) & test["endpoint_role"].eq("test_negative")
    ][["galaxy_id", "physical_group_id", "spatial_component_id"]].copy()
    allowed = inputs["eligible"].merge(
        test_negative,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
    )
    reserved_groups = set(
        pd.concat(
            [inputs["threshold_calibration"], inputs["achieved_fpr"]],
            ignore_index=True,
        )["evaluation_spatial_group_id"].astype(str)
    )
    allowed = allowed[
        ~allowed["evaluation_spatial_group_id"].astype(str).isin(reserved_groups)
    ].copy()
    if len(allowed) != EXPECTED_ALLOWED_ROWS:
        raise RepairError(
            f"allowed pool has {len(allowed)} rows, expected {EXPECTED_ALLOWED_ROWS}"
        )
    groups = allowed["evaluation_spatial_group_id"].nunique()
    if groups != EXPECTED_ALLOWED_GROUPS:
        raise RepairError(
            f"allowed pool has {groups} groups, expected {EXPECTED_ALLOWED_GROUPS}"
        )
    digest = id_manifest_sha256(allowed["galaxy_id"])
    if digest != EXPECTED_ALLOWED_ID_SHA256:
        raise RepairError(f"allowed pool identity digest differs: {digest}")
    return allowed


def verify_selection(
    selected: pd.DataFrame,
    allowed: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    prior = inputs["prior_selection"]
    selected_ids = set(selected["galaxy_id"].astype(str))
    allowed_ids = set(allowed["galaxy_id"].astype(str))
    prior_ids = set(prior["galaxy_id"].astype(str))
    if len(selected) != 4_500 or len(selected_ids) != 4_500:
        raise RepairError("repaired selection is not 4,500 unique hosts")
    if not selected_ids <= allowed_ids:
        raise RepairError("repaired selection contains an ineligible host")
    selected_digest = id_manifest_sha256(selected["galaxy_id"])
    if selected_digest != EXPECTED_SELECTED_ID_SHA256:
        raise RepairError(f"selected host identity digest differs: {selected_digest}")
    retained = selected_ids & prior_ids
    removed = prior_ids - selected_ids
    added = selected_ids - prior_ids
    if (
        len(retained) != EXPECTED_RETAINED
        or len(removed) != EXPECTED_REPLACED
        or len(added) != EXPECTED_REPLACED
    ):
        raise RepairError(
            "repair continuity differs: "
            f"retained={len(retained)} removed={len(removed)} added={len(added)}"
        )
    counts = selected.groupby(["psf_bin", "depth_bin"]).size()
    if len(counts) != 9 or not counts.eq(500).all():
        raise RepairError("repaired selection is not nine cells of 500")
    group_sizes = selected.groupby(
        ["psf_bin", "depth_bin", "evaluation_spatial_group_id"]
    ).size()
    if group_sizes.max() > MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL:
        raise RepairError("repaired selection exceeds the per-cell group cap")
    if (
        selected.groupby(["psf_bin", "depth_bin"])[
            "evaluation_spatial_group_id"
        ]
        .nunique()
        .lt(100)
        .any()
    ):
        raise RepairError("a repaired cell has fewer than 100 spatial groups")
    if selected["payload_sha256"].duplicated().any():
        raise RepairError("repaired selection contains duplicate payload bytes")
    reserved = pd.concat(
        [inputs["threshold_calibration"], inputs["achieved_fpr"]],
        ignore_index=True,
    )
    if selected_ids & set(reserved["galaxy_id"].astype(str)):
        raise RepairError("repaired selection overlaps a frozen negative role")
    if set(selected["evaluation_spatial_group_id"].astype(str)) & set(
        reserved["evaluation_spatial_group_id"].astype(str)
    ):
        raise RepairError("repaired selection shares a frozen role spatial group")
    prior_cell = prior.set_index("galaxy_id")[["psf_bin", "depth_bin"]]
    retained_rows = selected[selected["galaxy_id"].astype(str).isin(retained)]
    per_cell_retained = (
        retained_rows.groupby(["psf_bin", "depth_bin"]).size().sort_index()
    )
    for row in retained_rows.itertuples(index=False):
        old = prior_cell.loc[str(row.galaxy_id)]
        if (
            float(old["psf_bin"]) != float(row.psf_bin)
            or float(old["depth_bin"]) != float(row.depth_bin)
        ):
            raise RepairError("a retained host changed condition cell")
    return {
        "selected_id_sha256": selected_digest,
        "retained_ids": sorted(retained),
        "removed_ids": sorted(removed),
        "added_ids": sorted(added),
        "per_cell_retained": {
            f"{psf:.2f}/{depth:.1f}": int(value)
            for (psf, depth), value in per_cell_retained.items()
        },
    }


def select(output: Path) -> None:
    inputs = load_frozen_inputs()
    allowed = build_allowed_pool(inputs)
    selected, support, selected_group_counts = select_grec_hosts(allowed)
    continuity = verify_selection(selected, allowed, inputs)
    allowed_path = output / "allowed_pool.parquet"
    selected_path = output / "proposed_grec_hosts.parquet"
    support_path = output / "condition_support.parquet"
    atomic_parquet(allowed_path, allowed.sort_values("galaxy_id"))
    atomic_parquet(
        selected_path,
        selected.sort_values(["psf_bin", "depth_bin", "galaxy_id"]),
    )
    atomic_parquet(
        support_path, support.sort_values(["psf_bin", "depth_bin"])
    )
    summary = {
        "schema_version": 1,
        "status": "score_blind_role_proposal_not_final_assignment",
        "repair_status": "canonical_test_negative_reselection_frozen",
        "role_seed": ROLE_SEED,
        "pool_rule": (
            "eligible intersect canonical test_negative; exclude frozen "
            "threshold-calibration and achieved-FPR spatial groups"
        ),
        "input_rows": len(allowed),
        "input_spatial_groups": int(
            allowed["evaluation_spatial_group_id"].nunique()
        ),
        "proposed_grec_hosts": len(selected),
        "proposed_grec_rows_at_eleven_radii": len(selected) * 11,
        "retained_condition_cells": int(
            selected[["psf_bin", "depth_bin"]].drop_duplicates().shape[0]
        ),
        "minimum_spatial_groups_per_retained_cell": 100,
        "maximum_hosts_per_spatial_group_per_cell": (
            MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL
        ),
        "retained_cell_spatial_groups": {
            f"{psf:.2f}/{depth:.1f}": int(value)
            for (psf, depth), value in selected_group_counts.items()
        },
        "retained_from_prior_galaxy_ids": len(continuity["retained_ids"]),
        "replaced_galaxy_ids": len(continuity["added_ids"]),
        "per_cell_retained_from_prior": continuity["per_cell_retained"],
        "detector_scores_used": False,
        "input_sha256": {
            name: EXPECTED_SHA256[name] for name in sorted(EXPECTED_SHA256)
        },
        "allowed_pool_sha256": sha256_file(allowed_path),
        "allowed_pool_galaxy_id_manifest_sha256": EXPECTED_ALLOWED_ID_SHA256,
        "condition_support_sha256": sha256_file(support_path),
        "proposed_grec_hosts_sha256": sha256_file(selected_path),
        "proposed_grec_host_ids_sha256": continuity["selected_id_sha256"],
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    atomic_json(
        output / "ROLE_PROPOSAL_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "score_blind_role_proposal_complete_not_final",
            "repair_status": "canonical_test_negative_reselection_frozen",
            "summary_sha256": sha256_file(summary_path),
            "condition_support_sha256": sha256_file(support_path),
            "proposed_grec_hosts_sha256": sha256_file(selected_path),
        },
    )
    atomic_json(
        output / "GREC_TEST_NEGATIVE_RESELECTION_RECEIPT.json",
        {
            "schema_version": 1,
            "status": "grec_test_negative_reselection_frozen",
            "selection_rule": (
                "frozen G_rec nine-cell support, group cap, rank, and hash rule"
            ),
            "role_seed_grec": ROLE_SEED,
            "allowed_pool_rows": len(allowed),
            "allowed_pool_spatial_groups": int(
                allowed["evaluation_spatial_group_id"].nunique()
            ),
            "allowed_pool_galaxy_id_manifest_sha256": (
                EXPECTED_ALLOWED_ID_SHA256
            ),
            "prior_selection_manifest_sha256": (
                EXPECTED_SHA256["prior_selection"]
            ),
            "new_selection_manifest_sha256": sha256_file(selected_path),
            "new_selection_galaxy_id_manifest_sha256": (
                continuity["selected_id_sha256"]
            ),
            "rows": len(selected),
            "cells": 9,
            "hosts_per_cell": 500,
            "retained_from_prior_galaxy_ids": len(continuity["retained_ids"]),
            "replaced_galaxy_ids": len(continuity["added_ids"]),
            "per_cell_retained_from_prior": continuity["per_cell_retained"],
            "all_hosts_canonical_test_negative": True,
            "disjoint_calibration_galaxy_ids": 0,
            "disjoint_achieved_fpr_galaxy_ids": 0,
            "disjoint_calibration_spatial_groups": 0,
            "disjoint_achieved_fpr_spatial_groups": 0,
            "detector_scores_used": False,
            "input_sha256": {
                name: EXPECTED_SHA256[name] for name in sorted(EXPECTED_SHA256)
            },
        },
    )
    print(selected_path)


def tractor_record(row: Any) -> dict[str, Any]:
    return {
        column: getattr(row, f"{column}_tractor", getattr(row, column, None))
        for column in TRACTOR_COLUMNS
    }


def fetch_new(output: Path, tractor_dir: Path, workers: int) -> None:
    if not 1 <= workers <= 16:
        raise RepairError("workers must be between 1 and 16")
    selected_path = output / "proposed_grec_hosts.parquet"
    if not selected_path.is_file():
        raise RepairError("run select before fetching repaired Tractor rows")
    selected = pd.read_parquet(selected_path)
    prior = pd.read_parquet(input_paths()["prior_selection"])
    new_hosts = selected[
        ~selected["galaxy_id"].astype(str).isin(
            set(prior["galaxy_id"].astype(str))
        )
    ].copy()
    if len(new_hosts) != EXPECTED_REPLACED:
        raise RepairError(
            f"new-host count is {len(new_hosts)}, expected {EXPECTED_REPLACED}"
        )
    new_hosts["objid"] = (
        new_hosts["galaxy_id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    )
    expected_pairs = set(
        zip(new_hosts["brickname"].astype(str), new_hosts["objid"], strict=True)
    )
    groups = {
        str(brick): sorted(group["objid"].astype(int).tolist())
        for brick, group in new_hosts.groupby("brickname")
    }
    tractor_dir.mkdir(parents=True, exist_ok=True)
    new_host_path = tractor_dir / "new_host_request.parquet"
    atomic_parquet(
        new_host_path,
        new_hosts.sort_values("galaxy_id")[
            ["galaxy_id", "brickname", "objid", "ra", "dec"]
        ],
    )
    connection = initialize(
        tractor_dir / "new_tractor_queries.sqlite",
        sha256_file(new_host_path),
        sha256_file(Path(__file__).resolve()),
    )
    completed: set[str] = set()
    for brick, requested_json, rows_json in connection.execute(
        """
        SELECT brickname,requested_ids_json,rows_json
        FROM queries WHERE status='complete'
        """
    ):
        if brick not in groups or json.loads(requested_json) != groups[brick]:
            raise RepairError(f"resumed Tractor request differs for brick {brick}")
        frame = pd.read_json(rows_json, orient="records")
        if list(frame.columns) != FETCH_COLUMNS:
            raise RepairError(f"resumed Tractor schema differs for brick {brick}")
        completed.add(str(brick))
    pending = [(brick, ids) for brick, ids in groups.items() if brick not in completed]
    print(
        f"new_hosts={len(new_hosts)} bricks={len(groups)} "
        f"resumed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    limiter = RateLimiter(requests_per_second=4.0)

    def work(item: tuple[str, list[int]]) -> tuple[str, str, str | None, str | None]:
        brick, ids = item
        try:
            frame = query_brick(brick, ids, limiter)
            returned = set(
                zip(frame["brickname"], frame["objid"].astype(int), strict=True)
            )
            expected = {(brick, value) for value in ids}
            if returned != expected or len(frame) != len(ids):
                raise ValueError("catalogue response membership differs")
            return brick, "complete", None, frame.to_json(orient="records")
        except Exception as error:  # noqa: BLE001
            return brick, "failed", f"{type(error).__name__}: {error}", None

    executor = ThreadPoolExecutor(max_workers=workers)
    iterator = iter(pending)
    in_flight = {
        executor.submit(work, item)
        for item in (next(iterator, None) for _ in range(max(workers * 2, 1)))
        if item is not None
    }
    processed = 0
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
                replacement = next(iterator, None)
                if replacement is not None:
                    in_flight.add(executor.submit(work, replacement))
            if processed % 25 == 0:
                connection.commit()
                print(f"processed={processed}/{len(pending)}", flush=True)
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
    failures: list[dict[str, str]] = []
    for brick, status, error, rows_json in connection.execute(
        "SELECT brickname,status,error,rows_json FROM queries ORDER BY brickname"
    ):
        if status != "complete":
            failures.append({"brickname": brick, "error": str(error)})
        else:
            records.extend(json.loads(rows_json))
    rows = pd.DataFrame(records)
    returned_pairs = set(
        zip(rows["brickname"], rows["objid"].astype(int), strict=True)
    )
    if failures or returned_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise RepairError(
            f"Tractor fetch incomplete: failures={len(failures)} "
            f"returned={len(returned_pairs)} expected={len(expected_pairs)}"
        )
    rows = rows[FETCH_COLUMNS].sort_values(["brickname", "objid"])
    rows_path = tractor_dir / "new_tractor_rows.parquet"
    atomic_parquet(rows_path, rows)
    atomic_json(
        tractor_dir / "NEW_TRACTOR_FETCH_RECEIPT.json",
        {
            "schema_version": 1,
            "status": "canonical_replacement_tractor_rows_complete",
            "new_hosts": len(new_hosts),
            "bricks": len(groups),
            "new_host_request_sha256": sha256_file(new_host_path),
            "tractor_rows_sha256": sha256_file(rows_path),
            "query_contract_columns": FETCH_COLUMNS,
            "failures": [],
        },
    )
    print(rows_path)


def query_tractor_batch(pairs: list[tuple[str, int]]) -> pd.DataFrame:
    conditions = " OR ".join(
        f"(brickname='{brick}' AND objid={objid})" for brick, objid in pairs
    )
    sql = (
        f"SELECT {','.join(FETCH_COLUMNS)} FROM ls_dr10.tractor "
        f"WHERE {conditions}"
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
        url, headers={"X-DL-AuthToken": ANONYMOUS_TOKEN}
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read(1_000_001)
                if len(body) > 1_000_000:
                    raise ValueError("catalogue batch response exceeds size limit")
            frame = pd.read_csv(io.BytesIO(body))
            if list(frame.columns) != FETCH_COLUMNS:
                raise ValueError("catalogue batch response has unexpected columns")
            return frame
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(float(2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def fetch_new_batched(output: Path, tractor_dir: Path, workers: int) -> None:
    if not 1 <= workers <= 16:
        raise RepairError("workers must be between 1 and 16")
    selected_path = output / "proposed_grec_hosts.parquet"
    if not selected_path.is_file():
        raise RepairError("run select before fetching repaired Tractor rows")
    selected = pd.read_parquet(selected_path)
    prior = pd.read_parquet(input_paths()["prior_selection"])
    new_hosts = selected[
        ~selected["galaxy_id"].astype(str).isin(
            set(prior["galaxy_id"].astype(str))
        )
    ].copy()
    if len(new_hosts) != EXPECTED_REPLACED:
        raise RepairError("replacement-host count differs before batch fetch")
    new_hosts["objid"] = (
        new_hosts["galaxy_id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    )
    pairs = sorted(
        zip(new_hosts["brickname"].astype(str), new_hosts["objid"], strict=True)
    )
    batch_size = 25
    batches = [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]
    print(
        f"new_hosts={len(new_hosts)} batches={len(batches)} workers={workers}",
        flush=True,
    )
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    def work(
        item: tuple[int, list[tuple[str, int]]],
    ) -> tuple[int, pd.DataFrame | None, str | None]:
        index, batch = item
        try:
            frame = query_tractor_batch(batch)
            returned = set(
                zip(frame["brickname"], frame["objid"].astype(int), strict=True)
            )
            if returned != set(batch) or len(frame) != len(batch):
                raise ValueError("catalogue batch membership differs")
            return index, frame, None
        except Exception as error:  # noqa: BLE001
            return index, None, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(work, item) for item in enumerate(batches)
        ]
        for completed, future in enumerate(futures, start=1):
            index, frame, error = future.result()
            if error is not None:
                failures.append(f"batch {index}: {error}")
            elif frame is not None:
                frames.append(frame)
            if completed % 10 == 0 or completed == len(futures):
                print(f"completed={completed}/{len(futures)}", flush=True)
    if failures:
        raise RepairError(f"Tractor batch fetch failed: {failures}")
    rows = pd.concat(frames, ignore_index=True)
    returned_pairs = set(
        zip(rows["brickname"], rows["objid"].astype(int), strict=True)
    )
    if returned_pairs != set(pairs) or len(rows) != len(pairs):
        raise RepairError("combined Tractor batch membership differs")
    tractor_dir.mkdir(parents=True, exist_ok=True)
    new_host_path = tractor_dir / "new_host_request.parquet"
    atomic_parquet(
        new_host_path,
        new_hosts.sort_values("galaxy_id")[
            ["galaxy_id", "brickname", "objid", "ra", "dec"]
        ],
    )
    rows = rows[FETCH_COLUMNS].sort_values(["brickname", "objid"])
    rows_path = tractor_dir / "new_tractor_rows.parquet"
    atomic_parquet(rows_path, rows)
    atomic_json(
        tractor_dir / "NEW_TRACTOR_FETCH_RECEIPT.json",
        {
            "schema_version": 1,
            "status": "canonical_replacement_tractor_rows_complete",
            "new_hosts": len(new_hosts),
            "batches": len(batches),
            "batch_size_maximum": batch_size,
            "new_host_request_sha256": sha256_file(new_host_path),
            "tractor_rows_sha256": sha256_file(rows_path),
            "query_contract_columns": FETCH_COLUMNS,
            "failures": [],
        },
    )
    print(rows_path)


def historical_catalogue_rows(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not HISTORICAL_SAMPLED_DIR.is_dir():
        raise RepairError("historical sampled-negative authority is absent")
    columns = [
        "galaxy_id",
        "brickname",
        "objid",
        "ra",
        "dec",
        "type",
        "nobs_z",
        "flux_g",
        "flux_r",
        "flux_z",
        "psfsize_g",
        "psfsize_r",
        "psfsize_z",
        "psfdepth_g",
        "psfdepth_r",
        "psfdepth_z",
        "galdepth_r",
        "maskbits",
        "shape_r",
        "shape_e1",
        "shape_e2",
        "sersic",
    ]
    dataset = pads.dataset(HISTORICAL_SAMPLED_DIR, format="parquet")
    ids = sorted(selected["galaxy_id"].astype(str))
    historical = dataset.to_table(
        columns=columns,
        filter=pads.field("galaxy_id").isin(ids),
    ).to_pandas()
    if (
        len(historical) != 4_500
        or historical["galaxy_id"].astype(str).nunique() != 4_500
    ):
        raise RepairError("historical authority does not cover all repaired hosts")
    fetched = selected.merge(
        historical,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_payload", "_tractor"),
    )
    fetched["brickname"] = fetched["brickname_payload"]
    if not fetched["brickname"].eq(fetched["brickname_tractor"]).all():
        raise RepairError("historical brick identity differs from payload identity")
    fetched["brick_primary"] = 1
    fetched["nobs_g"] = pd.NA
    fetched["nobs_r"] = pd.NA
    fetched["_tractor_source"] = "frozen_historical_sampled_negative"
    fetched["_tractor_found"] = True
    delta_ra = np.deg2rad(
        fetched["ra_payload"].to_numpy(float)
        - fetched["ra_tractor"].to_numpy(float)
    )
    delta_dec = np.deg2rad(
        fetched["dec_payload"].to_numpy(float)
        - fetched["dec_tractor"].to_numpy(float)
    )
    dec_payload = np.deg2rad(fetched["dec_payload"].to_numpy(float))
    dec_tractor = np.deg2rad(fetched["dec_tractor"].to_numpy(float))
    haversine = np.sin(delta_dec / 2.0) ** 2 + (
        np.cos(dec_payload) * np.cos(dec_tractor) * np.sin(delta_ra / 2.0) ** 2
    )
    fetched["payload_tractor_separation_arcsec"] = (
        2.0
        * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        * 180.0
        / np.pi
        * 3600.0
    )
    if fetched["payload_tractor_separation_arcsec"].max() > 0.05:
        raise RepairError("historical catalogue coordinates differ from payloads")

    prior = pd.read_parquet(
        ROOT / "evaluation_dataset_prep/tractor_fallback_full/tractor_hosts.parquet"
    )
    retained_ids = set(fetched["galaxy_id"].astype(str)) & set(
        prior["galaxy_id"].astype(str)
    )
    if len(retained_ids) != EXPECTED_RETAINED:
        raise RepairError("historical/current comparison support differs")
    comparison = fetched[
        fetched["galaxy_id"].astype(str).isin(retained_ids)
    ].merge(
        prior,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_historical", "_current"),
    )
    numeric_fields = [
        "ra",
        "dec",
        "flux_g",
        "flux_r",
        "flux_z",
        "psfsize_g",
        "psfsize_r",
        "psfsize_z",
        "psfdepth_g",
        "psfdepth_r",
        "psfdepth_z",
        "galdepth_r",
        "shape_r",
        "shape_e1",
        "shape_e2",
        "sersic",
    ]
    maximum_differences: dict[str, float] = {}
    for field in numeric_fields:
        left_name = (
            f"{field}_tractor_historical"
            if f"{field}_tractor_historical" in comparison
            else f"{field}_historical"
        )
        right_name = (
            f"{field}_tractor_current"
            if f"{field}_tractor_current" in comparison
            else f"{field}_current"
        )
        left = comparison[left_name].to_numpy(dtype=float)
        right = comparison[right_name].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=True):
            raise RepairError(f"historical/current field differs: {field}")
        maximum_differences[field] = float(np.nanmax(np.abs(left - right)))
    dispositions = pd.DataFrame(
        {
            "galaxy_id": fetched["galaxy_id"].astype(str),
            "eligible": True,
            "source": "frozen_canonical_test_negative_membership",
            "disposition_reasons_json": "[]",
            "eligibility_basis": (
                "retained 409551-negative universe and canonical test assignment"
            ),
        }
    )
    evidence = {
        "catalogue_source": "frozen_historical_sampled_negative",
        "historical_sampled_negative_aggregate_sha256": (
            HISTORICAL_SAMPLED_AGGREGATE_SHA256
        ),
        "historical_current_comparison_rows": len(comparison),
        "historical_current_allclose_rtol": 1e-6,
        "historical_current_allclose_atol": 1e-8,
        "historical_current_maximum_absolute_differences": maximum_differences,
    }
    prior_by_id = prior.set_index(prior["galaxy_id"].astype(str))
    retained_mask = fetched["galaxy_id"].astype(str).isin(retained_ids)
    exact_fields = [
        "brickname",
        "objid",
        "brick_primary",
        "maskbits",
        "type",
        "ra_tractor",
        "dec_tractor",
        "flux_g",
        "flux_r",
        "flux_z",
        "nobs_z_tractor",
        "psfsize_g",
        "psfsize_r_tractor",
        "psfsize_z",
        "psfdepth_g",
        "psfdepth_r_tractor",
        "psfdepth_z",
        "galdepth_r_tractor",
        "sersic",
        "shape_r",
        "shape_e1",
        "shape_e2",
        "_tractor_source",
        "_tractor_found",
        "payload_tractor_separation_arcsec",
    ]
    for field in exact_fields:
        fetched[field] = fetched[field].astype(prior[field].dtype)
        replacements = fetched.loc[retained_mask, "galaxy_id"].astype(str).map(
            prior_by_id[field]
        )
        replacements = replacements.astype(prior[field].dtype)
        fetched.loc[retained_mask, field] = replacements.to_numpy()
    evidence["retained_current_tractor_rows_carried_forward"] = len(retained_ids)
    return fetched, dispositions, evidence


def finalize(
    output: Path,
    tractor_dir: Path,
    catalogue_source: str,
) -> None:
    inputs = load_frozen_inputs()
    selected_path = output / "proposed_grec_hosts.parquet"
    reselection_receipt_path = output / "GREC_TEST_NEGATIVE_RESELECTION_RECEIPT.json"
    fallback_tractor_path = tractor_dir / "tractor_hosts.parquet"
    fallback_dispositions_path = tractor_dir / "eligibility_dispositions.parquet"
    fallback_summary_path = tractor_dir / "summary.json"
    fallback_complete_path = tractor_dir / "COMPLETE.json"
    new_tractor_path = tractor_dir / "new_tractor_rows.parquet"
    tractor_fetch_receipt_path = tractor_dir / "NEW_TRACTOR_FETCH_RECEIPT.json"
    prior_tractor_path = (
        ROOT / "evaluation_dataset_prep/tractor_fallback_full/tractor_hosts.parquet"
    )
    for path in (selected_path, reselection_receipt_path):
        if not path.is_file():
            raise RepairError(f"required finalization input is absent: {path}")
    selected = pd.read_parquet(selected_path)
    tractor_evidence: dict[str, Any]
    if catalogue_source == "historical":
        fetched, dispositions_frame, tractor_evidence = historical_catalogue_rows(
            selected
        )
        failed = dispositions_frame[~dispositions_frame["eligible"].astype(bool)]
    elif catalogue_source == "fallback":
        for path in (
            fallback_dispositions_path,
            fallback_summary_path,
            fallback_complete_path,
        ):
            if not path.is_file():
                raise RepairError(f"fallback finalization input is absent: {path}")
        fetched = pd.read_parquet(fallback_tractor_path)
        dispositions_frame = pd.read_parquet(fallback_dispositions_path)
        failed = dispositions_frame[~dispositions_frame["eligible"].astype(bool)]
        tractor_evidence = {
            "tractor_manifest_sha256": sha256_file(fallback_tractor_path),
            "tractor_summary_sha256": sha256_file(fallback_summary_path),
            "tractor_completion_sha256": sha256_file(fallback_complete_path),
        }
    elif catalogue_source == "datalab":
        for path in (
            new_tractor_path,
            tractor_fetch_receipt_path,
            prior_tractor_path,
        ):
            if not path.is_file():
                raise RepairError(f"required finalization input is absent: {path}")
        prior_tractor = pd.read_parquet(prior_tractor_path)
        retained_ids = set(selected["galaxy_id"].astype(str)) & set(
            prior_tractor["galaxy_id"].astype(str)
        )
        if len(retained_ids) != EXPECTED_RETAINED:
            raise RepairError("prior Tractor rows do not cover exactly 3,364 hosts")
        retained = prior_tractor[
            prior_tractor["galaxy_id"].astype(str).isin(retained_ids)
        ].copy()
        new_selected = selected[
            ~selected["galaxy_id"].astype(str).isin(retained_ids)
        ].copy()
        new_selected["objid"] = (
            new_selected["galaxy_id"]
            .astype(str)
            .str.rsplit("_", n=1)
            .str[1]
            .astype(int)
        )
        new_tractor = pd.read_parquet(new_tractor_path)
        new_fetched = new_selected.merge(
            new_tractor,
            on=["brickname", "objid"],
            how="inner",
            validate="one_to_one",
            suffixes=("_payload", "_tractor"),
        )
        new_fetched["_tractor_source"] = "datalab_repair_query"
        new_fetched["_tractor_found"] = True
        fetched = pd.concat([retained, new_fetched], ignore_index=True, sort=False)
        coordinates = {
            (str(row.brickname), int(row.objid)): (
                float(row.ra_payload),
                float(row.dec_payload),
            )
            for row in fetched.itertuples(index=False)
        }
        dispositions = [
            evaluate_eligibility(
                tractor_record(row),
                coordinates,
                source="datalab_repair_query",
            )
            for row in fetched.itertuples(index=False)
        ]
        dispositions_frame = pd.DataFrame(dispositions)
        failed = dispositions_frame[~dispositions_frame["eligible"].astype(bool)]
        tractor_evidence = {
            "prior_tractor_manifest_sha256": sha256_file(prior_tractor_path),
            "new_tractor_rows_sha256": sha256_file(new_tractor_path),
            "new_tractor_fetch_receipt_sha256": sha256_file(
                tractor_fetch_receipt_path
            ),
        }
    else:
        raise RepairError(f"unsupported catalogue source: {catalogue_source}")
    if set(selected["galaxy_id"].astype(str)) != set(
        fetched["galaxy_id"].astype(str)
    ):
        raise RepairError("fetched Tractor rows differ from repaired hosts")
    atomic_parquet(output / "eligibility_dispositions.parquet", dispositions_frame)
    if len(failed):
        reasons: dict[str, int] = {}
        for value in failed.to_dict(orient="records"):
            disposition_reasons = value.get(
                "disposition_reasons",
                json.loads(value.get("disposition_reasons_json", "[]")),
            )
            for reason in disposition_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        raise RepairError(f"{len(failed)} repaired hosts failed eligibility: {reasons}")

    test_negative = inputs["test"][
        inputs["test"]["label"].eq(0)
        & inputs["test"]["endpoint_role"].eq("test_negative")
    ][["galaxy_id", "physical_group_id", "spatial_component_id"]]
    final = fetched.merge(
        test_negative,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_test"),
    )
    if len(final) != 4_500 or final["physical_group_id"].isna().any():
        raise RepairError("canonical physical groups do not cover all repaired hosts")
    if "physical_group_id_test" in final:
        if not final["physical_group_id"].astype(str).eq(
            final["physical_group_id_test"].astype(str)
        ).all():
            raise RepairError("canonical physical-group bindings differ")
        final = final.drop(
            columns=["physical_group_id_test", "spatial_component_id_test"],
            errors="ignore",
        )
    reserved = pd.concat(
        [inputs["threshold_calibration"], inputs["achieved_fpr"]],
        ignore_index=True,
    )
    if set(final["physical_group_id"].astype(str)) & set(
        reserved["physical_group_id"].astype(str)
    ):
        raise RepairError("repaired host physical groups cross a frozen role")
    final["final_role"] = "experiment_host"
    final["pixel_unit"] = "nanomaggy_per_native_pixel"
    final["flux_calibration_system"] = "AB_nanomaggy_zeropoint_22.5"
    final["ra"] = final["ra_payload"]
    final["dec"] = final["dec_payload"]
    final["psfsize_r"] = final["psfsize_r_tractor"]
    final["psfdepth_r"] = final["psfdepth_r_tractor"]
    final["pixel_scale"] = final["pixel_scale"].astype(float)
    final = final.drop(columns=["proposed_role"], errors="ignore")
    calibration_path = (
        ROOT / "evaluation_dataset_prep/final_selection/host_flux_calibration.json"
    )
    if not calibration_path.is_file():
        raise RepairError("host flux-calibration authority is absent")
    final["calibration_provenance_path"] = str(calibration_path)
    final["calibration_provenance_sha256"] = sha256_file(calibration_path)
    final_path = output / "final_grec_selection.parquet"
    atomic_parquet(
        final_path, final.sort_values(["psf_bin", "depth_bin", "galaxy_id"])
    )
    prior = inputs["prior_selection"]
    overlap = set(final["galaxy_id"].astype(str)) & set(
        prior["galaxy_id"].astype(str)
    )
    if len(overlap) != EXPECTED_RETAINED:
        raise RepairError("finalized repair does not retain exactly 3,364 hosts")
    receipt = {
        "schema_version": 1,
        "status": "final_experiment_host_selection_approved",
        "selection_manifest_sha256": sha256_file(final_path),
        "rows": len(final),
        "cells": 9,
        "hosts_per_cell": 500,
        "retained_from_prior_galaxy_ids": len(overlap),
        "replaced_galaxy_ids": EXPECTED_REPLACED,
        "detector_scores_used": False,
        "full_tractor_eligibility": "passed",
        "physical_group_checks": "passed",
        "physical_group_method": "canonical_test_negative_physical_group_only",
        "role_overlap_checks": "passed",
        "role_overlap_scope": "frozen_calibration_and_achieved_fpr_roles",
        "pre_score_review": "approved",
        "pre_score_review_basis": (
            "author_requested_uncommitted_canonical_population_comparison"
        ),
        "reference_population": "canonical_test_negative",
        "reselection_receipt_sha256": sha256_file(reselection_receipt_path),
        **tractor_evidence,
        "eligibility_dispositions_sha256": sha256_file(
            output / "eligibility_dispositions.parquet"
        ),
        "input_sha256": {
            name: EXPECTED_SHA256[name] for name in sorted(EXPECTED_SHA256)
        },
    }
    atomic_json(output / "FINAL_SELECTION_RECEIPT.json", receipt)
    print(final_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "fetch-new", "finalize"),
        help="build the score-blind proposal or finalize fetched Tractor rows",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tractor-dir",
        type=Path,
        default=ROOT / "results_v2/canonical_test_host_repair_v1/tractor",
    )
    parser.add_argument(
        "--catalogue-source",
        choices=("historical", "fallback", "datalab"),
        default="historical",
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.command == "select":
            select(arguments.output_dir)
        elif arguments.command == "fetch-new":
            fetch_new_batched(
                arguments.output_dir, arguments.tractor_dir, arguments.workers
            )
        else:
            finalize(
                arguments.output_dir,
                arguments.tractor_dir,
                arguments.catalogue_source,
            )
    except (RepairError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
