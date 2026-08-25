#!/usr/bin/env python3
"""Finalize latest-test negative roles after reserving G_rec spatial groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "evaluation_dataset_prep"
ROLE_SEED = "latest-test-negative-roles-v2"
MIN_CALIBRATION_ROWS = 20_000
MIN_ACHIEVED_FPR_ROWS = 10_000
EXPECTED = {
    "eligible": "91147024bd10b046024abee3012ced583a128b33cb8cbda2eb901f0f1290e43e",
    "final_hosts": "60429cc8f00de5ffcb05b44a93a61ae30d7109ee99842b56ac192d73bd94b023",
    "test": "3a7a2a616bca2b31d0c0ec00bf86d15e87356d298087e14fa7ca6021481b6e2b",
    "train": "a0caa3824568f96010275e59d89d013a23ac7e6e6ae280de2c636a401fa38e5b",
    "validation": "809729e63c9bd6fca96f51b4f8d244aeda604beca4135edcc428f431a9941ee9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_key(group_id: str) -> str:
    return hashlib.sha256(f"{ROLE_SEED}:{group_id}".encode()).hexdigest()


def main() -> int:
    eligible_path = PREP / "spatial_selection/eligible_negative_candidates.parquet"
    hosts_path = PREP / "final_selection/final_grec_selection.parquet"
    test_path = ROOT / "data/manifests/test.parquet"
    train_path = ROOT / "data/manifests/train.parquet"
    validation_path = ROOT / "data/manifests/validation.parquet"
    paths = {
        "eligible": eligible_path,
        "final_hosts": hosts_path,
        "test": test_path,
        "train": train_path,
        "validation": validation_path,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != EXPECTED[name]:
            raise ValueError(f"frozen input changed or is missing: {name}")

    eligible = pd.read_parquet(eligible_path)
    hosts = pd.read_parquet(hosts_path)
    test = pd.read_parquet(test_path)
    train = pd.read_parquet(train_path, columns=["galaxy_id"])
    validation = pd.read_parquet(validation_path, columns=["galaxy_id"])

    test_negative = test[
        test["label"].eq(0) & test["endpoint_role"].eq("test_negative")
    ].copy()
    pool = eligible.merge(
        test_negative[
            [
                "galaxy_id",
                "physical_group_id",
                "spatial_component_id",
                "cutout_relpath",
            ]
        ],
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_test"),
    )
    pool = pool.rename(
        columns={"cutout_relpath": "frozen_test_cutout_relpath"}
    )
    pool["uses_exact_frozen_test_payload"] = (
        pool["s3_key"].astype(str) == pool["frozen_test_cutout_relpath"].astype(str)
    )
    host_groups = set(hosts["evaluation_spatial_group_id"].astype(str))
    pool = pool[
        ~pool["evaluation_spatial_group_id"].astype(str).isin(host_groups)
    ].copy()
    if len(pool) < MIN_CALIBRATION_ROWS + MIN_ACHIEVED_FPR_ROWS:
        raise ValueError("insufficient audited test negatives after host exclusion")
    forbidden = set(train["galaxy_id"].astype(str)) | set(
        validation["galaxy_id"].astype(str)
    )
    if set(pool["galaxy_id"].astype(str)) & forbidden:
        raise ValueError("latest-test negative pool overlaps development")
    if pool["payload_sha256"].duplicated().any():
        raise ValueError("latest-test negative pool has duplicate pixels")

    groups = (
        pool.groupby("evaluation_spatial_group_id")
        .size()
        .rename("rows")
        .reset_index()
    )
    groups["order"] = groups["evaluation_spatial_group_id"].astype(str).map(
        order_key
    )
    groups = groups.sort_values("order").reset_index(drop=True)
    calibration_groups: set[str] = set()
    calibration_rows = 0
    for row in groups.itertuples():
        if calibration_rows >= MIN_CALIBRATION_ROWS:
            break
        calibration_groups.add(str(row.evaluation_spatial_group_id))
        calibration_rows += int(row.rows)

    pool["final_role"] = "achieved_fpr"
    pool.loc[
        pool["evaluation_spatial_group_id"]
        .astype(str)
        .isin(calibration_groups),
        "final_role",
    ] = "threshold_calibration"
    counts = pool["final_role"].value_counts()
    if (
        counts.get("threshold_calibration", 0) < MIN_CALIBRATION_ROWS
        or counts.get("achieved_fpr", 0) < MIN_ACHIEVED_FPR_ROWS
    ):
        raise ValueError("negative roles do not meet their minimum row counts")
    if (
        pool.groupby("evaluation_spatial_group_id")["final_role"].nunique().max()
        != 1
    ):
        raise ValueError("one spatial group crosses negative roles")

    output_dir = ROOT / "results_v2/evaluation_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for role in ("threshold_calibration", "achieved_fpr"):
        frame = pool[pool["final_role"].eq(role)].copy()
        frame["label"] = 0
        frame["label_scope"] = "catalogue_negative"
        frame["global_role"] = role
        frame["sample_weight"] = 1.0 / len(frame)
        frame = frame.sort_values("galaxy_id").reset_index(drop=True)
        path = output_dir / f"{role}.parquet"
        partial = output_dir / f"{role}.partial.parquet"
        frame.to_parquet(partial, index=False)
        partial.replace(path)
        outputs[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "rows": len(frame),
            "spatial_groups": frame["evaluation_spatial_group_id"].nunique(),
            "physical_groups": frame["physical_group_id"].nunique(),
        }

    calibration = pd.read_parquet(ROOT / outputs["threshold_calibration"]["path"])
    achieved = pd.read_parquet(ROOT / outputs["achieved_fpr"]["path"])
    for column in (
        "galaxy_id",
        "payload_sha256",
        "evaluation_spatial_group_id",
        "physical_group_id",
    ):
        if set(calibration[column].astype(str)) & set(achieved[column].astype(str)):
            raise ValueError(f"negative roles overlap through {column}")

    receipt = {
        "schema_version": 1,
        "status": "latest_test_negative_roles_frozen_pre_score",
        "selection_rule": "whole spatial groups in stable hash order",
        "role_seed": ROLE_SEED,
        "detector_scores_used": False,
        "source_population": "latest detector frozen test identities and labels with canonical current S3 pixels",
        "pixel_population": "canonical_current_S3_cutout_version_with_timestamp_only_fallbacks",
        "exact_historical_test_pixel_claim": False,
        "exact_frozen_test_payload_rows": int(
            pool["uses_exact_frozen_test_payload"].sum()
        ),
        "canonical_pixel_rows_differing_from_frozen_test_path": int(
            (~pool["uses_exact_frozen_test_payload"]).sum()
        ),
        "source_test_manifest_sha256": sha256(test_path),
        "train_manifest_sha256": sha256(train_path),
        "validation_manifest_sha256": sha256(validation_path),
        "final_host_manifest_sha256": sha256(hosts_path),
        "eligible_input_sha256": sha256(eligible_path),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "roles": outputs,
        "cross_role_id_pixel_spatial_physical_overlap": 0,
    }
    receipt_path = output_dir / "NEGATIVE_ROLE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
