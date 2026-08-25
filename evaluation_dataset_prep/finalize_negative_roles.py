#!/usr/bin/env python3
"""Finalize score-blind calibration and achieved-FPR negative manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "evaluation_dataset_prep"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    roles_path = PREP / "role_proposal/proposed_roles.parquet"
    role_summary_path = PREP / "role_proposal/summary.json"
    role_receipt_path = PREP / "role_proposal/ROLE_PROPOSAL_COMPLETE.json"
    final_hosts_path = PREP / "final_selection/final_grec_selection.parquet"
    final_host_receipt_path = PREP / "final_selection/FINAL_SELECTION_RECEIPT.json"
    train_path = ROOT / "data/manifests/train.parquet"
    validation_path = ROOT / "data/manifests/validation.parquet"

    roles = pd.read_parquet(roles_path)
    role_summary = json.loads(role_summary_path.read_text())
    role_receipt = json.loads(role_receipt_path.read_text())
    hosts = pd.read_parquet(final_hosts_path)
    host_receipt = json.loads(final_host_receipt_path.read_text())

    if (
        role_receipt.get("status")
        != "score_blind_role_proposal_complete_not_final"
        or role_receipt.get("proposed_roles_sha256") != sha256(roles_path)
        or role_receipt.get("summary_sha256") != sha256(role_summary_path)
        or role_summary.get("detector_scores_used") is not False
        or host_receipt.get("status") != "final_experiment_host_selection_approved"
        or host_receipt.get("selection_manifest_sha256") != sha256(final_hosts_path)
    ):
        raise ValueError("role or host evidence is incomplete")

    forbidden = set(
        pd.read_parquet(train_path, columns=["galaxy_id"])["galaxy_id"].astype(str)
    ) | set(
        pd.read_parquet(validation_path, columns=["galaxy_id"])["galaxy_id"].astype(
            str
        )
    )
    score_columns = [
        column
        for column in roles.columns
        if any(token in column.lower() for token in ("score", "logit", "probability"))
    ]
    if score_columns or set(roles["galaxy_id"].astype(str)) & forbidden:
        raise ValueError("role proposal is not score-blind or development-disjoint")
    if (
        roles.groupby("evaluation_spatial_group_id")["proposed_role"].nunique().max()
        != 1
    ):
        raise ValueError("one spatial group crosses evaluation roles")

    host_groups = set(hosts["evaluation_spatial_group_id"].astype(str))
    output_dir = ROOT / "results_v2/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for proposal_role, final_role, expected_minimum in (
        ("threshold_calibration", "threshold_calibration", 20_000),
        ("achieved_fpr", "achieved_fpr", 20_000),
    ):
        frame = roles[roles["proposed_role"].eq(proposal_role)].copy()
        if len(frame) < expected_minimum:
            raise ValueError(f"{final_role} has fewer than {expected_minimum} rows")
        groups = set(frame["evaluation_spatial_group_id"].astype(str))
        if groups & host_groups:
            raise ValueError(f"{final_role} shares a spatial group with experiment hosts")
        frame["final_role"] = final_role
        frame["label"] = 0
        frame["label_scope"] = "catalogue_negative"
        frame["sample_weight"] = 1.0 / len(frame)
        frame = frame.drop(columns=["proposed_role"], errors="ignore")
        frame = frame.sort_values("galaxy_id").reset_index(drop=True)
        path = output_dir / f"{final_role}.parquet"
        temporary = output_dir / f"{final_role}.partial.parquet"
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        outputs[final_role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "rows": len(frame),
            "spatial_groups": frame["evaluation_spatial_group_id"].nunique(),
        }

    threshold = pd.read_parquet(ROOT / outputs["threshold_calibration"]["path"])
    achieved = pd.read_parquet(ROOT / outputs["achieved_fpr"]["path"])
    for column in ("galaxy_id", "payload_sha256", "evaluation_spatial_group_id"):
        if set(threshold[column].astype(str)) & set(achieved[column].astype(str)):
            raise ValueError(f"negative roles overlap through {column}")

    receipt = {
        "schema_version": 1,
        "status": "negative_evaluation_roles_frozen_pre_score",
        "detector_scores_used": False,
        "train_manifest_sha256": sha256(train_path),
        "validation_manifest_sha256": sha256(validation_path),
        "role_proposal_sha256": sha256(roles_path),
        "final_host_selection_sha256": sha256(final_hosts_path),
        "roles": outputs,
        "cross_role_id_pixel_spatial_overlap": 0,
    }
    receipt_path = output_dir / "NEGATIVE_ROLE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
