#!/usr/bin/env python3
"""Create a score-blind, whole-group evaluation-role proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from grec_host_selection import (
    MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL,
    MIN_SPATIAL_GROUPS_PER_CONDITION_CELL,
    ROLE_SEED,
    assign_condition_cells,
    select_grec_hosts,
    stable_hash,
)

MIN_CALIBRATION_ROWS = 20_000
MIN_ACHIEVED_FPR_ROWS = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_groups(frame: pd.DataFrame) -> dict[str, str]:
    groups = (
        frame.groupby("evaluation_spatial_group_id")
        .size()
        .rename("rows")
        .reset_index()
    )
    groups["ordering"] = groups["evaluation_spatial_group_id"].map(stable_hash)
    groups = groups.sort_values("ordering").reset_index(drop=True)
    assignment: dict[str, str] = {}
    position = 0
    for role, minimum in (
        ("threshold_calibration", MIN_CALIBRATION_ROWS),
        ("achieved_fpr", MIN_ACHIEVED_FPR_ROWS),
    ):
        selected = 0
        while selected < minimum and position < len(groups):
            row = groups.iloc[position]
            assignment[str(row["evaluation_spatial_group_id"])] = role
            selected += int(row["rows"])
            position += 1
        if selected < minimum:
            raise ValueError(f"insufficient whole-group support for {role}")
    for group_id in groups.iloc[position:]["evaluation_spatial_group_id"]:
        assignment[str(group_id)] = "experiment_host"
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligible",
        type=Path,
        default=Path(
            "evaluation_dataset_prep/spatial_selection/"
            "eligible_negative_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/role_proposal"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = arguments.eligible.parent / "SPATIAL_STAGE_COMPLETE.json"
    summary_path = arguments.eligible.parent / "summary.json"
    if not completion_path.is_file() or not summary_path.is_file():
        raise ValueError("spatial-stage completion evidence is missing")
    completion = json.loads(completion_path.read_text())
    spatial_summary = json.loads(summary_path.read_text())
    input_digest = sha256_file(arguments.eligible)
    if (
        completion.get("eligible_manifest_sha256") != input_digest
        or spatial_summary.get("eligible_manifest_sha256") != input_digest
        or completion.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError("eligible manifest does not match spatial-stage evidence")
    (arguments.output_dir / "ROLE_PROPOSAL_COMPLETE.json").unlink(missing_ok=True)

    frame = pd.read_parquet(arguments.eligible)
    required = {
        "galaxy_id",
        "evaluation_spatial_group_id",
        "psfsize_r",
        "psfdepth_r",
        "tractor_type",
        "pool",
        "payload_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"eligible manifest lacks columns: {missing}")
    if len(frame) != spatial_summary.get("eligible_rows"):
        raise ValueError("eligible row count differs from spatial-stage summary")
    if frame["galaxy_id"].duplicated().any():
        raise ValueError("eligible galaxy IDs are not unique")

    assignment = assign_groups(frame)
    frame["proposed_role"] = frame["evaluation_spatial_group_id"].map(assignment)
    if frame["proposed_role"].isna().any():
        raise AssertionError("a spatial group lacks a proposed role")
    if (
        frame.groupby("evaluation_spatial_group_id")["proposed_role"].nunique().max()
        != 1
    ):
        raise AssertionError("one spatial group crosses proposed roles")

    frame = assign_condition_cells(frame)

    experiment = frame[frame["proposed_role"].eq("experiment_host")].copy()
    grec_hosts, support, selected_group_counts = select_grec_hosts(experiment)
    counts = grec_hosts.groupby(["psf_bin", "depth_bin"]).size()

    role_path = arguments.output_dir / "proposed_roles.parquet"
    support_path = arguments.output_dir / "condition_support.parquet"
    grec_path = arguments.output_dir / "proposed_grec_hosts.parquet"
    role_tmp = arguments.output_dir / "proposed_roles.partial.parquet"
    support_tmp = arguments.output_dir / "condition_support.partial.parquet"
    grec_tmp = arguments.output_dir / "proposed_grec_hosts.partial.parquet"
    frame.sort_values("galaxy_id").to_parquet(role_tmp, index=False)
    support.to_parquet(support_tmp, index=False)
    grec_hosts.sort_values(["psf_bin", "depth_bin", "galaxy_id"]).to_parquet(
        grec_tmp, index=False
    )
    role_tmp.replace(role_path)
    support_tmp.replace(support_path)
    grec_tmp.replace(grec_path)

    role_counts = frame["proposed_role"].value_counts().to_dict()
    summary = {
        "schema_version": 1,
        "status": "score_blind_role_proposal_not_final_assignment",
        "role_seed": ROLE_SEED,
        "input_rows": len(frame),
        "input_spatial_groups": int(
            frame["evaluation_spatial_group_id"].nunique()
        ),
        "role_rows": role_counts,
        "role_spatial_groups": frame.groupby("proposed_role")[
            "evaluation_spatial_group_id"
        ]
        .nunique()
        .to_dict(),
        "retained_condition_cells": len(counts),
        "minimum_spatial_groups_per_retained_cell": (
            MIN_SPATIAL_GROUPS_PER_CONDITION_CELL
        ),
        "maximum_hosts_per_spatial_group_per_cell": (
            MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL
        ),
        "retained_cell_spatial_groups": {
            f"{psf_bin:.2f}/{depth_bin:.1f}": int(value)
            for (psf_bin, depth_bin), value in selected_group_counts.items()
        },
        "proposed_grec_hosts": len(grec_hosts),
        "proposed_grec_rows_at_eleven_radii": len(grec_hosts) * 11,
        "detector_scores_used": False,
        "physical_group_and_full_tractor_eligibility_pending": True,
        "input_sha256": input_digest,
        "spatial_completion_sha256": sha256_file(completion_path),
        "spatial_summary_sha256": sha256_file(summary_path),
        "proposed_roles_sha256": sha256_file(role_path),
        "condition_support_sha256": sha256_file(support_path),
        "proposed_grec_hosts_sha256": sha256_file(grec_path),
    }
    temporary_summary = arguments.output_dir / "summary.partial.json"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_summary = arguments.output_dir / "summary.json"
    temporary_summary.replace(final_summary)
    temporary_completion = arguments.output_dir / "ROLE_PROPOSAL_COMPLETE.partial.json"
    temporary_completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "score_blind_role_proposal_complete_not_final",
                "summary_sha256": sha256_file(final_summary),
                "proposed_roles_sha256": sha256_file(role_path),
                "condition_support_sha256": sha256_file(support_path),
                "proposed_grec_hosts_sha256": sha256_file(grec_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_completion.replace(
        arguments.output_dir / "ROLE_PROPOSAL_COMPLETE.json"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
