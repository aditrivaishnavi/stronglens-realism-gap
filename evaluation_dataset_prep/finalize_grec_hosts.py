#!/usr/bin/env python3
"""Finalize the score-blind G_rec host selection after Tractor validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "evaluation_dataset_prep"
EXPECTED_HASHES = {
    "role_completion": "739e779e3f193175213faf1113062995684f9a2044e383d51cee5fd0f07bafe0",
    "role_summary": "eae3ef0b5ed25a64c5734efa49b8f717cc1c02b44bed0e64564fe759996bb854",
    "tractor_dispositions": "559b55646c328a87d4d471f717be54096029b0fccc50be5c898f3e029341bdc6",
    "tractor_summary": "f61ac0c22d3efe20b5130aac37832061bd482248c387a00d1a68d0a4faf9af41",
    "tractor_completion": "f147b6810098f9d41a8d5ed71cc03741ca00195e5601fca5eb80939bd5f5bd15",
    "manuscript_data_section": "003066b2e352a9c68adc6877553807367d2e19b7ead3565dca27b9659fd752a2",
    "train_manifest": "a0caa3824568f96010275e59d89d013a23ac7e6e6ae280de2c636a401fa38e5b",
    "validation_manifest": "809729e63c9bd6fca96f51b4f8d244aeda604beca4135edcc428f431a9941ee9",
    "test_manifest": "3a7a2a616bca2b31d0c0ec00bf86d15e87356d298087e14fa7ca6021481b6e2b",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    tractor_dir = PREP / "tractor_fallback_full"
    tractor_path = tractor_dir / "tractor_hosts.parquet"
    summary_path = tractor_dir / "summary.json"
    completion_path = tractor_dir / "COMPLETE.json"
    roles_path = PREP / "role_proposal/proposed_roles.parquet"
    grec_path = PREP / "role_proposal/proposed_grec_hosts.parquet"
    role_completion_path = PREP / "role_proposal/ROLE_PROPOSAL_COMPLETE.json"
    role_summary_path = PREP / "role_proposal/summary.json"
    dispositions_path = tractor_dir / "eligibility_dispositions.parquet"

    evidence_paths = {
        "role_completion": role_completion_path,
        "role_summary": role_summary_path,
        "tractor_dispositions": dispositions_path,
        "tractor_summary": summary_path,
        "tractor_completion": completion_path,
        "manuscript_data_section": ROOT / "manuscript/03_data_samples_detector.tex",
    }
    for name, path in evidence_paths.items():
        if not path.is_file() or sha256(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"selection evidence changed or is missing: {name}")

    summary = json.loads(summary_path.read_text())
    completion = json.loads(completion_path.read_text())
    role_completion = json.loads(role_completion_path.read_text())
    role_summary = json.loads(role_summary_path.read_text())
    if (
        summary.get("status") != "complete"
        or summary.get("manifest_hosts") != 4_500
        or summary.get("manifest_tractor_rows_found") != 4_500
        or summary.get("unresolved_hosts") != 0
        or completion.get("manifest_sha256") != sha256(tractor_path)
        or completion.get("summary_sha256") != sha256(summary_path)
        or role_completion.get("status")
        != "score_blind_role_proposal_complete_not_final"
        or role_completion.get("summary_sha256") != sha256(role_summary_path)
        or role_summary.get("detector_scores_used") is not False
    ):
        raise ValueError("Tractor completion evidence is incomplete")

    hosts = pd.read_parquet(tractor_path)
    proposal = pd.read_parquet(grec_path)
    roles = pd.read_parquet(roles_path)
    dispositions = pd.read_parquet(dispositions_path)
    if len(hosts) != 4_500 or hosts["galaxy_id"].nunique() != 4_500:
        raise ValueError("Tractor host table is not exactly 4,500 unique objects")
    if set(hosts["galaxy_id"]) != set(proposal["galaxy_id"]):
        raise ValueError("Tractor hosts differ from the score-blind proposal")
    if (
        role_completion.get("proposed_roles_sha256") != sha256(roles_path)
        or role_completion.get("proposed_grec_hosts_sha256") != sha256(grec_path)
        or role_summary.get("proposed_roles_sha256") != sha256(roles_path)
        or role_summary.get("proposed_grec_hosts_sha256") != sha256(grec_path)
    ):
        raise ValueError("role proposal parquets differ from their receipts")
    if len(dispositions) != 4_500 or dispositions["source_record_key"].duplicated().any():
        raise ValueError("eligibility disposition table is incomplete")
    status_column = next(
        (
            column
            for column in ("eligibility_pass", "eligible", "passes")
            if column in dispositions
        ),
        None,
    )
    if status_column is not None and not dispositions[status_column].astype(bool).all():
        raise ValueError("at least one host failed Tractor eligibility")
    if not dispositions["disposition_reasons_json"].eq("[]").all():
        raise ValueError("at least one host has a failed eligibility disposition")
    expected_pairs = set(zip(hosts["brickname"], hosts["objid"], strict=True))
    disposition_pairs = set(
        zip(dispositions["brickname"], dispositions["objid"], strict=True)
    )
    if disposition_pairs != expected_pairs:
        raise ValueError("eligibility dispositions do not cover exact hosts")

    split_paths = {
        "train_manifest": ROOT / "data/manifests/train.parquet",
        "validation_manifest": ROOT / "data/manifests/validation.parquet",
        "test_manifest": ROOT / "data/manifests/test.parquet",
    }
    for name, path in split_paths.items():
        if sha256(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"detector split manifest changed: {name}")
    train = pd.read_parquet(split_paths["train_manifest"])
    validation = pd.read_parquet(split_paths["validation_manifest"])
    forbidden = pd.concat(
        [
            train[["galaxy_id"]],
            validation[["galaxy_id"]],
        ]
    )
    if set(hosts["galaxy_id"]) & set(forbidden["galaxy_id"]):
        raise ValueError("final hosts overlap detector training or validation")

    host_groups = set(hosts["evaluation_spatial_group_id"].astype(str))
    other_roles = roles[
        roles["proposed_role"].isin(["threshold_calibration", "achieved_fpr"])
    ]
    if host_groups & set(other_roles["evaluation_spatial_group_id"].astype(str)):
        raise ValueError("final host spatial group crosses a negative evaluation role")
    if (
        roles.groupby("evaluation_spatial_group_id")["proposed_role"].nunique().max()
        != 1
    ):
        raise ValueError("one spatial group crosses proposed negative roles")
    if hosts["payload_sha256"].duplicated().any():
        raise ValueError("final hosts contain duplicate payload pixels")
    if hosts[["ra_payload", "dec_payload"]].duplicated().any():
        raise ValueError("final hosts contain duplicate coordinates")

    cell_counts = hosts.groupby(["psf_bin", "depth_bin"]).size()
    if len(cell_counts) != 9 or not cell_counts.eq(500).all():
        raise ValueError("final hosts do not contain nine cells of 500")

    test = pd.read_parquet(split_paths["test_manifest"])[
        ["galaxy_id", "physical_group_id"]
    ]
    selection = hosts.merge(test, on="galaxy_id", how="left")
    selection["physical_group_id"] = selection["physical_group_id"].fillna(
        "spatial:" + selection["evaluation_spatial_group_id"].astype(str)
    )
    if selection["physical_group_id"].isna().any():
        raise ValueError("a final host lacks a conservative physical group")
    development_physical = set(
        pd.concat([train, validation])["physical_group_id"].dropna().astype(str)
    )
    if set(selection["physical_group_id"].astype(str)) & development_physical:
        raise ValueError("final host physical group overlaps detector development")
    comparison = roles[roles["proposed_role"].isin(
        ["threshold_calibration", "achieved_fpr"]
    )].merge(test, on="galaxy_id", how="left")
    comparison["physical_group_id"] = comparison["physical_group_id"].fillna(
        "spatial:" + comparison["evaluation_spatial_group_id"].astype(str)
    )
    if set(selection["physical_group_id"].astype(str)) & set(
        comparison["physical_group_id"].astype(str)
    ):
        raise ValueError("final host physical group crosses a negative role")
    selection["final_role"] = "experiment_host"
    selection["pixel_unit"] = "nanomaggy_per_native_pixel"
    selection["flux_calibration_system"] = "AB_nanomaggy_zeropoint_22.5"
    selection["ra"] = selection["ra_payload"]
    selection["dec"] = selection["dec_payload"]
    selection["psfsize_r"] = selection["psfsize_r_tractor"]
    selection["psfdepth_r"] = selection["psfdepth_r_tractor"]
    selection["pixel_scale"] = selection["pixel_scale"].astype(float)
    selection = selection.drop(columns=["proposed_role"], errors="ignore")

    output_dir = PREP / "final_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = {
        "schema_version": 1,
        "status": "host_flux_calibration_frozen",
        "pixel_unit": "nanomaggy_per_native_pixel",
        "flux_calibration_system": "AB_nanomaggy_zeropoint_22.5",
        "source": "Legacy Survey DR10 FITS cutouts and Tractor catalogue",
        "unit_authority": "Legacy Survey nanomaggy convention documented in the manuscript data section",
        "manuscript_data_section_sha256": EXPECTED_HASHES[
            "manuscript_data_section"
        ],
        "payload_manifest_sha256": sha256(
            PREP / "candidate_payloads/payload_manifest.parquet"
        ),
    }
    calibration_path = output_dir / "host_flux_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    selection["calibration_provenance_path"] = str(calibration_path)
    selection["calibration_provenance_sha256"] = sha256(calibration_path)

    selection_path = output_dir / "final_grec_selection.parquet"
    selection.sort_values(["psf_bin", "depth_bin", "galaxy_id"]).to_parquet(
        selection_path, index=False
    )
    receipt = {
        "schema_version": 1,
        "status": "final_experiment_host_selection_candidate",
        "selection_manifest_sha256": sha256(selection_path),
        "rows": 4_500,
        "cells": 9,
        "hosts_per_cell": 500,
        "detector_scores_used": False,
        "full_tractor_eligibility": "passed",
        "physical_group_checks": "passed",
        "physical_group_method": "known_test_group_else_conservative_spatial_group",
        "role_overlap_checks": "passed",
        "role_overlap_scope": "complete_score_blind_role_proposal",
        "pre_score_review": "pending",
        "tractor_manifest_sha256": sha256(tractor_path),
        "tractor_summary_sha256": sha256(summary_path),
        "role_proposal_sha256": sha256(roles_path),
        "reference_population": "none",
        "evidence_hashes": EXPECTED_HASHES,
    }
    receipt_path = output_dir / "FINAL_SELECTION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
