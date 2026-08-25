#!/usr/bin/env python3
"""Materialize raw CHW inputs for the frozen threshold-calibration role."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLE_RECEIPT_SHA256 = (
    "ebdabda7735c6ccfbf17df5d35b9b79324d1c7b5ec94ed4b555048bd54689acb"
)
EXPECTED_TRAIN_SHA256 = (
    "a0caa3824568f96010275e59d89d013a23ac7e6e6ae280de2c636a401fa38e5b"
)
EXPECTED_VALIDATION_SHA256 = (
    "809729e63c9bd6fca96f51b4f8d244aeda604beca4135edcc428f431a9941ee9"
)
EXPECTED_HOST_SHA256 = (
    "60429cc8f00de5ffcb05b44a93a61ae30d7109ee99842b56ac192d73bd94b023"
)
sys.path.insert(0, str(ROOT / "experiments_v2/grec_pnoise/src"))

from grec_pnoise.contracts import (  # noqa: E402
    array_bytes_sha256,
    config_provenance,
    load_config,
    sha256_file,
)


def stable_id(value: str) -> str:
    return "cal-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def main() -> int:
    role_path = ROOT / "results_v2/evaluation_v2/threshold_calibration.parquet"
    role_receipt_path = ROOT / "results_v2/evaluation_v2/NEGATIVE_ROLE_RECEIPT.json"
    payload_root = ROOT / "evaluation_dataset_prep/candidate_payloads"
    output_dir = ROOT / "results_v2/evaluation_v2/calibration_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(
        ROOT / "experiments_v2/grec_pnoise/config/production.toml"
    )
    role_receipt = json.loads(role_receipt_path.read_text())
    if (
        sha256_file(role_receipt_path) != EXPECTED_ROLE_RECEIPT_SHA256
        or
        role_receipt.get("status") != "latest_test_negative_roles_frozen_pre_score"
        or role_receipt["roles"]["threshold_calibration"]["sha256"]
        != sha256_file(role_path)
        or role_receipt.get("detector_scores_used") is not False
        or role_receipt.get("train_manifest_sha256") != EXPECTED_TRAIN_SHA256
        or role_receipt.get("validation_manifest_sha256")
        != EXPECTED_VALIDATION_SHA256
        or role_receipt.get("final_host_manifest_sha256") != EXPECTED_HOST_SHA256
        or role_receipt.get("cross_role_id_pixel_spatial_physical_overlap") != 0
    ):
        raise ValueError("calibration role receipt is invalid")
    frame = pd.read_parquet(role_path)
    if len(frame) != 20_002 or not frame["final_role"].eq(
        "threshold_calibration"
    ).all():
        raise ValueError("calibration role has unexpected membership")

    def materialize(row: dict[str, Any]) -> dict[str, Any]:
        source = payload_root / str(row["local_relpath"])
        if sha256_file(source) != str(row["payload_sha256"]):
            raise ValueError(f"payload hash mismatch: {row['galaxy_id']}")
        with np.load(source, allow_pickle=False) as archive:
            cutout = np.asarray(archive["cutout"])
            bands = np.asarray(archive["meta_bands_requested"]).item()
        if bands != "grz":
            raise ValueError(f"unexpected payload band order: {row['galaxy_id']}")
        if (
            cutout.dtype != np.float32
            or cutout.shape != (101, 101, 3)
            or not np.isfinite(cutout).all()
        ):
            raise ValueError(f"invalid cutout: {row['galaxy_id']}")
        image = np.ascontiguousarray(cutout.transpose(2, 0, 1))
        destination = (
            output_dir
            / "pixels"
            / str(row["galaxy_id"])[:4]
            / f"{row['galaxy_id']}.npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        np.savez(temporary, image_chw=image)
        temporary.replace(destination)
        return {
            "calibration_id": stable_id(str(row["galaxy_id"])),
            "galaxy_id": str(row["galaxy_id"]),
            "global_role": "threshold_calibration",
            "evaluation_spatial_group_id": str(
                row["evaluation_spatial_group_id"]
            ),
            "host_pixel_sha256": array_bytes_sha256(cutout),
            "input_pixel_path": str(destination),
            "input_artifact_sha256": sha256_file(destination),
            "input_pixel_sha256": array_bytes_sha256(image),
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(executor.map(materialize, frame.to_dict(orient="records")))
    manifest = pd.DataFrame(rows).sort_values("calibration_id").reset_index(drop=True)
    if (
        len(manifest) != len(frame)
        or manifest["calibration_id"].duplicated().any()
        or manifest["input_pixel_sha256"].duplicated().any()
    ):
        raise ValueError("calibration materialization is incomplete or duplicated")
    manifest_path = output_dir / "calibration_manifest.parquet"
    partial = output_dir / "calibration_manifest.partial.parquet"
    manifest.to_parquet(partial, index=False)
    partial.replace(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": config["threshold"]["required_calibration_manifest_status"],
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "source_role_manifest_sha256": sha256_file(role_path),
        "source_role_receipt_sha256": sha256_file(role_receipt_path),
        "rows": len(manifest),
        "pixel_artifacts_verified": len(manifest),
        "detector_scores_used": False,
        "materializer_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "config": config_provenance(config),
    }
    receipt_path = output_dir / "CALIBRATION_MANIFEST_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
