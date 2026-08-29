#!/usr/bin/env python3
"""Verify best-checkpoint selection records and calibration-role identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def strict_best(metrics_path: Path) -> tuple[int, float, int]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best_epoch = -1
    best_value = float("-inf")
    improvements = 0
    for row in rows:
        value = float(row["selection_value"])
        if value > best_value:
            best_value = value
            best_epoch = int(row["epoch"])
            improvements += 1
    return best_epoch, best_value, len(rows)


def checkpoint_ledger(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 5 or fields[1] != "MATCH":
            continue
        key = fields[0].removeprefix("runs/")
        result[key] = fields[2]
    return result


def verify_selection(
    panel_path: Path,
    training_root: Path,
    ledger_path: Path,
) -> list[dict[str, Any]]:
    panel = load_json(panel_path)
    ledger = checkpoint_ledger(ledger_path)
    records = []
    for model in panel["checkpoints"]:
        relative = Path(model["relative_path"])
        run_dir = training_root / relative.parent
        metrics_path = run_dir / "metrics.jsonl"
        run_info_path = run_dir / "run_info.json"
        setup_path = run_dir / "training_setup.json"
        for path in (metrics_path, run_info_path, setup_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        epoch, value, metric_rows = strict_best(metrics_path)
        run_info = load_json(run_info_path)
        if (
            epoch != int(model["checkpoint_epoch"])
            or value != float(run_info["best_selection_value"])
            or model["sha256"] != run_info["best_checkpoint_sha256"]
            or ledger.get(relative.as_posix()) != model["sha256"]
        ):
            raise RuntimeError(f"selection evidence differs for {model['model_id']}")
        records.append(
            {
                "model_id": model["model_id"],
                "selected_epoch": epoch,
                "selected_validation_average_precision": value,
                "metric_rows": metric_rows,
                "checkpoint_sha256": model["sha256"],
                "metrics_sha256": sha256(metrics_path),
                "run_info_sha256": sha256(run_info_path),
                "training_setup_sha256": sha256(setup_path),
                "strict_improvement_tie_rule": "earliest maximum retained",
            }
        )
    if len(records) != 12:
        raise RuntimeError("selection evidence does not cover 12 runs")
    return records


def verify_calibration(
    calibration_path: Path,
    calibration_receipt_path: Path,
    achieved_path: Path,
) -> dict[str, Any]:
    calibration = pd.read_parquet(calibration_path)
    achieved = pd.read_parquet(achieved_path)
    receipt = load_json(calibration_receipt_path)
    if (
        len(calibration) != 20_002
        or calibration["calibration_id"].duplicated().any()
        or not calibration["global_role"].eq("threshold_calibration").all()
        or receipt["calibration_manifest_sha256"] != sha256(calibration_path)
    ):
        raise RuntimeError("calibration identity manifest differs")
    checks = {
        "galaxy_id_overlap": len(
            set(calibration["galaxy_id"].astype(str))
            & set(achieved["galaxy_id"].astype(str))
        ),
        "pixel_hash_overlap": len(
            set(calibration["input_pixel_sha256"].astype(str))
            & set(achieved["pixel_array_sha256"].astype(str))
        ),
        "spatial_group_overlap": len(
            set(calibration["evaluation_spatial_group_id"].astype(str))
            & set(achieved["spatial_group_id"].astype(str))
        ),
    }
    if any(checks.values()):
        raise RuntimeError(f"calibration/achieved role overlap: {checks}")
    return {
        "calibration_rows": len(calibration),
        "calibration_ids_unique": True,
        "calibration_manifest_sha256": sha256(calibration_path),
        "calibration_receipt_sha256": sha256(calibration_receipt_path),
        "achieved_manifest_sha256": sha256(achieved_path),
        **checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--checkpoint-ledger", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--achieved-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    run_names = {
        json.loads(line)["run_name"]
        for line in arguments.run_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not {"gen6_amendment3_r3", "gen7_diagnostics_v2"} <= run_names:
        raise RuntimeError("run ledger lacks required training executions")

    payload = {
        "schema_version": 1,
        "status": "checkpoint_selection_and_calibration_identity_pass",
        "selection_metric": "development-validation average precision",
        "tie_rule": "strict improvement; earliest tied maximum retained",
        "panel_sha256": sha256(arguments.panel),
        "checkpoint_ledger_sha256": sha256(arguments.checkpoint_ledger),
        "run_ledger_sha256": sha256(arguments.run_ledger),
        "models": verify_selection(
            arguments.panel,
            arguments.training_root,
            arguments.checkpoint_ledger,
        ),
        "calibration": verify_calibration(
            arguments.calibration_manifest,
            arguments.calibration_receipt,
            arguments.achieved_manifest,
        ),
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CHECKPOINT_SELECTION_AND_CALIBRATION_IDENTITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
