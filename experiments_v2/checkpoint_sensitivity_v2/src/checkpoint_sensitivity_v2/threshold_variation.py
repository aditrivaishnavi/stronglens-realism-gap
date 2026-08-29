from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _seed(model_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"checkpoint-threshold-variation:{model_id}".encode()).digest()[
            :16
        ],
        "big",
    )


def _threshold(values: np.ndarray, alpha: float) -> np.float32:
    n = len(values)
    k = math.floor(alpha * n)
    ascending_index = n - k - 1
    s_k_plus_1 = np.float32(np.partition(values, ascending_index)[ascending_index])
    return np.nextafter(s_k_plus_1, np.float32(np.inf), dtype=np.float32)


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": int(values.min()),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
        "maximum": int(values.max()),
    }


def _model_bootstrap(
    arguments: tuple[
        str,
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        int,
        float,
    ],
) -> dict[str, Any]:
    (
        model_id,
        calibration_logits,
        group_indices,
        achieved_logits,
        selected_logits,
        draws,
        alpha,
    ) = arguments
    grouped = [calibration_logits[indices] for indices in group_indices]
    rng = np.random.Generator(np.random.PCG64DXSM(_seed(model_id)))
    threshold_draws = np.empty(draws, dtype=np.float32)
    achieved_counts = np.empty(draws, dtype=np.int64)
    selected_counts = np.empty(draws, dtype=np.int64)
    n_groups = len(grouped)
    for draw in range(draws):
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        sampled = np.concatenate([grouped[index] for index in sampled_groups])
        threshold = _threshold(sampled, alpha)
        threshold_draws[draw] = threshold
        achieved_counts[draw] = np.count_nonzero(achieved_logits >= threshold)
        selected_counts[draw] = np.count_nonzero(selected_logits >= threshold)
    point_threshold = _threshold(calibration_logits, alpha)
    return {
        "model_id": model_id,
        "draws": draws,
        "spatial_groups": n_groups,
        "point_threshold": float(point_threshold),
        "point_achieved_fpr_count": int(
            np.count_nonzero(achieved_logits >= point_threshold)
        ),
        "point_selected_count": int(
            np.count_nonzero(selected_logits >= point_threshold)
        ),
        "bootstrap_threshold": {
            "minimum": float(threshold_draws.min()),
            "q025": float(np.quantile(threshold_draws, 0.025)),
            "median": float(np.quantile(threshold_draws, 0.5)),
            "q975": float(np.quantile(threshold_draws, 0.975)),
            "maximum": float(threshold_draws.max()),
        },
        "bootstrap_achieved_fpr_count": _quantiles(achieved_counts),
        "bootstrap_selected_count": _quantiles(selected_counts),
    }


def run(root: Path, draws: int, workers: int, output: Path) -> dict[str, Any]:
    panel = json.loads(
        (
            root / "experiments_v2/checkpoint_sensitivity_v2/config/panel.json"
        ).read_text()
    )
    result_root = root / "results_v2/checkpoint_sensitivity_v2"
    calibration_manifest = pd.read_parquet(
        root
        / "results_v2/evaluation_v2/calibration_inputs/calibration_manifest.parquet"
    )
    calibration = pd.read_parquet(result_root / "calibration/panel_logits.parquet")
    achieved = pd.read_parquet(result_root / "achieved_fpr/panel_logits.parquet")
    selected = pd.read_parquet(result_root / "terminal_confirmed/panel_logits.parquet")
    if (
        len(calibration) != 20_002
        or len(achieved) != 12_512
        or len(selected) != 43
        or not calibration["calibration_id"].equals(
            calibration_manifest["calibration_id"]
        )
    ):
        raise RuntimeError("threshold-variation input support differs")
    group_indices = [
        indices.to_numpy()
        for _, indices in calibration_manifest.groupby(
            "evaluation_spatial_group_id", sort=True
        ).groups.items()
    ]
    work = []
    for model in panel["checkpoints"]:
        model_id = str(model["model_id"])
        column = f"{model_id}__raw_logit_float32"
        work.append(
            (
                model_id,
                calibration[column].to_numpy(copy=True),
                group_indices,
                achieved[column].to_numpy(copy=True),
                selected[column].to_numpy(copy=True),
                draws,
                float(panel["alpha"]),
            )
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        models = list(executor.map(_model_bootstrap, work))
    payload = {
        "schema_version": 1,
        "status": "checkpoint_threshold_spatial_bootstrap_complete",
        "method": (
            "Resample the 1,768 calibration spatial groups with replacement; "
            "recompute the native-float32 threshold in every draw; apply it to "
            "fixed achieved-FPR and selected-system logits."
        ),
        "interpretation": (
            "Diagnostic calibration-sample variation conditional on each "
            "checkpoint's fixed logits; not a population interval."
        ),
        "draws_per_checkpoint": draws,
        "workers": workers,
        "models": models,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.root.resolve(),
        arguments.draws,
        arguments.workers,
        arguments.output.resolve(),
    )
    for model in result["models"]:
        print(
            model["model_id"],
            model["point_selected_count"],
            model["bootstrap_selected_count"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
