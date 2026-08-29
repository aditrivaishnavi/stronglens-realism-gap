#!/usr/bin/env python3
"""Summarize validation and terminal predictions for the primary detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = ROOT / "results/primary_detector/validation_logits.parquet"
DEFAULT_TERMINAL = ROOT / "results/primary_detector/terminal_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "results/primary_detector"
CHECKPOINT_SHA256 = (
    "5d1082839aac7a0bfb89342f84a9a533b22e1111c15cb4711255ed1b524a2b8d"
)
THRESHOLDS = {
    "fpr_1e3": 1.3337492942810059,
    "fpr_1e4": 17.135698318481445,
}


def wilson_interval(successes: int, total: int, level: float = 0.95) -> list[float]:
    z = float(norm.ppf(0.5 + level / 2.0))
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [float(centre - half), float(centre + half)]


def analyse(validation: pd.DataFrame, terminal: pd.DataFrame) -> tuple[dict, dict]:
    labels = validation["label"].to_numpy(dtype=int)
    logits = validation["logit"].to_numpy(dtype=float)
    probability32 = validation["probability_float32"].to_numpy(dtype=float)
    probability64 = validation["probability_float64"].to_numpy(dtype=float)
    negative = labels == 0
    operating_points = {}
    for name, threshold in THRESHOLDS.items():
        false_positives = int((logits[negative] >= threshold).sum())
        operating_points[name] = {
            "logit_threshold": threshold,
            "false_positives": false_positives,
            "realised_fpr": false_positives / int(negative.sum()),
        }
    validation_result = {
        "schema_version": 1,
        "model": "EfficientNetV2-S",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_seed": 42,
        "epoch": 160,
        "training_positive_mixture": "candidate plus Tier-B catalogue positives",
        "rows": int(len(validation)),
        "validation_negatives": int(negative.sum()),
        "positive_labels": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "roc_auc_native_logit": float(roc_auc_score(labels, logits)),
        "average_precision_float32_probability": float(
            average_precision_score(labels, probability32)
        ),
        "average_precision_native_logit": float(
            average_precision_score(labels, logits)
        ),
        "brier_score_float64_probability": float(
            brier_score_loss(labels, probability64)
        ),
        "float32_probabilities_equal_to_one": int((probability32 == 1.0).sum()),
        "operating_points": operating_points,
        "threshold_rule": (
            "smallest float32 strictly greater than s_(k+1); "
            "classify when logit >= threshold"
        ),
    }

    endpoints = {}
    for name, threshold in THRESHOLDS.items():
        detected = int((terminal["logit"].to_numpy(dtype=float) >= threshold).sum())
        total = int(len(terminal))
        endpoints[
            "primary_fpr_1e3" if name == "fpr_1e3" else "secondary_fpr_1e4"
        ] = {
            "confidence_level": 0.95,
            "detected": detected,
            "independent_unit": "confirmed physical system",
            "response": detected / total,
            "systems": total,
            "wilson_interval": wilson_interval(detected, total),
        }
    terminal_result = {
        "schema_version": 1,
        "checkpoint": "primary_seed42_epoch160.pt",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "terminal_systems": int(len(terminal)),
        "cluster_scale_systems": int(terminal["cluster_scale"].sum()),
        "endpoints": endpoints,
        "method": {
            "confidence_level": 0.95,
            "name": "Wilson score interval",
            "scope": "binomial uncertainty over the 43 selected terminal systems",
        },
    }
    return validation_result, terminal_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    validation_result, terminal_result = analyse(
        pd.read_parquet(arguments.validation),
        pd.read_parquet(arguments.terminal),
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "validation_metrics.json").write_text(
        json.dumps(validation_result, indent=2) + "\n"
    )
    (arguments.output / "test_response.json").write_text(
        json.dumps(terminal_result, indent=2) + "\n"
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
