#!/usr/bin/env python3
"""Summarize the six terminal training-mixture and seed sensitivity fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/derived/terminal_sensitivity_logits.npz"
DEFAULT_OUTPUT = ROOT / "results/primary_detector/sensitivity.json"
THRESHOLDS = {"fpr_1e3": 1.3337492942810059, "fpr_1e4": 17.135698318481445}


def analyse(path: Path) -> dict:
    models = {}
    with np.load(path) as payload:
        for name in payload.files:
            logits = payload[name].astype(float)
            if len(logits) != 43:
                raise ValueError(f"{name} has {len(logits)} systems, expected 43")
            detected = {
                key: int((logits >= threshold).sum())
                for key, threshold in THRESHOLDS.items()
            }
            models[name] = {
                "systems": 43,
                "detected_fpr_1e3": detected["fpr_1e3"],
                "detected_fpr_1e4": detected["fpr_1e4"],
                "recall_fpr_1e3": detected["fpr_1e3"] / 43,
                "recall_fpr_1e4": detected["fpr_1e4"] / 43,
                "logits": logits.tolist(),
                "mixture": (
                    "candidate_plus_tier_b"
                    if name.startswith("tier_ab_historical")
                    else "candidate_only"
                ),
            }
    return {
        "schema_version": 1,
        "scope": (
            "training-mixture and seed sensitivity on the same 43 terminal systems"
        ),
        "systems": 43,
        "thresholds": THRESHOLDS,
        "models": models,
        "interpretation": (
            "All fits use the primary detector's numerical thresholds; "
            "non-primary rows are correlated robustness fits rather than "
            "independent detectors."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = analyse(arguments.input)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
