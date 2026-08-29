#!/usr/bin/env python3
"""Score the 43 selected terminal systems with the primary detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dhs.fixed_width_scoring import (
    CONTRACT_FILLER,
    array_batches,
    assert_no_batch_coupling,
    fixed_width_logits,
    pin_deterministic_kernels,
)
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from scripts.analyze_primary_detector import CHECKPOINT_SHA256, THRESHOLDS

ROOT = Path(__file__).resolve().parents[1]


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def images(
    frame: pd.DataFrame,
    *,
    cutout_root: Path,
    preprocessing: dict,
):
    for relative in frame["cutout_relpath"].astype(str):
        with np.load(cutout_root / relative, allow_pickle=False) as payload:
            array = payload["image"] if "image" in payload else payload["cutout"]
        value = np.asarray(array, dtype=np.float32)
        if value.shape == (101, 101, 3):
            value = np.transpose(value, (2, 0, 1))
        yield preprocess_stack(value, **preprocessing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/downloads/models/primary_seed42_epoch160.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/test.parquet",
    )
    parser.add_argument(
        "--cutout-root",
        type=Path,
        default=ROOT / "data/downloads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/terminal_evaluation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()

    frame = pd.read_parquet(arguments.manifest)
    frame = frame.loc[frame["endpoint_role"].eq("terminal_confirmed")].copy()
    if len(frame) != 43:
        raise ValueError(f"expected 43 terminal systems, found {len(frame)}")

    device = select_device(arguments.device)
    model, preprocessing = load_model_and_spec(str(arguments.checkpoint), device)
    model = model.to(device).eval()
    decoupling = assert_no_batch_coupling(model)
    kernels = pin_deterministic_kernels()
    logits, padding = fixed_width_logits(
        model,
        array_batches(
            images(
                frame,
                cutout_root=arguments.cutout_root,
                preprocessing=dict(preprocessing or {}),
            ),
            arguments.batch_size,
        ),
        batch_size=arguments.batch_size,
        device=device,
        filler=CONTRACT_FILLER,
    )

    output = frame[["galaxy_id"]].copy()
    output["logit"] = logits
    output["probability"] = 1.0 / (1.0 + np.exp(-logits))
    for name, threshold in THRESHOLDS.items():
        output[f"detected_{name}"] = logits >= threshold

    arguments.output.mkdir(parents=True, exist_ok=True)
    output.to_parquet(arguments.output / "terminal_predictions.parquet", index=False)
    summary = {
        "schema_version": 1,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "systems": len(output),
        "detected_fpr_1e3": int(output["detected_fpr_1e3"].sum()),
        "detected_fpr_1e4": int(output["detected_fpr_1e4"].sum()),
        "scoring": {
            "device": device,
            "batch_decoupling": decoupling,
            **kernels,
            **padding,
        },
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
