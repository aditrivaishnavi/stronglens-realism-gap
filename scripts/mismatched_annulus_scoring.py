#!/usr/bin/env python3
"""
Mismatched Annulus Scoring: v4 model with (32.5,45) preprocessing.

"Run existing model (v4) on 200 injections preprocessed
with (32.5, 45.0) (MISMATCH with training annulus). Compare completeness vs
standard (20, 32). If completeness changes dramatically -> annulus matters.
If unchanged -> it doesn't."

Hypothesis: "Almost certainly hurts." But the magnitude of degradation
is informative. A small change means the model is robust to normalization;
a large change means normalization materially controls detectability.

This is a sensitivity test, NOT a correctness test.

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python scripts/mismatched_annulus_scoring.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/mismatched_annulus \\
        --n-samples 500

Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec

PIXEL_SCALE = 0.262


@torch.no_grad()
def score_batch(
    model: nn.Module,
    paths: list,
    device: torch.device,
    pp_kwargs: dict,
) -> np.ndarray:
    """Score cutouts, returning sigmoid probabilities."""
    scores = []
    for p in paths:
        try:
            with np.load(p) as z:
                hwc = z["cutout"].astype(np.float32)
            chw = np.transpose(hwc, (2, 0, 1))
            proc = preprocess_stack(chw, **pp_kwargs)
            x = torch.from_numpy(proc[None]).float().to(device)
            logit = model(x).squeeze().cpu().item()
            scores.append(1.0 / (1.0 + np.exp(-logit)))
        except Exception:
            scores.append(float("nan"))
    return np.array(scores)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mismatched annulus scoring: v4 model with (32.5,45) preprocessing",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model + its native preprocessing config
    print("Loading model...")
    model, pp_kwargs_native = load_model_and_spec(args.checkpoint, device)
    print(f"  Native preprocessing: {pp_kwargs_native}")

    # Build mismatched preprocessing: same mode/clip but different annulus
    pp_kwargs_mismatched = dict(pp_kwargs_native)
    pp_kwargs_mismatched["annulus_r_in"] = 32.5
    pp_kwargs_mismatched["annulus_r_out"] = 45.0
    print(f"  Mismatched preprocessing: {pp_kwargs_mismatched}")

    # Load manifest
    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    val_df = df[df["split"] == "val"].copy()

    # Sample negatives and positives
    neg_df = val_df[val_df["label"] == 0].dropna(subset=["cutout_path"])
    pos_df = val_df[val_df["label"] == 1].dropna(subset=["cutout_path"])

    n_neg = min(args.n_samples, len(neg_df))
    n_pos = min(args.n_samples, len(pos_df))
    neg_sample = neg_df.sample(n=n_neg, random_state=args.seed)
    pos_sample = pos_df.sample(n=n_pos, random_state=args.seed)

    print(f"\n  Scoring {n_pos} positives + {n_neg} negatives with NATIVE annulus...")
    pos_paths = pos_sample["cutout_path"].astype(str).tolist()
    neg_paths = neg_sample["cutout_path"].astype(str).tolist()

    # Score with native preprocessing
    pos_scores_native = score_batch(model, pos_paths, device, pp_kwargs_native)
    neg_scores_native = score_batch(model, neg_paths, device, pp_kwargs_native)

    print(f"  Scoring {n_pos} positives + {n_neg} negatives with MISMATCHED annulus...")

    # Score with mismatched preprocessing
    pos_scores_mismatch = score_batch(model, pos_paths, device, pp_kwargs_mismatched)
    neg_scores_mismatch = score_batch(model, neg_paths, device, pp_kwargs_mismatched)

    # Compute metrics
    thresholds = [0.3, 0.5, 0.7]

    def compute_metrics(pos_scores, neg_scores, label):
        valid_pos = pos_scores[np.isfinite(pos_scores)]
        valid_neg = neg_scores[np.isfinite(neg_scores)]
        m = {"label": label, "n_pos": len(valid_pos), "n_neg": len(valid_neg)}
        for thr in thresholds:
            recall = float((valid_pos >= thr).mean()) if len(valid_pos) > 0 else float("nan")
            fpr = float((valid_neg >= thr).mean()) if len(valid_neg) > 0 else float("nan")
            m[f"recall_p{thr}"] = recall
            m[f"fpr_p{thr}"] = fpr
        m["median_pos_score"] = float(np.median(valid_pos)) if len(valid_pos) > 0 else float("nan")
        m["median_neg_score"] = float(np.median(valid_neg)) if len(valid_neg) > 0 else float("nan")
        return m

    native_metrics = compute_metrics(pos_scores_native, neg_scores_native, "native")
    mismatch_metrics = compute_metrics(pos_scores_mismatch, neg_scores_mismatch, "mismatched")

    # Delta
    deltas = {}
    for thr in thresholds:
        key = f"recall_p{thr}"
        deltas[f"delta_{key}"] = mismatch_metrics[key] - native_metrics[key]
        key_fpr = f"fpr_p{thr}"
        deltas[f"delta_{key_fpr}"] = mismatch_metrics[key_fpr] - native_metrics[key_fpr]

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "native_annulus": [
            pp_kwargs_native.get("annulus_r_in", 20),
            pp_kwargs_native.get("annulus_r_out", 32),
        ],
        "mismatched_annulus": [32.5, 45.0],
        "native_metrics": native_metrics,
        "mismatched_metrics": mismatch_metrics,
        "deltas": deltas,
        "interpretation": {
            "small_delta": "Model is robust to annulus choice -> retraining unlikely to help via annulus fix.",
            "large_delta": "Model is sensitive to normalization -> annulus fix during retraining may help.",
            "note": "Hypothesis: 'almost certainly hurts' because model expects (20,32)-normalized inputs.",
        },
    }

    json_path = os.path.join(args.out_dir, "mismatched_annulus_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("MISMATCHED ANNULUS SCORING: native (20,32) vs mismatched (32.5,45)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Native':>12} {'Mismatched':>12} {'Delta':>10}")
    print("-" * 60)
    for thr in thresholds:
        r_n = native_metrics[f"recall_p{thr}"]
        r_m = mismatch_metrics[f"recall_p{thr}"]
        d = r_m - r_n
        print(f"  Recall (p>{thr})        {r_n:>11.3f} {r_m:>12.3f} {d:>+10.3f}")
    for thr in thresholds:
        f_n = native_metrics[f"fpr_p{thr}"]
        f_m = mismatch_metrics[f"fpr_p{thr}"]
        d = f_m - f_n
        print(f"  FPR (p>{thr})           {f_n:>11.4f} {f_m:>12.4f} {d:>+10.4f}")
    print(f"  Median pos score       {native_metrics['median_pos_score']:>11.4f} "
          f"{mismatch_metrics['median_pos_score']:>12.4f}")
    print(f"  Median neg score       {native_metrics['median_neg_score']:>11.4f} "
          f"{mismatch_metrics['median_neg_score']:>12.4f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
