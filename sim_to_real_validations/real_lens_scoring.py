#!/usr/bin/env python3
"""
Real Lens Scoring: Score all real confirmed lenses in the validation split
using a frozen trained model and report recall at multiple detection thresholds.
Also scores a sample of negatives for FPR comparison.

This script provides a focused diagnostic for sim-to-real generalization:
  - Recall of real confirmed lenses at fixed (p>0.3, p>0.5) and FPR-derived
    thresholds (p>0.806 for FPR=0.1%, p>0.995 for FPR=0.01%)
  - Full score distribution statistics (percentiles, mean, tail fractions)
  - Negative FPR at the same thresholds for comparison

Outputs:
  - real_lens_scoring_results.json: comprehensive numerical summary
  - real_lens_scores.npz: raw score arrays (real_scores, neg_scores)
  - Console summary table

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python sim_to_real_validations/real_lens_scoring.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/real_lens_scoring \\
        --host-split val \\
        --n-negatives 3000

Author: stronglens-realism-gap project
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
from scipy.stats import beta as beta_dist

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec


TIER_COL = "tier"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load model + preprocessing kwargs from checkpoint.

    Uses scoring_utils.load_model_and_spec to ensure preprocessing
    parameters match what the model was trained with.
    """
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    # Extract arch/epoch for reporting (need to re-read checkpoint metadata)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    epoch = ckpt.get("epoch", -1)
    return model, arch, epoch, pp_kwargs


@torch.no_grad()
def score_one(model: nn.Module, cutout_path: str, device: torch.device,
              pp_kwargs: dict | None = None) -> float | None:
    """Score a single cutout. Returns probability or None on error.

    Args:
        pp_kwargs: Preprocessing kwargs from checkpoint. If None,
                   uses defaults (raw_robust, crop=False, clip_range=10.0).
    """
    if pp_kwargs is None:
        pp_kwargs = {"mode": "raw_robust", "crop": False, "clip_range": 10.0}
    try:
        with np.load(cutout_path) as z:
            hwc = z["cutout"].astype(np.float32)
        chw = np.transpose(hwc, (2, 0, 1))
        proc = preprocess_stack(chw, **pp_kwargs)
        x = torch.from_numpy(proc[None]).float().to(device)
        logit = model(x).squeeze().cpu().item()
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(p)
    except Exception:
        return None


def score_paths(
    model: nn.Module,
    paths: list[str],
    device: torch.device,
    progress_every: int = 200,
    pp_kwargs: dict | None = None,
) -> np.ndarray:
    """Score a list of cutout paths. Returns array of probabilities (NaN on error)."""
    scores = []
    for i, path in enumerate(paths):
        p = score_one(model, path, device, pp_kwargs=pp_kwargs)
        scores.append(p if p is not None else np.nan)
        if (i + 1) % progress_every == 0:
            print(f"  Scored {i + 1}/{len(paths)}", flush=True)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "p>0.3": 0.3,
    "p>0.5": 0.5,
    "p>0.806 (FPR-derived)": 0.806,
    "p>0.995 (FPR-derived)": 0.995,
}


def compute_stats(scores: np.ndarray) -> dict:
    """Compute percentiles, mean, and tail fractions."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {
            "p5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "mean": np.nan,
            "frac_above_0.9": np.nan,
            "frac_below_0.1": np.nan,
            "n_valid": 0,
        }
    pctls = np.percentile(valid, [5, 25, 50, 75, 95])
    return {
        "p5": float(pctls[0]),
        "p25": float(pctls[1]),
        "median": float(pctls[2]),
        "p75": float(pctls[3]),
        "p95": float(pctls[4]),
        "mean": float(np.mean(valid)),
        "frac_above_0.9": float((valid > 0.9).mean()),
        "frac_below_0.1": float((valid < 0.1).mean()),
        "n_valid": int(len(valid)),
    }


def recall_at_thresholds(scores: np.ndarray) -> dict[str, float]:
    """Compute recall at each threshold."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {k: np.nan for k in THRESHOLDS}
    return {
        label: float((valid >= thr).mean())
        for label, thr in THRESHOLDS.items()
    }


def fpr_at_thresholds(scores: np.ndarray) -> dict[str, float]:
    """Compute FPR (fraction above threshold) at each threshold."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {k: np.nan for k in THRESHOLDS}
    return {
        label: float((valid >= thr).mean())
        for label, thr in THRESHOLDS.items()
    }


def binomial_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Bayesian binomial credible interval using Jeffreys prior Beta(0.5, 0.5)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - level) / 2.0
    a = k + 0.5
    b = n - k + 0.5
    lo = float(beta_dist.ppf(alpha, a, b))
    hi = float(beta_dist.ppf(1.0 - alpha, a, b))
    return (lo, hi)


def recall_by_tier(
    df: pd.DataFrame,
    scores: np.ndarray,
) -> Dict[str, dict]:
    """Compute recall at each threshold stratified by tier (A, B).

    Returns a dict keyed by tier value (e.g. "A", "B") with recall at
    each threshold and 95% binomial CIs.
    """
    results: Dict[str, dict] = {}
    if TIER_COL not in df.columns:
        return results

    for tier_val in sorted(df[TIER_COL].dropna().unique()):
        mask = df[TIER_COL] == tier_val
        tier_scores = scores[mask.values]
        valid = tier_scores[np.isfinite(tier_scores)]
        n_total = len(tier_scores)
        n_valid = len(valid)

        tier_result: dict = {
            "n_total": n_total,
            "n_valid": n_valid,
        }

        # Recall at each threshold with binomial CI
        for label, thr in THRESHOLDS.items():
            if n_valid == 0:
                tier_result[label] = {
                    "recall": float("nan"),
                    "n_detected": 0,
                    "ci_95": (float("nan"), float("nan")),
                }
            else:
                n_det = int((valid >= thr).sum())
                recall = n_det / n_valid
                ci = binomial_ci(n_det, n_valid, level=0.95)
                tier_result[label] = {
                    "recall": float(recall),
                    "n_detected": n_det,
                    "ci_95": ci,
                }

        results[f"tier_{tier_val}"] = tier_result

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Score real lenses and negatives; report recall and FPR at thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (e.g. best.pt)")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument(
        "--host-split",
        default="val",
        help="Split to use (default: val)",
    )
    ap.add_argument(
        "--n-negatives",
        type=int,
        default=3000,
        help="Number of random negatives to score (default: 3000)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for JSON and NPZ results",
    )
    ap.add_argument("--device", default="cuda", help="Device for inference (default: cuda)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling")
    ap.add_argument(
        "--tier-a-only",
        action="store_true",
        help="Restrict positive evaluation to Tier-A (confirmed) lenses only. "
             "headline recall should be Tier-A only.",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("REAL LENS SCORING")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest:   {args.manifest}")
    print(f"Split:      {args.host_split}")
    print(f"Device:     {device}")
    print()

    # Load model + preprocessing config from checkpoint
    print("Loading model...")
    model, arch, epoch, pp_kwargs = load_model(args.checkpoint, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")
    print(f"  Preprocessing: {pp_kwargs}")
    print()

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_parquet(args.manifest)

    # --- Training-split leakage guard ---
    # Report Tier-A counts per split so user can verify no leakage
    if TIER_COL in df.columns:
        print("\n  Tier-A counts per split (leakage guard):")
        for split_name in sorted(df["split"].unique()):
            tier_a_in_split = df[(df["split"] == split_name) & (df["label"] == 1) & (df[TIER_COL] == "A")]
            tier_b_in_split = df[(df["split"] == split_name) & (df["label"] == 1) & (df[TIER_COL] == "B")]
            print(f"    {split_name}: {len(tier_a_in_split)} Tier-A, {len(tier_b_in_split)} Tier-B")
        print()

    pos_rows = df[(df["split"] == args.host_split) & (df["label"] == 1)]
    neg_rows = df[(df["split"] == args.host_split) & (df["label"] == 0)]

    # --- Tier-A-only mode ---
    if args.tier_a_only:
        if TIER_COL in pos_rows.columns:
            n_before = len(pos_rows)
            pos_rows = pos_rows[pos_rows[TIER_COL] == "A"]
            print(f"  --tier-a-only: filtered {n_before} -> {len(pos_rows)} Tier-A positives")
        else:
            print("  WARNING: --tier-a-only requested but no 'tier' column in manifest")

    print(f"  Positives (label=1): {len(pos_rows)}")
    print(f"  Negatives (label=0): {len(neg_rows)}")
    print()

    if len(pos_rows) == 0:
        print("ERROR: No positives in selected split. Exiting.")
        return 1

    # Score all positives
    print(f"Scoring {len(pos_rows)} real confirmed lenses...")
    pos_paths = pos_rows["cutout_path"].astype(str).tolist()
    real_scores = score_paths(model, pos_paths, device, progress_every=200, pp_kwargs=pp_kwargs)
    n_errors = int(np.isnan(real_scores).sum())
    valid_real = real_scores[np.isfinite(real_scores)]
    print(f"  Done. Valid: {len(valid_real)}, Errors: {n_errors}")
    print()

    # Score negatives
    n_neg = min(args.n_negatives, len(neg_rows))
    neg_sample = neg_rows.sample(n=n_neg, random_state=args.seed)
    print(f"Scoring {n_neg} random negatives...")
    neg_paths = neg_sample["cutout_path"].astype(str).tolist()
    neg_scores = score_paths(model, neg_paths, device, progress_every=200, pp_kwargs=pp_kwargs)
    n_neg_errors = int(np.isnan(neg_scores).sum())
    valid_neg = neg_scores[np.isfinite(neg_scores)]
    print(f"  Done. Valid: {len(valid_neg)}, Errors: {n_neg_errors}")
    print()

    # Compute metrics
    real_recall = recall_at_thresholds(real_scores)
    real_stats = compute_stats(real_scores)
    neg_fpr = fpr_at_thresholds(neg_scores)
    neg_stats = compute_stats(neg_scores)

    # Per-tier recall (Tier-A vs Tier-B) with binomial CIs
    tier_recall = recall_by_tier(pos_rows.reset_index(drop=True), real_scores)
    if tier_recall:
        print("Per-tier recall computed.")
        for tk, tv in tier_recall.items():
            print(f"  {tk}: n_valid={tv['n_valid']}")
    else:
        print("WARNING: No 'tier' column found in manifest. Per-tier recall not available.")

    # Build results dict
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "host_split": args.host_split,
        "arch": arch,
        "epoch": epoch,
        "device": str(device),
        "seed": args.seed,
        "real_lens": {
            "n_total": int(len(real_scores)),
            "n_valid": int(len(valid_real)),
            "n_errors": n_errors,
            "recall_at_thresholds": real_recall,
            "score_stats": {
                "p5": real_stats["p5"],
                "p25": real_stats["p25"],
                "median": real_stats["median"],
                "p75": real_stats["p75"],
                "p95": real_stats["p95"],
                "mean": real_stats["mean"],
                "frac_above_0.9": real_stats["frac_above_0.9"],
                "frac_below_0.1": real_stats["frac_below_0.1"],
            },
        },
        "negatives": {
            "n_total": int(len(neg_scores)),
            "n_valid": int(len(valid_neg)),
            "n_errors": n_neg_errors,
            "fpr_at_thresholds": neg_fpr,
            "score_stats": {
                "p5": neg_stats["p5"],
                "p25": neg_stats["p25"],
                "median": neg_stats["median"],
                "p75": neg_stats["p75"],
                "p95": neg_stats["p95"],
                "mean": neg_stats["mean"],
                "frac_above_0.9": neg_stats["frac_above_0.9"],
                "frac_below_0.1": neg_stats["frac_below_0.1"],
            },
        },
        "recall_by_tier": tier_recall,
    }

    # Save JSON
    json_path = os.path.join(args.out_dir, "real_lens_scoring_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {json_path}")

    # Save NPZ
    npz_path = os.path.join(args.out_dir, "real_lens_scores.npz")
    np.savez(npz_path, real_scores=real_scores, neg_scores=neg_scores)
    print(f"Scores saved:  {npz_path}")
    print()

    # Print summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Real lens recall:")
    for label, thr in THRESHOLDS.items():
        r = real_recall[label]
        pct = r * 100 if np.isfinite(r) else "N/A"
        print(f"  {label:30s}  recall = {pct:.1f}%")
    print()
    print("Real lens score distribution:")
    print(f"  p5={real_stats['p5']:.4f}  p25={real_stats['p25']:.4f}  "
          f"median={real_stats['median']:.4f}  p75={real_stats['p75']:.4f}  "
          f"p95={real_stats['p95']:.4f}")
    print(f"  mean={real_stats['mean']:.4f}  frac>0.9={real_stats['frac_above_0.9']:.4f}  "
          f"frac<0.1={real_stats['frac_below_0.1']:.4f}")
    print()
    print("Negative FPR at same thresholds:")
    for label, thr in THRESHOLDS.items():
        fpr = neg_fpr[label]
        pct = fpr * 100 if np.isfinite(fpr) else "N/A"
        print(f"  {label:30s}  FPR = {pct:.2f}%")
    print()
    print("Negative score distribution:")
    print(f"  p5={neg_stats['p5']:.4f}  p25={neg_stats['p25']:.4f}  "
          f"median={neg_stats['median']:.4f}  p75={neg_stats['p75']:.4f}  "
          f"p95={neg_stats['p95']:.4f}")
    print(f"  mean={neg_stats['mean']:.4f}  frac>0.9={neg_stats['frac_above_0.9']:.4f}  "
          f"frac<0.1={neg_stats['frac_below_0.1']:.4f}")
    print()

    # Per-tier recall breakdown
    if tier_recall:
        print("PER-TIER RECALL (Tier-A = confirmed, Tier-B = visual candidates):")
        print("-" * 70)
        for tier_key, tier_data in sorted(tier_recall.items()):
            print(f"  {tier_key}: n_valid={tier_data['n_valid']}")
            for label in THRESHOLDS:
                tr = tier_data.get(label, {})
                rec = tr.get("recall", float("nan"))
                n_det = tr.get("n_detected", 0)
                ci = tr.get("ci_95", (float("nan"), float("nan")))
                if np.isfinite(rec):
                    print(f"    {label:30s}  recall={rec*100:.1f}%  "
                          f"({n_det}/{tier_data['n_valid']})  "
                          f"95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
                else:
                    print(f"    {label:30s}  recall=N/A")
            print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
