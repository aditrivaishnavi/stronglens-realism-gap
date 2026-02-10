#!/usr/bin/env python3
"""
Confuser Morphology Test: Test whether the lens detection model is detecting
deflector galaxy morphology rather than genuine arc signal.

SCIENTIFIC MOTIVATION
---------------------
Strong lens classifiers are trained to distinguish true gravitational lens systems
(arcs + counter-images around a deflector) from non-lenses. A concern is that the
model might "cheat" by learning deflector galaxy morphology (e.g., ring-like
structure, edge-on disks, large smooth galaxies, blue clumpy star-forming regions)
rather than the genuine lensed arc signal. These morphologies can superficially
resemble arcs or partial arcs.

To test this, we score negatives that were explicitly categorized as confusers:
- ring_proxy:     galaxies with ring-like structure (can mimic Einstein rings)
- edge_on_proxy:  edge-on disks (can mimic straight arcs)
- large_galaxy:   large smooth galaxies (can mimic extended arcs)
- blue_clumpy:    blue clumpy star-forming regions (can mimic distorted sources)

INTERPRETATION
--------------
If confuser categories score LOW (median near zero, small fraction above 0.3/0.5),
the model is NOT cheating by detecting galaxy morphology—it is responding to
genuine arc signal. If confuser categories score HIGH (similar to or above
random negatives), the model may be exploiting morphological shortcuts.

A random-negative baseline provides context: typical negatives should score low.
Confusers that score similarly high would indicate potential shortcut learning.

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.
    python sim_to_real_validations/confuser_morphology_test.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/confuser_morphology_test

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add parent for dhs imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFUSER_CATEGORIES = ["ring_proxy", "edge_on_proxy", "large_galaxy", "blue_clumpy"]
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
CONFUSER_COL = "confuser_category"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load model + preprocessing kwargs from checkpoint.

    Uses scoring_utils.load_model_and_spec to ensure preprocessing
    parameters match what the model was trained with.
    """
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    return model, arch, ckpt.get("epoch", -1), pp_kwargs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_cutout(model: nn.Module, cutout_path: str, device: torch.device,
                 pp_kwargs: dict | None = None) -> Optional[float]:
    """Score a single cutout. Returns probability or None on error."""
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


def score_batch(
    model: nn.Module, cutout_paths: List[str], device: torch.device,
    pp_kwargs: dict | None = None,
) -> tuple[np.ndarray, int]:
    """Score a list of cutouts. Returns (scores array, n_errors)."""
    scores = []
    n_errors = 0
    for i, path in enumerate(cutout_paths):
        p = score_cutout(model, path, device, pp_kwargs=pp_kwargs)
        if p is not None:
            scores.append(p)
        else:
            scores.append(np.nan)
            n_errors += 1
        if (i + 1) % 200 == 0:
            print(f"    Scored {i + 1}/{len(cutout_paths)}", flush=True)
    return np.array(scores), n_errors


# ---------------------------------------------------------------------------
# Per-category stats
# ---------------------------------------------------------------------------
def compute_category_stats(scores: np.ndarray) -> Dict[str, Any]:
    """Compute N, median, frac>0.3, frac>0.5 for a score array."""
    valid = scores[np.isfinite(scores)]
    n = len(valid)
    if n == 0:
        return {
            "n_scored": 0,
            "median_score": None,
            "frac_above_0_3": None,
            "frac_above_0_5": None,
        }
    return {
        "n_scored": int(n),
        "median_score": float(np.median(valid)),
        "frac_above_0_3": float((valid > 0.3).mean()),
        "frac_above_0_5": float((valid > 0.5).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Test whether model detects deflector morphology vs genuine arc signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument(
        "--host-split",
        default="val",
        help="Split to use for negative hosts (default: val)",
    )
    ap.add_argument(
        "--n-per-category",
        type=int,
        default=200,
        help="Number of negatives per category to score (default: 200)",
    )
    ap.add_argument(
        "--out-dir",
        default="results/confuser_morphology_test",
        help="Output directory for results JSON",
    )
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("CONFUSER MORPHOLOGY TEST")
    print("=" * 70)
    print(f"\nScientific motivation: Testing whether the model detects deflector")
    print(f"galaxy morphology (confusers) rather than genuine arc signal.")
    print(f"If confusers score LOW, the model is NOT cheating.")

    # Load model + preprocessing config from checkpoint
    print(f"\n1. Loading model: {args.checkpoint}")
    model, arch, epoch, pp_kwargs = load_model(args.checkpoint, device)
    print(f"   Architecture: {arch}, Epoch: {epoch}")
    print(f"   Preprocessing: {pp_kwargs}")

    # Load manifest, filter val negatives
    print(f"\n2. Loading manifest: {args.manifest}")
    manifest = pd.read_parquet(args.manifest)
    val_neg = manifest[
        (manifest[SPLIT_COL] == args.host_split) & (manifest[LABEL_COL] == 0)
    ].copy()
    print(f"   Val negatives in '{args.host_split}': {len(val_neg):,}")

    if len(val_neg) == 0:
        print("ERROR: No negatives found in host split. Aborting.")
        sys.exit(1)

    # Check for confuser_category column
    has_confuser = CONFUSER_COL in val_neg.columns
    if not has_confuser:
        print(f"\nWARNING: Manifest has no '{CONFUSER_COL}' column.")
        print("   Cannot score confuser categories. Will score random baseline only.")

    results: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "host_split": args.host_split,
        "arch": arch,
        "epoch": epoch,
        "n_per_category": args.n_per_category,
        "seed": args.seed,
        "categories": {},
        "baseline_random": None,
        "key_finding": None,
    }

    # Score each confuser category
    if has_confuser:
        for cat in CONFUSER_CATEGORIES:
            cat_rows = val_neg[val_neg[CONFUSER_COL] == cat]
            n_avail = len(cat_rows)
            if n_avail == 0:
                print(f"\n3.{CONFUSER_CATEGORIES.index(cat)+1}. {cat}: 0 available (skipping)")
                results["categories"][cat] = {"n_available": 0, "n_scored": 0}
                continue

            n_sample = min(args.n_per_category, n_avail)
            sample = cat_rows.sample(n=n_sample, random_state=rng)
            paths = sample[CUTOUT_PATH_COL].tolist()

            print(f"\n3.{CONFUSER_CATEGORIES.index(cat)+1}. {cat}: scoring {n_sample} (of {n_avail} available)")
            scores, n_err = score_batch(model, paths, device, pp_kwargs=pp_kwargs)
            stats = compute_category_stats(scores)
            stats["n_available"] = n_avail
            stats["n_errors"] = n_err
            results["categories"][cat] = stats
            print(f"    N={stats['n_scored']}, median={stats['median_score']:.6f}, "
                  f">0.3: {stats['frac_above_0_3']*100:.2f}%, >0.5: {stats['frac_above_0_5']*100:.2f}%")

    # Score random baseline (regardless of category)
    n_baseline = min(args.n_per_category, len(val_neg))
    baseline_sample = val_neg.sample(n=n_baseline, random_state=rng)
    baseline_paths = baseline_sample[CUTOUT_PATH_COL].tolist()

    print(f"\n4. Baseline (random negatives): scoring {n_baseline}")
    baseline_scores, n_err = score_batch(model, baseline_paths, device, pp_kwargs=pp_kwargs)
    baseline_stats = compute_category_stats(baseline_scores)
    baseline_stats["n_errors"] = n_err
    results["baseline_random"] = baseline_stats
    print(f"    N={baseline_stats['n_scored']}, median={baseline_stats['median_score']:.6f}, "
          f">0.3: {baseline_stats['frac_above_0_3']*100:.2f}%, >0.5: {baseline_stats['frac_above_0_5']*100:.2f}%")

    # Key finding
    if has_confuser and results["categories"]:
        medians = [
            (cat, s["median_score"])
            for cat, s in results["categories"].items()
            if s["median_score"] is not None
        ]
        baseline_med = results["baseline_random"]["median_score"] or 0.0
        if medians:
            max_confuser_med = max(m for _, m in medians)
            if max_confuser_med < 0.1 and baseline_med < 0.1:
                key = (
                    "All confuser categories score very low (median < 0.1). "
                    "The model is NOT cheating by detecting galaxy morphology."
                )
            elif max_confuser_med < baseline_med * 1.5:
                key = (
                    "Confuser categories score similarly to or lower than random negatives. "
                    "No strong evidence of morphology shortcut learning."
                )
            else:
                key = (
                    "Some confuser categories score notably higher than baseline. "
                    "Consider investigating potential morphology shortcut learning."
                )
        else:
            key = "Insufficient confuser data for conclusion."
        results["key_finding"] = key
        print(f"\n{'='*70}")
        print("KEY FINDING")
        print(f"{'='*70}")
        print(f"  {key}")

    # Save results
    out_path = os.path.join(args.out_dir, "confuser_morphology_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Results saved: {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Category':<20} {'N':>6} {'Median':>10} {'>0.3':>8} {'>0.5':>8}")
    print("-" * 56)
    if has_confuser:
        for cat in CONFUSER_CATEGORIES:
            s = results["categories"].get(cat, {})
            n = s.get("n_scored", 0)
            med = s.get("median_score")
            f03 = s.get("frac_above_0_3")
            f05 = s.get("frac_above_0_5")
            med_s = f"{med:.4f}" if med is not None else "N/A"
            f03_s = f"{f03*100:.2f}%" if f03 is not None else "N/A"
            f05_s = f"{f05*100:.2f}%" if f05 is not None else "N/A"
            print(f"{cat:<20} {n:>6} {med_s:>10} {f03_s:>8} {f05_s:>8}")
    b = results["baseline_random"]
    if b:
        med_s = f"{b['median_score']:.4f}" if b.get("median_score") is not None else "N/A"
        f03_s = f"{b['frac_above_0_3']*100:.2f}%" if b.get("frac_above_0_3") is not None else "N/A"
        f05_s = f"{b['frac_above_0_5']*100:.2f}%" if b.get("frac_above_0_5") is not None else "N/A"
        print(f"{'random (baseline)':<20} {b.get('n_scored', 0):>6} {med_s:>10} {f03_s:>8} {f05_s:>8}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
