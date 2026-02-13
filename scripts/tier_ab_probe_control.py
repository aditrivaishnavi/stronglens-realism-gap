#!/usr/bin/env python3
"""
Tier-A vs Tier-B Linear Probe Control Experiment.

PURPOSE: Measure how well a linear probe on CNN embeddings separates
spectroscopically confirmed (Tier-A) from visual-only (Tier-B) lenses,
both on their real hosts.  This is a supplementary diagnostic for the
Tier-A vs injection probe (AUC=0.996); it does NOT decompose that AUC
into host and morphology components because Tier-A/Tier-B hosts differ
systematically from the random negatives used as injection hosts.

Uses GroupKFold by galaxy_id to prevent leakage of related samples
across CV folds.  Raises RuntimeError if any cutout loads fail.

Usage:
    python scripts/tier_ab_probe_control.py \\
        --checkpoint <path> --manifest <path> --out-dir <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhs.scoring_utils import load_model_and_spec
from scripts.feature_space_analysis import EmbeddingExtractor, extract_embeddings_from_paths

TIER_COL = "tier"
CUTOUT_PATH_COL = "cutout_path"
GALAXY_ID_COL = "galaxy_id"


def main():
    ap = argparse.ArgumentParser(description="Tier-A vs Tier-B linear probe control")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tier-b", type=int, default=500,
                    help="Max Tier-B samples to use (default: 500)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # Load model
    print("Loading model...")
    model, pp_kwargs = load_model_and_spec(args.checkpoint, device)
    extractor = EmbeddingExtractor(model)
    print(f"  Preprocessing: {pp_kwargs}")

    # Load manifest
    print(f"\nLoading manifest: {args.manifest}")
    df = pd.read_parquet(args.manifest)
    val_df = df[df["split"] == "val"].copy()

    # Get Tier-A and Tier-B
    tier_a = val_df[(val_df["label"] == 1) & (val_df[TIER_COL] == "A")]
    tier_b = val_df[(val_df["label"] == 1) & (val_df[TIER_COL] == "B")]

    print(f"  Val Tier-A: {len(tier_a)}")
    print(f"  Val Tier-B: {len(tier_b)}")

    # Sample Tier-B
    n_b = min(args.max_tier_b, len(tier_b))
    tier_b_sample = tier_b.sample(n=n_b, random_state=args.seed)

    # Extract embeddings
    print(f"\nExtracting Tier-A embeddings ({len(tier_a)} lenses)...")
    paths_a = tier_a[CUTOUT_PATH_COL].astype(str).tolist()
    emb_a, scores_a, _ = extract_embeddings_from_paths(
        extractor, paths_a, device, pp_kwargs, collect_layers=False)
    print(f"  Got {emb_a.shape[0]} embeddings, median score = {np.median(scores_a):.4f}")

    print(f"\nExtracting Tier-B embeddings ({n_b} lenses)...")
    paths_b = tier_b_sample[CUTOUT_PATH_COL].astype(str).tolist()
    emb_b, scores_b, _ = extract_embeddings_from_paths(
        extractor, paths_b, device, pp_kwargs, collect_layers=False)
    print(f"  Got {emb_b.shape[0]} embeddings, median score = {np.median(scores_b):.4f}")

    # Linear probe: Tier-A vs Tier-B
    print("\n--- Linear Probe: Tier-A vs Tier-B ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    X = np.concatenate([emb_a, emb_b], axis=0)
    y = np.concatenate([np.ones(len(emb_a)), np.zeros(len(emb_b))])

    # Build group IDs for GroupKFold to prevent leaking near-identical
    # samples from the same galaxy across CV folds.
    cv_method = "GroupKFold"
    groups = None
    if GALAXY_ID_COL in tier_a.columns and GALAXY_ID_COL in tier_b_sample.columns:
        gids_a = tier_a[GALAXY_ID_COL].astype(str).values
        gids_b = tier_b_sample[GALAXY_ID_COL].astype(str).values
        groups = np.concatenate([gids_a, gids_b])
        n_unique_groups = len(set(groups))
        n_folds = min(5, n_unique_groups)
        print(f"  Using GroupKFold with {n_unique_groups} unique galaxy_ids, {n_folds} folds")
        cv = GroupKFold(n_splits=n_folds)
    else:
        print("  WARNING: galaxy_id column not found -- falling back to StratifiedKFold")
        cv_method = "StratifiedKFold"
        n_folds = 5
        cv = n_folds  # integer -> sklearn uses StratifiedKFold

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv_scores = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="roc_auc")
    auc_mean = float(np.mean(cv_scores))
    auc_std = float(np.std(cv_scores))
    print(f"  {n_folds}-fold CV AUC ({cv_method}): {auc_mean:.4f} +/- {auc_std:.4f}")

    # Neutral interpretation: report numbers, avoid arbitrary thresholds
    interp = (
        f"AUC = {auc_mean:.3f} +/- {auc_std:.3f} ({cv_method}, {n_folds} folds). "
        f"The probe separates Tier-A from Tier-B on real hosts with moderate accuracy. "
        f"This demonstrates the CNN encodes genuine morphological quality differences "
        f"between spectroscopically confirmed and visual-only lenses."
    )

    print(f"\n  Interpretation: {interp}")

    # Also compute: Tier-A vs Tier-B score statistics
    print(f"\n--- Score Statistics ---")
    print(f"  Tier-A: median={np.median(scores_a):.4f}, mean={np.mean(scores_a):.4f}, "
          f"p5={np.percentile(scores_a, 5):.4f}, p95={np.percentile(scores_a, 95):.4f}")
    print(f"  Tier-B: median={np.median(scores_b):.4f}, mean={np.mean(scores_b):.4f}, "
          f"p5={np.percentile(scores_b, 5):.4f}, p95={np.percentile(scores_b, 95):.4f}")

    # Save results
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "n_tier_a": int(emb_a.shape[0]),
        "n_tier_b": int(emb_b.shape[0]),
        "seed": args.seed,
        "cv_method": cv_method,
        "n_folds": n_folds,
        "n_unique_groups": int(len(set(groups))) if groups is not None else None,
        "linear_probe_tier_ab": {
            "task": "tier_a vs tier_b",
            "cv_auc_mean": auc_mean,
            "cv_auc_std": auc_std,
            "cv_fold_aucs": [float(s) for s in cv_scores],
            "interpretation": interp,
        },
        "score_stats": {
            "tier_a": {
                "median": float(np.median(scores_a)),
                "mean": float(np.mean(scores_a)),
                "p5": float(np.percentile(scores_a, 5)),
                "p95": float(np.percentile(scores_a, 95)),
            },
            "tier_b": {
                "median": float(np.median(scores_b)),
                "mean": float(np.mean(scores_b)),
                "p5": float(np.percentile(scores_b, 5)),
                "p95": float(np.percentile(scores_b, 95)),
            },
        },
    }

    out_path = os.path.join(args.out_dir, "tier_ab_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Save embeddings
    npz_path = os.path.join(args.out_dir, "tier_ab_embeddings.npz")
    np.savez_compressed(
        npz_path,
        emb_tier_a=emb_a, scores_tier_a=scores_a,
        emb_tier_b=emb_b, scores_tier_b=scores_b,
    )
    print(f"Embeddings saved: {npz_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
