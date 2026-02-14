#!/usr/bin/env python3
"""
Annulus Comparison: Compare (20,32) vs (32.5,45) normalization statistics.

"Compute annulus median and MAD for 1000 real training
cutouts with BOTH (20,32) and (32.5,45). Compare distributions. If nearly
identical, bug is cosmetic and retraining won't help."

This is one of four cheap pre-retrain experiments that should be run BEFORE
committing GPU-hours to retraining. Estimated runtime: 1-5 minutes CPU.

What this decides:
  - If median/MAD distributions barely change -> annulus issue is cosmetic
  - If they shift materially AND correlate with PSFsize/depth/host size
    -> annulus may inject condition-dependent distortions worth retraining

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.

    python scripts/annulus_comparison.py \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/annulus_comparison \\
        --n-samples 1000

Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from dhs.utils import normalize_outer_annulus, radial_mask, robust_median_mad


def compute_annulus_stats(
    img_2d: np.ndarray,
    r_in: float,
    r_out: float,
) -> Dict[str, float]:
    """Compute median and MAD in a given annulus for a single 2D image."""
    H, W = img_2d.shape
    mask = radial_mask(H, W, r_in, r_out)
    vals = img_2d[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return {"median": float("nan"), "mad": float("nan"), "n_pix": 0}
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return {"median": med, "mad": mad, "n_pix": len(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare annulus (20,32) vs (32.5,45) normalization stats",
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load manifest
    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    split_df = df[df["split"] == args.split].copy()

    # Sample: mix of positives and negatives
    n = min(args.n_samples, len(split_df))
    sample = split_df.sample(n=n, random_state=args.seed)
    print(f"Sampled {n} cutouts from split={args.split}")

    # Two annulus configs
    configs = {
        "old_20_32": (20.0, 32.0),
        "new_32p5_45": (32.5, 45.0),
    }

    # Compute stats
    results_per_cutout: List[Dict] = []
    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
        except Exception:
            continue

        entry = {
            "label": int(row.get("label", -1)),
            "psfsize_r": float(row.get("psfsize_r", float("nan"))),
            "psfdepth_r": float(row.get("psfdepth_r", float("nan"))),
        }

        # r-band stats for both annulus configs
        r_band = hwc[:, :, 1] if hwc.shape[2] == 3 else hwc[:, :, 0]
        for config_name, (r_in, r_out) in configs.items():
            s = compute_annulus_stats(r_band, r_in, r_out)
            entry[f"median_{config_name}"] = s["median"]
            entry[f"mad_{config_name}"] = s["mad"]
            entry[f"npix_{config_name}"] = s["n_pix"]

        results_per_cutout.append(entry)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}", end="\r")

    print(f"\nComputed stats for {len(results_per_cutout)} cutouts")

    # Aggregate
    rdf = pd.DataFrame(results_per_cutout)

    # Summary statistics
    summary = {}
    for config_name in configs:
        med_col = f"median_{config_name}"
        mad_col = f"mad_{config_name}"
        valid = rdf[rdf[med_col].notna()]
        summary[config_name] = {
            "median_of_medians": float(valid[med_col].median()),
            "mean_of_medians": float(valid[med_col].mean()),
            "std_of_medians": float(valid[med_col].std()),
            "median_of_mads": float(valid[mad_col].median()),
            "mean_of_mads": float(valid[mad_col].mean()),
            "std_of_mads": float(valid[mad_col].std()),
            "n_valid": int(len(valid)),
        }

    # Paired comparison
    valid_both = rdf.dropna(subset=[
        "median_old_20_32", "median_new_32p5_45",
        "mad_old_20_32", "mad_new_32p5_45",
    ])

    med_diff = valid_both["median_new_32p5_45"] - valid_both["median_old_20_32"]
    mad_diff = valid_both["mad_new_32p5_45"] - valid_both["mad_old_20_32"]

    comparison = {
        "n_paired": int(len(valid_both)),
        "median_diff": {
            "mean": float(med_diff.mean()),
            "std": float(med_diff.std()),
            "median": float(med_diff.median()),
            "p25": float(med_diff.quantile(0.25)),
            "p75": float(med_diff.quantile(0.75)),
        },
        "mad_diff": {
            "mean": float(mad_diff.mean()),
            "std": float(mad_diff.std()),
            "median": float(mad_diff.median()),
            "p25": float(mad_diff.quantile(0.25)),
            "p75": float(mad_diff.quantile(0.75)),
        },
    }

    # KS test: are the distributions significantly different?
    if len(valid_both) > 10:
        ks_med = sp_stats.ks_2samp(
            valid_both["median_old_20_32"].values,
            valid_both["median_new_32p5_45"].values,
        )
        ks_mad = sp_stats.ks_2samp(
            valid_both["mad_old_20_32"].values,
            valid_both["mad_new_32p5_45"].values,
        )
        comparison["ks_test_median"] = {
            "statistic": float(ks_med.statistic),
            "pvalue": float(ks_med.pvalue),
        }
        comparison["ks_test_mad"] = {
            "statistic": float(ks_mad.statistic),
            "pvalue": float(ks_mad.pvalue),
        }

    # Correlation of shift with PSF/depth
    corr_with_psf = {}
    if "psfsize_r" in valid_both.columns:
        psf = valid_both["psfsize_r"].values
        finite_psf = np.isfinite(psf)
        if finite_psf.sum() > 20:
            r_med, p_med = sp_stats.pearsonr(
                psf[finite_psf], med_diff.values[finite_psf]
            )
            r_mad, p_mad = sp_stats.pearsonr(
                psf[finite_psf], mad_diff.values[finite_psf]
            )
            corr_with_psf = {
                "median_diff_vs_psf": {"r": float(r_med), "p": float(p_med)},
                "mad_diff_vs_psf": {"r": float(r_mad), "p": float(p_mad)},
            }

    corr_with_depth = {}
    if "psfdepth_r" in valid_both.columns:
        depth = valid_both["psfdepth_r"].values
        finite_depth = np.isfinite(depth)
        if finite_depth.sum() > 20:
            r_med, p_med = sp_stats.pearsonr(
                depth[finite_depth], med_diff.values[finite_depth]
            )
            r_mad, p_mad = sp_stats.pearsonr(
                depth[finite_depth], mad_diff.values[finite_depth]
            )
            corr_with_depth = {
                "median_diff_vs_depth": {"r": float(r_med), "p": float(p_med)},
                "mad_diff_vs_depth": {"r": float(r_mad), "p": float(p_mad)},
            }

    # Positive vs negative comparison
    pos_shift = {}
    neg_shift = {}
    if "label" in valid_both.columns:
        pos_mask = valid_both["label"] == 1
        neg_mask = valid_both["label"] == 0
        if pos_mask.sum() > 5:
            pos_shift = {
                "n": int(pos_mask.sum()),
                "mean_median_diff": float(med_diff[pos_mask].mean()),
                "mean_mad_diff": float(mad_diff[pos_mask].mean()),
            }
        if neg_mask.sum() > 5:
            neg_shift = {
                "n": int(neg_mask.sum()),
                "mean_median_diff": float(med_diff[neg_mask].mean()),
                "mean_mad_diff": float(mad_diff[neg_mask].mean()),
            }

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_cutouts": len(results_per_cutout),
        "split": args.split,
        "annulus_configs": {k: list(v) for k, v in configs.items()},
        "per_config_summary": summary,
        "paired_comparison": comparison,
        "correlation_with_psf": corr_with_psf,
        "correlation_with_depth": corr_with_depth,
        "shift_by_label": {
            "positives": pos_shift,
            "negatives": neg_shift,
        },
        "interpretation": {
            "if_ks_pvalue_large": "Distributions barely change -> annulus issue is cosmetic.",
            "if_ks_pvalue_small_and_correlated": (
                "Material, condition-dependent shifts -> annulus may inject "
                "nuisance variation worth retraining to fix."
            ),
            "if_pos_neg_shift_differs": (
                "Different shifts for positives vs negatives -> annulus creates "
                "a systematic real-vs-injection mismatch."
            ),
        },
    }

    json_path = os.path.join(args.out_dir, "annulus_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANNULUS COMPARISON: (20,32) vs (32.5,45)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"\n  {name}:")
        print(f"    Median of medians: {s['median_of_medians']:.6f}")
        print(f"    Median of MADs:    {s['median_of_mads']:.6f}")

    print(f"\n  Paired differences (new - old), N={comparison['n_paired']}:")
    print(f"    Median diff:  mean={comparison['median_diff']['mean']:.6f}, "
          f"std={comparison['median_diff']['std']:.6f}")
    print(f"    MAD diff:     mean={comparison['mad_diff']['mean']:.6f}, "
          f"std={comparison['mad_diff']['std']:.6f}")

    if "ks_test_median" in comparison:
        print(f"\n  KS test (median): stat={comparison['ks_test_median']['statistic']:.4f}, "
              f"p={comparison['ks_test_median']['pvalue']:.4e}")
        print(f"  KS test (MAD):    stat={comparison['ks_test_mad']['statistic']:.4f}, "
              f"p={comparison['ks_test_mad']['pvalue']:.4e}")

    if corr_with_psf:
        print(f"\n  Correlation with PSF:")
        print(f"    median_diff: r={corr_with_psf['median_diff_vs_psf']['r']:.4f}, "
              f"p={corr_with_psf['median_diff_vs_psf']['p']:.4e}")
        print(f"    mad_diff:    r={corr_with_psf['mad_diff_vs_psf']['r']:.4f}, "
              f"p={corr_with_psf['mad_diff_vs_psf']['p']:.4e}")

    if pos_shift and neg_shift:
        print(f"\n  Shift by label:")
        print(f"    Positives (N={pos_shift['n']}): median_diff={pos_shift['mean_median_diff']:.6f}")
        print(f"    Negatives (N={neg_shift['n']}): median_diff={neg_shift['mean_median_diff']:.6f}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
