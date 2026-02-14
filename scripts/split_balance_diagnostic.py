#!/usr/bin/env python3
"""
Split Balance Diagnostic: Verify PSF/depth balance and positive spatial
distribution across train/val/test splits.

Code review:
  Q3.9: "Are positives spatially correlated? Many DESI candidates from
         deep-coverage regions. If most positives in a few pixels,
         train/val split heavily unbalanced."
  Q3.10: "Your spatial splits prevent field-level leakage. But do they
          prevent PSF/depth condition leakage? If high-PSF in train and
          low-PSF in val, the model learns PSF-specific features."

This script loads the manifest and reports:
  1. Distribution of positives per HEALPix pixel
  2. PSF/depth distributions in train vs val vs test (KS test)
  3. Tier-A counts per split
  4. Masked pixel fraction statistics (Q3.3 overlap)

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.

    python scripts/split_balance_diagnostic.py \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/split_balance

Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


TIER_COL = "tier"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split balance diagnostic: verify PSF/depth balance across splits",
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    results: Dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_total": len(df),
    }

    # ---- 1. Per-split counts ----
    print("\n=== 1. Per-split counts ===")
    split_counts = {}
    for split_name in sorted(df["split"].unique()):
        sdf = df[df["split"] == split_name]
        n_pos = int((sdf["label"] == 1).sum())
        n_neg = int((sdf["label"] == 0).sum())
        tier_a = 0
        tier_b = 0
        if TIER_COL in sdf.columns:
            tier_a = int(((sdf["label"] == 1) & (sdf[TIER_COL] == "A")).sum())
            tier_b = int(((sdf["label"] == 1) & (sdf[TIER_COL] == "B")).sum())
        split_counts[split_name] = {
            "n_total": len(sdf), "n_pos": n_pos, "n_neg": n_neg,
            "n_tier_a": tier_a, "n_tier_b": tier_b,
        }
        print(f"  {split_name}: {len(sdf)} total, {n_pos} pos ({tier_a} A, {tier_b} B), {n_neg} neg")
    results["split_counts"] = split_counts

    # ---- 2. Positive spatial distribution ----
    print("\n=== 2. Positive spatial distribution (HEALPix) ===")
    healpix_col = None
    for col in ["healpix_128", "healpix", "hpx"]:
        if col in df.columns:
            healpix_col = col
            break

    if healpix_col:
        pos_df = df[df["label"] == 1]
        # Drop NaN healpix values (objects with missing ra/dec)
        hp_vals = pos_df[healpix_col].dropna()
        n_nan = len(pos_df) - len(hp_vals)
        if n_nan > 0:
            print(f"  WARNING: {n_nan} positives have NaN healpix (missing ra/dec)")

        if len(hp_vals) > 0:
            hp_counts = Counter(hp_vals.values)
            n_pixels = len(hp_counts)
            counts_arr = np.array(list(hp_counts.values()))
            top5 = hp_counts.most_common(5)

            spatial = {
                "healpix_col": healpix_col,
                "n_unique_pixels": n_pixels,
                "n_nan_healpix": n_nan,
                "max_positives_per_pixel": int(counts_arr.max()),
                "median_positives_per_pixel": float(np.median(counts_arr)),
                "mean_positives_per_pixel": float(np.mean(counts_arr)),
                "top5_pixels": [{"pixel": int(p), "count": int(c)} for p, c in top5],
                "pct_in_top5": float(sum(c for _, c in top5) / len(pos_df) * 100),
            }
            print(f"  {n_pixels} unique HEALPix pixels with positives")
            print(f"  Max positives per pixel: {counts_arr.max()}")
            print(f"  Top 5 pixels hold {spatial['pct_in_top5']:.1f}% of all positives")
            for p, c in top5:
                print(f"    Pixel {p}: {c} positives")
        else:
            spatial = {
                "healpix_col": healpix_col,
                "n_unique_pixels": 0,
                "n_nan_healpix": n_nan,
                "note": "All positive healpix values are NaN",
            }
            print("  All positive healpix values are NaN — spatial analysis skipped")

        results["positive_spatial_distribution"] = spatial
    else:
        print("  No HEALPix column found — skipping spatial analysis")
        results["positive_spatial_distribution"] = None

    # ---- 3. PSF/depth balance across splits ----
    print("\n=== 3. PSF/depth balance across splits ===")
    condition_cols = [c for c in ["psfsize_r", "psfdepth_r", "psfsize_g", "psfdepth_g"] if c in df.columns]

    ks_results = {}
    splits = sorted(df["split"].unique())
    for col in condition_cols:
        ks_results[col] = {}
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1 = splits[i]
                s2 = splits[j]
                v1 = df.loc[df["split"] == s1, col].dropna().values
                v2 = df.loc[df["split"] == s2, col].dropna().values
                if len(v1) > 10 and len(v2) > 10:
                    ks = sp_stats.ks_2samp(v1, v2)
                    pair_key = f"{s1}_vs_{s2}"
                    ks_results[col][pair_key] = {
                        "ks_stat": float(ks.statistic),
                        "p_value": float(ks.pvalue),
                        "mean_1": float(np.mean(v1)),
                        "mean_2": float(np.mean(v2)),
                        "n_1": len(v1),
                        "n_2": len(v2),
                    }
                    sig = "***" if ks.pvalue < 0.001 else ("**" if ks.pvalue < 0.01 else ("*" if ks.pvalue < 0.05 else ""))
                    print(f"  {col} {s1} vs {s2}: KS={ks.statistic:.4f}, p={ks.pvalue:.4e} {sig}")
                    print(f"    mean({s1})={np.mean(v1):.4f}, mean({s2})={np.mean(v2):.4f}")

    results["psf_depth_balance"] = ks_results

    # ---- 4. PSF/depth balance for POSITIVES ONLY ----
    print("\n=== 4. PSF/depth balance for POSITIVES across splits ===")
    pos_ks = {}
    pos_df = df[df["label"] == 1]
    for col in condition_cols[:2]:  # Just psfsize_r and psfdepth_r
        pos_ks[col] = {}
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1 = splits[i]
                s2 = splits[j]
                v1 = pos_df.loc[pos_df["split"] == s1, col].dropna().values
                v2 = pos_df.loc[pos_df["split"] == s2, col].dropna().values
                if len(v1) > 5 and len(v2) > 5:
                    ks = sp_stats.ks_2samp(v1, v2)
                    pair_key = f"{s1}_vs_{s2}"
                    pos_ks[col][pair_key] = {
                        "ks_stat": float(ks.statistic),
                        "p_value": float(ks.pvalue),
                        "n_1": len(v1),
                        "n_2": len(v2),
                    }
                    sig = "***" if ks.pvalue < 0.001 else ""
                    print(f"  {col} (pos) {s1} vs {s2}: KS={ks.statistic:.4f}, p={ks.pvalue:.4e} {sig}")
    results["positive_psf_depth_balance"] = pos_ks

    # Save
    json_path = os.path.join(args.out_dir, "split_balance_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # ---- Interpretation ----
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  - KS p < 0.05: split distributions differ significantly")
    print("  - Top 5 pixels >50%: positives are highly clustered")
    print("  - Large mean PSF difference: condition leakage risk")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
