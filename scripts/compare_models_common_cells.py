#!/usr/bin/env python3
"""
Compare Model 1 vs Model 2 on common (intersection) grid cells.

Model 2 has 209 populated cells vs Model 1's 220.
The 11 missing cells in Model 2 could bias the comparison if those cells
have higher-than-average completeness. This script restricts both models
to the intersection of populated cells for a fair comparison.

Usage:
    python scripts/compare_models_common_cells.py \\
        --model1-results results/grid_model1.parquet \\
        --model2-results results/grid_model2.parquet \\
        --out results/common_cell_comparison.json

Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def cell_key(row: pd.Series) -> tuple:
    """Create a unique cell identifier from grid coordinates."""
    return (
        round(float(row["theta_e"]), 3),
        round(float(row["psf_fwhm"]), 3),
        round(float(row["depth_5sig"]), 3),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare Model 1 vs Model 2 on common grid cells"
    )
    ap.add_argument("--model1-results", required=True,
                    help="Path to Model 1 grid results (parquet or CSV)")
    ap.add_argument("--model2-results", required=True,
                    help="Path to Model 2 grid results (parquet or CSV)")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Detection threshold (default: 0.3)")
    ap.add_argument("--out", required=True,
                    help="Output JSON path")
    args = ap.parse_args()

    # Load results
    def load_df(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    df1 = load_df(args.model1_results)
    df2 = load_df(args.model2_results)

    # Filter to "all" source mag bin and target threshold
    df1_all = df1[(df1["source_mag_bin"] == "all") &
                  (np.abs(df1["threshold"] - args.threshold) < 0.001)].copy()
    df2_all = df2[(df2["source_mag_bin"] == "all") &
                  (np.abs(df2["threshold"] - args.threshold) < 0.001)].copy()

    # Add cell keys
    df1_all["cell_key"] = df1_all.apply(cell_key, axis=1)
    df2_all["cell_key"] = df2_all.apply(cell_key, axis=1)

    # Find populated cells (sufficient=True or n_injections > 0)
    cells1 = set(df1_all[df1_all["n_injections"] > 0]["cell_key"].values)
    cells2 = set(df2_all[df2_all["n_injections"] > 0]["cell_key"].values)
    common = cells1 & cells2
    only_m1 = cells1 - cells2
    only_m2 = cells2 - cells1

    print(f"Model 1 populated cells: {len(cells1)}")
    print(f"Model 2 populated cells: {len(cells2)}")
    print(f"Common cells:            {len(common)}")
    print(f"Only in Model 1:         {len(only_m1)}")
    print(f"Only in Model 2:         {len(only_m2)}")

    # Restrict to common cells
    df1_common = df1_all[df1_all["cell_key"].isin(common)].copy()
    df2_common = df2_all[df2_all["cell_key"].isin(common)].copy()

    # Compute weighted mean completeness
    def weighted_completeness(df):
        valid = df[df["n_injections"] > 0]
        if len(valid) == 0:
            return float("nan"), 0
        total_inj = valid["n_injections"].sum()
        total_det = valid["n_detected"].sum()
        return float(total_det / total_inj), int(total_inj)

    comp1_all, n1_all = weighted_completeness(
        df1_all[df1_all["n_injections"] > 0])
    comp2_all, n2_all = weighted_completeness(
        df2_all[df2_all["n_injections"] > 0])
    comp1_common, n1_common = weighted_completeness(df1_common)
    comp2_common, n2_common = weighted_completeness(df2_common)

    print(f"\nAll cells:")
    print(f"  Model 1: {comp1_all*100:.2f}% ({n1_all} injections)")
    print(f"  Model 2: {comp2_all*100:.2f}% ({n2_all} injections)")
    print(f"  Delta:   {(comp2_all - comp1_all)*100:+.2f} pp")

    print(f"\nCommon cells only:")
    print(f"  Model 1: {comp1_common*100:.2f}% ({n1_common} injections)")
    print(f"  Model 2: {comp2_common*100:.2f}% ({n2_common} injections)")
    print(f"  Delta:   {(comp2_common - comp1_common)*100:+.2f} pp")

    # Check if excluded cells are systematically different
    df1_only = df1_all[df1_all["cell_key"].isin(only_m1)]
    comp1_excluded, n1_excluded = weighted_completeness(
        df1_only[df1_only["n_injections"] > 0])

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model1_results": args.model1_results,
        "model2_results": args.model2_results,
        "threshold": args.threshold,
        "cell_counts": {
            "model1_populated": len(cells1),
            "model2_populated": len(cells2),
            "common": len(common),
            "only_model1": len(only_m1),
            "only_model2": len(only_m2),
        },
        "all_cells": {
            "model1_completeness": comp1_all,
            "model1_n_injections": n1_all,
            "model2_completeness": comp2_all,
            "model2_n_injections": n2_all,
            "delta_pp": (comp2_all - comp1_all) * 100,
        },
        "common_cells": {
            "model1_completeness": comp1_common,
            "model1_n_injections": n1_common,
            "model2_completeness": comp2_common,
            "model2_n_injections": n2_common,
            "delta_pp": (comp2_common - comp1_common) * 100,
        },
        "excluded_model1_only": {
            "completeness": comp1_excluded,
            "n_injections": n1_excluded,
            "note": "Completeness of cells in Model 1 but not Model 2. "
                    "If much higher/lower than overall, cell-count bias exists.",
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
