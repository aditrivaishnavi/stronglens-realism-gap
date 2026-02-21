#!/usr/bin/env python3
"""
Poisson Diagnostics for D06 — Paired delta, conditioned grid, score histograms.

Three targeted analyses:

1. PAIRED DELTA ANALYSIS
   For each injection in the bright-arc test, compute delta_p = score_poisson -
   score_baseline. Same seed (42), same hosts, same geometry — only Poisson
   differs. Reports per-magnitude-bin statistics to distinguish "systematic
   score uplift" (mechanism B) from "threshold scatter" (mechanism A).

2. CONDITIONED GRID SUBSET (high-value diagnostic)
   From the 110k grid CSVs, restrict to theta_E near 1.5 and lensed mag 20-23
   to directly reconcile the grid vs bright-arc Poisson results. Also produces
   a full 2D cross-tabulation: theta_E x lensed_mag for both conditions.

3. SCORE HISTOGRAMS
   Plots score distributions from bright-arc scored parquets (baseline vs
   Poisson) per magnitude bin. Saves as PNG.

Usage:
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap
    PYTHONPATH=. python scripts/analyze_poisson_diagnostics.py \
        --d06-dir results/D06_corrected_priors \
        --out-dir results/D06_corrected_priors/poisson_diagnostics
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# matplotlib may not be available on headless lambda3 — use Agg backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. PAIRED DELTA ANALYSIS
# ---------------------------------------------------------------------------

def paired_delta_analysis(d06_dir: str) -> dict:
    """
    Join ba_baseline and ba_poisson scored parquets on (mag_bin, host_idx).
    Compute per-injection delta_p = score_poisson - score_baseline.
    Report distribution statistics by magnitude bin.
    """
    bl_path = os.path.join(d06_dir, "ba_baseline", "injection_metadata_scored.parquet")
    po_path = os.path.join(d06_dir, "ba_poisson", "injection_metadata_scored.parquet")

    if not os.path.exists(bl_path):
        print(f"  ERROR: baseline parquet not found: {bl_path}")
        return {"error": f"file not found: {bl_path}"}
    if not os.path.exists(po_path):
        print(f"  ERROR: poisson parquet not found: {po_path}")
        return {"error": f"file not found: {po_path}"}

    bl = pd.read_parquet(bl_path)
    po = pd.read_parquet(po_path)

    print(f"  Loaded baseline: {len(bl)} rows, poisson: {len(po)} rows")
    print(f"  Baseline columns: {list(bl.columns)}")

    # Join on (mag_bin, host_idx) — these uniquely identify each injection
    # because both conditions used seed=42 and the same host selection.
    merged = bl.merge(
        po[["mag_bin", "host_idx", "cnn_score"]],
        on=["mag_bin", "host_idx"],
        how="inner",
        suffixes=("_baseline", "_poisson"),
    )
    print(f"  Merged: {len(merged)} paired injections")

    if len(merged) == 0:
        return {"error": "no paired injections found after merge"}

    merged["delta_p"] = merged["cnn_score_poisson"] - merged["cnn_score_baseline"]

    # Per-magnitude-bin statistics
    results_by_bin = {}
    detection_threshold = 0.3

    for mb, grp in merged.groupby("mag_bin"):
        dp = grp["delta_p"].dropna()
        bl_scores = grp["cnn_score_baseline"].dropna()
        po_scores = grp["cnn_score_poisson"].dropna()

        n = len(dp)
        if n == 0:
            continue

        # Detection crossings: how many cross 0.3 threshold in each direction
        bl_above = bl_scores >= detection_threshold
        po_above = po_scores >= detection_threshold
        gained = int(((~bl_above) & po_above).sum())   # below -> above
        lost = int((bl_above & (~po_above)).sum())      # above -> below
        net_crossings = gained - lost

        results_by_bin[str(mb)] = {
            "n_paired": n,
            "mean_delta": float(dp.mean()),
            "median_delta": float(dp.median()),
            "std_delta": float(dp.std()),
            "pct_positive": float((dp > 0).mean()) * 100,
            "pct_negative": float((dp < 0).mean()) * 100,
            "pct_zero": float((dp == 0).mean()) * 100,
            "p25_delta": float(dp.quantile(0.25)),
            "p75_delta": float(dp.quantile(0.75)),
            "min_delta": float(dp.min()),
            "max_delta": float(dp.max()),
            # Detection threshold crossings
            "n_gained_above_03": gained,
            "n_lost_below_03": lost,
            "net_crossings": net_crossings,
            # Score distribution shifts
            "baseline_median_score": float(bl_scores.median()),
            "poisson_median_score": float(po_scores.median()),
            "baseline_detection_rate": float(bl_above.mean()),
            "poisson_detection_rate": float(po_above.mean()),
        }

    # Determine mechanism: A (threshold scatter) vs B (systematic uplift)
    # If the mean delta is large and positive AND pct_positive >> 50%,
    # it's mechanism B (systematic uplift). If mean delta ~ 0 but there
    # are many crossings in both directions, it's mechanism A.
    mechanism_assessment = {}
    for mb, stats in results_by_bin.items():
        mean_d = stats["mean_delta"]
        pct_pos = stats["pct_positive"]
        net = stats["net_crossings"]

        if abs(mean_d) > 0.02 and pct_pos > 60:
            mech = "B_systematic_uplift"
        elif abs(mean_d) > 0.02 and pct_pos < 40:
            mech = "B_systematic_decrease"
        elif abs(mean_d) < 0.01 and stats["n_gained_above_03"] > 0:
            mech = "A_threshold_scatter"
        else:
            mech = "mixed"

        mechanism_assessment[mb] = mech

    return {
        "n_total_paired": len(merged),
        "results_by_bin": results_by_bin,
        "mechanism_assessment": mechanism_assessment,
    }


def plot_score_histograms(d06_dir: str, out_dir: str) -> str:
    """
    Plot overlapping score histograms (baseline vs Poisson) per mag bin.
    Returns path to saved figure.
    """
    bl_path = os.path.join(d06_dir, "ba_baseline", "injection_metadata_scored.parquet")
    po_path = os.path.join(d06_dir, "ba_poisson", "injection_metadata_scored.parquet")

    bl = pd.read_parquet(bl_path)
    po = pd.read_parquet(po_path)

    mag_bins = sorted(bl["mag_bin"].unique())
    n_bins = len(mag_bins)

    fig, axes = plt.subplots(2, (n_bins + 1) // 2, figsize=(4 * ((n_bins + 1) // 2), 8))
    axes = axes.flatten()

    for i, mb in enumerate(mag_bins):
        ax = axes[i]
        bl_scores = bl[bl["mag_bin"] == mb]["cnn_score"].dropna()
        po_scores = po[po["mag_bin"] == mb]["cnn_score"].dropna()

        bins_h = np.linspace(0, 1, 50)
        ax.hist(bl_scores, bins=bins_h, alpha=0.5, label="Baseline", density=True)
        ax.hist(po_scores, bins=bins_h, alpha=0.5, label="Poisson", density=True)
        ax.axvline(0.3, color="red", linestyle="--", linewidth=1, label="p=0.3")
        ax.set_title(f"Mag {mb}")
        ax.set_xlabel("CNN score")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("D06 Bright-Arc Score Distributions: Baseline vs Poisson", fontsize=12)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "score_histograms_baseline_vs_poisson.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved score histograms: {fig_path}")
    return fig_path


def plot_delta_histograms(d06_dir: str, out_dir: str) -> str:
    """
    Plot delta_p histograms per mag bin (how much each injection's score changes).
    """
    bl_path = os.path.join(d06_dir, "ba_baseline", "injection_metadata_scored.parquet")
    po_path = os.path.join(d06_dir, "ba_poisson", "injection_metadata_scored.parquet")

    bl = pd.read_parquet(bl_path)
    po = pd.read_parquet(po_path)

    merged = bl.merge(
        po[["mag_bin", "host_idx", "cnn_score"]],
        on=["mag_bin", "host_idx"],
        how="inner",
        suffixes=("_baseline", "_poisson"),
    )
    merged["delta_p"] = merged["cnn_score_poisson"] - merged["cnn_score_baseline"]

    mag_bins = sorted(merged["mag_bin"].unique())
    n_bins = len(mag_bins)

    fig, axes = plt.subplots(2, (n_bins + 1) // 2, figsize=(4 * ((n_bins + 1) // 2), 8))
    axes = axes.flatten()

    for i, mb in enumerate(mag_bins):
        ax = axes[i]
        dp = merged[merged["mag_bin"] == mb]["delta_p"].dropna()

        ax.hist(dp, bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        mean_dp = dp.mean()
        ax.axvline(mean_dp, color="orange", linestyle="-", linewidth=1.5,
                   label=f"mean={mean_dp:.3f}")
        ax.set_title(f"Mag {mb} (n={len(dp)})")
        ax.set_xlabel("delta_p (Poisson - Baseline)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("D06 Per-Injection Score Change: Poisson - Baseline", fontsize=12)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "delta_p_histograms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved delta_p histograms: {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 2. CONDITIONED GRID SUBSET
# ---------------------------------------------------------------------------

def conditioned_grid_analysis(d06_dir: str) -> dict:
    """
    From the 110k grid CSVs, produce:
    - Subset at theta_E ~ 1.5, lensed mag 20-23 (bright-arc comparable regime)
    - Full 2D cross-tabulation: theta_E x lensed_mag for no-Poisson and Poisson
    - Poisson delta by theta_E (reproducing the earlier finding)
    """
    results = {}

    for label, subdir in [("no_poisson", "grid_no_poisson"),
                          ("poisson", "grid_poisson")]:
        csv_path = os.path.join(d06_dir, subdir, "selection_function.csv")
        df = pd.read_csv(csv_path)

        # Filter to p>0.3 threshold, source_mag_bin='all' for theta_E analysis
        df_all = df[
            (df["threshold"] == 0.3)
            & (df["source_mag_bin"] == "all")
            & (df["sufficient"] == True)
        ]

        # Completeness by theta_E (marginal over PSF/depth)
        by_te = {}
        for te in sorted(df_all["theta_e"].unique()):
            sub = df_all[df_all["theta_e"] == te]
            n = int(sub["n_injections"].sum())
            d = int(sub["n_detected"].sum())
            by_te[f"{te:.2f}"] = {
                "n_injections": n,
                "n_detected": d,
                "completeness": d / n if n > 0 else 0.0,
            }

        # Completeness by lensed mag bin
        df_lensed = df[
            (df["threshold"] == 0.3)
            & (df["source_mag_bin"].str.startswith("lensed_"))
            & (df["sufficient"] == True)
        ]
        by_lmag = {}
        for smb in sorted(df_lensed["source_mag_bin"].unique()):
            sub = df_lensed[df_lensed["source_mag_bin"] == smb]
            n = int(sub["n_injections"].sum())
            d = int(sub["n_detected"].sum())
            by_lmag[smb] = {
                "n_injections": n,
                "n_detected": d,
                "completeness": d / n if n > 0 else 0.0,
            }

        # 2D cross-tabulation: theta_E x lensed_mag
        # For each theta_E, break down by lensed_mag bin
        cross_tab = {}
        for te in sorted(df_lensed["theta_e"].unique()):
            te_key = f"{te:.2f}"
            cross_tab[te_key] = {}
            for smb in sorted(df_lensed["source_mag_bin"].unique()):
                sub = df_lensed[
                    (df_lensed["theta_e"] == te)
                    & (df_lensed["source_mag_bin"] == smb)
                ]
                n = int(sub["n_injections"].sum())
                d = int(sub["n_detected"].sum())
                cross_tab[te_key][smb] = {
                    "n_injections": n,
                    "n_detected": d,
                    "completeness": d / n if n > 0 else 0.0,
                }

        results[label] = {
            "by_theta_e": by_te,
            "by_lensed_mag": by_lmag,
            "cross_tab_theta_e_x_lensed_mag": cross_tab,
        }

    # Poisson delta by theta_E (the key finding from the Poisson analysis)
    poisson_delta_by_te = {}
    all_tes = set(results["no_poisson"]["by_theta_e"].keys()) & set(
        results["poisson"]["by_theta_e"].keys()
    )
    for te in sorted(all_tes):
        np_c = results["no_poisson"]["by_theta_e"][te]["completeness"]
        p_c = results["poisson"]["by_theta_e"][te]["completeness"]
        delta_pp = (p_c - np_c) * 100
        poisson_delta_by_te[te] = {
            "no_poisson": np_c,
            "poisson": p_c,
            "delta_pp": delta_pp,
            "poisson_helps": delta_pp > 0,
        }

    # Conditioned subset: theta_E = 1.50 and lensed mag 20-23
    # This is the bright-arc-comparable regime
    conditioned_subset = {}
    for label in ["no_poisson", "poisson"]:
        ct = results[label]["cross_tab_theta_e_x_lensed_mag"]
        te_key = "1.50"
        if te_key in ct:
            relevant_bins = {k: v for k, v in ct[te_key].items()
                            if any(k.startswith(f"lensed_{m}")
                                   for m in ["20", "21", "22"])}
            total_n = sum(v["n_injections"] for v in relevant_bins.values())
            total_d = sum(v["n_detected"] for v in relevant_bins.values())
            conditioned_subset[label] = {
                "theta_e": te_key,
                "lensed_mag_range": "20-23",
                "bins": relevant_bins,
                "total_injections": total_n,
                "total_detected": total_d,
                "completeness": total_d / total_n if total_n > 0 else 0.0,
            }
        else:
            conditioned_subset[label] = {"error": f"theta_E={te_key} not in grid"}

    # Poisson delta by lensed mag
    poisson_delta_by_lmag = {}
    np_lmag = results["no_poisson"]["by_lensed_mag"]
    p_lmag = results["poisson"]["by_lensed_mag"]
    for mb in sorted(set(np_lmag.keys()) & set(p_lmag.keys())):
        np_c = np_lmag[mb]["completeness"]
        p_c = p_lmag[mb]["completeness"]
        delta_pp = (p_c - np_c) * 100
        poisson_delta_by_lmag[mb] = {
            "no_poisson": np_c,
            "poisson": p_c,
            "delta_pp": delta_pp,
            "poisson_helps": delta_pp > 0,
        }

    return {
        "by_condition": results,
        "poisson_delta_by_theta_e": poisson_delta_by_te,
        "poisson_delta_by_lensed_mag": poisson_delta_by_lmag,
        "conditioned_subset_te1p5_mag20_23": conditioned_subset,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="D06 Poisson diagnostics: paired delta, conditioned grid, histograms"
    )
    ap.add_argument("--d06-dir", required=True,
                    help="Path to D06 results directory")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write diagnostic outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("D06 POISSON DIAGNOSTICS")
    print("=" * 70)

    # ---- 1. Paired delta analysis ----
    print("\n--- PAIRED DELTA ANALYSIS ---")
    delta = paired_delta_analysis(args.d06_dir)

    if "error" not in delta:
        print(f"\n  Total paired injections: {delta['n_total_paired']}")
        header = (f"{'Mag bin':<10} {'N':>5} {'Mean Δp':>10} {'Med Δp':>10} "
                  f"{'%pos':>6} {'Gained':>7} {'Lost':>6} {'Net':>5} {'Mechanism':>25}")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for mb in sorted(delta["results_by_bin"].keys()):
            s = delta["results_by_bin"][mb]
            m = delta["mechanism_assessment"][mb]
            print(f"  {mb:<10} {s['n_paired']:>5} {s['mean_delta']:>+10.4f} "
                  f"{s['median_delta']:>+10.4f} {s['pct_positive']:>5.1f}% "
                  f"{s['n_gained_above_03']:>7} {s['n_lost_below_03']:>6} "
                  f"{s['net_crossings']:>+5} {m:>25}")

        # Plot histograms
        print("\n--- SCORE HISTOGRAMS ---")
        plot_score_histograms(args.d06_dir, args.out_dir)
        plot_delta_histograms(args.d06_dir, args.out_dir)
    else:
        print(f"  SKIPPED: {delta['error']}")

    # ---- 2. Conditioned grid analysis ----
    print("\n--- CONDITIONED GRID ANALYSIS ---")
    grid_diag = conditioned_grid_analysis(args.d06_dir)

    # Print Poisson delta by theta_E
    print("\n  Poisson delta by theta_E (p>0.3):")
    print(f"  {'θ_E':>6} {'No-Poisson':>12} {'Poisson':>10} {'Δ (pp)':>10} {'Helps?':>8}")
    print(f"  {'-'*50}")
    for te in sorted(grid_diag["poisson_delta_by_theta_e"].keys()):
        v = grid_diag["poisson_delta_by_theta_e"][te]
        helps = "YES" if v["poisson_helps"] else "no"
        print(f"  {te:>6} {v['no_poisson']*100:>11.2f}% {v['poisson']*100:>9.2f}% "
              f"{v['delta_pp']:>+9.2f} {helps:>8}")

    # Print conditioned subset
    print("\n  Conditioned subset (θ_E=1.50, lensed mag 20-23):")
    for label in ["no_poisson", "poisson"]:
        cs = grid_diag["conditioned_subset_te1p5_mag20_23"][label]
        if "error" not in cs:
            print(f"    {label}: {cs['completeness']*100:.2f}% "
                  f"({cs['total_detected']}/{cs['total_injections']})")
            for bk, bv in cs["bins"].items():
                print(f"      {bk}: {bv['completeness']*100:.2f}% "
                      f"({bv['n_detected']}/{bv['n_injections']})")
        else:
            print(f"    {label}: {cs['error']}")

    # Print Poisson delta by lensed mag
    print("\n  Poisson delta by lensed magnitude (p>0.3):")
    print(f"  {'Lensed mag':>16} {'No-Poisson':>12} {'Poisson':>10} {'Δ (pp)':>10} {'Helps?':>8}")
    print(f"  {'-'*60}")
    for mb in sorted(grid_diag["poisson_delta_by_lensed_mag"].keys()):
        v = grid_diag["poisson_delta_by_lensed_mag"][mb]
        helps = "YES" if v["poisson_helps"] else "no"
        print(f"  {mb:>16} {v['no_poisson']*100:>11.2f}% {v['poisson']*100:>9.2f}% "
              f"{v['delta_pp']:>+9.2f} {helps:>8}")

    # ---- Save all diagnostics ----
    output = {
        "paired_delta": delta,
        "conditioned_grid": grid_diag,
    }

    out_path = os.path.join(args.out_dir, "d06_poisson_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved diagnostics to: {out_path}")


if __name__ == "__main__":
    main()
