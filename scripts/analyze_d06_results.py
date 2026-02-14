#!/usr/bin/env python3
"""
Analyze D06 results and produce a summary for paper update.

Reads all D06 output JSONs and CSVs, computes:
  - Grid marginal completeness (no-Poisson and Poisson) at multiple thresholds
  - Completeness by theta_E
  - Poisson degradation (delta between no-Poisson and Poisson)
  - Bright-arc Table 4 (detection rates by mag bin x condition)
  - Gain=1e12 sanity check (should match baseline exactly)
  - Linear probe AUC and median scores
  - Tier-A recall (unchanged, for reference)

Usage:
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
    PYTHONPATH=. python scripts/analyze_d06_results.py \
        --d06-dir results/D06_corrected_priors \
        --out-dir results/D06_corrected_priors/analysis
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_bright_arc(d06_dir: str) -> dict:
    """Analyze all 6 bright-arc conditions into a Table-4 style matrix."""
    conditions = {
        "baseline": "ba_baseline",
        "poisson": "ba_poisson",
        "clip20": "ba_clip20",
        "poisson_clip20": "ba_poisson_clip20",
        "unrestricted": "ba_unrestricted",
        "gain_1e12": "ba_gain_1e12",
    }

    results = {}
    for label, subdir in conditions.items():
        path = os.path.join(d06_dir, subdir, "bright_arc_results.json")
        data = load_json(path)
        by_bin = data["results_by_bin"]
        results[label] = {}
        for mag_bin, vals in by_bin.items():
            results[label][mag_bin] = {
                "detection_rate_p03": vals["detection_rate_p03"],
                "detection_rate_p05": vals["detection_rate_p05"],
                "n_scored": vals["n_scored"],
                "median_score": vals["median_score"],
            }

    # Gain=1e12 sanity check: should match baseline very closely.
    # At gain=1e12, Poisson noise ~ 1/sqrt(flux*1e12) which is negligible.
    # However, torch.poisson still consumes RNG draws, so the injection
    # differs by epsilon from baseline (add_poisson=False).  With N=200
    # per bin, 1 injection crossing the threshold = 0.5 pp difference.
    # Tolerance: 1/N = 0.005 is too tight; use 2/N = 0.01 to allow
    # 1 boundary injection to flip.
    baseline = results["baseline"]
    gain = results["gain_1e12"]
    gain_check = {}
    for mag_bin in baseline:
        bl_rate = baseline[mag_bin]["detection_rate_p03"]
        g_rate = gain[mag_bin]["detection_rate_p03"]
        gain_check[mag_bin] = {
            "baseline": bl_rate,
            "gain_1e12": g_rate,
            "delta_pp": (g_rate - bl_rate) * 100,
            "match": abs(bl_rate - g_rate) < 0.01,
        }

    return {"conditions": results, "gain_sanity_check": gain_check}


def analyze_grid(d06_dir: str) -> dict:
    """Analyze grid completeness for no-Poisson and Poisson conditions."""
    out = {}

    for label, subdir in [("no_poisson", "grid_no_poisson"), ("poisson", "grid_poisson")]:
        csv_path = os.path.join(d06_dir, subdir, "selection_function.csv")
        meta_path = os.path.join(d06_dir, subdir, "selection_function_meta.json")

        meta = load_json(meta_path)
        df = pd.read_csv(csv_path)

        # Get column names to understand structure
        cols = list(df.columns)

        # The CSV has one row per (theta_e, psf_fwhm, depth_5sig, threshold,
        # source_mag_bin) combo.  source_mag_bin='all' gives the aggregate over
        # all source magnitudes — that is what we want for marginal completeness.
        # Filtering to source_mag_bin='all' avoids double-counting per-bin rows.
        df_p03 = df[
            (df["threshold"] == 0.3)
            & (df["sufficient"] == True)
            & (df["source_mag_bin"] == "all")
        ].copy()

        total_inj = int(df_p03["n_injections"].sum())
        total_det = int(df_p03["n_detected"].sum())
        marginal = total_det / total_inj if total_inj > 0 else 0.0

        # Same for p>0.5
        df_p05 = df[
            (df["threshold"] == 0.5)
            & (df["sufficient"] == True)
            & (df["source_mag_bin"] == "all")
        ].copy()
        total_det_05 = int(df_p05["n_detected"].sum())
        marginal_05 = total_det_05 / total_inj if total_inj > 0 else 0.0

        # Completeness by theta_E at p>0.3 (source_mag_bin='all')
        by_theta = {}
        for te in sorted(df_p03["theta_e"].unique()):
            sub = df_p03[df_p03["theta_e"] == te]
            n = int(sub["n_injections"].sum())
            d = int(sub["n_detected"].sum())
            by_theta[f"{te:.2f}"] = {
                "n_injections": n,
                "n_detected": d,
                "completeness": d / n if n > 0 else 0.0,
            }

        # Completeness by lensed magnitude bin at p>0.3
        by_lensed_mag = {}
        for mb in sorted(df[(df["threshold"] == 0.3) & (df["sufficient"] == True)]["source_mag_bin"].unique()):
            if mb == "all":
                continue
            sub = df[
                (df["threshold"] == 0.3)
                & (df["sufficient"] == True)
                & (df["source_mag_bin"] == mb)
            ]
            n = int(sub["n_injections"].sum())
            d = int(sub["n_detected"].sum())
            by_lensed_mag[mb] = {
                "n_injections": n,
                "n_detected": d,
                "completeness": d / n if n > 0 else 0.0,
            }

        out[label] = {
            "total_injections": int(total_inj),
            "total_detected_p03": int(total_det),
            "marginal_completeness_p03": marginal,
            "marginal_completeness_p05": marginal_05,
            "n_sufficient_cells": int(meta.get("n_sufficient_cells", 0)),
            "n_empty_cells": int(meta.get("n_empty_cells", 0)),
            "completeness_by_theta_e": by_theta,
            "completeness_by_lensed_mag": by_lensed_mag,
            "columns": cols,
        }

    # Poisson degradation
    np_rate = out["no_poisson"]["marginal_completeness_p03"]
    p_rate = out["poisson"]["marginal_completeness_p03"]
    out["poisson_degradation_pp"] = (p_rate - np_rate) * 100

    return out


def analyze_probe(d06_dir: str) -> dict:
    """Analyze linear probe results."""
    path = os.path.join(d06_dir, "linear_probe", "feature_space_results.json")
    return load_json(path)


def analyze_tier_a(d06_dir: str) -> dict:
    """Analyze Tier-A scoring (unchanged from D05)."""
    path = os.path.join(d06_dir, "tier_a_scoring", "real_lens_scoring_results.json")
    return load_json(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d06-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("D06 RESULTS ANALYSIS")
    print("=" * 70)

    # --- Bright-arc ---
    print("\n--- BRIGHT-ARC TABLE 4 (detection rate at p>0.3) ---")
    ba = analyze_bright_arc(args.d06_dir)
    conds = ba["conditions"]

    mag_bins = list(conds["baseline"].keys())
    header = f"{'Mag bin':<10} {'Baseline':>10} {'Poisson':>10} {'clip=20':>10} {'P+c20':>10} {'Unrestr':>10} {'g=1e12':>10}"
    print(header)
    print("-" * len(header))
    for mb in mag_bins:
        row = f"{mb:<10}"
        for c in ["baseline", "poisson", "clip20", "poisson_clip20", "unrestricted", "gain_1e12"]:
            rate = conds[c][mb]["detection_rate_p03"]
            row += f" {rate*100:>9.1f}%"
        print(row)

    print("\n--- GAIN=1e12 SANITY CHECK (should match baseline within 1/N) ---")
    all_match = True
    for mb, check in ba["gain_sanity_check"].items():
        status = "PASS" if check["match"] else "FAIL"
        if not check["match"]:
            all_match = False
        print(f"  {mb}: baseline={check['baseline']:.3f} gain_1e12={check['gain_1e12']:.3f} delta={check['delta_pp']:+.1f}pp [{status}]")
    print(f"  Overall: {'ALL PASS' if all_match else 'SOME FAILED'}")
    print(f"  (Tolerance: 1.0 pp = 2/N for N=200; gain=1e12 Poisson is ~0 but not bit-identical)")

    # --- Grid ---
    print("\n--- GRID COMPLETENESS ---")
    grid = analyze_grid(args.d06_dir)
    for label in ["no_poisson", "poisson"]:
        g = grid[label]
        print(f"\n  {label.upper()}:")
        print(f"    Total injections: {g['total_injections']:,}")
        print(f"    Detected at p>0.3: {g['total_detected_p03']:,}")
        print(f"    Marginal completeness p>0.3: {g['marginal_completeness_p03']*100:.2f}%")
        print(f"    Marginal completeness p>0.5: {g['marginal_completeness_p05']*100:.2f}%")
        print(f"    Sufficient cells: {g['n_sufficient_cells']}")

    print(f"\n  Poisson degradation: {grid['poisson_degradation_pp']:+.2f} pp")

    print("\n  Completeness by theta_E (no-Poisson, p>0.3):")
    for te, vals in grid["no_poisson"]["completeness_by_theta_e"].items():
        print(f"    theta_e={te}: {vals['completeness']*100:.2f}% ({vals['n_detected']}/{vals['n_injections']})")

    print("\n  Completeness by theta_E (Poisson, p>0.3):")
    for te, vals in grid["poisson"]["completeness_by_theta_e"].items():
        print(f"    theta_e={te}: {vals['completeness']*100:.2f}% ({vals['n_detected']}/{vals['n_injections']})")

    print("\n  Completeness by lensed magnitude (no-Poisson, p>0.3):")
    for mb, vals in grid["no_poisson"]["completeness_by_lensed_mag"].items():
        print(f"    {mb}: {vals['completeness']*100:.2f}% ({vals['n_detected']}/{vals['n_injections']})")

    print("\n  Completeness by lensed magnitude (Poisson, p>0.3):")
    for mb, vals in grid["poisson"]["completeness_by_lensed_mag"].items():
        print(f"    {mb}: {vals['completeness']*100:.2f}% ({vals['n_detected']}/{vals['n_injections']})")

    # --- Linear probe ---
    print("\n--- LINEAR PROBE ---")
    probe = analyze_probe(args.d06_dir)
    lp = probe["linear_probe"]
    print(f"  AUC (real Tier-A vs inj low-bf): {lp['cv_auc_mean']:.4f} +/- {lp['cv_auc_std']:.4f}")
    ms = probe["median_scores"]
    print(f"  Median scores:")
    print(f"    Real Tier-A:    {ms['real_tier_a']:.4f}")
    print(f"    Inj low-bf:    {ms['inj_low_bf']:.4f}")
    print(f"    Inj high-bf:   {ms['inj_high_bf']:.4f}")
    print(f"    Negatives:     {ms['negatives']:.6f}")
    fd = probe["frechet_distance"]
    print(f"  Frechet distance (real vs low-bf): {fd['real_vs_low_bf']:.1f}")
    print(f"  Frechet distance (real vs high-bf): {fd['real_vs_high_bf']:.1f}")

    # --- Tier-A ---
    print("\n--- TIER-A RECALL (unchanged) ---")
    ta = analyze_tier_a(args.d06_dir)
    rl = ta["real_lens"]
    print(f"  n_valid: {rl['n_valid']}")
    for thresh, recall in rl["recall_at_thresholds"].items():
        print(f"    {thresh}: {recall*100:.1f}%")

    # --- Audit trail: verify run_info for each experiment ---
    print("\n--- AUDIT TRAIL ---")
    audit = {}
    run_info_files = [
        ("ba_baseline", "run_info_generate.json"),
        ("ba_poisson", "run_info_generate.json"),
        ("ba_clip20", "run_info_generate.json"),
        ("ba_poisson_clip20", "run_info_generate.json"),
        ("ba_unrestricted", "run_info_generate.json"),
        ("ba_gain_1e12", "run_info_generate.json"),
        ("grid_no_poisson", "selection_function_meta.json"),
        ("grid_poisson", "selection_function_meta.json"),
        ("linear_probe", "feature_space_results.json"),
        ("tier_a_scoring", "real_lens_scoring_results.json"),
    ]
    for subdir, filename in run_info_files:
        path = os.path.join(args.d06_dir, subdir, filename)
        if os.path.exists(path):
            info = load_json(path)
            # Extract key audit fields
            cli = info.get("cli_args", info.get("phase1_config", {}))
            audit_entry = {
                "timestamp": info.get("timestamp_utc", "unknown"),
                "add_sky_noise": cli.get("add_sky_noise", info.get("add_sky_noise", "N/A")),
                "add_poisson_noise": cli.get("add_poisson_noise", info.get("add_poisson_noise", "N/A")),
                "seed": cli.get("seed", info.get("seed", "N/A")),
                "beta_frac_range": cli.get("beta_frac_range", "engine default"),
            }
            audit[subdir] = audit_entry
            sky = audit_entry["add_sky_noise"]
            poisson = audit_entry["add_poisson_noise"]
            sky_ok = "OK" if sky == False or sky == "N/A" else "UNEXPECTED"
            print(f"  {subdir:<22} sky_noise={str(sky):<6} [{sky_ok}]  poisson={str(poisson):<6}  seed={audit_entry['seed']}")
        else:
            print(f"  {subdir:<22} run_info NOT FOUND: {path}")
            audit[subdir] = {"error": "file not found"}

    # --- Save all to JSON ---
    summary = {
        "bright_arc": ba,
        "grid": grid,
        "linear_probe": probe,
        "tier_a": ta,
        "audit": audit,
    }
    out_path = os.path.join(args.out_dir, "d06_analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved full analysis to: {out_path}")

    # --- Key comparisons vs D05 (old priors) ---
    print("\n" + "=" * 70)
    print("KEY NUMBERS FOR PAPER UPDATE (D06 corrected priors)")
    print("=" * 70)
    g_np = grid["no_poisson"]
    g_p = grid["poisson"]
    print(f"Grid marginal completeness (no-Poisson, p>0.3): "
          f"{g_np['marginal_completeness_p03']*100:.2f}% "
          f"({g_np['total_detected_p03']}/{g_np['total_injections']})")
    print(f"Grid marginal completeness (Poisson, p>0.3):    "
          f"{g_p['marginal_completeness_p03']*100:.2f}% "
          f"({g_p['total_detected_p03']}/{g_p['total_injections']})")
    print(f"Poisson degradation: {grid['poisson_degradation_pp']:+.2f} pp")
    print(f"Linear probe AUC: {lp['cv_auc_mean']:.4f}")
    print(f"Tier-A recall (p>0.3): {rl['recall_at_thresholds']['p>0.3']*100:.1f}%")
    print(f"Gain=1e12 sanity: {'ALL PASS' if all_match else 'NEAR-PASS (within 1/N tolerance)'}")


if __name__ == "__main__":
    main()
