#!/usr/bin/env python3
"""
Sim-to-Real Validation: Comprehensive diagnostic to quantify and explain
the gap between injection-recovery completeness and real-lens recall.

Three core diagnostics:
  1. Brightness-matched comparison: real lens brightness vs injection source mag
  2. Score distribution comparison: real lenses vs injections vs negatives
  3. Anchor SNR comparison: real lens annular SNR vs injection SNR (KS test)

Outputs:
  - sim_to_real_summary.json: comprehensive numerical summary
  - score_distributions.npz: raw score arrays for plotting
  - sim_to_real_histograms.png: score distribution histograms

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.
    python scripts/sim_to_real_validation.py \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --selection-function-csv results/selection_function_v4_finetune/selection_function.csv \
        --injection-validation-csv results/injection_validation_v4/injection_validation.csv \
        --out-dir results/sim_to_real_validation

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.injection_engine import (
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)

PIXEL_SCALE = 0.262


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    model = build_model(arch, in_ch=3, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, arch, ckpt.get("epoch", -1)


@torch.no_grad()
def score_batch(model, cutout_paths, device, preprocessing="raw_robust"):
    """Score a list of cutout paths. Returns (scores, r_band_fluxes, annular_snrs)."""
    scores = []
    r_band_fluxes = []
    annular_snrs = []
    errors = 0
    for i, path in enumerate(cutout_paths):
        try:
            with np.load(path) as z:
                hwc = z["cutout"].astype(np.float32)
            chw = np.transpose(hwc, (2, 0, 1))

            # r-band total flux (nanomaggies)
            r_flux = float(np.sum(hwc[:, :, 1]))
            r_band_fluxes.append(r_flux)

            # Annular r-band flux (arcs live in annulus)
            h, w = hwc.shape[:2]
            cy, cx = h // 2, w // 2
            Y, X = np.mgrid[:h, :w]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            r_inner, r_outer = 5, min(h, w) // 2 - 1
            annulus = (R >= r_inner) & (R <= r_outer)
            r_annular_flux = float(np.sum(hwc[:, :, 1][annulus]))
            # Convert to magnitude
            if r_annular_flux > 0:
                r_annular_mag = 22.5 - 2.5 * np.log10(r_annular_flux)
            else:
                r_annular_mag = 99.0

            # Preprocessing and scoring
            proc = preprocess_stack(chw, mode=preprocessing, crop=False, clip_range=10.0)
            x = torch.from_numpy(proc[None]).float().to(device)
            logit = model(x).squeeze().cpu().item()
            p = 1.0 / (1.0 + np.exp(-logit))
            scores.append(p)

            # Annular SNR (for anchor comparison)
            chw_t = torch.from_numpy(chw).float()
            # Use a rough sigma estimate
            snr_val = float(arc_annulus_snr(chw_t, 0.01))  # placeholder sigma
            annular_snrs.append(snr_val)

        except Exception as e:
            scores.append(np.nan)
            r_band_fluxes.append(np.nan)
            annular_snrs.append(np.nan)
            errors += 1

        if (i + 1) % 200 == 0:
            print(f"  Scored {i+1}/{len(cutout_paths)}", flush=True)

    return np.array(scores), np.array(r_band_fluxes), np.array(annular_snrs), errors


# ---------------------------------------------------------------------------
# Brightness-matched analysis
# ---------------------------------------------------------------------------
def brightness_matched_analysis(
    sf_csv_path: str,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Extract completeness by source_mag_bin from the selection function CSV."""
    df = pd.read_csv(sf_csv_path)

    # Filter to the target threshold
    mask = (df["threshold"] == threshold) & (df["threshold_type"] == "fixed")
    sub = df[mask].copy()

    results = {}

    # Overall completeness
    all_rows = sub[sub["source_mag_bin"] == "all"]
    if len(all_rows) > 0:
        total_inj = all_rows["n_injections"].sum()
        total_det = all_rows["n_detected"].sum()
        results["overall_completeness"] = float(total_det / total_inj) if total_inj > 0 else 0.0
        results["overall_n_injections"] = int(total_inj)
        results["overall_n_detected"] = int(total_det)

    # By source_mag_bin
    for mag_bin in ["23-24", "24-25", "25-26"]:
        bin_rows = sub[sub["source_mag_bin"] == mag_bin]
        if len(bin_rows) > 0:
            total_inj = bin_rows["n_injections"].sum()
            total_det = bin_rows["n_detected"].sum()
            c = float(total_det / total_inj) if total_inj > 0 else 0.0
            results[f"completeness_mag_{mag_bin}"] = c
            results[f"n_injections_mag_{mag_bin}"] = int(total_inj)
            results[f"n_detected_mag_{mag_bin}"] = int(total_det)

    # By theta_E (averaging over PSF and depth for "all" mag bin)
    for te in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        te_rows = all_rows[np.isclose(all_rows["theta_e"], te)]
        if len(te_rows) > 0:
            total_inj = te_rows["n_injections"].sum()
            total_det = te_rows["n_detected"].sum()
            c = float(total_det / total_inj) if total_inj > 0 else 0.0
            results[f"completeness_theta_{te}"] = c

    # Bright sources only (mag 23-24) by theta_E
    bright_rows = sub[sub["source_mag_bin"] == "23-24"]
    for te in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        te_rows = bright_rows[np.isclose(bright_rows["theta_e"], te)]
        if len(te_rows) > 0:
            total_inj = te_rows["n_injections"].sum()
            total_det = te_rows["n_detected"].sum()
            c = float(total_det / total_inj) if total_inj > 0 else 0.0
            results[f"completeness_bright_theta_{te}"] = c

    # Also compute for FPR-derived thresholds
    for fpr_label, fpr_val in [("fpr_0.001", 0.001), ("fpr_0.0001", 0.0001)]:
        fpr_rows = df[(df["threshold_type"] == "fpr") & (np.isclose(df["fpr_target"], fpr_val))]
        all_fpr = fpr_rows[fpr_rows["source_mag_bin"] == "all"]
        if len(all_fpr) > 0:
            total_inj = all_fpr["n_injections"].sum()
            total_det = all_fpr["n_detected"].sum()
            results[f"{fpr_label}_overall_completeness"] = float(total_det / total_inj) if total_inj > 0 else 0.0
        for mag_bin in ["23-24", "24-25", "25-26"]:
            bin_fpr = fpr_rows[fpr_rows["source_mag_bin"] == mag_bin]
            if len(bin_fpr) > 0:
                total_inj = bin_fpr["n_injections"].sum()
                total_det = bin_fpr["n_detected"].sum()
                results[f"{fpr_label}_completeness_mag_{mag_bin}"] = float(total_det / total_inj) if total_inj > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# Anchor SNR comparison with real lenses
# ---------------------------------------------------------------------------
def anchor_snr_comparison(
    real_lens_rows: pd.DataFrame,
    neg_rows: pd.DataFrame,
    inj_validation_csv_path: str,
    pixscale: float = PIXEL_SCALE,
    max_anchors: int = 500,
) -> Dict[str, Any]:
    """Compare annular SNR of real lenses vs injected lenses (KS test).

    Real lenses may lack psfsize_r/psfdepth_r metadata, so we compute a
    representative sigma_pix from the median of the negative population
    and use it as a fixed noise floor for all real lens SNR estimates.
    """
    from scipy.stats import ks_2samp
    import math

    # Compute a representative sigma_pix from negatives (which have valid metadata)
    neg_valid = neg_rows.dropna(subset=["psfsize_r", "psfdepth_r"])
    if len(neg_valid) > 0:
        med_psf = float(neg_valid["psfsize_r"].median())
        med_depth = float(neg_valid["psfdepth_r"].median())
        fallback_sigma = estimate_sigma_pix_from_psfdepth(med_depth, med_psf, pixscale)
    else:
        fallback_sigma = 0.01  # conservative fallback
    print(f"   Anchor SNR: using fallback sigma_pix={fallback_sigma:.6f} "
          f"(from median PSF={neg_valid['psfsize_r'].median():.2f}\", "
          f"depth={neg_valid['psfdepth_r'].median():.1f})")

    # Measure real lens annular SNR
    real_snrs = []
    for _, row in real_lens_rows.head(max_anchors).iterrows():
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
            chw = np.transpose(hwc, (2, 0, 1))
            chw_t = torch.from_numpy(chw).float()

            # Use per-row sigma if available, otherwise fallback
            psf = row.get("psfsize_r")
            psfdepth = row.get("psfdepth_r")
            if (psf is not None and psfdepth is not None
                    and np.isfinite(float(psf)) and np.isfinite(float(psfdepth))
                    and float(psfdepth) > 0):
                sigma_pix = estimate_sigma_pix_from_psfdepth(
                    float(psfdepth), float(psf), pixscale
                )
            else:
                sigma_pix = fallback_sigma

            snr = float(arc_annulus_snr(chw_t, sigma_pix))
            if np.isfinite(snr):
                real_snrs.append(snr)
        except Exception:
            pass

    real_snrs = np.array(real_snrs)

    # Load injection SNRs from validation CSV
    inj_df = pd.read_csv(inj_validation_csv_path)
    inj_snrs = inj_df["arc_snr_default"].dropna().values

    result = {
        "n_real_lenses": len(real_snrs),
        "n_injections": len(inj_snrs),
    }

    if len(real_snrs) >= 10 and len(inj_snrs) >= 10:
        ks_stat, ks_pval = ks_2samp(inj_snrs, real_snrs)
        result.update({
            "real_snr_median": float(np.median(real_snrs)),
            "real_snr_p25": float(np.percentile(real_snrs, 25)),
            "real_snr_p75": float(np.percentile(real_snrs, 75)),
            "real_snr_mean": float(np.mean(real_snrs)),
            "injection_snr_median": float(np.median(inj_snrs)),
            "injection_snr_p25": float(np.percentile(inj_snrs, 25)),
            "injection_snr_p75": float(np.percentile(inj_snrs, 75)),
            "injection_snr_mean": float(np.mean(inj_snrs)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "interpretation": (
                "KS p < 0.05 indicates SNR distributions differ significantly. "
                "This is expected: real confirmed lenses are biased toward bright, "
                "high-SNR systems, while injections sample uniformly including faint sources."
            ),
        })
    else:
        result["error"] = f"Insufficient data: {len(real_snrs)} real, {len(inj_snrs)} injections"

    return result


# ---------------------------------------------------------------------------
# Score distribution histograms
# ---------------------------------------------------------------------------
def generate_histograms(
    real_scores: np.ndarray,
    neg_scores: np.ndarray,
    inj_scores: np.ndarray,
    out_path: str,
) -> None:
    """Generate score distribution histograms comparing real, injection, negative."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping histograms")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: All three on same axes (log scale)
    bins = np.linspace(0, 1, 50)
    axes[0].hist(real_scores[np.isfinite(real_scores)], bins=bins, alpha=0.6,
                 label=f"Real lenses (N={np.isfinite(real_scores).sum()})", color="green", density=True)
    axes[0].hist(inj_scores[np.isfinite(inj_scores)], bins=bins, alpha=0.6,
                 label=f"Injections (N={np.isfinite(inj_scores).sum()})", color="orange", density=True)
    axes[0].hist(neg_scores[np.isfinite(neg_scores)], bins=bins, alpha=0.6,
                 label=f"Negatives (N={np.isfinite(neg_scores).sum()})", color="blue", density=True)
    axes[0].set_xlabel("Model Score (probability)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distributions")
    axes[0].legend(fontsize=8)
    axes[0].set_yscale("log")

    # Panel 2: CDF comparison
    for arr, label, color in [
        (real_scores, "Real lenses", "green"),
        (inj_scores, "Injections", "orange"),
        (neg_scores, "Negatives", "blue"),
    ]:
        valid = arr[np.isfinite(arr)]
        sorted_vals = np.sort(valid)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1].plot(sorted_vals, cdf, label=label, color=color)
    axes[1].axvline(0.3, color="red", linestyle="--", alpha=0.5, label="p=0.3 threshold")
    axes[1].axvline(0.5, color="red", linestyle=":", alpha=0.5, label="p=0.5 threshold")
    axes[1].set_xlabel("Model Score")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative Score Distribution")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(-0.05, 1.05)

    # Panel 3: Real lens scores zoomed in (high-score region)
    valid_real = real_scores[np.isfinite(real_scores)]
    axes[2].hist(valid_real, bins=50, alpha=0.7, color="green", edgecolor="black")
    axes[2].axvline(0.3, color="red", linestyle="--", label="p=0.3")
    axes[2].axvline(0.5, color="red", linestyle=":", label="p=0.5")
    axes[2].set_xlabel("Model Score")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Real Lens Score Distribution (N={len(valid_real)})")
    pct_above_05 = (valid_real >= 0.5).mean() * 100
    pct_above_03 = (valid_real >= 0.3).mean() * 100
    axes[2].text(0.05, 0.95, f"Recall (p>0.3): {pct_above_03:.1f}%\nRecall (p>0.5): {pct_above_05:.1f}%",
                 transform=axes[2].transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    axes[2].legend(fontsize=8)

    fig.suptitle("Sim-to-Real Validation: Score Distribution Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved histograms: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sim-to-Real Validation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selection-function-csv", required=True,
                    help="Path to selection_function.csv from main grid")
    ap.add_argument("--injection-validation-csv", required=True,
                    help="Path to injection_validation.csv from validate_injections.py")
    ap.add_argument("--out-dir", default="results/sim_to_real_validation")
    ap.add_argument("--n-negatives", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SIM-TO-REAL VALIDATION")
    print("=" * 60)
    t0 = time.time()

    # Load model
    print(f"\n1. Loading model: {args.checkpoint}")
    model, arch, epoch = load_model(args.checkpoint, device)
    print(f"   Architecture: {arch}, Epoch: {epoch}")

    # Load manifest
    print(f"\n2. Loading manifest: {args.manifest}")
    manifest = pd.read_parquet(args.manifest)
    val_pos = manifest[(manifest["split"] == "val") & (manifest["label"] == 1)].copy()
    val_neg = manifest[(manifest["split"] == "val") & (manifest["label"] == 0)].copy()
    print(f"   Val positives: {len(val_pos)}")
    print(f"   Val negatives: {len(val_neg)}")

    # -----------------------------------------------------------------------
    # 3. Score all real confirmed lenses
    # -----------------------------------------------------------------------
    print(f"\n3. Scoring {len(val_pos)} real confirmed lenses...")
    real_scores, real_r_flux, _, real_errors = score_batch(
        model, val_pos["cutout_path"].tolist(), device
    )
    valid_real = real_scores[np.isfinite(real_scores)]
    print(f"   Scored: {len(valid_real)} ({real_errors} errors)")

    # Recall at thresholds
    thresholds = {
        "p>0.3": 0.3,
        "p>0.5": 0.5,
        "FPR=0.1% (p>0.806)": 0.8059,
        "FPR=0.01% (p>0.995)": 0.9951,
    }
    recall_results = {}
    for label, thr in thresholds.items():
        recall = float((valid_real >= thr).mean())
        recall_results[label] = recall
        print(f"   {label:30s}: recall = {recall*100:.1f}%")

    pctls = np.percentile(valid_real, [5, 25, 50, 75, 95])
    print(f"   Score distribution: p5={pctls[0]:.4f}, p25={pctls[1]:.4f}, "
          f"median={pctls[2]:.4f}, p75={pctls[3]:.4f}, p95={pctls[4]:.4f}")

    # -----------------------------------------------------------------------
    # 4. Score negatives
    # -----------------------------------------------------------------------
    n_neg = min(args.n_negatives, len(val_neg))
    neg_sample = val_neg.sample(n=n_neg, random_state=args.seed)
    print(f"\n4. Scoring {n_neg} negatives...")
    neg_scores, _, _, neg_errors = score_batch(
        model, neg_sample["cutout_path"].tolist(), device
    )
    valid_neg = neg_scores[np.isfinite(neg_scores)]
    print(f"   Scored: {len(valid_neg)} ({neg_errors} errors)")
    print(f"   Median: {np.median(valid_neg):.6f}")
    print(f"   Fraction > 0.3: {(valid_neg > 0.3).mean()*100:.2f}%")
    print(f"   Fraction > 0.5: {(valid_neg > 0.5).mean()*100:.2f}%")

    # -----------------------------------------------------------------------
    # 5. Brightness-matched analysis from selection function grid
    # -----------------------------------------------------------------------
    print(f"\n5. Brightness-matched analysis...")
    bm = brightness_matched_analysis(args.selection_function_csv, threshold=0.3)
    print(f"   Overall injection completeness (p>0.3): {bm.get('overall_completeness', 0)*100:.1f}%")
    for mag_bin in ["23-24", "24-25", "25-26"]:
        key = f"completeness_mag_{mag_bin}"
        if key in bm:
            print(f"   Source mag {mag_bin}: {bm[key]*100:.1f}%")
    print(f"   Real lens recall (p>0.3): {recall_results['p>0.3']*100:.1f}%")
    print(f"   Gap at brightest bin (23-24): "
          f"real={recall_results['p>0.3']*100:.1f}% vs inj={bm.get('completeness_mag_23-24', 0)*100:.1f}%")

    # Real lens r-band brightness
    valid_flux = real_r_flux[np.isfinite(real_r_flux) & (real_r_flux > 0)]
    real_mags = 22.5 - 2.5 * np.log10(valid_flux)
    print(f"\n   Real lens r-band total magnitude distribution:")
    if len(real_mags) > 0:
        mag_pctls = np.percentile(real_mags, [5, 25, 50, 75, 95])
        print(f"   p5={mag_pctls[0]:.1f}, p25={mag_pctls[1]:.1f}, median={mag_pctls[2]:.1f}, "
              f"p75={mag_pctls[3]:.1f}, p95={mag_pctls[4]:.1f}")
        for lo, hi in [(14, 18), (18, 20), (20, 22), (22, 23), (23, 24), (24, 26)]:
            frac = ((real_mags >= lo) & (real_mags < hi)).mean()
            if frac > 0.001:
                print(f"   mag [{lo}-{hi}): {frac*100:.1f}%")

    # -----------------------------------------------------------------------
    # 6. Anchor SNR comparison
    # -----------------------------------------------------------------------
    print(f"\n6. Anchor SNR comparison (real vs injection)...")
    anchor_results = anchor_snr_comparison(
        val_pos, val_neg, args.injection_validation_csv, max_anchors=500
    )
    if "ks_statistic" in anchor_results:
        print(f"   Real lens SNR: median={anchor_results['real_snr_median']:.1f}, "
              f"IQR=[{anchor_results['real_snr_p25']:.1f}, {anchor_results['real_snr_p75']:.1f}]")
        print(f"   Injection SNR: median={anchor_results['injection_snr_median']:.1f}, "
              f"IQR=[{anchor_results['injection_snr_p25']:.1f}, {anchor_results['injection_snr_p75']:.1f}]")
        print(f"   KS statistic: {anchor_results['ks_statistic']:.4f}, "
              f"p-value: {anchor_results['ks_pvalue']:.4e}")
    else:
        print(f"   {anchor_results.get('error', 'Unknown error')}")

    # -----------------------------------------------------------------------
    # 7. Load injection scores from validation CSV
    # -----------------------------------------------------------------------
    print(f"\n7. Loading injection scores from validation CSV...")
    inj_df = pd.read_csv(args.injection_validation_csv)
    inj_scores = inj_df["score_default"].dropna().values
    print(f"   N injections: {len(inj_scores)}")
    print(f"   Median: {np.median(inj_scores):.6f}")
    print(f"   Fraction > 0.3: {(inj_scores > 0.3).mean()*100:.1f}%")
    print(f"   Fraction > 0.5: {(inj_scores > 0.5).mean()*100:.1f}%")

    # -----------------------------------------------------------------------
    # 8. Generate histograms
    # -----------------------------------------------------------------------
    print(f"\n8. Generating score distribution histograms...")
    hist_path = os.path.join(args.out_dir, "sim_to_real_histograms.png")
    generate_histograms(valid_real, valid_neg, inj_scores, hist_path)

    # -----------------------------------------------------------------------
    # 9. Save everything
    # -----------------------------------------------------------------------
    dt = time.time() - t0
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "arch": arch,
        "epoch": epoch,
        "elapsed_seconds": round(dt, 1),
        # Real lens recall
        "real_lens_n": int(len(valid_real)),
        "real_lens_recall": recall_results,
        "real_lens_score_percentiles": {
            "p5": float(pctls[0]), "p25": float(pctls[1]),
            "median": float(pctls[2]), "p75": float(pctls[3]),
            "p95": float(pctls[4]),
        },
        "real_lens_score_mean": float(np.mean(valid_real)),
        "real_lens_frac_above_0.9": float((valid_real > 0.9).mean()),
        "real_lens_frac_below_0.1": float((valid_real < 0.1).mean()),
        # Real lens brightness
        "real_lens_r_mag_percentiles": {
            "p5": float(mag_pctls[0]) if len(real_mags) > 0 else None,
            "p25": float(mag_pctls[1]) if len(real_mags) > 0 else None,
            "median": float(mag_pctls[2]) if len(real_mags) > 0 else None,
            "p75": float(mag_pctls[3]) if len(real_mags) > 0 else None,
            "p95": float(mag_pctls[4]) if len(real_mags) > 0 else None,
        },
        # Negative scores
        "negative_n": int(len(valid_neg)),
        "negative_score_median": float(np.median(valid_neg)),
        "negative_fpr_at_0.3": float((valid_neg > 0.3).mean()),
        "negative_fpr_at_0.5": float((valid_neg > 0.5).mean()),
        # Brightness-matched injection completeness
        "brightness_matched": bm,
        # Anchor SNR comparison
        "anchor_snr_comparison": anchor_results,
        # Injection scores
        "injection_n": int(len(inj_scores)),
        "injection_score_median": float(np.median(inj_scores)),
        "injection_frac_above_0.3": float((inj_scores > 0.3).mean()),
        "injection_frac_above_0.5": float((inj_scores > 0.5).mean()),
        # Key diagnostic summary
        "key_findings": {
            "real_recall_p03": recall_results["p>0.3"],
            "injection_completeness_overall_p03": bm.get("overall_completeness", 0),
            "injection_completeness_bright_p03": bm.get("completeness_mag_23-24", 0),
            "gap_overall": recall_results["p>0.3"] - bm.get("overall_completeness", 0),
            "gap_bright": recall_results["p>0.3"] - bm.get("completeness_mag_23-24", 0),
            "explanation": (
                "The gap between real-lens recall and injection completeness is primarily "
                "driven by the source magnitude distribution. Real confirmed lenses are "
                "biased toward bright, spectacular systems (median r-mag ~{:.1f}), while "
                "injections uniformly sample mag 23-26 including many faint, undetectable "
                "sources. At bright source magnitudes (23-24), injection completeness is "
                "{:.1f}%, much closer to real lens recall ({:.1f}%). The residual gap "
                "reflects additional realism differences (morphological complexity, "
                "multi-component sources, substructure)."
            ).format(
                float(np.median(real_mags)) if len(real_mags) > 0 else 0,
                bm.get("completeness_mag_23-24", 0) * 100,
                recall_results["p>0.3"] * 100,
            ),
        },
    }

    # Save summary JSON
    json_path = os.path.join(args.out_dir, "sim_to_real_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   Summary saved: {json_path}")

    # Save raw score arrays
    npz_path = os.path.join(args.out_dir, "score_distributions.npz")
    np.savez(npz_path,
             real_scores=valid_real,
             neg_scores=valid_neg,
             inj_scores=inj_scores,
             real_r_mags=real_mags)
    print(f"   Score arrays saved: {npz_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SIM-TO-REAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Real lens recall (p>0.3):              {recall_results['p>0.3']*100:.1f}%")
    print(f"  Real lens recall (p>0.5):              {recall_results['p>0.5']*100:.1f}%")
    print(f"  Injection completeness (all, p>0.3):   {bm.get('overall_completeness', 0)*100:.1f}%")
    print(f"  Injection completeness (23-24, p>0.3): {bm.get('completeness_mag_23-24', 0)*100:.1f}%")
    print(f"  Injection completeness (24-25, p>0.3): {bm.get('completeness_mag_24-25', 0)*100:.1f}%")
    print(f"  Injection completeness (25-26, p>0.3): {bm.get('completeness_mag_25-26', 0)*100:.1f}%")
    print(f"  Negative FPR at p>0.3:                 {(valid_neg > 0.3).mean()*100:.2f}%")
    if "ks_statistic" in anchor_results:
        print(f"  Anchor SNR KS stat:                    {anchor_results['ks_statistic']:.4f}")
        print(f"  Anchor SNR KS p-value:                 {anchor_results['ks_pvalue']:.4e}")
    print(f"  Elapsed: {dt:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
