#!/usr/bin/env python3
"""
Arc Morphology Statistics: Pixel-level diagnostics comparing real vs injected arcs.

SCIENTIFIC MOTIVATION
============================================
Sersic injections differ from real arcs in specific, measurable
pixel-level properties. This script computes:

1. **High-frequency power**: Radial average of 2D power spectrum in the arc
   annulus region. Real arcs (clumpy star-forming galaxies) have higher high-k
   power than smooth Sersic injections.

2. **Anisotropy / elongation**: Distribution of structure tensor eigenvalue
   ratios in an annulus near theta_E. Real arcs are tangentially elongated;
   low-magnification injections are barely stretched.

3. **Color-gradient coherence**: Real arcs have spatially varying colors
   (blue knots); Sersic injections have uniform color across the arc.
   Measured as the correlation between per-band spatial structure.

4. **Local variance ratio**: In an arc annulus, the ratio of local variance
   to expected Poisson + sky noise. Real arcs have variance from clumpy
   structure; smooth injections have only sky noise.

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python scripts/arc_morphology_statistics.py \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/arc_morphology \\
        --n-samples 200

Date: 2026-02-13
References:
  - anisotropy, high-freq power, wavelet kurtosis, color coherence
  - surface brightness profile, noise properties, color gradients
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PIXEL_SCALE = 0.262
TIER_COL = "tier"


def arc_annulus_mask(
    H: int, W: int,
    r_inner_pix: float = 4.0,
    r_outer_pix: float = 16.0,
) -> np.ndarray:
    """Create a boolean mask for the arc annulus region."""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = np.arange(H) - cy
    xx = np.arange(W) - cx
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rr = np.sqrt(X * X + Y * Y)
    return (rr >= r_inner_pix) & (rr <= r_outer_pix)


def high_freq_power(img_2d: np.ndarray, mask: np.ndarray, k_lo: float = 0.2) -> float:
    """Fraction of 2D power spectrum above spatial frequency k_lo (cycles/pixel).

    Higher values = more high-frequency structure (clumps, knots).
    """
    masked = np.where(mask, img_2d - np.median(img_2d[mask]), 0.0)
    F = np.fft.fft2(masked)
    P = np.abs(F) ** 2
    H, W = img_2d.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    KY, KX = np.meshgrid(fy, fx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    total = P.sum()
    if total <= 0:
        return float("nan")
    hi_frac = float(P[K >= k_lo].sum() / total)
    return hi_frac


def structure_tensor_anisotropy(img_2d: np.ndarray, mask: np.ndarray) -> float:
    """Mean anisotropy from the structure tensor in the arc annulus.

    Anisotropy = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    where lambda are eigenvalues of the local structure tensor.
    High anisotropy = elongated features (arcs). Low = isotropic (blobs).
    """
    # Sobel-like gradients
    gy = np.gradient(img_2d, axis=0)
    gx = np.gradient(img_2d, axis=1)

    # Structure tensor components (smooth locally with 3x3 mean)
    from scipy.ndimage import uniform_filter
    Jxx = uniform_filter(gx * gx, size=3)
    Jyy = uniform_filter(gy * gy, size=3)
    Jxy = uniform_filter(gx * gy, size=3)

    # Eigenvalues at each pixel
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy**2
    disc = np.sqrt(np.maximum(trace**2 - 4 * det, 0.0))
    lam1 = (trace + disc) / 2.0
    lam2 = (trace - disc) / 2.0

    denom = lam1 + lam2
    aniso = np.where(denom > 1e-10, (lam1 - lam2) / denom, 0.0)

    vals = aniso[mask]
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def color_gradient_coherence(
    img_g: np.ndarray, img_r: np.ndarray, img_z: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Cross-band spatial correlation in the arc annulus.

    If the arc has spatially varying colors (real arcs: blue knots, red
    regions), per-band structures differ. If uniform color (Sersic), all
    bands are perfectly correlated.

    Returns mean pairwise correlation coefficient (0 = uncorrelated, 1 = identical).
    """
    g_vals = img_g[mask] - np.mean(img_g[mask])
    r_vals = img_r[mask] - np.mean(img_r[mask])
    z_vals = img_z[mask] - np.mean(img_z[mask])

    def corr(a, b):
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom < 1e-10:
            return float("nan")
        return float(np.sum(a * b) / denom)

    c_gr = corr(g_vals, r_vals)
    c_rz = corr(r_vals, z_vals)
    c_gz = corr(g_vals, z_vals)
    return float(np.nanmean([c_gr, c_rz, c_gz]))


def local_variance_ratio(img_2d: np.ndarray, mask: np.ndarray, patch_size: int = 5) -> float:
    """Ratio of local variance in arc region to global sky variance.

    Real arcs with clumpy structure have higher local variance (from H II
    regions, dust lanes). Smooth Sersic injections have only sky noise.
    """
    from scipy.ndimage import uniform_filter

    # Local variance = E[X^2] - E[X]^2
    local_mean = uniform_filter(img_2d, size=patch_size)
    local_sq = uniform_filter(img_2d**2, size=patch_size)
    local_var = local_sq - local_mean**2

    arc_var = local_var[mask]
    arc_var = arc_var[np.isfinite(arc_var)]

    # Sky variance from outer ring
    H, W = img_2d.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = np.arange(H) - cy
    xx = np.arange(W) - cx
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rr = np.sqrt(X * X + Y * Y)
    sky_mask = (rr >= 35) & (rr <= 48)
    sky_var = local_var[sky_mask]
    sky_var = sky_var[np.isfinite(sky_var)]

    if len(arc_var) < 5 or len(sky_var) < 5:
        return float("nan")

    med_arc = float(np.median(arc_var))
    med_sky = float(np.median(sky_var))
    if med_sky <= 0:
        return float("nan")
    return med_arc / med_sky


def compute_stats_for_cutout(hwc: np.ndarray) -> Dict[str, float]:
    """Compute all morphology statistics for a single cutout (H,W,3)."""
    chw = np.transpose(hwc, (2, 0, 1)).astype(np.float64)
    H, W = chw.shape[1], chw.shape[2]
    mask = arc_annulus_mask(H, W, r_inner_pix=4.0, r_outer_pix=16.0)

    r_band = chw[1]  # r-band

    return {
        "high_freq_power_r": high_freq_power(r_band, mask),
        "anisotropy_r": structure_tensor_anisotropy(r_band, mask),
        "color_coherence": color_gradient_coherence(chw[0], chw[1], chw[2], mask),
        "local_var_ratio_r": local_variance_ratio(r_band, mask),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Arc morphology pixel statistics: real vs injected",
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theta-e", type=float, default=1.5)
    ap.add_argument("--target-mag", type=float, default=19.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load manifest
    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    val_df = df[df["split"] == "val"].copy()

    # Real Tier-A lenses
    print("\n--- Real Tier-A lenses ---")
    if TIER_COL in val_df.columns:
        tier_a = val_df[(val_df["label"] == 1) & (val_df[TIER_COL] == "A")]
    else:
        tier_a = val_df[val_df["label"] == 1]

    n_a = min(args.n_samples, len(tier_a))
    tier_a_sample = tier_a.sample(n=n_a, random_state=args.seed)

    real_stats = []
    for i, (_, row) in enumerate(tier_a_sample.iterrows()):
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
            stats = compute_stats_for_cutout(hwc)
            real_stats.append(stats)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_a}", end="\r")
    print(f"  Computed stats for {len(real_stats)} real lenses")

    # Injected arcs (bright, low beta_frac)
    print("\n--- Generating bright injections (low beta_frac) ---")
    neg_df = val_df[(val_df["label"] == 0)].dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts = neg_df.sample(n=min(2000, len(neg_df)), random_state=args.seed + 1)

    import torch
    from dhs.injection_engine import (
        sample_lens_params, sample_source_params, inject_sis_shear, SourceParams,
    )
    import dataclasses

    inj_stats = []
    for i in range(min(args.n_samples, len(hosts))):
        host_row = hosts.iloc[i % len(hosts)]
        try:
            with np.load(str(host_row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])

            lens = sample_lens_params(rng, args.theta_e)
            source = sample_source_params(rng, args.theta_e, beta_frac_range=(0.1, 0.3))

            # Scale to bright magnitude
            target_flux = 10.0 ** ((22.5 - args.target_mag) / 2.5)
            if source.flux_nmgy_r > 0:
                scale = target_flux / source.flux_nmgy_r
            else:
                scale = 1.0
            fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
            fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
            fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
            fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
            source = SourceParams(**fields)

            result = inject_sis_shear(
                host_nmgy_hwc=host_t, lens=lens, source=source,
                pixel_scale=PIXEL_SCALE, psf_fwhm_r_arcsec=host_psf,
                seed=42 + i,
            )
            inj_hwc = result.injected[0].permute(1, 2, 0).numpy()
            stats = compute_stats_for_cutout(inj_hwc)
            inj_stats.append(stats)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_samples}", end="\r")
    print(f"  Computed stats for {len(inj_stats)} injections")

    # Negatives (non-lenses)
    print("\n--- Val negatives ---")
    neg_sample = neg_df.sample(n=min(args.n_samples, len(neg_df)), random_state=args.seed + 2)
    neg_stats = []
    for i, (_, row) in enumerate(neg_sample.iterrows()):
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
            stats = compute_stats_for_cutout(hwc)
            neg_stats.append(stats)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{min(args.n_samples, len(neg_df))}", end="\r")
    print(f"  Computed stats for {len(neg_stats)} negatives")

    # Aggregate and compare
    def summarize(stats_list, label):
        if not stats_list:
            return {}
        keys = stats_list[0].keys()
        summary = {}
        for k in keys:
            vals = [s[k] for s in stats_list if np.isfinite(s[k])]
            if vals:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "n": len(vals),
                }
            else:
                summary[k] = {"mean": float("nan"), "n": 0}
        return summary

    real_summary = summarize(real_stats, "real")
    inj_summary = summarize(inj_stats, "injection")
    neg_summary = summarize(neg_stats, "negative")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_real": len(real_stats),
        "n_injection": len(inj_stats),
        "n_negative": len(neg_stats),
        "injection_params": {
            "theta_e": args.theta_e,
            "target_mag": args.target_mag,
            "beta_frac_range": [0.1, 0.3],
        },
        "real_tier_a": real_summary,
        "injection_low_bf": inj_summary,
        "negative": neg_summary,
        "interpretation": {
            "high_freq_power": "Higher = more small-scale structure (clumps). Real arcs should be higher than Sersic injections.",
            "anisotropy": "Higher = more elongated features. Real arcs should show tangential elongation.",
            "color_coherence": "Lower = more color variation across arc (spatially varying SED). Sersic injections are uniform (coherence~1).",
            "local_var_ratio": "Higher = arc region is noisier than sky (clumps + Poisson). Smooth injections should be lower.",
        },
    }

    json_path = os.path.join(args.out_dir, "arc_morphology_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("ARC MORPHOLOGY STATISTICS COMPARISON")
    print("=" * 80)
    keys = ["high_freq_power_r", "anisotropy_r", "color_coherence", "local_var_ratio_r"]
    print(f"{'Statistic':<25} {'Real Tier-A':>15} {'Injection':>15} {'Negative':>15}")
    print("-" * 70)
    for k in keys:
        r = real_summary.get(k, {}).get("median", float("nan"))
        inj = inj_summary.get(k, {}).get("median", float("nan"))
        n = neg_summary.get(k, {}).get("median", float("nan"))
        print(f"{k:<25} {r:>15.4f} {inj:>15.4f} {n:>15.4f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
