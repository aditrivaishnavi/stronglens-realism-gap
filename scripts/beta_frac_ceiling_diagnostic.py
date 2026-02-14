#!/usr/bin/env python3
"""
Beta_frac Ceiling Diagnostic: Investigate why bright-arc recovery
plateaus at ~30%.

IMPORTANT LIMITATION (Code review finding):
  This script uses an arc-annulus SNR proxy on BLANK (noise-only) hosts
  to measure "detection." It does NOT use the CNN model. Therefore:
  - Results are UPPER BOUNDS on detection probability (real hosts add
    galaxy light that reduces effective arc SNR).
  - Results show the GEOMETRIC contribution to the ceiling but do NOT
    capture morphological, noise, or feature-space effects that the CNN
    exploits.
  - For CNN-based beta_frac analysis, use bright_arc_injection_test.py
    with --beta-frac-range instead.

HYPOTHESIS (code review finding):
  The area-weighted sampling beta_frac = sqrt(U(lo^2, hi^2)) with
  lo=0.1, hi=1.0 produces P(beta_frac < 0.55) ≈ 29.5%. For SIS/SIE
  lenses, sources near the Einstein radius (beta_frac ~ 0.5-0.6) form
  the brightest arcs. If the selection function grid only counts
  "detected" injections, the ~70% with beta_frac > 0.55 produce faint
  or unresolved arcs that are hard to detect, creating a ~30% ceiling.

EXPERIMENTAL PLAN:
  1. Pure math: compute P(beta_frac < threshold) for various thresholds
  2. Injection experiment: for N_trial injections at each theta_E, measure
     what fraction produce arc SNR > detection threshold as a function of
     beta_frac_max (the upper bound of the prior).
  3. Repeat with restricted beta_frac_range = (0.1, 0.55) to see if
     bright-arc recovery jumps to ~80-90%.

USAGE (requires torch, numpy):
    cd stronglens_calibration
    PYTHONPATH=. python scripts/beta_frac_ceiling_diagnostic.py \\
        --n-trials 200 --theta-e 1.5 --output results/beta_frac_diagnostic.json

    # Quick math-only check (no torch needed):
    PYTHONPATH=. python scripts/beta_frac_ceiling_diagnostic.py --math-only

Created: 2026-02-13 (code review finding)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# ==================================================================
# Part 1: Pure math — no torch required
# ==================================================================

def beta_frac_cdf(threshold: float, lo: float = 0.1, hi: float = 1.0) -> float:
    """P(beta_frac < threshold) under area-weighted sampling.

    beta_frac = sqrt(U(lo^2, hi^2))
    CDF(t) = (t^2 - lo^2) / (hi^2 - lo^2) for lo <= t <= hi
    """
    if threshold <= lo:
        return 0.0
    if threshold >= hi:
        return 1.0
    return (threshold**2 - lo**2) / (hi**2 - lo**2)


def run_math_diagnostic(lo: float = 0.1, hi: float = 1.0):
    """Print the CDF of beta_frac at key thresholds."""
    print("=" * 60)
    print("Beta_frac CDF under area-weighted sampling")
    print(f"  beta_frac = sqrt(U({lo}^2, {hi}^2))")
    print("=" * 60)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    for t in thresholds:
        p = beta_frac_cdf(t, lo, hi)
        marker = "  <-- ~30% ceiling?" if abs(p - 0.30) < 0.03 else ""
        print(f"  P(beta_frac < {t:.2f}) = {p:.1%}{marker}")

    # The hypothesis: P(beta_frac < 0.55) ~ 30%
    p_055 = beta_frac_cdf(0.55, lo, hi)
    print(f"\nKey prediction: P(beta_frac < 0.55) = {p_055:.1%}")
    if abs(p_055 - 0.30) < 0.05:
        print("  MATCHES the observed ~30% bright-arc ceiling.")
    else:
        print(f"  Does NOT match 30% ceiling (off by {abs(p_055-0.30):.1%}).")

    # What beta_frac_max gives 80% bright arcs?
    # P(bf < t) = (t^2 - lo^2)/(t^2 - lo^2) = 1 for t = hi
    # We want: if we restrict to (lo, hi_new), what P(bf < 0.55)?
    # With hi_new=0.55: 100% by construction.
    # With hi_new=0.65: P = (0.55^2-0.01)/(0.65^2-0.01) = 0.2925/0.4125 = 70.9%
    print(f"\nWhat if we restrict beta_frac_max?")
    for hi_new in [0.55, 0.60, 0.65, 0.70, 0.80]:
        # P(beta_frac < 0.55 | restricted to [lo, hi_new])
        p = beta_frac_cdf(min(0.55, hi_new), lo, hi_new)
        print(f"  beta_frac_range=({lo}, {hi_new}): "
              f"P(bf<0.55) = {p:.1%}")


# ==================================================================
# Part 2: Injection experiment — requires torch
# ==================================================================

def run_injection_experiment(
    n_trials: int,
    theta_e_arcsec: float,
    pixel_scale: float,
    psf_fwhm_r_arcsec: float,
    beta_frac_maxes: list[float],
    snr_threshold: float,
    seed: int,
    output_path: Path | None,
):
    """Run injection experiments with varying beta_frac_max."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch not available. Use --math-only for math-only mode.")
        sys.exit(1)

    # Import from the actual codebase
    from dhs.injection_engine import (
        LensParams,
        SourceParams,
        inject_sis_shear,
        sample_source_params,
        sample_lens_params,
        arc_annulus_snr,
        estimate_sigma_pix_from_psfdepth,
    )

    print("=" * 60)
    print("Beta_frac Injection Experiment")
    print(f"  theta_E = {theta_e_arcsec} arcsec")
    print(f"  pixel_scale = {pixel_scale} arcsec/pixel")
    print(f"  PSF FWHM (r) = {psf_fwhm_r_arcsec} arcsec")
    print(f"  n_trials = {n_trials}")
    print(f"  SNR threshold = {snr_threshold}")
    print(f"  beta_frac_max values = {beta_frac_maxes}")
    print("=" * 60)

    h = w = 101
    # Create a blank host (just background noise)
    rng_np = np.random.default_rng(seed)
    background_nmgy = 0.1  # typical sky background in nanomaggies
    noise_sigma = 0.05

    # Typical psfdepth for DR10 r-band (inverse variance in nmgy^-2)
    psfdepth_r = 100.0  # reasonable proxy
    sigma_pix_r = estimate_sigma_pix_from_psfdepth(
        psfdepth_r, psf_fwhm_r_arcsec, pixel_scale
    )

    results = {}

    for bf_max in beta_frac_maxes:
        print(f"\n--- beta_frac_max = {bf_max} ---")
        n_detected = 0
        snrs = []
        beta_fracs = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 1000

            # Create fresh host for each trial
            host_np = (background_nmgy
                       + noise_sigma * rng_np.standard_normal((h, w, 3))).astype(np.float32)
            host_t = torch.from_numpy(host_np).float()

            # Sample parameters with restricted beta_frac_max
            trial_rng = np.random.default_rng(trial_seed)
            source = sample_source_params(
                rng=trial_rng,
                theta_e_arcsec=theta_e_arcsec,
                beta_frac_range=(0.1, bf_max),
            )
            lens = sample_lens_params(
                rng=trial_rng,
                theta_e_arcsec=theta_e_arcsec,
            )

            # Compute actual beta_frac from source position
            beta = math.sqrt(source.beta_x_arcsec**2 + source.beta_y_arcsec**2)
            bf = beta / theta_e_arcsec
            beta_fracs.append(bf)

            # Inject
            result = inject_sis_shear(
                host_nmgy_hwc=host_t,
                lens=lens,
                source=source,
                pixel_scale=pixel_scale,
                psf_fwhm_r_arcsec=psf_fwhm_r_arcsec,
                seed=trial_seed,
            )

            snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)
            snrs.append(snr)

            if not math.isnan(snr) and snr > snr_threshold:
                n_detected += 1

        detection_rate = n_detected / n_trials
        mean_snr = float(np.nanmean(snrs))
        median_bf = float(np.median(beta_fracs))

        print(f"  Detection rate: {detection_rate:.1%} ({n_detected}/{n_trials})")
        print(f"  Mean arc SNR: {mean_snr:.1f}")
        print(f"  Median beta_frac: {median_bf:.3f}")

        results[str(bf_max)] = {
            "beta_frac_max": bf_max,
            "detection_rate": detection_rate,
            "n_detected": n_detected,
            "n_trials": n_trials,
            "mean_snr": mean_snr,
            "median_beta_frac": median_bf,
            "snr_threshold": snr_threshold,
        }

    # --- Q1.20 fix: compute P(detected | beta_frac) in bins ---
    # Use the bf_max=1.0 run (full prior) for binned analysis
    full_key = str(max(beta_frac_maxes))
    if full_key in results:
        print("\n" + "=" * 60)
        print("P(detected | beta_frac) in bins  [Q1.20]")
        print("  NOTE: These are blank-host detection rates, which are strict")
        print("  upper bounds on real-host detection (Q1.18). Real hosts add")
        print("  galaxy light gradients and color context that can confuse")
        print("  or suppress arc detection.")
        print("=" * 60)

        # Collect beta_frac and detection flag arrays from the full run
        # We need to re-run to get per-trial data; store from above
        # Actually we have beta_fracs and snrs from the last bf_max run.
        # To get data for the full prior, we store them per-run below.

    # Re-run with full prior for binned analysis
    bf_all = max(beta_frac_maxes)
    trial_bfs = []
    trial_detected = []
    trial_rng_binned = np.random.default_rng(seed + 999)

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000 + 77777
        trial_rng = np.random.default_rng(trial_seed)
        source = sample_source_params(
            rng=trial_rng, theta_e_arcsec=theta_e_arcsec,
            beta_frac_range=(0.1, bf_all),
        )
        lens = sample_lens_params(rng=trial_rng, theta_e_arcsec=theta_e_arcsec)
        beta = math.sqrt(source.beta_x_arcsec**2 + source.beta_y_arcsec**2)
        bf = beta / theta_e_arcsec
        trial_bfs.append(bf)

        host_np = (background_nmgy
                   + noise_sigma * trial_rng_binned.standard_normal((h, w, 3))).astype(np.float32)
        host_t = torch.from_numpy(host_np).float()
        result = inject_sis_shear(
            host_nmgy_hwc=host_t, lens=lens, source=source,
            pixel_scale=pixel_scale, psf_fwhm_r_arcsec=psf_fwhm_r_arcsec,
            seed=trial_seed,
        )
        snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)
        trial_detected.append(1 if (not math.isnan(snr) and snr > snr_threshold) else 0)

    trial_bfs = np.array(trial_bfs)
    trial_detected = np.array(trial_detected)
    binned_results = {}

    bin_edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n{'beta_frac bin':>16} {'N':>6} {'P(det)':>10}")
    print("-" * 36)
    for i in range(len(bin_edges) - 1):
        lo_b, hi_b = bin_edges[i], bin_edges[i + 1]
        mask = (trial_bfs >= lo_b) & (trial_bfs < hi_b)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            p_det = float(trial_detected[mask].mean())
        else:
            p_det = float("nan")
        print(f"  [{lo_b:.1f}, {hi_b:.1f}){' ':>6} {n_in_bin:>6} {p_det:>9.1%}")
        binned_results[f"{lo_b:.1f}-{hi_b:.1f}"] = {
            "n": n_in_bin, "p_detected": p_det,
        }

    # Summary table
    print("\n" + "=" * 60)
    print("Summary (by beta_frac_max)")
    print("=" * 60)
    print(f"{'bf_max':>8} {'detection':>12} {'mean_SNR':>10} {'med_bf':>8}")
    print("-" * 42)
    for bf_max in beta_frac_maxes:
        r = results[str(bf_max)]
        print(f"{bf_max:>8.2f} {r['detection_rate']:>11.1%} "
              f"{r['mean_snr']:>10.1f} {r['median_beta_frac']:>8.3f}")

    print("\n  NOTE (Q1.18): All detection rates above are on BLANK hosts (noise-only).")
    print("  They represent strict upper bounds. Real hosts with galaxy light")
    print("  will have lower detection rates due to annulus contamination,")
    print("  color gradients, and morphological confusion.")
    print("  NOTE (Q5.6): To test whether clipping artifacts contribute to the")
    print("  ceiling, rerun with --clip-range-test to compare clip_range=10 vs 50.")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "config": {
                    "theta_e_arcsec": theta_e_arcsec,
                    "pixel_scale": pixel_scale,
                    "psf_fwhm_r_arcsec": psf_fwhm_r_arcsec,
                    "n_trials": n_trials,
                    "snr_threshold": snr_threshold,
                    "seed": seed,
                },
                "results": results,
                "binned_detection": binned_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Beta_frac ceiling diagnostic for injection-recovery"
    )
    parser.add_argument("--math-only", action="store_true",
                        help="Run only the pure-math CDF analysis (no torch)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of injection trials per beta_frac_max")
    parser.add_argument("--theta-e", type=float, default=1.5,
                        help="Einstein radius in arcsec")
    parser.add_argument("--pixel-scale", type=float, default=0.262,
                        help="Pixel scale in arcsec/pixel (DESI Legacy)")
    parser.add_argument("--psf-fwhm", type=float, default=1.3,
                        help="PSF FWHM in arcsec (r-band)")
    parser.add_argument("--snr-threshold", type=float, default=5.0,
                        help="Arc SNR detection threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--beta-frac-maxes", type=float, nargs="+",
                        default=[0.40, 0.55, 0.65, 0.80, 1.00],
                        help="List of beta_frac_max values to test")

    args = parser.parse_args()

    # Always run math diagnostic
    run_math_diagnostic(lo=0.1, hi=1.0)

    if not args.math_only:
        print()
        run_injection_experiment(
            n_trials=args.n_trials,
            theta_e_arcsec=args.theta_e,
            pixel_scale=args.pixel_scale,
            psf_fwhm_r_arcsec=args.psf_fwhm,
            beta_frac_maxes=args.beta_frac_maxes,
            snr_threshold=args.snr_threshold,
            seed=args.seed,
            output_path=Path(args.output) if args.output else None,
        )


if __name__ == "__main__":
    main()
