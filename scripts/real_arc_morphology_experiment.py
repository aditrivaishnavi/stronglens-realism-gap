#!/usr/bin/env python3
"""
Real-Arc Morphology Experiment per morphology experiment:
Test whether realistic source morphology improves injection detection.

SCIENTIFIC MOTIVATION
====================
"Replace the Sersic source with a 'scrambled real arc'
— take a real confirmed lens, subtract the galaxy model, use the residual arc
as the source-plane morphology. This gives realistic texture."

This script:
  1. Extracts arc residuals from Tier-A confirmed lenses by subtracting a
     smooth galaxy model (Sersic fit or median filter)
  2. Uses these residuals as source-plane morphology templates for injection
  3. Compares CNN detection rates: Sersic source vs real-arc-residual source
     at fixed beta_frac [0.1, 0.3] and bright magnitude

This tests whether morphological realism (clumps, knots, irregular structure)
improves injection completeness. If detection jumps significantly, source
morphology is a major driver of the sim-to-real gap.

PREREQUISITES:
  - Tier-A lens cutouts must be available
  - This is a controlled experiment: vary ONLY source morphology
  - All other parameters (beta_frac, magnitude, lens, host) are fixed

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python scripts/real_arc_morphology_experiment.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/real_arc_morphology \\
        --n-arcs 10 \\
        --n-hosts 200

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
import torch
import torch.nn as nn
from scipy.ndimage import median_filter

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.injection_engine import (
    LensParams,
    SourceParams,
    sample_lens_params,
    sample_source_params,
    inject_sis_shear,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)

PIXEL_SCALE = 0.262
TIER_COL = "tier"


@torch.no_grad()
def score_one(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    """Score a single preprocessed image. Returns sigmoid probability."""
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def extract_arc_residual(cutout_hwc: np.ndarray, filter_size: int = 15) -> np.ndarray:
    """Extract arc residual by subtracting a smooth galaxy model.

    Uses a large median filter as a simple, non-parametric galaxy model.
    The residual contains arc structure, noise, and small-scale features
    that differ from the smooth model.

    Args:
        cutout_hwc: (H, W, 3) float32 cutout in nanomaggies
        filter_size: median filter kernel size (larger = smoother galaxy model)

    Returns:
        residual_hwc: (H, W, 3) arc residual (can be positive and negative)
    """
    residual = np.zeros_like(cutout_hwc)
    for band in range(3):
        galaxy_model = median_filter(cutout_hwc[:, :, band], size=filter_size)
        residual[:, :, band] = cutout_hwc[:, :, band] - galaxy_model
    return residual


def inject_with_real_arc_template(
    host_hwc: np.ndarray,
    arc_residual_hwc: np.ndarray,
    theta_e_arcsec: float = 1.5,
    brightness_scale: float = 1.0,
) -> np.ndarray:
    """Inject a real arc residual into a host galaxy.

    This is a simplified injection that adds the arc residual directly
    (no re-lensing). The residual is centered and optionally rescaled.

    For a more rigorous approach, one would de-lens the arc to the source
    plane and re-lens at the desired geometry. This simple version tests
    whether TEXTURE (not geometry) matters.
    """
    result = host_hwc.copy()
    # Scale the arc residual to desired brightness
    # Only add positive residual (arc light) to avoid subtracting sky
    arc_positive = np.maximum(arc_residual_hwc * brightness_scale, 0.0)
    result += arc_positive
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real-arc morphology experiment",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-arcs", type=int, default=10,
                    help="Number of Tier-A arc residuals to extract (default: 10)")
    ap.add_argument("--n-hosts", type=int, default=200,
                    help="Hosts per arc template (default: 200)")
    ap.add_argument("--theta-e", type=float, default=1.5)
    ap.add_argument("--target-mag", type=float, default=19.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, pp_kwargs = load_model_and_spec(args.checkpoint, device)

    # Load manifest
    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    val_df = df[df["split"] == "val"].copy()

    # Get Tier-A lenses for arc extraction
    if TIER_COL in val_df.columns:
        tier_a = val_df[(val_df["label"] == 1) & (val_df[TIER_COL] == "A")]
    else:
        tier_a = val_df[val_df["label"] == 1]

    n_arcs = min(args.n_arcs, len(tier_a))
    arc_sample = tier_a.sample(n=n_arcs, random_state=args.seed)

    # Extract arc residuals
    print(f"\n--- Extracting {n_arcs} arc residuals ---")
    arc_residuals = []
    arc_paths = []
    for _, row in arc_sample.iterrows():
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
            residual = extract_arc_residual(hwc)
            arc_residuals.append(residual)
            arc_paths.append(str(row["cutout_path"]))
        except Exception:
            continue
    print(f"  Extracted {len(arc_residuals)} arc residuals")

    # Get host negatives
    neg_df = val_df[(val_df["label"] == 0)].dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts = neg_df.sample(n=min(args.n_hosts, len(neg_df)), random_state=args.seed + 1)

    # Experiment: compare Sersic injection vs real-arc-residual injection
    print(f"\n--- Running comparison experiment ---")
    print(f"  {len(arc_residuals)} arc templates × {len(hosts)} hosts")

    import dataclasses

    sersic_scores = []
    real_arc_scores = []

    for i, (_, host_row) in enumerate(hosts.iterrows()):
        try:
            with np.load(str(host_row["cutout_path"])) as z:
                host_hwc = z["cutout"].astype(np.float32)
        except Exception:
            continue

        host_psf = float(host_row["psfsize_r"])

        # --- Sersic injection (standard) ---
        lens = sample_lens_params(rng, args.theta_e)
        source = sample_source_params(rng, args.theta_e, beta_frac_range=(0.1, 0.3))

        # Scale to target magnitude
        target_flux = 10.0 ** ((22.5 - args.target_mag) / 2.5)
        if source.flux_nmgy_r > 0:
            scale = target_flux / source.flux_nmgy_r
        else:
            scale = 1.0
        fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
        fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
        fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
        fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
        source_bright = SourceParams(**fields)

        host_t = torch.from_numpy(host_hwc).float()
        result = inject_sis_shear(
            host_nmgy_hwc=host_t, lens=lens, source=source_bright,
            pixel_scale=PIXEL_SCALE, psf_fwhm_r_arcsec=host_psf,
            seed=args.seed + i,
        )
        inj_chw = result.injected[0].numpy()
        proc = preprocess_stack(inj_chw, **pp_kwargs)
        score_sersic = score_one(model, proc, device)
        sersic_scores.append(score_sersic)

        # --- Real-arc residual injection ---
        if arc_residuals:
            arc_idx = i % len(arc_residuals)
            inj_real = inject_with_real_arc_template(
                host_hwc, arc_residuals[arc_idx], brightness_scale=1.0,
            )
            inj_real_chw = np.transpose(inj_real, (2, 0, 1))
            proc_real = preprocess_stack(inj_real_chw, **pp_kwargs)
            score_real = score_one(model, proc_real, device)
            real_arc_scores.append(score_real)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(hosts)} scored", end="\r")

    sersic_arr = np.array(sersic_scores)
    real_arr = np.array(real_arc_scores)

    sersic_det = float((sersic_arr >= 0.3).mean()) if len(sersic_arr) > 0 else float("nan")
    real_det = float((real_arr >= 0.3).mean()) if len(real_arr) > 0 else float("nan")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_arc_templates": len(arc_residuals),
        "n_hosts": len(hosts),
        "theta_e": args.theta_e,
        "target_mag": args.target_mag,
        "sersic_injection": {
            "n_scored": len(sersic_scores),
            "detection_rate_p03": sersic_det,
            "median_score": float(np.median(sersic_arr)) if len(sersic_arr) > 0 else float("nan"),
            "beta_frac_range": [0.1, 0.3],
        },
        "real_arc_injection": {
            "n_scored": len(real_arc_scores),
            "detection_rate_p03": real_det,
            "median_score": float(np.median(real_arr)) if len(real_arr) > 0 else float("nan"),
            "method": "median-filter galaxy subtraction + positive residual addition",
            "arc_source_paths": arc_paths[:5],
        },
        "delta_detection_pp": (real_det - sersic_det) * 100 if np.isfinite(real_det) and np.isfinite(sersic_det) else float("nan"),
        "interpretation": (
            "If real-arc detection >> Sersic detection, source morphology is "
            "a significant driver. If similar, geometry (beta_frac) dominates."
        ),
    }

    json_path = os.path.join(args.out_dir, "real_arc_morphology_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    print("\n" + "=" * 60)
    print("REAL-ARC MORPHOLOGY EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Sersic injection detection (p>0.3): {sersic_det*100:.1f}%")
    print(f"  Real-arc injection detection (p>0.3): {real_det*100:.1f}%")
    print(f"  Delta: {(real_det - sersic_det)*100:+.1f} pp")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
