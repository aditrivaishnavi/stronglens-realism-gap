#!/usr/bin/env python3
"""
Bright Arc Injection Test: Injection completeness at varying source magnitudes.

SCIENTIFIC MOTIVATION
====================
Strong-lens selection-function calibration reveals a persistent sim-to-real gap:
  - Real confirmed lenses: ~73% recall at p>0.5
  - Injection-recovery completeness: 4–8% on the standard grid (source mag 23–26)

This ~70 percentage-point gap could be driven by:
  1. Brightness: Training positives and real lenses are brighter than 23–26 mag.
  2. Morphology: Real lenses may have extended arcs/clumps not well captured by
     the Sérsic injection model.
  3. Environment: Host galaxy structure, blending, color gradients.
  4. Observational systematics: Astrometric scatter, PSF variations.

This script tests hypothesis (1) by injecting sources across a wide magnitude
range (18–26 mag) and measuring detection completeness. If brightness alone
explains the gap, we should see completeness approach real-lens recall in
bright bins (e.g., 18–20 mag). If not, the gap persists and other factors
(morphology, environment) are likely contributors.

TWO-PHASE EXECUTION (2026-02-16 refactor):
  Phase 1 (--phase generate): Generate all injections, save .npz cutouts and
    metadata CSV. No model loading, no inference. Safe to interrupt and resume
    via --resume flag.
  Phase 2 (--phase score): Load saved cutouts from Phase 1, run CNN inference,
    write results JSON. Reads from Phase 1 output directory.

This separation ensures that expensive injection generation is never lost if
inference crashes, and allows re-scoring with different models/checkpoints
without regenerating injections.

Date: 2026-02-13 (original), 2026-02-16 (2-phase refactor)
References:
  - real_lens_scoring.py: real-lens recall baseline
  - selection_function_grid.py: standard injection-recovery grid
  - MNRAS_RAW_NOTES.md Section 9.3: injection-recovery methodology

Paper IV Cross-Model Check :
  If Paper IV's pre-trained checkpoint is available, run this script with
  --checkpoint pointing to it. If Paper IV achieves >50% detection at
  bright magnitudes (vs our ~30%), it suggests their model learned different
  features. If similar (~25-35%), the gap is fundamental to Sersic injections,
  not our training. This is a 30-minute experiment that could clarify whether
  the gap is model-dependent or injection-dependent.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.injection_engine import (
    SourceParams,
    sample_lens_params,
    sample_source_params,
    inject_sis_shear,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)


PIXEL_SCALE = 0.262  # arcsec/pixel (DESI Legacy Survey)

MAGNITUDE_BINS: List[Tuple[float, float]] = [
    (18.0, 19.0),
    (19.0, 20.0),
    (20.0, 21.0),
    (21.0, 22.0),
    (22.0, 23.0),
    (23.0, 24.0),
    (24.0, 25.0),
    (25.0, 26.0),
]

CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"

# Metadata CSV columns for Phase 1 output
META_COLUMNS = [
    "injection_id", "mag_bin", "host_idx", "cutout_filename",
    "host_cutout_path", "theta_e", "psf_fwhm_r", "psfdepth_r",
    "target_mag", "source_r_mag", "lensed_r_mag",
    "beta_frac", "source_re", "source_n_sersic", "source_q",
    "source_flux_r", "source_flux_g", "source_flux_z",
    "lens_q", "lens_shear_g1", "lens_shear_g2",
    "arc_snr", "injection_seed",
]


# ---------------------------------------------------------------------------
# Reproducibility: save run_info.json with git hash, CLI args, pip freeze
# ---------------------------------------------------------------------------
def save_run_info(out_dir: str, args: argparse.Namespace, phase: str) -> str:
    """Save full experiment provenance for reproducibility."""
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "cli_args": vars(args),
        "python_version": sys.version,
    }
    # Git hash (best-effort, don't fail if not in a repo)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_hash"] = git_hash
    except Exception:
        info["git_hash"] = "unknown"
    # pip freeze (best-effort)
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")
        info["pip_freeze"] = freeze
    except Exception:
        info["pip_freeze"] = []

    path = os.path.join(out_dir, f"run_info_{phase}.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  Run info saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Model loading (Phase 2 only)
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, str, int, dict]:
    """Load model + preprocessing kwargs from checkpoint.

    Uses scoring_utils.load_model_and_spec to ensure preprocessing
    parameters match what the model was trained with.
    """
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    epoch = ckpt.get("epoch", -1)
    return model, arch, epoch, pp_kwargs


@torch.no_grad()
def score_one(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    """Score a single preprocessed image. Returns sigmoid probability."""
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def scale_source_to_magnitude(
    source: SourceParams,
    target_mag: float,
) -> SourceParams:
    """Create new SourceParams with fluxes scaled to achieve target r-band magnitude."""
    target_flux = 10.0 ** ((22.5 - target_mag) / 2.5)
    if source.flux_nmgy_r <= 0:
        scale = 1.0
    else:
        scale = target_flux / source.flux_nmgy_r

    fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
    fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
    fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
    fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
    return SourceParams(**fields)


def compute_beta_frac(source: SourceParams, theta_e: float) -> float:
    """Compute true beta_frac = |beta| / theta_E."""
    if theta_e <= 0:
        return float("nan")
    r = math.sqrt(source.beta_x_arcsec ** 2 + source.beta_y_arcsec ** 2)
    return r / theta_e


# ---------------------------------------------------------------------------
# Phase 1: Generate injections and save cutouts
# ---------------------------------------------------------------------------
def phase1_generate(
    manifest_path: str,
    host_split: str,
    n_hosts: int,
    theta_e: float,
    out_dir: str,
    seed: int,
    beta_frac_range: Optional[Tuple[float, float]],
    # FIX (2026-02-16): add_poisson_noise and add_sky_noise now both exposed
    # as CLI flags. Previous version only had add_poisson_noise and was missing
    # add_sky_noise, causing injections to appear unnaturally clean.
    add_poisson_noise: bool,
    gain_e_per_nmgy: float,
    add_sky_noise: bool,
    resume: bool = False,
) -> str:
    """Phase 1: Generate all injections and save to disk.

    For each (magnitude_bin, host) pair, generates one injection and saves:
      - {cutout_dir}/{injection_id}.npz  (injected_chw, injection_only_chw, host_hwc)
      - {out_dir}/injection_metadata.parquet  (full metadata for all injections)

    Returns the path to the metadata parquet.

    If resume=True, skips already-generated injections found in existing metadata.
    The RNG is re-seeded per-injection to ensure determinism regardless of
    resume state (each injection's RNG state depends only on the global seed
    plus the injection index, not on execution order).
    """
    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "injection_metadata.parquet")

    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    neg = df[(df[SPLIT_COL] == host_split) & (df[LABEL_COL] == 0)].copy()
    neg = neg.dropna(subset=["psfsize_r", "psfdepth_r"])
    if neg.empty:
        raise ValueError("No val negatives with valid psfsize_r and psfdepth_r")
    print(f"  Val negatives (valid PSF/depth): {len(neg)}")

    n_sample = min(n_hosts, len(neg))
    hosts = neg.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {n_sample} host galaxies")

    # Resume support: load existing metadata to skip completed injections
    existing_ids = set()
    existing_records = []
    if resume and os.path.exists(meta_path):
        existing_df = pd.read_parquet(meta_path)
        existing_ids = set(existing_df["injection_id"].tolist())
        existing_records = existing_df.to_dict("records")
        print(f"  Resuming: {len(existing_ids)} injections already exist, skipping")

    records = list(existing_records)
    total_generated = 0
    total_skipped = 0

    for bin_idx, (mag_lo, mag_hi) in enumerate(MAGNITUDE_BINS):
        bin_key = f"{mag_lo:.0f}-{mag_hi:.0f}"
        print(f"\n--- Magnitude bin {bin_key} ---")

        for i in range(n_sample):
            host_row = hosts.iloc[i]
            injection_id = f"bright_arc_{bin_key}_host{i:04d}"

            # Skip if already generated (resume support)
            if injection_id in existing_ids:
                total_skipped += 1
                continue

            # Deterministic per-injection RNG: seed depends on (global_seed,
            # bin_index, host_index) so each injection is reproducible
            # regardless of execution order or resume state.
            inj_seed_val = seed + bin_idx * 10000 + i
            rng = np.random.default_rng(inj_seed_val)

            try:
                with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed to load host {i}: {e}")
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])
            host_psfdepth = float(host_row["psfdepth_r"])

            # Sample lens and source with updated priors
            # FIX (2026-02-16): beta_frac_range now defaults to (0.10, 0.40)
            # in injection_engine.py, favouring near-caustic arc morphologies.
            lens = sample_lens_params(rng, theta_e)
            bf_kwargs: Dict[str, Any] = {}
            if beta_frac_range is not None:
                bf_kwargs["beta_frac_range"] = beta_frac_range
            source = sample_source_params(rng, theta_e, **bf_kwargs)

            # Override source magnitude to target bin
            target_mag = float(rng.uniform(mag_lo, mag_hi))
            source = scale_source_to_magnitude(source, target_mag)

            # Inject with all noise flags
            # FIX (2026-02-16): add_sky_noise now available — adds per-band
            # Gaussian noise matching host background (MAD from outer annulus).
            result = inject_sis_shear(
                host_nmgy_hwc=host_t,
                lens=lens,
                source=source,
                pixel_scale=PIXEL_SCALE,
                psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=None,
                seed=inj_seed_val,
                add_poisson_noise=add_poisson_noise,
                gain_e_per_nmgy=gain_e_per_nmgy,
                add_sky_noise=add_sky_noise,
            )

            inj_chw = result.injected[0].detach().cpu().numpy()
            inj_only_chw = result.injection_only[0].detach().cpu().numpy()

            # Compute diagnostics
            sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                host_psfdepth, host_psf, PIXEL_SCALE
            )
            snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)
            total_lensed_flux_r = float(inj_only_chw[1].sum())
            lensed_r_mag = (
                22.5 - 2.5 * math.log10(total_lensed_flux_r)
                if total_lensed_flux_r > 0 else float("nan")
            )

            # Save cutout as .npz
            filename = f"{injection_id}.npz"
            save_path = os.path.join(cutout_dir, filename)
            np.savez_compressed(
                save_path,
                injected_chw=inj_chw,
                injection_only_chw=inj_only_chw,
                host_hwc=hwc,
            )

            records.append({
                "injection_id": injection_id,
                "mag_bin": bin_key,
                "host_idx": i,
                "cutout_filename": filename,
                "host_cutout_path": str(host_row[CUTOUT_PATH_COL]),
                "theta_e": theta_e,
                "psf_fwhm_r": host_psf,
                "psfdepth_r": host_psfdepth,
                "target_mag": target_mag,
                "source_r_mag": target_mag,
                "lensed_r_mag": lensed_r_mag,
                "beta_frac": compute_beta_frac(source, theta_e),
                "source_re": source.re_arcsec,
                "source_n_sersic": source.n_sersic,
                "source_q": source.q,
                "source_flux_r": source.flux_nmgy_r,
                "source_flux_g": source.flux_nmgy_g,
                "source_flux_z": source.flux_nmgy_z,
                "lens_q": lens.q_lens,
                "lens_shear_g1": lens.shear_g1,
                "lens_shear_g2": lens.shear_g2,
                "arc_snr": float(snr),
                "injection_seed": inj_seed_val,
            })
            total_generated += 1

            if (total_generated) % 100 == 0:
                # Checkpoint metadata periodically so progress is not lost
                meta_df = pd.DataFrame(records)
                meta_df.to_parquet(meta_path, index=False)
                print(f"  Checkpoint: {total_generated} generated, "
                      f"{total_skipped} skipped (bin {bin_key}, host {i})")

        print(f"  Bin {bin_key} done")

    # Final save of metadata
    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(meta_path, index=False)
    print(f"\nPhase 1 complete: {total_generated} new injections "
          f"({total_skipped} resumed/skipped)")
    print(f"  Cutouts: {cutout_dir}/")
    print(f"  Metadata: {meta_path}")
    print(f"  Total records: {len(records)}")

    return meta_path


# ---------------------------------------------------------------------------
# Phase 2: Score saved injections with CNN
# ---------------------------------------------------------------------------
def phase2_score(
    checkpoint_path: str,
    out_dir: str,
    device_str: str = "cuda",
    clip_range_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Phase 2: Load saved injections from Phase 1 and run CNN inference.

    Reads injection_metadata.parquet and cutouts/ from out_dir.
    Appends cnn_score column and writes scored metadata + results JSON.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(out_dir, "injection_metadata.parquet")
    cutout_dir = os.path.join(out_dir, "cutouts")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No injection_metadata.parquet found in {out_dir}. Run Phase 1 first."
        )

    # Load model
    print("Loading model...")
    model, arch, epoch, pp_kwargs = load_model(checkpoint_path, device)
    if clip_range_override is not None:
        pp_kwargs["clip_range"] = clip_range_override
        print(f"  clip_range overridden to {clip_range_override}")
    print(f"  Architecture: {arch}, Epoch: {epoch}")
    print(f"  Preprocessing: {pp_kwargs}")

    # Load metadata
    meta_df = pd.read_parquet(meta_path)
    print(f"\nLoaded {len(meta_df)} injection records from Phase 1")

    # Score each injection
    scores = []
    n_total = len(meta_df)
    n_failed = 0

    for idx, row in meta_df.iterrows():
        npz_path = os.path.join(cutout_dir, row["cutout_filename"])
        try:
            with np.load(npz_path) as z:
                inj_chw = z["injected_chw"]
        except Exception as e:
            print(f"  WARNING: Failed to load {npz_path}: {e}")
            scores.append(float("nan"))
            n_failed += 1
            continue

        proc = preprocess_stack(inj_chw, **pp_kwargs)
        score = score_one(model, proc, device)
        scores.append(score)

        if (idx + 1) % 200 == 0:
            print(f"  Scored {idx + 1}/{n_total}", end="\r")

    meta_df["cnn_score"] = scores
    print(f"\nScoring complete: {n_total - n_failed} scored, {n_failed} failed")

    # Save scored metadata
    scored_path = os.path.join(out_dir, "injection_metadata_scored.parquet")
    meta_df.to_parquet(scored_path, index=False)
    print(f"  Scored metadata: {scored_path}")

    # Aggregate results by magnitude bin
    results_by_bin: Dict[str, Dict[str, Any]] = {}
    for bin_key, grp in meta_df.groupby("mag_bin"):
        valid = grp["cnn_score"].dropna()
        n_scored = len(valid)
        if n_scored > 0:
            det_p03 = float((valid >= 0.3).mean())
            det_p05 = float((valid >= 0.5).mean())
            median_score = float(valid.median())
        else:
            det_p03 = det_p05 = median_score = float("nan")

        valid_snrs = grp["arc_snr"].dropna()
        median_snr = float(valid_snrs.median()) if len(valid_snrs) > 0 else float("nan")

        results_by_bin[str(bin_key)] = {
            "mag_bin": str(bin_key),
            "n_scored": n_scored,
            "detection_rate_p03": det_p03,
            "detection_rate_p05": det_p05,
            "median_score": median_score,
            "median_arc_snr": median_snr,
        }

    # Read Phase 1 run_info for provenance
    phase1_info_path = os.path.join(out_dir, "run_info_generate.json")
    phase1_info = {}
    if os.path.exists(phase1_info_path):
        with open(phase1_info_path) as f:
            phase1_info = json.load(f)

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "arch": arch,
        "epoch": epoch,
        "device": str(device),
        "clip_range_override": clip_range_override,
        "preprocessing": pp_kwargs,
        "phase1_config": phase1_info.get("cli_args", {}),
        "n_total_injections": n_total,
        "n_scored": n_total - n_failed,
        "n_failed": n_failed,
        "magnitude_bins": sorted(results_by_bin.keys()),
        "results_by_bin": results_by_bin,
        "notes": [
            "Phase 2 scoring of pre-generated injections from Phase 1.",
            "Tests whether sim-to-real gap (~70 pp) is explained by brightness.",
            "Real lens recall ~73% at p>0.5; standard injection completeness 4–8%.",
            "FIX (2026-02-16): uses corrected injection priors (K-corrected colours,",
            "  narrowed beta_frac, raised Re_min, lowered n_max, disabled clumps,",
            "  add_sky_noise for background texture matching).",
        ],
    }

    # Save results JSON
    results_path = os.path.join(out_dir, "bright_arc_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results JSON: {results_path}")

    return output


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(output: Dict[str, Any]) -> None:
    """Print clean summary table to console."""
    print("\n" + "=" * 80)
    print("BRIGHT ARC INJECTION TEST — SUMMARY")
    print("=" * 80)
    print(f"Checkpoint: {output.get('checkpoint', 'N/A')}")
    phase1_cfg = output.get("phase1_config", {})
    print(f"Host split: {phase1_cfg.get('host_split', 'val')}  |  "
          f"n_hosts: {phase1_cfg.get('n_hosts', 'N/A')}")
    print(f"theta_E: {phase1_cfg.get('theta_e', 'N/A')}\"  |  "
          f"seed: {phase1_cfg.get('seed', 'N/A')}")
    print(f"add_poisson_noise: {phase1_cfg.get('add_poisson_noise', False)}  |  "
          f"add_sky_noise: {phase1_cfg.get('add_sky_noise', False)}")
    print()

    header = (
        f"{'Mag bin':<10} {'N scored':>10} {'p>0.3':>10} {'p>0.5':>10} "
        f"{'median_p':>12} {'median_SNR':>12}"
    )
    print(header)
    print("-" * len(header))

    for bin_key in sorted(output.get("magnitude_bins", [])):
        r = output["results_by_bin"].get(bin_key, {})
        n = r.get("n_scored", 0)
        p03 = r.get("detection_rate_p03", float("nan"))
        p05 = r.get("detection_rate_p05", float("nan"))
        med_p = r.get("median_score", float("nan"))
        med_snr = r.get("median_arc_snr", float("nan"))
        p03_str = f"{p03*100:.1f}%" if np.isfinite(p03) else "N/A"
        p05_str = f"{p05*100:.1f}%" if np.isfinite(p05) else "N/A"
        med_p_str = f"{med_p:.4f}" if np.isfinite(med_p) else "N/A"
        med_snr_str = f"{med_snr:.1f}" if np.isfinite(med_snr) else "N/A"
        print(f"{bin_key:<10} {n:>10} {p03_str:>10} {p05_str:>10} "
              f"{med_p_str:>12} {med_snr_str:>12}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bright Arc Injection Test: completeness at varying source magnitudes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Phase selection
    ap.add_argument("--phase", choices=["generate", "score", "both"], default="both",
                    help="Which phase to run: 'generate' (Phase 1 only), "
                         "'score' (Phase 2 only), or 'both' (default).")

    # Phase 1 args
    ap.add_argument("--manifest", help="Path to training manifest parquet (Phase 1)")
    ap.add_argument("--host-split", default="val",
                    help="Split for host negatives (default: val)")
    ap.add_argument("--n-hosts", type=int, default=200,
                    help="Number of host galaxies per magnitude bin (default: 200)")
    ap.add_argument("--theta-e", type=float, default=1.5,
                    help="Einstein radius in arcsec (default: 1.5)")
    ap.add_argument("--beta-frac-range", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="Override beta_frac range (e.g. --beta-frac-range 0.1 0.3). "
                         "Default: engine default (0.10, 0.40).")
    # FIX (2026-02-16): add_poisson_noise and add_sky_noise now both available.
    # Previous version only exposed add_poisson_noise, missing add_sky_noise
    # which caused injections to appear unnaturally clean.
    ap.add_argument("--add-poisson-noise", action="store_true", default=False,
                    help="Add Poisson noise to injected arcs.")
    ap.add_argument("--gain-e-per-nmgy", type=float, default=150.0,
                    help="Gain in e-/nmgy for Poisson noise (default: 150).")
    ap.add_argument("--add-sky-noise", action="store_true", default=False,
                    help="Add per-band Gaussian sky noise matching host background. "
                         "FIX (2026-02-16): previously missing, causing arcs to "
                         "appear sharper than real survey images.")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume Phase 1 from last checkpoint (skip existing injections).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Phase 2 args
    ap.add_argument("--checkpoint", help="Path to model checkpoint (Phase 2)")
    ap.add_argument("--clip-range", type=float, default=None,
                    help="Override clip_range in preprocessing.")
    ap.add_argument("--device", default="cuda",
                    help="Device for inference (default: cuda)")

    # Shared
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (shared between Phase 1 and Phase 2)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Phase 1: Generate ---
    if args.phase in ("generate", "both"):
        if not args.manifest:
            ap.error("--manifest is required for Phase 1 (generate)")

        print("=" * 80)
        print("PHASE 1: GENERATE INJECTIONS")
        print("=" * 80)

        save_run_info(args.out_dir, args, phase="generate")

        bf_range = tuple(args.beta_frac_range) if args.beta_frac_range else None
        phase1_generate(
            manifest_path=args.manifest,
            host_split=args.host_split,
            n_hosts=args.n_hosts,
            theta_e=args.theta_e,
            out_dir=args.out_dir,
            seed=args.seed,
            beta_frac_range=bf_range,
            add_poisson_noise=args.add_poisson_noise,
            gain_e_per_nmgy=args.gain_e_per_nmgy,
            add_sky_noise=args.add_sky_noise,
            resume=args.resume,
        )

    # --- Phase 2: Score ---
    if args.phase in ("score", "both"):
        if not args.checkpoint:
            ap.error("--checkpoint is required for Phase 2 (score)")

        print("\n" + "=" * 80)
        print("PHASE 2: SCORE INJECTIONS")
        print("=" * 80)

        save_run_info(args.out_dir, args, phase="score")

        output = phase2_score(
            checkpoint_path=args.checkpoint,
            out_dir=args.out_dir,
            device_str=args.device,
            clip_range_override=args.clip_range,
        )

        print_summary_table(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
