#!/usr/bin/env python3
"""
Selection Function: Injection-Recovery Grid.

Measures the detection completeness C(theta_E, PSF, depth) by injecting
physically-motivated SIS+shear lensed sources into real DR10 negative cutouts
and scoring with a frozen trained model.

Grid axes (MNRAS_RAW_NOTES.md Section 9.3):
  - theta_E (Einstein radius): 0.5 to 3.0 arcsec
  - PSF FWHM: 0.9 to 1.8 arcsec (r-band)
  - 5-sigma depth: 22.5 to 24.5 mag (r-band)

For each grid cell:
  1. Select host galaxies (negatives) matching the PSF/depth bin
  2. Sample lens + source parameters; inject via SIS+shear ray-shooting
  3. Apply the same preprocessing as training (raw_robust, 101x101)
  4. Score with the frozen model
  5. Compute detection fraction with Bayesian binomial confidence intervals
  6. Report arc annulus SNR for diagnostics

Injection engine: dhs.injection_engine (SIE+shear, Sersic source, nanomaggy
flux calibration, FFT-based PSF blur). No external lensing dependencies.

Detection thresholds:
  - Fixed probability thresholds (e.g., 0.3, 0.5, 0.7): --thresholds
  - Fixed FPR thresholds (scientifically defensible): --fpr-targets
    Scores all negatives in the host split to derive the probability threshold
    corresponding to target FPR (e.g. 1e-3, 1e-4). This is the preferred
    method for publication as it ties completeness to a controlled false
    positive rate.

Supports any architecture via build_model() factory (auto-detected from
checkpoint).

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    # Run with EfficientNetV2-S (multiple thresholds)
    python scripts/selection_function_grid.py \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --host-split val \
        --thresholds 0.3 0.5 0.7 \
        --out-dir results/selection_function_efficientnet

    # Run with ResNet-18 for cross-model comparison
    python scripts/selection_function_grid.py \
        --checkpoint checkpoints/paperIV_resnet18/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --host-split val \
        --out-dir results/selection_function_resnet18

    # Output to S3
    python scripts/selection_function_grid.py \
        --checkpoint checkpoints/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out-dir s3://my-bucket/selection_function_run1

Author: stronglens-realism-gap project
Date: 2026-02-11
References:
  - MNRAS_RAW_NOTES.md Section 9.3
  - Paper IV (Inchausti et al. 2025): ensemble completeness comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# dhs imports
from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.constants import CUTOUT_SIZE, STAMP_SIZE
from dhs.injection_engine import (
    AB_ZP,
    LensParams,
    SourceParams,
    inject_sis_shear,
    sample_lens_params,
    sample_source_params,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)
from dhs.selection_function_utils import (
    m5_from_psfdepth,
    bayes_binomial_interval,
    build_edges,
    digitize,
)
from dhs.s3io import is_s3_uri, join_uri, write_bytes, write_text, write_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"

PIXEL_SCALE = 0.262  # arcsec/pixel (DESI Legacy Survey)


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, int, dict]:
    """Load model + preprocessing kwargs, return (model, arch, epoch, pp_kwargs).

    Uses scoring_utils.load_model_and_spec to ensure preprocessing
    parameters match what the model was trained with.  This fixes the
    BUG where preprocess_stack was called with hard-coded defaults,
    creating a train-score mismatch after annulus changes.
    """
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    epoch = ckpt.get("epoch", -1)
    return model, arch, epoch, pp_kwargs


@torch.no_grad()
def score_batch(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    """Score a batch of preprocessed images. Returns sigmoid probabilities."""
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


# ---------------------------------------------------------------------------
# Fixed-FPR threshold derivation
# ---------------------------------------------------------------------------
@torch.no_grad()
def derive_fpr_thresholds(
    model: nn.Module,
    manifest_df: pd.DataFrame,
    split: str,
    target_fprs: List[float],
    preprocessing: str,
    crop: bool,
    device: torch.device,
    pp_kwargs: Optional[dict] = None,
    data_root: Optional[str] = None,
    max_negatives: int = 50000,
    batch_size: int = 256,
    seed: int = 1337,
) -> Dict[float, float]:
    """Derive detection thresholds from a fixed False Positive Rate.

    Scores all (or a subsample of) negatives in the given split, sorts the
    scores, and finds the threshold tau such that the fraction of negatives
    scoring >= tau equals the target FPR.

    Parameters
    ----------
    target_fprs : list of float
        Target FPR values, e.g. [0.001, 0.0001].

    Returns
    -------
    dict mapping FPR -> threshold (probability).
    """
    neg = manifest_df[
        (manifest_df[SPLIT_COL] == split) & (manifest_df[LABEL_COL] == 0)
    ]
    if neg.empty:
        raise ValueError(f"No negatives in split '{split}' for FPR derivation")

    rng = np.random.default_rng(seed)
    if len(neg) > max_negatives:
        neg = neg.sample(n=max_negatives, random_state=seed).reset_index(drop=True)

    n_neg = len(neg)
    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE
    print(f"\nFPR threshold derivation: scoring {n_neg:,} negatives from '{split}'")

    all_scores = []
    n_load_ok = 0
    n_load_failed = 0
    for start in range(0, n_neg, batch_size):
        end = min(start + batch_size, n_neg)
        batch_size_actual = end - start
        batch_arr = np.zeros((batch_size_actual, 3, input_size, input_size), dtype=np.float32)
        valid_mask = np.zeros(batch_size_actual, dtype=bool)
        for j, (_, row) in enumerate(neg.iloc[start:end].iterrows()):
            try:
                with np.load(str(row[CUTOUT_PATH_COL])) as z:
                    img = z["cutout"].astype(np.float32)
                # img is HWC -> CHW
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                _pp = dict(pp_kwargs or {})
                _pp.setdefault("mode", preprocessing)
                _pp.setdefault("crop", crop)
                _pp.setdefault("clip_range", 10.0)
                proc = preprocess_stack(img, **_pp)
                batch_arr[j] = proc
                valid_mask[j] = True
                n_load_ok += 1
            except Exception as exc:
                n_load_failed += 1
                if n_load_failed <= 10:
                    print(f"  WARNING: FPR load failed ({n_load_failed}): "
                          f"{row.get(CUTOUT_PATH_COL, '?')}: {exc}")
        # Score the full batch but only keep scores for valid entries
        scores = score_batch(model, batch_arr, device)
        all_scores.append(scores[valid_mask])
        if (start // batch_size) % 10 == 0:
            print(f"  Scored {end}/{n_neg}", end="\r")

    all_scores = np.concatenate(all_scores)
    n_valid = len(all_scores)
    print(f"\n  Scored {n_valid:,} valid negatives "
          f"(loaded: {n_load_ok:,}, failed: {n_load_failed:,}). "
          f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

    if n_load_failed > 0:
        fail_pct = 100.0 * n_load_failed / n_neg
        print(f"  WARNING: {n_load_failed:,} load failures ({fail_pct:.2f}%) — "
              f"these are EXCLUDED from FPR calculation (not scored as zeros).")

    if n_valid == 0:
        raise RuntimeError("All negative loads failed — cannot derive FPR thresholds")

    # Sort descending to find threshold at rank k = FPR * N_valid
    sorted_scores = np.sort(all_scores)[::-1]

    fpr_to_threshold = {}
    for fpr in target_fprs:
        k = max(1, int(np.ceil(fpr * n_valid)))
        if k >= n_valid:
            tau = 0.0
        else:
            # threshold = score at rank k (0-indexed)
            tau = float(sorted_scores[k - 1])
        # Verify: fraction scoring >= tau
        actual_fpr = float((all_scores >= tau).sum()) / n_valid
        fpr_to_threshold[fpr] = tau
        print(f"  FPR target={fpr:.1e} -> threshold={tau:.6f} "
              f"(actual FPR={actual_fpr:.4e}, N_valid={n_valid:,})")

    return fpr_to_threshold


# ---------------------------------------------------------------------------
# Grid bin assignment
# ---------------------------------------------------------------------------
def nearest_bin(val: float, grid: List[float]) -> float:
    """Snap value to nearest grid bin center."""
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------
def run_selection_function(
    checkpoint_path: str,
    manifest_path: str,
    host_split: str = "val",
    host_max: int = 20000,
    thresholds: Optional[List[float]] = None,
    fpr_targets: Optional[List[float]] = None,
    injections_per_cell: int = 500,
    # Grid parameters
    theta_e_min: float = 0.5,
    theta_e_max: float = 3.0,
    theta_e_step: float = 0.25,
    psf_min: float = 0.9,
    psf_max: float = 1.8,
    psf_step: float = 0.15,
    depth_min: float = 22.5,
    depth_max: float = 24.5,
    depth_step: float = 0.5,
    # Processing
    preprocessing: str = "raw_robust",
    crop: bool = False,
    pixscale: float = PIXEL_SCALE,
    seed: int = 1337,
    device_str: str = "cuda",
    data_root: Optional[str] = None,
    # Injection realism options
    add_poisson_noise: bool = False,
    gain_e_per_nmgy: float = 150.0,
    # FIX (2026-02-16): add_sky_noise was previously missing, causing injected
    # arcs to appear unnaturally clean against the noisy host background.
    add_sky_noise: bool = False,
    # Save all injection cutouts as .npz files for reproducibility
    save_cutouts: bool = False,
    save_cutouts_dir: Optional[str] = None,
    # Sensitivity analysis overrides (all default to no-op)
    injection_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the injection-recovery grid. Returns (results_df, metadata)."""

    if thresholds is None:
        thresholds = [0.5]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Extract sensitivity analysis overrides (all default to no-op values)
    _ovr = injection_overrides or {}
    psf_scale_factor = _ovr.get("psf_scale_factor", 1.0)
    re_scale = _ovr.get("re_scale_factor", 1.0)
    gmr_shift = _ovr.get("g_minus_r_shift", 0.0)
    rmz_shift = _ovr.get("r_minus_z_shift", 0.0)
    q_lens_range_override = _ovr.get("q_lens_range", None)

    # Path overrides
    eff_manifest = manifest_path
    eff_ckpt = checkpoint_path
    if data_root:
        default_root = "/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap"
        eff_manifest = manifest_path.replace(default_root, data_root.rstrip("/"), 1)
        eff_ckpt = checkpoint_path.replace(default_root, data_root.rstrip("/"), 1)

    # Load model (with checkpoint-driven preprocessing spec)
    print(f"Loading model: {eff_ckpt}")
    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(eff_ckpt, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")
    print(f"  Preprocessing kwargs from checkpoint: {pp_kwargs}")

    # Load manifest
    print(f"Loading manifest: {eff_manifest}")
    df = pd.read_parquet(eff_manifest)

    # Derive FPR-based thresholds if requested
    fpr_threshold_map: Dict[float, float] = {}
    if fpr_targets:
        fpr_threshold_map = derive_fpr_thresholds(
            model=model,
            manifest_df=df,
            split=host_split,
            target_fprs=fpr_targets,
            preprocessing=preprocessing,
            crop=crop,
            device=device,
            pp_kwargs=pp_kwargs,
            data_root=data_root,
            seed=seed,
        )
        # Append the derived thresholds to the threshold list
        for fpr, tau in fpr_threshold_map.items():
            if tau not in thresholds:
                thresholds.append(tau)

    # Select negative hosts from the specified split
    hosts = df[(df[SPLIT_COL] == host_split) & (df[LABEL_COL] == 0)].copy()
    if hosts.empty:
        raise ValueError(f"No negative hosts found in split '{host_split}'")
    print(f"  Negative hosts in '{host_split}': {len(hosts):,}")

    # Require PSF and depth columns
    for col in ["psfsize_r", "psfdepth_r"]:
        if col not in hosts.columns:
            raise ValueError(f"Missing column '{col}' in manifest")

    # Subsample hosts if too many
    rng = np.random.default_rng(seed)
    if len(hosts) > host_max:
        hosts = hosts.sample(n=host_max, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {len(hosts):,} hosts")

    # Build grid
    theta_es = np.round(np.arange(theta_e_min, theta_e_max + 1e-9, theta_e_step), 4).tolist()
    psf_bins = np.round(np.arange(psf_min, psf_max + 1e-9, psf_step), 4).tolist()
    depth_bins = np.round(np.arange(depth_min, depth_max + 1e-9, depth_step), 4).tolist()

    n_cells = len(theta_es) * len(psf_bins) * len(depth_bins)
    print(f"\nGrid: {len(theta_es)} theta_E x {len(psf_bins)} PSF x {len(depth_bins)} depth = {n_cells} cells")
    print(f"  theta_E: {theta_es}")
    print(f"  PSF FWHM: {psf_bins}")
    print(f"  Depth (5sig mag): {depth_bins}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Injections/cell: {injections_per_cell}")
    print(f"  Poisson noise: {'ON (gain={:.0f} e-/nmgy)'.format(gain_e_per_nmgy) if add_poisson_noise else 'OFF'}")
    print(f"  Total injections: {n_cells * injections_per_cell:,}")

    # Assign hosts to PSF and depth bins using selection_function_utils
    depth_mag_arr = m5_from_psfdepth(hosts["psfdepth_r"].to_numpy(dtype=np.float64))
    hosts["depth_mag"] = depth_mag_arr
    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest_bin(float(v), psf_bins))
    hosts["depth_bin"] = hosts["depth_mag"].apply(lambda v: nearest_bin(float(v), depth_bins))

    # Group hosts by (psf_bin, depth_bin) for fast lookup
    # Store (cutout_path, psfsize_r, psfdepth_r) for each host
    host_groups: Dict[Tuple[float, float], pd.DataFrame] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        host_groups[(float(pb), float(db))] = g

    # Determine input size from preprocessing config
    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE

    # Cutout saving setup
    _cutout_dir: Optional[str] = None
    _cutout_meta: List[Dict[str, Any]] = []
    if save_cutouts:
        _cutout_dir = save_cutouts_dir or os.path.join(
            os.path.dirname(checkpoint_path) if checkpoint_path else ".",
            "grid_cutouts"
        )
        os.makedirs(_cutout_dir, exist_ok=True)
        print(f"  Saving cutouts to: {_cutout_dir}")

    # Run grid
    MAX_RETRIES = 5  # max retries per injection on failure before giving up
    print(f"\nRunning injection-recovery grid...")
    t0 = time.time()
    rows = []
    cell_idx = 0
    total_inject_ok = 0
    total_inject_failed = 0
    failure_log: List[Dict[str, Any]] = []  # track all failures for metadata

    for theta_e in theta_es:
        for pb in psf_bins:
            for db in depth_bins:
                cell_idx += 1
                host_df = host_groups.get((pb, db))

                # Build reverse lookup: threshold -> FPR label (if derived from FPR)
                thr_to_fpr = {tau: fpr for fpr, tau in fpr_threshold_map.items()}

                if host_df is None or len(host_df) == 0:
                    for thr in thresholds:
                        fpr_label = thr_to_fpr.get(thr, None)
                        threshold_type = f"FPR={fpr_label:.1e}" if fpr_label is not None else "fixed"
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": "all",
                            "n_injections": 0,
                            "n_detected": 0,
                            "n_failed": 0,
                            "completeness": float("nan"),
                            "ci68_lo": float("nan"),
                            "ci68_hi": float("nan"),
                            "ci95_lo": float("nan"),
                            "ci95_hi": float("nan"),
                            "mean_score": float("nan"),
                            "mean_arc_snr": float("nan"),
                            "sufficient": False,
                        })
                    continue

                n_target = injections_per_cell

                # Collect successful injections with retry-on-failure
                batch_list = []
                arc_snr_list = []
                src_mag_list = []
                lensed_mag_list = []  # Lensed (observed) arc magnitude
                cell_n_failed = 0

                attempt = 0
                while len(batch_list) < n_target and attempt < n_target + n_target * MAX_RETRIES:
                    attempt += 1
                    # Sample a random host from this cell's host pool
                    host_idx = int(rng.choice(len(host_df)))
                    host_row = host_df.iloc[host_idx]

                    try:
                        # Load host cutout (HWC, nanomaggies)
                        with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                            host_hwc = z["cutout"].astype(np.float32)

                        host_hwc_torch = torch.from_numpy(host_hwc).float()

                        # Get per-host PSF and depth
                        host_psf = float(host_row["psfsize_r"])
                        host_psfdepth = float(host_row["psfdepth_r"])

                        # Sample lens and source parameters (with sensitivity overrides)
                        lens = sample_lens_params(
                            rng, theta_e,
                            q_lens_range_override=q_lens_range_override,
                        )
                        source = sample_source_params(
                            rng, theta_e,
                            re_scale=re_scale,
                            gmr_shift=gmr_shift,
                            rmz_shift=rmz_shift,
                        )

                        # Record source r-magnitude for stratified analysis
                        src_r_mag = float("nan")
                        if source.flux_nmgy_r > 0:
                            src_r_mag = AB_ZP - 2.5 * math.log10(source.flux_nmgy_r)

                        # Inject (PSF may be scaled for sensitivity analysis)
                        eff_psf = host_psf * psf_scale_factor
                        inj_seed = seed + cell_idx * (n_target + n_target * MAX_RETRIES) + attempt
                        # FIX (2026-02-16): add_sky_noise now passed through
                        result = inject_sis_shear(
                            host_nmgy_hwc=host_hwc_torch,
                            lens=lens,
                            source=source,
                            pixel_scale=pixscale,
                            psf_fwhm_r_arcsec=eff_psf,
                            seed=inj_seed,
                            add_poisson_noise=add_poisson_noise,
                            gain_e_per_nmgy=gain_e_per_nmgy,
                            add_sky_noise=add_sky_noise,
                        )

                        # Compute arc annulus SNR diagnostic
                        sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                            host_psfdepth, eff_psf, pixscale
                        )
                        injection_chw = result.injection_only[0]  # (3, H, W)
                        snr_val = arc_annulus_snr(injection_chw, sigma_pix_r)

                        # Lensed (observed) arc magnitude from total injected flux
                        total_lensed_flux_r = float(injection_chw[1].sum().item())
                        if total_lensed_flux_r > 0:
                            lensed_r_mag = AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                        else:
                            lensed_r_mag = float("nan")

                        # Preprocess injected image for scoring
                        # Uses pp_kwargs from checkpoint for train-score consistency
                        injected_chw_np = result.injected[0].detach().cpu().numpy()  # (3, H, W)
                        _pp_inj = dict(pp_kwargs or {})
                        _pp_inj.setdefault("mode", preprocessing)
                        _pp_inj.setdefault("crop", crop)
                        _pp_inj.setdefault("clip_range", 10.0)
                        proc = preprocess_stack(injected_chw_np, **_pp_inj)

                        # SUCCESS — record this injection
                        batch_list.append(proc)
                        arc_snr_list.append(snr_val)
                        src_mag_list.append(src_r_mag)
                        lensed_mag_list.append(lensed_r_mag)

                        # Save cutout if requested
                        if save_cutouts and _cutout_dir is not None:
                            inj_id = f"cell{cell_idx:05d}_inj{len(batch_list)-1:05d}"
                            np.savez_compressed(
                                os.path.join(_cutout_dir, f"{inj_id}.npz"),
                                injected=injected_chw_np.astype(np.float32),
                                injection_only=injection_chw.detach().cpu().numpy().astype(np.float32),
                            )
                            _cutout_meta.append({
                                "inj_id": inj_id,
                                "cell_idx": cell_idx,
                                "theta_e": theta_e,
                                "psf_bin": pb,
                                "depth_bin": db,
                                "src_r_mag": src_r_mag,
                                "lensed_r_mag": lensed_r_mag,
                                "arc_snr": snr_val,
                                "inj_seed": inj_seed,
                                "cutout_path": str(host_row.get(CUTOUT_PATH_COL, "?")),
                            })

                    except Exception as exc:
                        cell_n_failed += 1
                        total_inject_failed += 1
                        if len(failure_log) < 100:  # cap logged failures
                            failure_log.append({
                                "cell_idx": cell_idx,
                                "theta_e": theta_e,
                                "psf_bin": pb,
                                "depth_bin": db,
                                "attempt": attempt,
                                "error": str(exc),
                                "cutout_path": str(host_row.get(CUTOUT_PATH_COL, "?")),
                            })
                        # Retry with a different host (loop continues)

                n_ok = len(batch_list)
                total_inject_ok += n_ok

                if cell_n_failed > 0:
                    print(f"\n  WARNING: Cell {cell_idx} "
                          f"[theta_E={theta_e}, PSF={pb}, depth={db}]: "
                          f"{cell_n_failed} injection failures, "
                          f"{n_ok}/{n_target} successful")

                # Score only successful injections
                if n_ok > 0:
                    batch = np.stack(batch_list, axis=0)
                    arc_snrs = np.array(arc_snr_list, dtype=np.float64)
                    source_r_mags = np.array(src_mag_list, dtype=np.float64)
                    lensed_r_mags = np.array(lensed_mag_list, dtype=np.float64)
                    scores = score_batch(model, batch, device)
                else:
                    scores = np.array([], dtype=np.float64)
                    arc_snrs = np.array([], dtype=np.float64)
                    source_r_mags = np.array([], dtype=np.float64)
                    lensed_r_mags = np.array([], dtype=np.float64)

                # Emit one row per threshold (+ per source-mag bin)
                src_mag_bins = [(23.0, 24.0), (24.0, 25.0), (25.0, 26.0)]

                for thr in thresholds:
                    if n_ok > 0:
                        detected = scores >= thr
                        k = int(detected.sum())
                        comp = k / n_ok
                        lo, hi = bayes_binomial_interval(k, n_ok, level=0.68)
                        lo95, hi95 = bayes_binomial_interval(k, n_ok, level=0.95)
                    else:
                        detected = np.array([], dtype=bool)
                        k = 0
                        comp = float("nan")
                        lo = hi = float("nan")

                    valid_snrs = arc_snrs[np.isfinite(arc_snrs)] if n_ok > 0 else np.array([])
                    mean_snr = float(valid_snrs.mean()) if len(valid_snrs) > 0 else float("nan")

                    # If this threshold was derived from a FPR target, record it
                    fpr_label = thr_to_fpr.get(thr, None)
                    threshold_type = f"FPR={fpr_label:.1e}" if fpr_label is not None else "fixed"

                    rows.append({
                        "theta_e": theta_e,
                        "psf_fwhm": pb,
                        "depth_5sig": db,
                        "threshold": thr,
                        "threshold_type": threshold_type,
                        "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                        "source_mag_bin": "all",
                        "n_injections": n_ok,
                        "n_detected": k,
                        "n_failed": cell_n_failed,
                        "completeness": float(comp),
                        "ci68_lo": lo,
                        "ci68_hi": hi,
                        "ci95_lo": lo95,
                        "ci95_hi": hi95,
                        "mean_score": float(scores.mean()) if n_ok > 0 else float("nan"),
                        "mean_arc_snr": mean_snr,
                        "sufficient": n_ok >= injections_per_cell,
                    })

                    # Source-magnitude stratified completeness
                    for mag_lo, mag_hi in src_mag_bins:
                        in_bin = (source_r_mags >= mag_lo) & (source_r_mags < mag_hi) if n_ok > 0 else np.array([], dtype=bool)
                        n_bin = int(in_bin.sum()) if n_ok > 0 else 0
                        if n_bin > 0:
                            k_bin = int((detected & in_bin).sum())
                            c_bin = k_bin / n_bin
                            lo_bin, hi_bin = bayes_binomial_interval(k_bin, n_bin, level=0.68)
                            lo95_bin, hi95_bin = bayes_binomial_interval(k_bin, n_bin, level=0.95)
                        else:
                            k_bin = 0
                            c_bin = float("nan")
                            lo_bin = hi_bin = float("nan")
                            lo95_bin = hi95_bin = float("nan")
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": f"{mag_lo:.0f}-{mag_hi:.0f}",
                            "n_injections": n_bin,
                            "n_detected": k_bin,
                            "n_failed": 0,  # failures not attributed to mag bins
                            "completeness": float(c_bin),
                            "ci68_lo": lo_bin,
                            "ci68_hi": hi_bin,
                            "ci95_lo": lo95_bin,
                            "ci95_hi": hi95_bin,
                            "mean_score": float(scores[in_bin].mean()) if n_bin > 0 else float("nan"),
                            "mean_arc_snr": float(arc_snrs[in_bin & np.isfinite(arc_snrs)].mean()) if n_ok > 0 and (in_bin & np.isfinite(arc_snrs)).sum() > 0 else float("nan"),
                            "sufficient": n_bin >= 10,
                        })

                    # Lensed (observed) magnitude stratified completeness
                    lensed_mag_bins = [
                        (18.0, 20.0), (20.0, 22.0), (22.0, 24.0), (24.0, 27.0),
                    ]
                    for lm_lo, lm_hi in lensed_mag_bins:
                        in_lbin = (lensed_r_mags >= lm_lo) & (lensed_r_mags < lm_hi) if n_ok > 0 else np.array([], dtype=bool)
                        n_lbin = int(in_lbin.sum()) if n_ok > 0 else 0
                        if n_lbin > 0:
                            k_lbin = int((detected & in_lbin).sum())
                            c_lbin = k_lbin / n_lbin
                            lo_lbin, hi_lbin = bayes_binomial_interval(k_lbin, n_lbin, level=0.68)
                            lo95_lbin, hi95_lbin = bayes_binomial_interval(k_lbin, n_lbin, level=0.95)
                        else:
                            k_lbin = 0
                            c_lbin = float("nan")
                            lo_lbin = hi_lbin = float("nan")
                            lo95_lbin = hi95_lbin = float("nan")
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": f"lensed_{lm_lo:.0f}-{lm_hi:.0f}",
                            "n_injections": n_lbin,
                            "n_detected": k_lbin,
                            "n_failed": 0,
                            "completeness": float(c_lbin),
                            "ci68_lo": lo_lbin,
                            "ci68_hi": hi_lbin,
                            "ci95_lo": lo95_lbin,
                            "ci95_hi": hi95_lbin,
                            "mean_score": float(scores[in_lbin].mean()) if n_lbin > 0 else float("nan"),
                            "mean_arc_snr": float(arc_snrs[in_lbin & np.isfinite(arc_snrs)].mean()) if n_ok > 0 and (in_lbin & np.isfinite(arc_snrs)).sum() > 0 else float("nan"),
                            "sufficient": n_lbin >= 10,
                        })

                if cell_idx % 10 == 0:
                    pct = 100 * cell_idx / n_cells
                    comp_display = comp if n_ok > 0 else float("nan")
                    print(f"  Cell {cell_idx}/{n_cells} ({pct:.0f}%) "
                          f"[theta_E={theta_e}, PSF={pb}, depth={db}] "
                          f"C(0.5)={comp_display:.3f}", end="\r")

    dt = time.time() - t0
    print(f"\n  Grid complete: {n_cells} cells, {dt:.1f}s ({dt/max(n_cells,1):.2f}s/cell)")
    print(f"  Total injections: {total_inject_ok:,} successful, "
          f"{total_inject_failed:,} failed "
          f"({100.0 * total_inject_failed / max(total_inject_ok + total_inject_failed, 1):.2f}% failure rate)")

    results_df = pd.DataFrame(rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "arch": arch,
        "epoch": epoch,
        "host_split": host_split,
        "host_max": host_max,
        "thresholds": thresholds,
        "injections_per_cell": injections_per_cell,
        "grid": {
            "theta_e": theta_es,
            "psf_fwhm": psf_bins,
            "depth_5sig": depth_bins,
        },
        "preprocessing": preprocessing,
        "crop": crop,
        "pixscale": pixscale,
        "seed": seed,
        "n_cells": n_cells,
        # BUGFIX: Filter to source_mag_bin='all' for cell counts.
        # Previous code counted across all mag sub-bins, inflating numbers.
        "n_sufficient_cells": int(results_df.loc[
            (results_df["threshold"] == thresholds[0]) &
            (results_df["source_mag_bin"] == "all"), "sufficient"
        ].sum()),
        "n_empty_cells": int(results_df.loc[
            (results_df["threshold"] == thresholds[0]) &
            (results_df["source_mag_bin"] == "all"), "n_injections"
        ].eq(0).sum()),
        "total_injections_ok": total_inject_ok,
        "total_injections_failed": total_inject_failed,
        "failure_rate_pct": round(100.0 * total_inject_failed / max(total_inject_ok + total_inject_failed, 1), 4),
        "max_retries_per_injection": MAX_RETRIES,
        "failure_log_sample": failure_log[:20],  # first 20 failures for debugging
        "fpr_targets": fpr_targets if fpr_targets else [],
        "fpr_derived_thresholds": {str(k): v for k, v in fpr_threshold_map.items()},
        "add_poisson_noise": add_poisson_noise,
        "gain_e_per_nmgy": gain_e_per_nmgy if add_poisson_noise else None,
        "add_sky_noise": add_sky_noise,
        "save_cutouts": save_cutouts,
        "save_cutouts_dir": _cutout_dir,
        "injection_engine": "SIE+shear (dhs.injection_engine)",
        "source_model": "Sersic (clumps disabled, clumps_prob=0.0)",
        "injection_overrides": injection_overrides if injection_overrides else {},
        "notes": [
            "Injection uses SIS+shear ray-shooting with Sersic source model.",
            "Flux units: nanomaggies (AB ZP=22.5) throughout.",
            "PSF blur via FFT-based Gaussian per band.",
            "Preprocessing: raw_robust outer-annulus median/MAD, clip [-10, 10].",
            "Scores are sigmoid(logit) from the frozen model.",
            "CIs are Bayesian binomial (Jeffreys prior, Beta(0.5, 0.5)).",
            "arc_annulus_snr reported per cell (mean over injections).",
            "Failed injections are retried with new hosts (up to MAX_RETRIES).",
            "Only successful injections count toward n_injections and completeness.",
            "n_failed column tracks failures per cell for transparency.",
        ],
    }

    # Save cutout metadata parquet alongside the npz files
    if save_cutouts and _cutout_meta:
        cutout_meta_df = pd.DataFrame(_cutout_meta)
        cutout_meta_path = os.path.join(_cutout_dir, "cutout_metadata.parquet")
        cutout_meta_df.to_parquet(cutout_meta_path, index=False)
        print(f"  Saved {len(_cutout_meta)} cutout metadata rows to {cutout_meta_path}")

    return results_df, metadata


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_outputs(
    results_df: pd.DataFrame,
    metadata: Dict[str, Any],
    out_dir: str,
) -> None:
    """Save CSV and JSON to local path or S3."""
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")

    csv_path = join_uri(out_dir, "selection_function.csv")
    json_path = join_uri(out_dir, "selection_function_meta.json")

    if is_s3_uri(out_dir):
        write_bytes(csv_path, csv_bytes, content_type="text/csv")
        write_bytes(json_path, json_bytes, content_type="application/json")
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        with open(json_path, "wb") as f:
            f.write(json_bytes)

    print(f"\nResults saved to: {csv_path}")
    print(f"Metadata saved to: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Selection Function: Injection-Recovery Grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/selection_function",
                    help="Output directory (local path or s3:// URI)")
    ap.add_argument("--host-split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--host-max", type=int, default=20000)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7],
                    help="Fixed detection thresholds (probability) to evaluate")
    ap.add_argument("--fpr-targets", nargs="*", type=float, default=None,
                    help="Target FPR values (e.g. 0.001 0.0001). Derives thresholds "
                         "by scoring negatives. Appended to --thresholds.")
    ap.add_argument("--injections-per-cell", type=int, default=500,
                    help="Injections per grid cell (default: 500). "
                         "Increased from 200: "
                         "at n=200 and p=3.5%%, 95%% CI is [1.7%%, 7.0%%] (too wide).")
    # Grid ranges
    ap.add_argument("--theta-e-min", type=float, default=0.5)
    ap.add_argument("--theta-e-max", type=float, default=3.0)
    ap.add_argument("--theta-e-step", type=float, default=0.25)
    ap.add_argument("--psf-min", type=float, default=0.9)
    ap.add_argument("--psf-max", type=float, default=1.8)
    ap.add_argument("--psf-step", type=float, default=0.15)
    ap.add_argument("--depth-min", type=float, default=22.5)
    ap.add_argument("--depth-max", type=float, default=24.5)
    ap.add_argument("--depth-step", type=float, default=0.5)
    # Processing
    ap.add_argument("--preprocessing", default="raw_robust")
    ap.add_argument("--crop", action="store_true", default=False)
    ap.add_argument("--pixscale", type=float, default=PIXEL_SCALE)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    # Injection realism
    ap.add_argument("--add-poisson-noise", action="store_true", default=False,
                    help="Add Poisson (shot) noise to injected arcs before adding to host. "
                         "Physically correct: real sources contribute photon noise. "
                         "Uses gain_e_per_nmgy=150 (approx DR10 coadd).")
    ap.add_argument("--gain-e-per-nmgy", type=float, default=150.0,
                    help="Gain in e-/nmgy for Poisson noise (default: 150)")
    ap.add_argument("--add-sky-noise", action="store_true", default=False,
                    help="Add realistic sky noise texture to injected arcs, "
                         "matching the host background noise level. "
                         "Without this, arcs appear unnaturally clean.")
    ap.add_argument("--save-cutouts", action="store_true", default=False,
                    help="Save all injection cutouts as .npz files for "
                         "reproducibility and downstream analysis.")
    ap.add_argument("--save-cutouts-dir", default=None,
                    help="Directory for saved cutouts (default: <checkpoint_dir>/grid_cutouts)")
    args = ap.parse_args()

    results_df, metadata = run_selection_function(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        host_split=args.host_split,
        host_max=args.host_max,
        thresholds=args.thresholds,
        fpr_targets=args.fpr_targets,
        injections_per_cell=args.injections_per_cell,
        theta_e_min=args.theta_e_min,
        theta_e_max=args.theta_e_max,
        theta_e_step=args.theta_e_step,
        psf_min=args.psf_min,
        psf_max=args.psf_max,
        psf_step=args.psf_step,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_step=args.depth_step,
        preprocessing=args.preprocessing,
        crop=args.crop,
        pixscale=args.pixscale,
        seed=args.seed,
        device_str=args.device,
        data_root=args.data_root,
        add_poisson_noise=args.add_poisson_noise,
        gain_e_per_nmgy=args.gain_e_per_nmgy,
        add_sky_noise=args.add_sky_noise,
        save_cutouts=args.save_cutouts,
        save_cutouts_dir=args.save_cutouts_dir,
    )

    # Save results
    save_outputs(results_df, metadata, args.out_dir)

    # Print summary
    # BUGFIX (D03 post-review): Filter to source_mag_bin='all' for marginal
    # completeness. Previous code averaged over ALL source_mag_bin types
    # (including lensed magnitude bins), giving inflated values.
    for thr in args.thresholds:
        # Use only the 'all' bin for marginal completeness
        thr_all = results_df[
            (results_df["threshold"] == thr) &
            (results_df["source_mag_bin"] == "all")
        ]
        valid_all = thr_all[thr_all["n_injections"] > 0]
        n_empty_host_cells = int(thr_all["n_injections"].eq(0).sum())

        print(f"\n{'='*60}")
        print(f"SELECTION FUNCTION SUMMARY (threshold={thr})")
        print(f"{'='*60}")
        print(f"  Model: {metadata['arch']}")
        print(f"  Grid cells: {metadata['n_cells']}")
        print(f"  Non-empty cells: {len(valid_all)}")
        print(f"  Empty cells (no hosts): {n_empty_host_cells}")

        # Injection-weighted marginal: total_detected / total_injected
        total_inj = int(valid_all["n_injections"].sum())
        total_det = int(valid_all["n_detected"].sum())
        mc = total_det / total_inj if total_inj > 0 else float("nan")
        print(f"  Marginal completeness: {mc:.4f} ({total_det}/{total_inj})")

        if len(valid_all) > 0:
            print(f"\n  Completeness by theta_E (source_mag_bin='all'):")
            for te in sorted(valid_all["theta_e"].unique()):
                te_rows = valid_all[valid_all["theta_e"] == te]
                te_inj = int(te_rows["n_injections"].sum())
                te_det = int(te_rows["n_detected"].sum())
                te_c = te_det / te_inj if te_inj > 0 else float("nan")
                snr = te_rows["mean_arc_snr"].dropna().mean()
                print(f"    theta_E={te:.2f}\": C={te_c:.4f} ({te_det}/{te_inj}), "
                      f"mean_arc_SNR={snr:.1f}")

            # Also show lensed-magnitude stratified completeness
            lensed_bins = results_df[
                (results_df["threshold"] == thr) &
                (results_df["source_mag_bin"].str.startswith("lensed_"))
            ]
            lensed_valid = lensed_bins[lensed_bins["n_injections"] > 0]
            if len(lensed_valid) > 0:
                print(f"\n  Completeness by lensed apparent magnitude:")
                for smb in sorted(lensed_valid["source_mag_bin"].unique()):
                    smb_rows = lensed_valid[lensed_valid["source_mag_bin"] == smb]
                    smb_inj = int(smb_rows["n_injections"].sum())
                    smb_det = int(smb_rows["n_detected"].sum())
                    smb_c = smb_det / smb_inj if smb_inj > 0 else float("nan")
                    print(f"    {smb}: C={smb_c:.4f} ({smb_det}/{smb_inj})")

        print(f"{'='*60}")


if __name__ == "__main__":
    main()
