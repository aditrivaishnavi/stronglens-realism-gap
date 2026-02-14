#!/usr/bin/env python3
"""
Recover injection cutouts from deterministic pipeline replay.

Replays the exact injection pipelines (same seeds, same hosts, same params)
that produced the paper's results, but this time saves each injection cutout
as a .npz file with full metadata.

Three modes corresponding to three experiments:
  --mode grid         : Selection function grid (110,000 injections, seed=1337)
  --mode bright_arc   : Bright-arc test (1,600 injections, seed=42)
  --mode linear_probe : Linear probe (1,000 injections, seed=42)

IMPORTANT: The replay logic must match the original scripts EXACTLY to produce
identical injections.  Any deviation in RNG consumption order will produce
different cutouts and invalidate the correspondence with published results.

Original scripts replayed:
  grid        -> selection_function_grid.py (lines 374-570)
  bright_arc  -> bright_arc_injection_test.py (lines 155-260)
  linear_probe -> feature_space_analysis.py (lines 225-300)

Safety features:
  - SHA256 manifest hash verification
  - Pre-flight scan of all host cutout files (hard-fail on missing)
  - Cell-level checkpointing for grid mode
  - Failure logging instead of silent swallowing

Usage (run on lambda3):
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
    export PYTHONPATH=.

    python scripts/recover_injection_cutouts.py --mode bright_arc \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out-dir results/recovered_injections/bright_arc

    python scripts/recover_injection_cutouts.py --mode linear_probe \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out-dir results/recovered_injections/linear_probe

    python scripts/recover_injection_cutouts.py --mode grid \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out-dir results/recovered_injections/grid

Date: 2026-02-16
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
)

CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
TIER_COL = "tier"
PIXEL_SCALE = 0.262

MAGNITUDE_BINS: List[Tuple[float, float]] = [
    (18.0, 19.0), (19.0, 20.0), (20.0, 21.0), (21.0, 22.0),
    (22.0, 23.0), (23.0, 24.0), (24.0, 25.0), (25.0, 26.0),
]


# ---------------------------------------------------------------------------
# Safety: manifest hash & pre-flight checks
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight_check_cutouts(paths: List[str], label: str = "hosts") -> None:
    """Verify all cutout .npz files exist and are readable. Hard-fail if any missing."""
    missing = []
    corrupt = []
    for p in paths:
        if not os.path.exists(p):
            missing.append(p)
        else:
            try:
                with np.load(p) as z:
                    arr = z["cutout"]
                    if arr.shape != (101, 101, 3):
                        corrupt.append((p, f"shape={arr.shape}"))
            except Exception as exc:
                corrupt.append((p, str(exc)))

    if missing or corrupt:
        msg_parts = []
        if missing:
            msg_parts.append(
                f"{len(missing)} missing {label} files (first 5): "
                + str(missing[:5])
            )
        if corrupt:
            msg_parts.append(
                f"{len(corrupt)} corrupt/unreadable {label} files (first 5): "
                + str(corrupt[:5])
            )
        raise RuntimeError(
            f"Pre-flight check FAILED for {label}. "
            + " | ".join(msg_parts)
            + "\nRecovery cannot proceed — missing files will corrupt RNG state."
        )
    print(f"  Pre-flight: all {len(paths)} {label} cutouts verified OK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, int, dict]:
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
def score_one(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


@torch.no_grad()
def score_batch(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def nearest_bin(val: float, grid: List[float]) -> float:
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


def scale_source_to_magnitude(source: SourceParams, target_mag: float) -> SourceParams:
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


def save_injection(out_dir: str, filename: str,
                   injected_chw: np.ndarray,
                   injection_only_chw: np.ndarray,
                   host_hwc: np.ndarray) -> str:
    """Save injection cutout as .npz. Returns the full path."""
    path = os.path.join(out_dir, filename)
    np.savez_compressed(
        path,
        injected_hwc=np.transpose(injected_chw, (1, 2, 0)),
        injection_only_chw=injection_only_chw,
        host_hwc=host_hwc,
    )
    return path


# ---------------------------------------------------------------------------
# Mode: bright_arc
# Replays bright_arc_injection_test.py run_bright_arc_test() exactly
# ---------------------------------------------------------------------------
def recover_bright_arc(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """Replay bright-arc test (seed=42, theta_E=1.5, beta_frac 0.1-0.55, 200 hosts x 8 bins).

    NOTE (2026-02-16): This function replays the ORIGINAL experiment's exact
    RNG sequence with beta_frac_range=(0.1, 0.55) — the value used when the
    experiment was first run. Changing this would break deterministic replay
    of old results. For NEW experiments with corrected priors, use
    bright_arc_injection_test.py which now defaults to (0.10, 0.40).
    """
    seed = 42
    theta_e = 1.5
    n_hosts = 200
    # Original experiment value — do NOT change for replay determinism
    beta_frac_range = (0.1, 0.55)

    print("=== Recovering bright_arc injections ===")
    print(f"  seed={seed}, theta_e={theta_e}, n_hosts={n_hosts}")
    print(f"  beta_frac_range={beta_frac_range}")

    rng = np.random.default_rng(seed)
    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(checkpoint_path, device)
    print(f"  Model: {arch}, epoch={epoch}, pp={pp_kwargs}")

    df = pd.read_parquet(manifest_path)
    neg = df[(df[SPLIT_COL] == "val") & (df[LABEL_COL] == 0)].copy()
    neg = neg.dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts = neg.sample(n=min(n_hosts, len(neg)), random_state=seed).reset_index(drop=True)
    n_sample = len(hosts)
    print(f"  Hosts sampled: {n_sample}")

    preflight_check_cutouts(
        hosts[CUTOUT_PATH_COL].astype(str).tolist(), label="bright_arc hosts"
    )

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)

    records = []
    total = 0

    for mag_lo, mag_hi in MAGNITUDE_BINS:
        bin_key = f"{mag_lo:.0f}-{mag_hi:.0f}"
        print(f"\n  Magnitude bin {bin_key}")

        for i, (_, host_row) in enumerate(hosts.iterrows()):
            try:
                with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception:
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])
            host_psfdepth = float(host_row["psfdepth_r"])

            lens = sample_lens_params(rng, theta_e)
            source = sample_source_params(rng, theta_e, beta_frac_range=beta_frac_range)
            target_mag = float(rng.uniform(mag_lo, mag_hi))
            source = scale_source_to_magnitude(source, target_mag)

            result = inject_sis_shear(
                host_nmgy_hwc=host_t,
                lens=lens,
                source=source,
                pixel_scale=PIXEL_SCALE,
                psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=None,
                seed=seed + i,
            )

            inj_chw = result.injected[0].numpy()
            inj_only_chw = result.injection_only[0].numpy()
            proc = preprocess_stack(inj_chw, **pp_kwargs)
            cnn_score = score_one(model, proc, device)

            sigma_pix_r = estimate_sigma_pix_from_psfdepth(host_psfdepth, host_psf, PIXEL_SCALE)
            snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)

            total_lensed_flux_r = float(inj_only_chw[1].sum())
            lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                           if total_lensed_flux_r > 0 else float("nan"))

            filename = f"bright_arc_{bin_key}_host{i:04d}.npz"
            save_injection(cutout_dir, filename, inj_chw, inj_only_chw, hwc)

            records.append({
                "injection_id": f"bright_arc_{bin_key}_{i:04d}",
                "experiment": "bright_arc",
                "cutout_filename": filename,
                "host_cutout_path": str(host_row[CUTOUT_PATH_COL]),
                "theta_e": theta_e,
                "psf_fwhm": host_psf,
                "depth_5sig": float("nan"),
                "cnn_score": cnn_score,
                "arc_snr": float(snr),
                "source_r_mag": target_mag,
                "lensed_r_mag": lensed_r_mag,
                "beta_frac": compute_beta_frac(source, theta_e),
                "source_re": source.re_arcsec,
                "source_n_sersic": source.n_sersic,
                "source_flux_r": source.flux_nmgy_r,
                "lens_q": lens.q_lens,
                "mag_bin": bin_key,
            })
            total += 1

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{n_sample}", end="\r")

        print(f"    Bin {bin_key} done, {total} total saved")

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    print(f"\n  Saved {total} bright_arc injections")
    return meta_df


# ---------------------------------------------------------------------------
# Mode: linear_probe
# Replays feature_space_analysis.py extract_embeddings_from_injections()
# ---------------------------------------------------------------------------
def recover_linear_probe(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """Replay linear probe injections (seed=42, theta_E=1.5, mag=19.0, 500+500)."""
    seed = 42
    theta_e = 1.5
    target_mag = 19.0
    n_samples = 500

    print("=== Recovering linear_probe injections ===")
    print(f"  seed={seed}, theta_e={theta_e}, target_mag={target_mag}, n_per_cat={n_samples}")

    rng = np.random.default_rng(seed)
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    print(f"  pp_kwargs={pp_kwargs}")

    df = pd.read_parquet(manifest_path)
    val_df = df[df["split"] == "val"].copy()

    neg_df = val_df[val_df["label"] == 0].dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts_df = neg_df.sample(n=min(2000, len(neg_df)), random_state=seed + 1)

    preflight_check_cutouts(
        hosts_df["cutout_path"].astype(str).tolist(), label="linear_probe hosts"
    )

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)

    records = []
    total = 0

    for cat_name, bf_range, seed_offset in [
        ("low_bf", (0.1, 0.3), 0),
        ("high_bf", (0.7, 1.0), 10000),
    ]:
        print(f"\n  Category: {cat_name}, beta_frac_range={bf_range}")
        host_indices = rng.choice(len(hosts_df), size=n_samples, replace=True)

        for i, host_idx in enumerate(host_indices):
            host_row = hosts_df.iloc[host_idx]
            try:
                with np.load(str(host_row["cutout_path"])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception:
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])

            lens = sample_lens_params(rng, theta_e)
            source = sample_source_params(rng, theta_e, beta_frac_range=bf_range)

            target_flux = 10.0 ** ((22.5 - target_mag) / 2.5)
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
                host_nmgy_hwc=host_t,
                lens=lens,
                source=source,
                pixel_scale=PIXEL_SCALE,
                psf_fwhm_r_arcsec=host_psf,
                seed=42 + seed_offset + i,
            )

            inj_chw = result.injected[0].numpy()
            inj_only_chw = result.injection_only[0].numpy()
            proc = preprocess_stack(inj_chw, **pp_kwargs)
            cnn_score = score_one(model, proc, device)

            total_lensed_flux_r = float(inj_only_chw[1].sum())
            lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                           if total_lensed_flux_r > 0 else float("nan"))

            filename = f"linear_probe_{cat_name}_{i:04d}.npz"
            save_injection(cutout_dir, filename, inj_chw, inj_only_chw, hwc)

            records.append({
                "injection_id": f"linear_probe_{cat_name}_{i:04d}",
                "experiment": "linear_probe",
                "cutout_filename": filename,
                "host_cutout_path": str(host_row["cutout_path"]),
                "theta_e": theta_e,
                "psf_fwhm": host_psf,
                "depth_5sig": float("nan"),
                "cnn_score": cnn_score,
                "arc_snr": float("nan"),
                "source_r_mag": target_mag,
                "lensed_r_mag": lensed_r_mag,
                "beta_frac": compute_beta_frac(source, theta_e),
                "source_re": source.re_arcsec,
                "source_n_sersic": source.n_sersic,
                "source_flux_r": source.flux_nmgy_r,
                "lens_q": lens.q_lens,
                "mag_bin": cat_name,
            })
            total += 1

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n_samples}", end="\r")

        print(f"    {cat_name} done")

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    print(f"\n  Saved {total} linear_probe injections")
    return meta_df


# ---------------------------------------------------------------------------
# Mode: grid
# Replays selection_function_grid.py run_selection_function() injection loop
# ---------------------------------------------------------------------------
def recover_grid(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """Replay selection function grid (seed=1337, 500/cell, 220 cells -> 110,000)."""
    seed = 1337
    host_max = 20000
    injections_per_cell = 500
    MAX_RETRIES = 5
    preprocessing = "raw_robust"
    crop = False

    theta_e_min, theta_e_max, theta_e_step = 0.5, 3.0, 0.25
    psf_min, psf_max, psf_step = 0.9, 1.8, 0.15
    depth_min, depth_max, depth_step = 22.5, 24.5, 0.5

    print("=== Recovering grid injections ===")
    print(f"  seed={seed}, injections_per_cell={injections_per_cell}")

    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(checkpoint_path, device)
    print(f"  Model: {arch}, epoch={epoch}, pp={pp_kwargs}")

    df = pd.read_parquet(manifest_path)
    hosts = df[(df[SPLIT_COL] == "val") & (df[LABEL_COL] == 0)].copy()
    for col in ["psfsize_r", "psfdepth_r"]:
        if col not in hosts.columns:
            raise ValueError(f"Missing column '{col}'")

    rng = np.random.default_rng(seed)
    if len(hosts) > host_max:
        hosts = hosts.sample(n=host_max, random_state=seed).reset_index(drop=True)
    print(f"  Hosts: {len(hosts)}")

    preflight_check_cutouts(
        hosts[CUTOUT_PATH_COL].astype(str).tolist(), label="grid hosts"
    )

    theta_es = np.round(np.arange(theta_e_min, theta_e_max + 1e-9, theta_e_step), 4).tolist()
    psf_bins = np.round(np.arange(psf_min, psf_max + 1e-9, psf_step), 4).tolist()
    depth_bins = np.round(np.arange(depth_min, depth_max + 1e-9, depth_step), 4).tolist()
    n_cells = len(theta_es) * len(psf_bins) * len(depth_bins)
    print(f"  Grid: {len(theta_es)} x {len(psf_bins)} x {len(depth_bins)} = {n_cells} cells")

    depth_mag_arr = m5_from_psfdepth(hosts["psfdepth_r"].to_numpy(dtype=np.float64))
    hosts["depth_mag"] = depth_mag_arr
    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest_bin(float(v), psf_bins))
    hosts["depth_bin"] = hosts["depth_mag"].apply(lambda v: nearest_bin(float(v), depth_bins))

    host_groups: Dict[Tuple[float, float], pd.DataFrame] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        host_groups[(float(pb), float(db))] = g

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)

    # Checkpointing: load existing records if resuming
    checkpoint_path_parquet = os.path.join(out_dir, "metadata_checkpoint.parquet")
    records = []
    completed_cells = set()
    if os.path.exists(checkpoint_path_parquet):
        existing = pd.read_parquet(checkpoint_path_parquet)
        records = existing.to_dict("records")
        for r in records:
            cid = r["injection_id"].split("_i")[0]
            completed_cells.add(cid)
        print(f"  Resuming from checkpoint: {len(records)} records, "
              f"{len(completed_cells)} cells completed")

    cell_idx = 0
    total = len(records)
    n_failures = 0
    t0 = time.time()

    for theta_e in theta_es:
        for pb in psf_bins:
            for db in depth_bins:
                cell_idx += 1
                cell_id = f"grid_c{cell_idx:04d}"

                if cell_id in completed_cells:
                    continue

                host_df = host_groups.get((pb, db))

                if host_df is None or len(host_df) == 0:
                    continue

                n_target = injections_per_cell
                n_saved_this_cell = 0
                cell_failures = 0
                attempt = 0

                while n_saved_this_cell < n_target and attempt < n_target + n_target * MAX_RETRIES:
                    attempt += 1
                    host_idx = int(rng.choice(len(host_df)))
                    host_row = host_df.iloc[host_idx]

                    try:
                        with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                            host_hwc = z["cutout"].astype(np.float32)

                        host_hwc_torch = torch.from_numpy(host_hwc).float()
                        host_psf = float(host_row["psfsize_r"])
                        host_psfdepth = float(host_row["psfdepth_r"])
                        host_depth_mag = float(host_row.get("depth_mag", float("nan")))

                        lens = sample_lens_params(rng, theta_e)
                        source = sample_source_params(rng, theta_e)

                        src_r_mag = float("nan")
                        if source.flux_nmgy_r > 0:
                            src_r_mag = AB_ZP - 2.5 * math.log10(source.flux_nmgy_r)

                        inj_seed = seed + cell_idx * (n_target + n_target * MAX_RETRIES) + attempt
                        result = inject_sis_shear(
                            host_nmgy_hwc=host_hwc_torch,
                            lens=lens,
                            source=source,
                            pixel_scale=PIXEL_SCALE,
                            psf_fwhm_r_arcsec=host_psf,
                            seed=inj_seed,
                        )

                        sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                            host_psfdepth, host_psf, PIXEL_SCALE
                        )
                        injection_chw = result.injection_only[0]
                        snr_val = arc_annulus_snr(injection_chw, sigma_pix_r)

                        total_lensed_flux_r = float(injection_chw[1].sum().item())
                        lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                                       if total_lensed_flux_r > 0 else float("nan"))

                        inj_chw = result.injected[0].numpy()
                        inj_only_chw = injection_chw.numpy()

                        _pp_inj = dict(pp_kwargs or {})
                        _pp_inj.setdefault("mode", preprocessing)
                        _pp_inj.setdefault("crop", crop)
                        _pp_inj.setdefault("clip_range", 10.0)
                        proc = preprocess_stack(inj_chw, **_pp_inj)
                        cnn_score = score_one(model, proc, device)

                        filename = f"grid_c{cell_idx:04d}_i{n_saved_this_cell:04d}.npz"
                        save_injection(cutout_dir, filename, inj_chw, inj_only_chw, host_hwc)

                        records.append({
                            "injection_id": f"grid_c{cell_idx:04d}_i{n_saved_this_cell:04d}",
                            "experiment": "grid",
                            "cutout_filename": filename,
                            "host_cutout_path": str(host_row[CUTOUT_PATH_COL]),
                            "theta_e": theta_e,
                            "psf_fwhm_bin": pb,
                            "depth_5sig_bin": db,
                            "host_psfsize_r": host_psf,
                            "host_depth_mag": host_depth_mag,
                            "cnn_score": cnn_score,
                            "arc_snr": float(snr_val),
                            "source_r_mag": src_r_mag,
                            "lensed_r_mag": lensed_r_mag,
                            "beta_frac": compute_beta_frac(source, theta_e),
                            "source_re": source.re_arcsec,
                            "source_n_sersic": source.n_sersic,
                            "source_flux_r": source.flux_nmgy_r,
                            "lens_q": lens.q_lens,
                            "mag_bin": f"theta_e={theta_e}",
                        })
                        n_saved_this_cell += 1
                        total += 1

                    except Exception as exc:
                        cell_failures += 1
                        n_failures += 1
                        if n_failures <= 50:
                            print(f"    WARNING: Cell {cell_idx} attempt {attempt} failed: "
                                  f"{type(exc).__name__}: {exc}")

                if cell_failures > 0:
                    print(f"    Cell {cell_idx}: {cell_failures} failures, "
                          f"{n_saved_this_cell}/{n_target} saved")

                # Checkpoint after each cell
                elapsed = time.time() - t0
                if cell_idx % 10 == 0:
                    pd.DataFrame(records).to_parquet(checkpoint_path_parquet, index=False)
                    rate = total / elapsed if elapsed > 0 else 0
                    print(f"  Cell {cell_idx}/{n_cells}: {total} injections saved "
                          f"({rate:.0f}/s, {elapsed:.0f}s elapsed, {n_failures} failures)")

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    if os.path.exists(checkpoint_path_parquet):
        os.remove(checkpoint_path_parquet)
    elapsed = time.time() - t0
    print(f"\n  Saved {total} grid injections in {elapsed:.0f}s "
          f"({n_failures} total failures)")
    return meta_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Recover injection cutouts from deterministic replay")
    ap.add_argument("--mode", required=True, choices=["grid", "bright_arc", "linear_probe"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Manifest integrity check
    manifest_hash = sha256_file(args.manifest)
    print(f"Manifest SHA-256: {manifest_hash}")

    t_start = time.time()

    if args.mode == "bright_arc":
        meta = recover_bright_arc(args.checkpoint, args.manifest, args.out_dir, device)
    elif args.mode == "linear_probe":
        meta = recover_linear_probe(args.checkpoint, args.manifest, args.out_dir, device)
    elif args.mode == "grid":
        meta = recover_grid(args.checkpoint, args.manifest, args.out_dir, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    elapsed = time.time() - t_start
    print(f"\nDone. Mode={args.mode}, total time={elapsed:.0f}s")
    print(f"  Metadata: {os.path.join(args.out_dir, 'metadata.parquet')}")
    print(f"  Cutouts:  {os.path.join(args.out_dir, 'cutouts/')}")

    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "manifest_sha256": manifest_hash,
        "n_recovered": len(meta),
        "elapsed_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
