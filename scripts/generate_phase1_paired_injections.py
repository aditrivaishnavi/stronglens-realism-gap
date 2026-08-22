#!/usr/bin/env python3
"""Generate matched real-COSMOS-source and single-Sersic-twin injections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch

from dhs import data_splits
from dhs.core import sha256_file
from dhs.injection_engine import (
    LensParams,
    arc_annulus_snr,
    estimate_sigma_pix_from_cutout,
)
from dhs.pixelized_source import inject_pixelized_source_sie


def safe_path(root: Path, relative_path: str) -> Path:
    parsed = PurePosixPath(str(relative_path))
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"Unsafe relative path: {relative_path}")
    return root.joinpath(*parsed.parts)


def restore_lens(row: pd.Series) -> LensParams:
    return LensParams(
        **{field.name: row[f"lens_{field.name}"] for field in fields(LensParams)}
    )


def paired_filename(index: int, host_id: str) -> str:
    safe_host = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in host_id
    )
    return f"pair_{index:04d}_{safe_host}.npz"


def add_poisson_noise(
    injection: torch.Tensor,
    *,
    gain_e_per_nmgy: float,
    seed: int,
) -> torch.Tensor:
    if gain_e_per_nmgy <= 0:
        return injection
    generator = torch.Generator(device=injection.device)
    generator.manual_seed(int(seed))
    expected = injection.clamp(min=0.0) * gain_e_per_nmgy
    observed = torch.poisson(expected, generator=generator)
    return injection + (observed - expected) / gain_e_per_nmgy


def generate(
    *,
    injection_metadata: pd.DataFrame,
    source_manifest: pd.DataFrame,
    host_data_root: Path,
    source_root: Path,
    output_pixel_dir: Path,
    source_mag_min: float,
    source_mag_max: float,
    source_mag_seed: int,
) -> pd.DataFrame:
    if len(injection_metadata) != len(source_manifest):
        raise ValueError(
            "Paired generation requires one unique source pair per matched host"
        )
    hosts = injection_metadata.sort_values(
        ["real_galaxy_id", "host_replicate"],
        kind="mergesort",
    ).reset_index(drop=True)
    sources = source_manifest.sort_values("pair_index", kind="mergesort").reset_index(
        drop=True
    )
    if hosts["host_galaxy_id"].astype(str).duplicated().any():
        raise ValueError("Paired generation cannot reuse hosts")
    if not source_mag_min < source_mag_max:
        raise ValueError("Paired source magnitude range is invalid")
    magnitude_rng = np.random.default_rng(source_mag_seed)
    magnitude_fractions = (
        np.arange(len(hosts), dtype=np.float64) + 0.5
    ) / len(hosts)
    target_magnitudes = source_mag_min + magnitude_rng.permutation(
        magnitude_fractions
    ) * (source_mag_max - source_mag_min)
    records: list[dict[str, Any]] = []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for index in range(len(hosts)):
        host_row = hosts.iloc[index]
        source_row = sources.iloc[index]
        host_path = safe_path(
            host_data_root,
            str(host_row["host_cutout_relpath"]),
        )
        source_path = safe_path(
            source_root,
            str(source_row["source_map_relpath"]),
        )
        with np.load(host_path, allow_pickle=False) as payload:
            host_hwc = np.asarray(payload["cutout"], dtype=np.float32)
        with np.load(source_path, allow_pickle=False) as payload:
            real_source = np.asarray(payload["real_source"], dtype=np.float32)
            sersic_source = np.asarray(payload["sersic_twin"], dtype=np.float32)
        lens = restore_lens(host_row)
        target_source_mag_r = float(target_magnitudes[index])
        target_source_flux_r = 10.0 ** (
            (22.5 - target_source_mag_r) / 2.5
        )
        base_source_flux_r = float(host_row["source_flux_nmgy_r"])
        if base_source_flux_r <= 0:
            raise ValueError("Base source has non-positive r-band flux")
        source_flux_scale = target_source_flux_r / base_source_flux_r
        common = {
            "host_nmgy_hwc": torch.from_numpy(host_hwc),
            "source_pixel_scale_arcsec": float(
                source_row["source_pixel_scale_arcsec"]
            ),
            "source_center_x_arcsec": float(host_row["source_beta_x_arcsec"]),
            "source_center_y_arcsec": float(host_row["source_beta_y_arcsec"]),
            "unlensed_flux_nmgy_grz": (
                float(host_row["source_flux_nmgy_g"]) * source_flux_scale,
                target_source_flux_r,
                float(host_row["source_flux_nmgy_z"]) * source_flux_scale,
            ),
            "lens": lens,
            "target_pixel_scale_arcsec": float(host_row["pixel_scale_arcsec"]),
            "psf_fwhm_r_arcsec": float(host_row["psfsize_r_matched"]),
            "seed": int(host_row["injection_seed"]),
            "subpixel_oversample": int(host_row["subpixel_oversample"]),
            "add_poisson_noise": False,
            "gain_e_per_nmgy": float(host_row["gain_e_per_nmgy"]),
        }
        real_result = inject_pixelized_source_sie(
            source_surface_brightness=torch.from_numpy(real_source),
            **common,
        )
        sersic_result = inject_pixelized_source_sie(
            source_surface_brightness=torch.from_numpy(sersic_source),
            **common,
        )
        real_noiseless = real_result.injection_only[0]
        sersic_noiseless = sersic_result.injection_only[0]
        real_noiseless_flux_r = float(real_noiseless[1].sum().item())
        sersic_noiseless_flux_r = float(sersic_noiseless[1].sum().item())
        if real_noiseless_flux_r <= 0 or sersic_noiseless_flux_r <= 0:
            raise ValueError("Paired source produced non-positive lensed r flux")
        sersic_lensed_flux_scale = (
            real_noiseless_flux_r / sersic_noiseless_flux_r
        )
        sersic_noiseless = sersic_noiseless * sersic_lensed_flux_scale
        if bool(host_row["add_poisson_noise"]):
            real_rendered = add_poisson_noise(
                real_noiseless,
                gain_e_per_nmgy=float(host_row["gain_e_per_nmgy"]),
                seed=int(host_row["injection_seed"]),
            )
            sersic_rendered = add_poisson_noise(
                sersic_noiseless,
                gain_e_per_nmgy=float(host_row["gain_e_per_nmgy"]),
                seed=int(host_row["injection_seed"]),
            )
        else:
            real_rendered = real_noiseless
            sersic_rendered = sersic_noiseless
        host_chw = torch.from_numpy(host_hwc).permute(2, 0, 1)
        real_cutout = (
            (host_chw + real_rendered).detach().cpu().numpy().astype(np.float32)
        )
        sersic_cutout = (
            (host_chw + sersic_rendered).detach().cpu().numpy().astype(np.float32)
        )
        real_only = real_rendered.detach().cpu().numpy().astype(np.float32)
        sersic_only = sersic_rendered.detach().cpu().numpy().astype(np.float32)
        sigma_r = estimate_sigma_pix_from_cutout(
            np.transpose(host_hwc, (2, 0, 1)),
            band_idx=1,
        )
        real_snr = arc_annulus_snr(torch.from_numpy(real_only), sigma_r)
        sersic_snr = arc_annulus_snr(torch.from_numpy(sersic_only), sigma_r)
        filename = paired_filename(index, str(host_row["host_galaxy_id"]))
        output_path = output_pixel_dir / filename
        np.savez_compressed(
            output_path,
            real_source_cutout=np.transpose(real_cutout, (1, 2, 0)),
            sersic_twin_cutout=np.transpose(sersic_cutout, (1, 2, 0)),
            real_source_injection_only=np.transpose(real_only, (1, 2, 0)),
            sersic_twin_injection_only=np.transpose(sersic_only, (1, 2, 0)),
        )
        real_flux_r = float(real_only[1].sum(dtype=np.float64))
        sersic_flux_r = float(sersic_only[1].sum(dtype=np.float64))
        record = {
            "pair_index": int(index),
            "real_galaxy_id": str(host_row["real_galaxy_id"]),
            "host_galaxy_id": str(host_row["host_galaxy_id"]),
            "host_spatial_component_id": str(
                host_row["host_spatial_component_id"]
            ),
            "cosmos_catalog_index": int(source_row["cosmos_catalog_index"]),
            "cosmos_ident": str(source_row["cosmos_ident"]),
            "pair_relpath": f"pairs/{filename}",
            "pair_sha256": sha256_file(output_path),
            "host_sigma_r_nmgy": float(sigma_r),
            "real_source_arc_annulus_snr": float(real_snr),
            "sersic_twin_arc_annulus_snr": float(sersic_snr),
            "pair_primary_snr_eligible": bool(
                np.isfinite(real_snr)
                and np.isfinite(sersic_snr)
                and min(real_snr, sersic_snr)
                >= float(host_row["primary_snr_minimum"])
            ),
            "real_source_lensed_flux_r_nmgy": real_flux_r,
            "sersic_twin_lensed_flux_r_nmgy": sersic_flux_r,
            "real_source_noiseless_lensed_flux_r_nmgy": (
                real_noiseless_flux_r
            ),
            "sersic_twin_noiseless_lensed_flux_r_before_match_nmgy": (
                sersic_noiseless_flux_r
            ),
            "sersic_twin_lensed_flux_match_scale": float(
                sersic_lensed_flux_scale
            ),
            "real_source_lensed_mag_r": (
                float(22.5 - 2.5 * np.log10(real_flux_r))
                if real_flux_r > 0
                else None
            ),
            "sersic_twin_lensed_mag_r": (
                float(22.5 - 2.5 * np.log10(sersic_flux_r))
                if sersic_flux_r > 0
                else None
            ),
            "unlensed_source_mag_r": target_source_mag_r,
            "lens_theta_e_arcsec": float(host_row["lens_theta_e_arcsec"]),
            "source_center_x_arcsec": float(host_row["source_beta_x_arcsec"]),
            "source_center_y_arcsec": float(host_row["source_beta_y_arcsec"]),
            "injection_seed": int(host_row["injection_seed"]),
        }
        records.append(record)
        if (index + 1) % 25 == 0 or index + 1 == len(hosts):
            print(f"Paired injection generation: {index + 1}/{len(hosts)}", flush=True)
    return pd.DataFrame(records)


def aggregate_digest(metadata: pd.DataFrame) -> str:
    value = hashlib.sha256()
    for row in metadata.sort_values("pair_relpath")[
        ["pair_relpath", "pair_sha256"]
    ].itertuples(index=False):
        value.update(f"{row.pair_relpath}\t{row.pair_sha256}\n".encode())
    return value.hexdigest()


def materialize(
    *,
    injection_metadata_path: Path,
    source_manifest_path: Path,
    host_data_root: Path,
    source_root: Path,
    output_dir: Path,
    source_mag_min: float,
    source_mag_max: float,
    source_mag_seed: int,
    test_manifest: Path | str = data_splits.DEFAULT_TEST_MANIFEST,
    allow_test_rows: bool = False,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace paired injections: {output_dir}")
    injection_metadata = pd.read_parquet(injection_metadata_path)
    source_manifest = pd.read_parquet(source_manifest_path)
    # Before any pixel is rendered: this is the last stage that could decline to
    # write a test host into the artifact the detector will later score.
    split_check = data_splits.check_selections(
        [
            data_splits.Selection(
                "injection_hosts", injection_metadata, "host_galaxy_id"
            ),
            data_splits.Selection(
                "injection_reals", injection_metadata, "real_galaxy_id"
            ),
        ],
        purpose=f"Phase 1 paired injections from {injection_metadata_path}",
        allow_test_rows=allow_test_rows,
        test_manifest=test_manifest,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        pixels = temporary / "pairs"
        pixels.mkdir()
        metadata = generate(
            injection_metadata=injection_metadata,
            source_manifest=source_manifest,
            host_data_root=host_data_root,
            source_root=source_root,
            output_pixel_dir=pixels,
            source_mag_min=source_mag_min,
            source_mag_max=source_mag_max,
            source_mag_seed=source_mag_seed,
        )
        metadata_path = temporary / "paired_injection_metadata.parquet"
        metadata.to_parquet(metadata_path, index=False)
        eligible = metadata.loc[
            metadata["pair_primary_snr_eligible"].astype(bool)
        ]
        summary = {
            "status": "pass" if len(eligible) >= 53 else "insufficient_pair_snr",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_is_cnn_score_blind": True,
            "split_check": split_check,
            "generated_pairs": int(len(metadata)),
            "primary_snr_eligible_pairs": int(len(eligible)),
            "real_lenses_represented_by_eligible_pairs": int(
                eligible["real_galaxy_id"].nunique()
            ),
            "pixel_inventory_aggregate_sha256": aggregate_digest(metadata),
            "paired_controls": {
                "same_host": True,
                "same_lens_geometry": True,
                "same_unlensed_band_fluxes_before_lensed_flux_matching": True,
                "same_noiseless_lensed_r_flux_after_matching": True,
                "same_psf": True,
                "same_poisson_seed": True,
                "primary_comparison": (
                    "source construction at matched noiseless lensed r-band flux"
                ),
            },
            "source_brightness_design": {
                "method": "permuted_midpoint_latin_hypercube",
                "unlensed_source_mag_r": [
                    float(source_mag_min),
                    float(source_mag_max),
                ],
                "seed": int(source_mag_seed),
                "purpose": (
                    "Bright, SNR-supported subset of the corrected 23-26 "
                    "magnitude source prior"
                ),
            },
            "limitation": (
                "The real-source member retains residual information from "
                "the noisy HST stamp; a residual/noise sensitivity must "
                "accompany any morphology-causal interpretation."
            ),
            "inputs": {
                "injection_metadata": str(injection_metadata_path),
                "injection_metadata_sha256": sha256_file(
                    injection_metadata_path
                ),
                "source_manifest": str(source_manifest_path),
                "source_manifest_sha256": sha256_file(source_manifest_path),
            },
        }
        summary_path = temporary / "generation_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        hashes = {
            metadata_path.name: sha256_file(metadata_path),
            summary_path.name: sha256_file(summary_path),
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{value}  {name}\n" for name, value in sorted(hashes.items())
            )
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injection-metadata", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--host-data-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-mag-min", type=float, default=23.0)
    parser.add_argument("--source-mag-max", type=float, default=24.0)
    parser.add_argument("--source-mag-seed", type=int, default=20260822)
    data_splits.add_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize(
        injection_metadata_path=args.injection_metadata,
        source_manifest_path=args.source_manifest,
        host_data_root=args.host_data_root,
        source_root=args.source_root,
        output_dir=args.output_dir,
        source_mag_min=args.source_mag_min,
        source_mag_max=args.source_mag_max,
        source_mag_seed=args.source_mag_seed,
        test_manifest=args.test_manifest,
        allow_test_rows=args.allow_test_rows,
    )
    print(f"Generated paired injections: {args.output_dir}")


if __name__ == "__main__":
    main()
