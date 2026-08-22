#!/usr/bin/env python3
"""Prepare provenance-bound COSMOS real-source and fitted-Sersic twins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import galsim
import numpy as np
import pandas as pd
from astropy.io import fits

from dhs.core import sha256_file

ZENODO_RECORD = "https://doi.org/10.5281/zenodo.3242143"
EXPECTED_ARCHIVE_MD5 = "e05cfe60c037c645d61ac70545cc2a99"
SOURCE_PIXEL_SCALE_ARCSEC = 0.03


def row_seed(seed: int, catalog_index: int) -> int:
    value = hashlib.sha256(f"{seed}:{catalog_index}".encode()).digest()
    return int.from_bytes(value[:4], "big")


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def candidate_indices(fit_catalog: np.ndarray) -> np.ndarray:
    sersic = np.asarray(fit_catalog["SERSICFIT"], dtype=np.float64)
    status = np.asarray(fit_catalog["FIT_STATUS"])
    viable = np.asarray(fit_catalog["VIABLE_SERSIC"]).astype(bool)
    magnitude = np.asarray(fit_catalog["MAG_AUTO"], dtype=np.float64)
    half_light_radius = (
        SOURCE_PIXEL_SCALE_ARCSEC
        * np.sqrt(sersic[:, 3])
        * sersic[:, 1]
    )
    return np.flatnonzero(
        viable
        & ~np.isin(status[:, 4], [0, 5])
        & np.isfinite(sersic).all(axis=1)
        & (magnitude >= 23.0)
        & (magnitude <= 25.2)
        & (half_light_radius >= 0.15)
        & (half_light_radius <= 0.50)
        & (sersic[:, 2] >= 0.5)
        & (sersic[:, 2] <= 2.0)
        & (sersic[:, 3] >= 0.3)
        & (sersic[:, 3] <= 1.0)
    )


def normalized_map(
    profile: galsim.GSObject,
    *,
    stamp_size: int,
    pixel_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    image = profile.withFlux(1.0).drawImage(
        nx=stamp_size,
        ny=stamp_size,
        scale=pixel_scale,
        method="no_pixel",
    )
    raw = np.asarray(image.array, dtype=np.float64)
    negative_flux = float(-raw[raw < 0].sum())
    positive = np.maximum(raw, 0.0)
    total = float(positive.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Rendered source has no positive finite flux")
    normalized = (positive / total).astype(np.float32)
    return normalized, {
        "raw_negative_pixel_fraction": float(np.mean(raw < 0)),
        "raw_negative_to_positive_flux_ratio": float(
            negative_flux / max(float(positive.sum()), 1e-12)
        ),
    }


def make_sersic_profile(sersic_fit: np.ndarray) -> galsim.GSObject:
    major_hlr_pixels = float(sersic_fit[1])
    n_sersic = float(sersic_fit[2])
    axis_ratio = float(sersic_fit[3])
    phi = float(sersic_fit[7])
    circularized_hlr = (
        SOURCE_PIXEL_SCALE_ARCSEC
        * np.sqrt(axis_ratio)
        * major_hlr_pixels
    )
    return galsim.Sersic(
        n=n_sersic,
        half_light_radius=circularized_hlr,
    ).shear(q=axis_ratio, beta=phi * galsim.radians)


def prepare(
    *,
    cosmos_dir: Path,
    archive_path: Path,
    output_dir: Path,
    count: int,
    seed: int,
    stamp_size: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace paired sources: {output_dir}")
    if file_digest(archive_path, "md5") != EXPECTED_ARCHIVE_MD5:
        raise ValueError("COSMOS archive MD5 does not match the Zenodo record")
    fit_path = cosmos_dir / "real_galaxy_catalog_25.2_fits.fits"
    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)
    fit_catalog = fits.getdata(fit_path, 1)
    eligible_original = candidate_indices(fit_catalog)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(cosmos_dir))
    usable_original_indices = np.asarray(catalog.orig_index, dtype=np.int64)
    eligible_catalog = np.flatnonzero(
        np.isin(usable_original_indices, eligible_original)
    )
    if len(eligible_catalog) < count:
        raise ValueError(
            f"Only {len(eligible_catalog)} viable usable COSMOS sources "
            f"for requested {count}"
        )
    rng = np.random.default_rng(seed)
    selected_catalog = np.sort(
        rng.choice(eligible_catalog, size=count, replace=False)
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        maps_dir = temporary / "source_maps"
        maps_dir.mkdir()
        records: list[dict[str, Any]] = []
        for position, catalog_index in enumerate(selected_catalog):
            original_index = int(usable_original_indices[catalog_index])
            fit = np.asarray(
                fit_catalog["SERSICFIT"][original_index],
                dtype=np.float64,
            )
            rotation = np.random.default_rng(
                row_seed(seed, original_index)
            ).uniform(0.0, 2.0 * np.pi)
            angle = float(rotation) * galsim.radians
            real_profile = catalog.makeGalaxy(
                index=int(catalog_index),
                gal_type="real",
                noise_pad_size=0,
            ).rotate(angle)
            sersic_profile = make_sersic_profile(fit).rotate(angle)
            real_map, real_diagnostics = normalized_map(
                real_profile,
                stamp_size=stamp_size,
                pixel_scale=SOURCE_PIXEL_SCALE_ARCSEC,
            )
            sersic_map, sersic_diagnostics = normalized_map(
                sersic_profile,
                stamp_size=stamp_size,
                pixel_scale=SOURCE_PIXEL_SCALE_ARCSEC,
            )
            filename = (
                f"source_{position:04d}_catalog_{catalog_index}"
                f"_original_{original_index}.npz"
            )
            path = maps_dir / filename
            np.savez_compressed(
                path,
                real_source=real_map,
                sersic_twin=sersic_map,
            )
            correlation = float(
                np.corrcoef(real_map.ravel(), sersic_map.ravel())[0, 1]
            )
            record = {
                "pair_index": int(position),
                "cosmos_catalog_index": int(catalog_index),
                "cosmos_original_index": original_index,
                "cosmos_ident": str(fit_catalog["IDENT"][original_index]),
                "f814w_mag_auto": float(
                    fit_catalog["MAG_AUTO"][original_index]
                ),
                "source_map_relpath": f"source_maps/{filename}",
                "source_map_sha256": sha256_file(path),
                "source_pixel_scale_arcsec": SOURCE_PIXEL_SCALE_ARCSEC,
                "source_stamp_size": int(stamp_size),
                "rotation_rad": float(rotation),
                "sersic_n": float(fit[2]),
                "sersic_q": float(fit[3]),
                "sersic_phi_catalog_rad": float(fit[7]),
                "sersic_circularized_hlr_arcsec": float(
                    SOURCE_PIXEL_SCALE_ARCSEC
                    * np.sqrt(fit[3])
                    * fit[1]
                ),
                "real_twin_pixel_correlation": correlation,
                **{
                    f"real_{key}": value
                    for key, value in real_diagnostics.items()
                },
                **{
                    f"sersic_{key}": value
                    for key, value in sersic_diagnostics.items()
                },
            }
            records.append(record)
            if (position + 1) % 25 == 0 or position + 1 == count:
                print(f"Paired source preparation: {position + 1}/{count}", flush=True)
        manifest = pd.DataFrame(records)
        manifest_path = temporary / "paired_source_manifest.parquet"
        manifest.to_parquet(manifest_path, index=False)
        summary = {
            "status": "prepared_paired_real_source_and_single_sersic_twins",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_is_cnn_score_blind": True,
            "seed": int(seed),
            "pairs": int(len(manifest)),
            "selection": {
                "eligible_original_fit_rows": int(len(eligible_original)),
                "eligible_after_real_stamp_quality_cuts": int(
                    len(eligible_catalog)
                ),
                "f814w_mag_auto": [23.0, 25.2],
                "single_sersic_n": [0.5, 2.0],
                "single_sersic_axis_ratio": [0.3, 1.0],
                "circularized_half_light_radius_arcsec": [0.15, 0.50],
                "fit_required_viable": True,
            },
            "rendering": {
                "source_pixel_scale_arcsec": SOURCE_PIXEL_SCALE_ARCSEC,
                "stamp_size": int(stamp_size),
                "same_catalog_object": True,
                "fitted_arm_uses_catalog_orientation": True,
                "same_additional_random_rotation": True,
                "same_working_grid_and_lens_source_map_origin": True,
                "realized_flux_centroids_explicitly_aligned": False,
                "each_map_positive_clipped_then_unit_flux_normalized": True,
            },
            "known_limit": (
                "RealGalaxy profiles originate from noisy HST stamps. Target "
                "seeing suppresses much of that noise, but a residual/noise "
                "control is required before attributing pair differences "
                "solely to intrinsic morphology."
            ),
            "inputs": {
                "zenodo_record": ZENODO_RECORD,
                "archive": str(archive_path),
                "archive_md5": EXPECTED_ARCHIVE_MD5,
                "fit_catalog": str(fit_path),
                "fit_catalog_sha256": sha256_file(fit_path),
                "galsim_version": galsim.__version__,
            },
        }
        summary_path = temporary / "source_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        inventory_digest = hashlib.sha256()
        for path in sorted(maps_dir.iterdir()):
            inventory_digest.update(
                f"{path.name}\t{sha256_file(path)}\n".encode()
            )
        summary["source_map_inventory_sha256"] = inventory_digest.hexdigest()
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        hashes = {
            manifest_path.name: sha256_file(manifest_path),
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
    parser.add_argument("--cosmos-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=265)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--stamp-size", type=int, default=161)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(
        cosmos_dir=args.cosmos_dir,
        archive_path=args.archive,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        stamp_size=args.stamp_size,
    )
    print(f"Published Phase 1 paired source inputs: {args.output_dir}")


if __name__ == "__main__":
    main()
