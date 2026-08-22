#!/usr/bin/env python3
"""Prepare COSMOS and fitted-Sersic source surface-brightness maps.

The script preserves the selected catalogue objects, order, and rotations from
the input manifest. It exposes source-noise, support, and control choices as
explicit switches.

The treatments are switches rather than defaults, because each one is a factor
in the probe design and has to be varied independently:

``--twin-noise catalogue``  give the twin a realisation of the source's own
                            correlated-noise model at its native variance, so
                            noise texture cannot separate the arms.
``--twin-noise none``       leave the twin noiseless, as the production
                            pipeline's analytic sources are.
``--support matched``       mask both arms to the real stamp's own support.
                            Required for the headline, not cosmetic: clipping
                            the sky at zero deposits a positive pedestal
                            proportional to the area carrying noise, so unequal
                            support makes that pedestal asymmetric by a third
                            of the arm's flux.
``--support full``          leave the twin's noise across the whole frame, as a
                            sensitivity arm.
``--arms twin_vs_twin``     the null control. Both arms become the same Sersic
                            profile under independent noise, so the morphology
                            difference is gone and nothing else is. Whatever
                            the experiment reports here it invented, and the
                            headline is uninterpretable until this comes back
                            empty.

The output schema is consumed directly by the paired injection generator.
"""

from __future__ import annotations

import argparse
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

import dhs.paired_sources
from dhs.core import sha256_file, write_sha256sums
from dhs.paired_sources import SOURCE_PIXEL_SCALE_ARCSEC, render_arms

CARRIED_COLUMNS = [
    "pair_index",
    "cosmos_catalog_index",
    "cosmos_original_index",
    "cosmos_ident",
    "f814w_mag_auto",
    "source_pixel_scale_arcsec",
    "source_stamp_size",
    "rotation_rad",
    "sersic_n",
    "sersic_q",
    "sersic_phi_catalog_rad",
    "sersic_circularized_hlr_arcsec",
]


def source_filename(pair_index: int, ident: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in ident
    )
    return f"source_{pair_index:04d}_{safe}.npz"


def build(
    *,
    manifest: pd.DataFrame,
    catalog: galsim.COSMOSCatalog,
    output_map_dir: Path,
    twin_noise: str,
    support_matched: bool,
    noise_seed: int,
    arms: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for record in manifest.sort_values("pair_index", kind="mergesort").itertuples():
        real_map, twin_map, diagnostics = render_arms(
            record=record,
            catalog=catalog,
            twin_noise=twin_noise,
            support_matched=support_matched,
            arms=arms,
            noise_seed=noise_seed + int(record.pair_index),
        )
        filename = source_filename(int(record.pair_index), str(record.cosmos_ident))
        path = output_map_dir / filename
        np.savez_compressed(
            path,
            real_source=real_map.astype(np.float32),
            sersic_twin=twin_map.astype(np.float32),
        )
        row = {column: getattr(record, column) for column in CARRIED_COLUMNS}
        row.update(diagnostics)
        row["source_map_relpath"] = f"source_maps/{filename}"
        row["source_map_sha256"] = sha256_file(path)
        records.append(row)
        if (len(records) % 25) == 0:
            print(f"Rendered {len(records)}/{len(manifest)}", flush=True)
    return pd.DataFrame(records)


def materialize(arguments: argparse.Namespace) -> None:
    output_dir = Path(arguments.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace paired sources: {output_dir}")
    manifest = pd.read_parquet(arguments.source_manifest)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(arguments.cosmos_dir))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        maps = temporary / "source_maps"
        maps.mkdir()
        built = build(
            manifest=manifest,
            catalog=catalog,
            output_map_dir=maps,
            twin_noise=arguments.twin_noise,
            support_matched=arguments.support == "matched",
            arms=arguments.arms,
            noise_seed=arguments.noise_seed,
        )
        truncated = (
            built["stamp_half_widths_per_half_light_radius"]
            < arguments.truncation_threshold
        )
        built["stamp_truncates_source"] = truncated

        manifest_path = temporary / "paired_source_manifest.parquet"
        built.to_parquet(manifest_path, index=False)

        summary = {
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "selection_is_cnn_score_blind": True,
            "pairs": int(len(built)),
            "treatments": {
                "real_arm": "RealGalaxy.original_gal, the interpolated observed stamp",
                "twin_arm": ("catalogue single-Sersic fit convolved with the same psf"),
                "twin_noise": arguments.twin_noise,
                "support": arguments.support,
                "arms": arguments.arms,
                "noise_seed": int(arguments.noise_seed),
            },
            "flux_within_half_light_radius": {
                "real_median": float(
                    built["real_flux_within_half_light_radius"].median()
                ),
                "twin_median": float(
                    built["twin_flux_within_half_light_radius"].median()
                ),
            },
            # Clipping the sky at zero deposits a positive pedestal. It is
            # harmless only while it is symmetric, so both arms are recorded and
            # retain the measured difference for interpretation.
            "clip_pedestal_fraction": {
                "real_median": float(built["real_pedestal_fraction"].median()),
                "twin_median": float(built["twin_pedestal_fraction"].median()),
                "median_absolute_difference": float(
                    (built["twin_pedestal_fraction"] - built["real_pedestal_fraction"])
                    .abs()
                    .median()
                ),
            },
            # The empirical annulus estimate this replaces measured zero padding
            # for 233 of 265 sources and came out about 150 times too small.
            "noise_model_over_measured_in_footprint_sky": {
                "median": float(
                    (
                        built["catalogue_noise_rms"] / built["in_footprint_sky_rms"]
                    ).median()
                ),
            },
            "truncation": {
                "threshold_stamp_half_widths_per_half_light_radius": float(
                    arguments.truncation_threshold
                ),
                "sources_flagged": int(truncated.sum()),
            },
            "inherits_selection_from": {
                "source_manifest": str(arguments.source_manifest),
                "source_manifest_sha256": sha256_file(arguments.source_manifest),
            },
            # Recorded here because stages after 13 are absent from
            # PHASE1_CODE_MANIFEST.json, so without a self-referential digest an
            # artifact cannot be tied to the code that produced it.
            "provenance": {
                "cosmos_dir": str(arguments.cosmos_dir),
                "galsim_version": galsim.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "rendering_module_sha256": sha256_file(
                    Path(dhs.paired_sources.__file__)
                ),
                "script_sha256": sha256_file(Path(__file__)),
                "source_pixel_scale_arcsec": SOURCE_PIXEL_SCALE_ARCSEC,
            },
        }
        summary_path = temporary / "source_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        write_sha256sums([manifest_path, summary_path], temporary / "SHA256SUMS")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Published corrected paired sources: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--cosmos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--twin-noise", choices=["catalogue", "none"], default="catalogue"
    )
    parser.add_argument("--support", choices=["matched", "full"], default="matched")
    parser.add_argument(
        "--arms", choices=["real_vs_twin", "twin_vs_twin"], default="real_vs_twin"
    )
    parser.add_argument("--noise-seed", type=int, default=20260822)
    parser.add_argument("--truncation-threshold", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    materialize(parse_args())


if __name__ == "__main__":
    main()
