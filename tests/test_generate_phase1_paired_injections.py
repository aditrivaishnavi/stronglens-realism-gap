from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from dhs.injection_engine import LensParams
from scripts.generate_phase1_paired_injections import generate


def test_generate_paired_injections_changes_only_source_map(tmp_path) -> None:
    host_root = tmp_path / "hosts"
    source_root = tmp_path / "sources"
    output = tmp_path / "output"
    host_root.mkdir()
    source_root.mkdir()
    output.mkdir()
    rng = np.random.default_rng(8)
    np.savez_compressed(
        host_root / "host.npz",
        cutout=rng.normal(0.0, 0.1, size=(101, 101, 3)).astype(np.float32),
    )
    real = np.zeros((41, 41), dtype=np.float32)
    real[20, 18:23] = 1.0
    sersic = np.zeros((41, 41), dtype=np.float32)
    sersic[20, 20] = 1.0
    np.savez_compressed(
        source_root / "source.npz",
        real_source=real,
        sersic_twin=sersic,
    )
    row = {
        "real_galaxy_id": "real-1",
        "host_galaxy_id": "host-1",
        "host_replicate": 0,
        "host_spatial_component_id": "component-1",
        "host_cutout_relpath": "host.npz",
        "source_beta_x_arcsec": 0.1,
        "source_beta_y_arcsec": 0.0,
        "source_flux_nmgy_g": 1.0,
        "source_flux_nmgy_r": 2.0,
        "source_flux_nmgy_z": 3.0,
        "pixel_scale_arcsec": 0.262,
        "psfsize_r_matched": 1.2,
        "injection_seed": 7,
        "subpixel_oversample": 2,
        "add_poisson_noise": False,
        "gain_e_per_nmgy": 150.0,
        "primary_snr_minimum": 0.0,
        "target_source_mag_r": 24.0,
    }
    row.update(
        {
            f"lens_{key}": value
            for key, value in asdict(LensParams(theta_e_arcsec=1.2)).items()
        }
    )
    sources = pd.DataFrame(
        [
            {
                "pair_index": 0,
                "source_map_relpath": "source.npz",
                "source_pixel_scale_arcsec": 0.03,
                "cosmos_catalog_index": 4,
                "cosmos_ident": "cosmos-4",
            }
        ]
    )

    result = generate(
        injection_metadata=pd.DataFrame([row]),
        source_manifest=sources,
        host_data_root=host_root,
        source_root=source_root,
        output_pixel_dir=output,
        source_mag_min=23.0,
        source_mag_max=24.0,
        source_mag_seed=3,
    )

    assert len(result) == 1
    assert (output / result.iloc[0]["pair_relpath"].split("/", 1)[1]).is_file()
    assert len(result.iloc[0]["pair_sha256"]) == 64
    assert result.iloc[0]["real_source_lensed_flux_r_nmgy"] > 0
    assert result.iloc[0]["sersic_twin_lensed_flux_r_nmgy"] > 0
    assert np.isclose(
        result.iloc[0]["real_source_noiseless_lensed_flux_r_nmgy"],
        result.iloc[0]["sersic_twin_noiseless_lensed_flux_r_before_match_nmgy"]
        * result.iloc[0]["sersic_twin_lensed_flux_match_scale"],
    )
