from __future__ import annotations

import torch

from dhs.injection_engine import LensParams
from dhs.pixelized_source import inject_pixelized_source_sie


def test_unlensed_pixelized_point_preserves_flux() -> None:
    host = torch.zeros((101, 101, 3), dtype=torch.float32)
    source = torch.zeros((101, 101), dtype=torch.float32)
    source[50, 50] = 1.0
    result = inject_pixelized_source_sie(
        host_nmgy_hwc=host,
        source_surface_brightness=source,
        source_pixel_scale_arcsec=0.03,
        source_center_x_arcsec=0.0,
        source_center_y_arcsec=0.0,
        unlensed_flux_nmgy_grz=(2.0, 3.0, 4.0),
        lens=LensParams(theta_e_arcsec=0.0),
        target_pixel_scale_arcsec=0.03,
        psf_fwhm_r_arcsec=1e-6,
        seed=7,
        subpixel_oversample=1,
        add_poisson_noise=False,
    )

    observed = result.injection_only[0].sum(dim=(1, 2))
    assert torch.allclose(observed, torch.tensor([2.0, 3.0, 4.0]), atol=1e-5)


def test_pixelized_poisson_render_is_seed_deterministic() -> None:
    host = torch.zeros((41, 41, 3), dtype=torch.float32)
    source = torch.ones((31, 31), dtype=torch.float32)
    kwargs = {
        "host_nmgy_hwc": host,
        "source_surface_brightness": source,
        "source_pixel_scale_arcsec": 0.05,
        "source_center_x_arcsec": 0.1,
        "source_center_y_arcsec": -0.1,
        "unlensed_flux_nmgy_grz": (1.0, 2.0, 3.0),
        "lens": LensParams(theta_e_arcsec=0.8),
        "target_pixel_scale_arcsec": 0.1,
        "psf_fwhm_r_arcsec": 0.7,
        "seed": 99,
        "subpixel_oversample": 2,
        "add_poisson_noise": True,
    }

    first = inject_pixelized_source_sie(**kwargs)
    second = inject_pixelized_source_sie(**kwargs)

    assert torch.equal(first.injection_only, second.injection_only)
