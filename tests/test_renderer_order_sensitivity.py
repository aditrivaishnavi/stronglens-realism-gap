"""Tests for the detector-free renderer-order sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.renderer_order_sensitivity import (
    RenderCase,
    build_cases,
    comparison_metrics,
    load_config,
    render_orders,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "renderer_order_sensitivity_v1.json"


def test_declared_case_count_and_ids_are_stable() -> None:
    config = load_config(CONFIG)
    cases = build_cases(config)
    assert len(cases) == 220
    assert cases[0].case_id == "corner_000"
    assert cases[63].case_id == "corner_063"
    assert cases[64].case_id == "random_000"
    assert cases[-1].case_id == "random_155"


def test_config_is_detector_free_and_noise_free() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["normalization"]["noise"] is False
    assert config["normalization"]["clumps"] is False
    assert "synthetic configurations only" in config["scope"]


def test_render_orders_preserve_flux_and_are_highly_correlated() -> None:
    case = RenderCase(
        case_id="unit",
        theta_e_arcsec=1.5,
        source_re_arcsec=0.2,
        source_n_sersic=1.0,
        source_q=0.7,
        source_phi_rad=0.3,
        beta_fraction=0.2,
        beta_angle_rad=0.8,
        lens_q=0.75,
        lens_phi_rad=0.4,
        lens_x0_arcsec=0.01,
        lens_y0_arcsec=-0.02,
        shear_g1=0.03,
        shear_g2=-0.02,
        psf_fwhm_arcsec=1.2,
    )
    executed, reference = render_orders(
        case,
        native_pixels=101,
        pixel_scale_arcsec=0.262,
        oversample=4,
    )
    metrics = comparison_metrics(executed, reference)
    assert metrics["relative_total_flux"] < 1e-12
    assert metrics["pixel_correlation"] > 0.999
    assert metrics["relative_l1"] < 0.03
