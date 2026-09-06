from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psrc.canaries import (
    bootstrap_coverage_canary,
    photon_canary,
    renderer_canary,
    synthetic_deconvolution_canary,
)
from psrc.cli import parser
from psrc.contracts import ContractError, load_config
from psrc.design import build_parent_bound_design, select_parent_scene_subset
from psrc.execution import render_expectation_bank, stratified_canary_subset
from psrc.inference import global_spatial_source_bootstrap, point_estimate
from psrc.rendering import (
    coupled_photon_draw,
    zero_padded_convolution,
)
from psrc.source import _align_nonnegative, select_sources
from psrc.utils import reject_outcome_columns

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/psrc_v2_parent_bound.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_source_roles_are_balanced_and_disjoint(config: dict) -> None:
    rows = []
    for index in range(400):
        rows.append(
            {
                "eligible": True,
                "cosmos_ident": index,
                "cosmos_catalog_index": index,
                "sersic_circularized_hlr_arcsec": float(index),
                "single_sersic_fit_mad": float(index % 20),
            }
        )
    ledger = pd.DataFrame(rows)
    _, experiment = select_sources(ledger, config, source_role="experiment")
    _, pilot = select_sources(ledger, config, source_role="pilot")
    assert len(experiment) == len(pilot) == 144
    assert set(experiment["source_id"]).isdisjoint(set(pilot["source_id"]))
    assert experiment.groupby("source_stratum").size().eq(9).all()
    assert pilot.groupby("source_stratum").size().eq(9).all()


def _parent_scene_population(config: dict) -> pd.DataFrame:
    radii = config["parent_grec"]["required_radii_arcsec"]
    rows = []
    for cell in range(9):
        for host_index in range(176):
            host = f"h{cell}_{host_index:03d}"
            magnitude = 23.0 + 3.0 * (host_index + 0.5) / 176
            for radius_index, radius in enumerate(radii):
                rows.append(
                    {
                        "scene_id": f"scene_{host}_{radius}",
                        "grec_row_id": f"grec_{host}_{radius}",
                        "parameter_row_sha256": f"{cell * 100000 + host_index * 20 + radius_index:064x}",
                        "host_galaxy_id": host,
                        "host_parent_id": f"parent_{host}",
                        "host_spatial_group_id": f"group_{cell}_{host_index // 2}",
                        "host_physical_group_id": f"physical_{cell}_{host_index}",
                        "host_spatial_coordinate_id": f"coord_{cell}_{host_index}",
                        "host_pixel_sha256": f"{cell * 1000 + host_index:064x}",
                        "host_payload_sha256": f"{cell * 1000 + host_index + 1:064x}",
                        "host_pixel_path": "/tmp/unused.npz",
                        "source_parameter_id": f"src_{host}_{radius}",
                        "source_nuisance_id": f"srcn_{host}",
                        "lens_parameter_id": f"lens_{host}_{radius}",
                        "lens_nuisance_id": f"lensn_{host}",
                        "render_config_id": "render_parent",
                        "render_contract_sha256": "a" * 64,
                        "base_draw_key": f"draw_{host}",
                        "psf_bin": 1.0 + 0.1 * cell,
                        "depth_bin": 24.0,
                        "source_r_mag": magnitude,
                        "theta_e_arcsec": float(radius),
                        "source_flux_nmgy_g": 0.1,
                        "source_flux_nmgy_r": 0.2,
                        "source_flux_nmgy_z": 0.3,
                        "source_beta_x_arcsec": float(radius) * 0.2,
                        "source_beta_y_arcsec": 0.0,
                        "lens_axis_ratio": 0.8,
                        "lens_phi_rad": 0.2,
                        "shear_g1": 0.01,
                        "shear_g2": -0.01,
                        "lens_x0_arcsec": 0.0,
                        "lens_y0_arcsec": 0.0,
                        "pixel_scale_arcsec": 0.262,
                        "psf_fwhm_g_arcsec": 1.2,
                        "psf_fwhm_r_arcsec": 1.1,
                        "psf_fwhm_z_arcsec": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": f"s{index:03d}",
                "source_stratum": f"q{index % 16:02d}",
                "source_analysis_weight": 1.0 + (index % 4) / 10,
                "source_population_sensitivity_weight": 1.0 + (index % 4) / 10,
                "source_map_relpath": f"maps/s{index:03d}.npz",
                "source_map_sha256": f"{index:064x}",
                "source_image_sha256": f"{index + 1000:064x}",
                "source_psf_image_sha256": f"{index + 2000:064x}",
                "selection_hash": f"{index:064x}",
            }
            for index in range(144)
        ]
    )


def test_parent_subset_uses_exact_registered_radius_and_magnitude_population(
    config: dict,
) -> None:
    scenes = _parent_scene_population(config)
    selected = select_parent_scene_subset(scenes, config)
    assert len(selected) == 1584
    assert selected["host_galaxy_id"].nunique() == 1584
    assert (
        sorted(selected["theta_e_arcsec"].unique())
        == config["parent_grec"]["required_radii_arcsec"]
    )
    assert (
        selected.groupby(
            ["host_cell_id", "theta_e_arcsec", "parent_magnitude_quantile"]
        )
        .size()
        .eq(4)
        .all()
    )
    original = scenes.set_index("scene_id")
    for row in selected.itertuples(index=False):
        assert row.source_r_mag == original.at[row.scene_id, "source_r_mag"]
        assert row.source_flux_nmgy_r == original.at[row.scene_id, "source_flux_nmgy_r"]


def test_child_design_preserves_parent_ids_and_parameters_exactly(
    config: dict,
) -> None:
    subset = select_parent_scene_subset(_parent_scene_population(config), config)
    design = build_parent_bound_design(subset, _sources(), config)
    assert len(design) == 1584
    assert design["host_galaxy_id"].nunique() == 1584
    assert design["parent_grec_scene_id"].nunique() == 1584
    assert design.groupby("source_id").size().eq(11).all()
    assert (
        design.groupby(["host_cell_id", "theta_e_arcsec"])["source_stratum"]
        .nunique()
        .eq(16)
        .all()
    )
    assert design["parent_scene_analysis_weight"].eq(1.0).all()
    assert design["source_analysis_weight"].eq(1.0).all()
    assert np.array_equal(design["scene_id"], design["parent_grec_scene_id"])
    assert np.array_equal(design["grec_row_id"], design["parent_grec_row_id"])
    assert design["intrinsic_flux_policy"].eq("fixed_parent_intrinsic_per_band").all()
    assert design["parent_grec_scene_id"].notna().all()


def test_outcome_columns_cannot_enter_parent_selection(config: dict) -> None:
    scenes = _parent_scene_population(config)
    scenes["historical_cnn_score"] = 0.0
    with pytest.raises(ValueError, match="outcome-like"):
        reject_outcome_columns(scenes, config["selection"]["forbidden_column_tokens"])
    with pytest.raises(ValueError, match="outcome-like"):
        select_parent_scene_subset(scenes, config)


def _bootstrap_frame() -> pd.DataFrame:
    rows = []
    for cell in range(3):
        for position in range(24):
            rows.append(
                {
                    "host_cell_id": f"c{cell}",
                    "host_spatial_group_id": f"g{position % 12}",
                    "source_id": f"s{position % 8}",
                    "source_stratum": f"q{position % 4}",
                    "source_analysis_weight": 1.0,
                    "paired_detection_risk_difference": (
                        0.2 if position % 3 == 0 else -0.1
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_uses_global_spatial_and_source_dependencies(config: dict) -> None:
    frame = _bootstrap_frame()
    first, design = global_spatial_source_bootstrap(
        frame, draws=100, seed_namespace="test"
    )
    second, _ = global_spatial_source_bootstrap(frame, draws=100, seed_namespace="test")
    np.testing.assert_array_equal(first, second)
    assert first.mean() == pytest.approx(point_estimate(frame), abs=1e-12)
    assert design["one_global_multiplier_per_spatial_group"] is True
    assert design["same_spatial_multiplier_across_all_host_cells"] is True
    assert design["one_global_multiplier_per_source"] is True
    assert design["source_multiplier_normalized_within_source_stratum"] is True
    assert design["host_cell_weighting"] == "equal_fixed_cells"
    assert design["spatial_groups_crossing_host_cells"] == 12


def test_real_canary_selector_spans_declared_dimensions(config: dict) -> None:
    subset = select_parent_scene_subset(_parent_scene_population(config), config)
    design = build_parent_bound_design(subset, _sources(), config)
    canary = stratified_canary_subset(design, 32)
    assert canary["source_stratum"].nunique() == 16
    assert canary["host_cell_id"].nunique() == 9
    assert canary["theta_e_arcsec"].nunique() >= 7


def test_support_projection_alignment_gate_is_explicit(config: dict) -> None:
    image = np.zeros((161, 161), dtype=np.float64)
    image[70:80, 100:110] = 1.0
    image[0, 0] = -2.0
    y, x = np.mgrid[:161, :161]
    support = np.hypot(x - 80, y - 80) * 0.03 <= 1.0
    aligned, diagnostic = _align_nonnegative(
        image,
        support=support,
        pixel_scale=0.03,
        tolerance_arcsec=3e-4,
    )
    assert np.all(aligned[~support] == 0)
    assert aligned.sum() * 0.03**2 == pytest.approx(1.0)
    assert diagnostic["centroid_residual_arcsec"] <= 3e-4
    assert (
        config["source_construction"]["maximum_structured_negative_to_positive_flux"]
        == 0.20
    )


def test_zero_padding_and_photon_keys_are_worker_invariant(config: dict) -> None:
    image = np.zeros((31, 31))
    image[0, 0] = 1.0
    kernel = np.ones((5, 5))
    convolved = zero_padded_convolution(image, kernel)
    assert convolved[-1, -1] == pytest.approx(0.0, abs=1e-14)
    a = np.full((2, 2, 3), 0.2)
    b = np.full((2, 2, 3), 0.3)
    forward = {
        draw: coupled_photon_draw(
            a, b, gain_e_per_nmgy=100, noise_key_base="k", draw_index=draw
        )[:2]
        for draw in range(8)
    }
    reverse = {
        draw: coupled_photon_draw(
            a, b, gain_e_per_nmgy=100, noise_key_base="k", draw_index=draw
        )[:2]
        for draw in reversed(range(8))
    }
    for draw in range(8):
        np.testing.assert_array_equal(forward[draw][0], reverse[draw][0])
        np.testing.assert_array_equal(forward[draw][1], reverse[draw][1])


def test_final_render_has_no_config_status_only_bypass(
    config: dict, tmp_path: Path
) -> None:
    with pytest.raises(ContractError, match="execution lock"):
        render_expectation_bank(
            design_path=tmp_path / "design.parquet",
            design_receipt_path=tmp_path / "receipt.json",
            source_root=tmp_path,
            config=config,
            output_dir=tmp_path / "out",
            execution_lock_path=None,
            design_only_pairs=None,
        )


def test_skip_host_hash_option_was_removed() -> None:
    help_text = parser().format_help()
    assert "skip-host" not in help_text


def test_dependencies_are_exactly_pinned() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    for dependency in (
        "numpy==2.5.3",
        "pandas==3.0.5",
        "torch==2.7.0",
        "galsim==2.8.5",
        "pytest==9.1.1",
        "lenstronomy==1.14.1",
    ):
        assert dependency in text
    assert ">=" not in "\n".join(
        line
        for line in text.splitlines()
        if '"' in line and "requires-python" not in line
    )


def test_all_design_only_science_canaries(config: dict) -> None:
    assert (
        synthetic_deconvolution_canary()["relative_l1_to_known_regularized_truth"]
        < 0.08
    )
    renderer = renderer_canary(config)
    assert renderer["central_delta_psf_flux"] == pytest.approx(1.0)
    assert renderer["opposite_corner_wrap_value"] == pytest.approx(0.0, abs=1e-14)
    photon = photon_canary()
    assert photon["worker_order_invariant"] is True
    assert photon["maximum_variance_relative_error"] <= 0.08
    coverage = bootstrap_coverage_canary(config)
    assert coverage["replications"] == 200
    assert coverage["pairs_per_replication"] == 1584
    assert coverage["binary_photon_draws_per_pair"] == 8
    assert coverage["interval_inflation"] == 1.0
    assert coverage["role"] == "conservative_diagnostic_only_never_a_pass_gate"
    assert coverage["exact_assignment"] == {
        "blocks": 99,
        "rows_per_block": 16,
        "source_strata_per_block_minimum": 16,
        "source_strata_per_block_maximum": 16,
        "distinct_sources": 144,
        "source_reuse_minimum": 11,
        "source_reuse_maximum": 11,
        "parent_scene_weight": 1.0,
    }
    assert 0.0 <= coverage["coverage"] <= 1.0
