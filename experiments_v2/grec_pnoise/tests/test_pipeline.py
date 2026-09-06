from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from grec_pnoise.backend import backend_profile, verify_backend_profile
from grec_pnoise.cells import derive_condition_cells, nearest_node_lower_midpoint
from grec_pnoise.cli import _analyse
from grec_pnoise.contracts import (
    ContractError,
    array_bytes_sha256,
    config_provenance,
    load_config,
    package_source_sha256,
    scientific_section_sha256,
    sha256_file,
    sha256_json,
    stable_id,
    verify_final_host_receipt,
)
from grec_pnoise.hosts import prepare_final_hosts
from grec_pnoise.inference import (
    _global_spatial_multiplier_bootstrap,
    analyse_grec,
    analyse_pnoise,
)
from grec_pnoise.pnoise import (
    choose_draw_count,
    draw_photon_arc,
    photon_moments_nmgy,
    render_photon_children,
    validate_draw_count_receipt,
    validate_parent_bank_for_pnoise,
    verify_design_role_ledger,
)
from grec_pnoise.rendering import (
    _grid,
    _sie_deflection,
    adaptive_render_source_expectation,
    render_parent_bank,
    render_source_expectation_at_order,
    zero_padded_gaussian_convolution,
)
from grec_pnoise.scene_table import bind_scene_identity, build_scene_table
from grec_pnoise.scoring import fixed_scorer_closure, validate_scoring_authority
from grec_pnoise.threshold import threshold_evidence

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/production.toml"


def psfdepth_for_mag(depth_mag: float) -> float:
    return float((5.0 * 10.0 ** ((depth_mag - 22.5) / 2.5)) ** 2)


@pytest.fixture()
def config() -> dict:
    return load_config(CONFIG_PATH)


def make_hosts(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index in range(4):
        rng = np.random.default_rng(index)
        host = rng.normal(0.0, 0.01, size=(101, 101, 3)).astype(np.float32)
        path = tmp_path / f"host_{index}.npz"
        np.savez_compressed(path, cutout=host)
        rows.append(
            {
                "host_galaxy_id": f"h{index}",
                "host_parent_id": f"p{index}",
                "host_pixel_path": str(path),
                "host_payload_sha256": sha256_file(path),
                "host_pixel_sha256": array_bytes_sha256(host),
                "host_spatial_group_id": f"g{index}",
                "host_spatial_coordinate_id": f"coord_{index}",
                "host_ra_deg": 10.0 + index,
                "host_dec_deg": -2.0 + index,
                "final_role": "experiment_host",
                "psf_bin": 1.2 if index < 2 else 1.35,
                "depth_bin": 24.0,
                "depth_mag": 24.0,
                "psfsize_r": 1.2 if index < 2 else 1.35,
                "psfdepth_r": psfdepth_for_mag(24.0),
                "psf_fwhm_r_arcsec": 1.2 if index < 2 else 1.35,
                "pixel_scale_arcsec": 0.262,
                "host_pixel_unit": "nanomaggy_per_native_pixel",
                "host_flux_calibration_system": "AB_nanomaggy_zeropoint_22.5",
                "host_calibration_provenance_sha256": f"{index + 1:064x}",
            }
        )
    return pd.DataFrame(rows)


def make_scenes(tmp_path: Path, config: dict) -> pd.DataFrame:
    return build_scene_table(
        make_hosts(tmp_path),
        config,
        expected_hosts=4,
        expected_cells=2,
        hosts_per_cell=2,
    )


def make_threshold_receipt(
    tmp_path: Path, config: dict, values: np.ndarray | None = None
) -> Path:
    n = int(config["threshold"]["minimum_calibration_rows"])
    values = (
        np.linspace(-8.0, 8.0, n, dtype=np.float32)
        if values is None
        else np.asarray(values, dtype=np.float32)
    )
    ids = [f"c{i}" for i in range(len(values))]
    manifest = tmp_path / "calibration_manifest.parquet"
    logits = tmp_path / "calibration_logits.parquet"
    pd.DataFrame(
        {
            "calibration_id": ids,
            "global_role": "threshold_calibration",
            "host_pixel_sha256": [f"{i:064x}" for i in range(len(values))],
            "evaluation_spatial_group_id": [f"g{i}" for i in range(len(values))],
        }
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        {
            "calibration_id": ids,
            "raw_logit_float32": values,
            "raw_logit_float32_bits": values.view(np.uint32),
        }
    ).to_parquet(logits, index=False)
    evidence, _ = threshold_evidence(values, float(config["threshold"]["alpha"]))
    device = config["scoring"]["authoritative_device"]
    closure = fixed_scorer_closure(config, device)
    profile = backend_profile(config, device)
    score_receipt_sha = "1" * 64
    approval = tmp_path / "external_approval.json"
    approval.write_text(
        json.dumps(
            {
                "status": config["approval"]["required_threshold_status"],
                "scope": config["approval"]["required_scope"],
                "calibration_manifest_sha256": sha256_file(manifest),
                "calibration_logits_sha256": sha256_file(logits),
                "calibration_score_receipt_sha256": score_receipt_sha,
                "config_contract_sha256": config_provenance(config)[
                    "config_contract_sha256"
                ],
                "canary_only": False,
                "release_authorized": True,
                "approved_by": "external-reviewer",
                "approved_at_utc": "2026-09-16T00:00:00Z",
            }
        )
    )
    receipt = tmp_path / "threshold.json"
    receipt.write_text(
        json.dumps(
            {
                "status": config["scoring"]["required_threshold_receipt_status"],
                "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
                "comparison": ">=",
                "native_logit_dtype": "float32",
                "calibration_role": "threshold_calibration",
                "pre_score_review": "external_receipt_bound_not_self_asserted",
                "released_for_experiment_scoring": False,
                "external_release_authorized": True,
                "threshold_rule": (
                    "nextafter_float32_above_s_k_plus_1; "
                    "classify_logit_greater_equal"
                ),
                "calibration_manifest_path": str(manifest),
                "calibration_manifest_sha256": sha256_file(manifest),
                "calibration_logits_path": str(logits),
                "calibration_logits_sha256": sha256_file(logits),
                "calibration_score_receipt_sha256": score_receipt_sha,
                "external_approval_receipt_path": str(approval),
                "external_approval_receipt_sha256": sha256_file(approval),
                "scorer_closure": closure,
                "scorer_closure_sha256": sha256_json(closure),
                "backend_profile": profile,
                "config": config_provenance(config),
                **evidence,
            }
        )
    )
    return receipt


def test_scene_table_is_x11_and_shares_nuisance(tmp_path: Path, config: dict):
    scenes = make_scenes(tmp_path, config)
    assert len(scenes) == 44
    grouped = scenes.groupby("host_galaxy_id")
    assert grouped["theta_e_arcsec"].nunique().eq(11).all()
    for column in [
        "base_draw_key",
        "source_nuisance_id",
        "lens_nuisance_id",
        "source_r_mag",
        "source_beta_fraction",
        "shear_g1",
    ]:
        assert grouped[column].nunique().eq(1).all()
    first = scenes.sort_values("theta_e_arcsec").iloc[0]
    last = scenes.sort_values("theta_e_arcsec").iloc[-1]
    assert 23 <= scenes["source_r_mag"].min()
    assert scenes["source_r_mag"].max() < 26
    assert first["source_parameter_id"] != last["source_parameter_id"]
    assert scenes["scene_id"].is_unique


def test_scene_table_rejects_outcome_column(tmp_path: Path, config: dict):
    hosts = make_hosts(tmp_path)
    hosts["cnn_score"] = 0.0
    with pytest.raises(ContractError, match="outcome columns"):
        build_scene_table(
            hosts,
            config,
            expected_hosts=4,
            expected_cells=2,
            hosts_per_cell=2,
        )


def test_nonfinal_role_proposal_cannot_be_promoted_by_code(tmp_path: Path):
    manifest = tmp_path / "hosts.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(manifest)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "score_blind_role_proposal_complete_not_final",
                "host_manifest_sha256": sha256_file(manifest),
            }
        )
    )
    with pytest.raises(ContractError, match="receipt status"):
        verify_final_host_receipt(
            manifest, receipt, "final_pre_score_grec_hosts_frozen"
        )


def test_prepare_final_hosts_hashes_exact_cutout_bytes(
    tmp_path: Path, config: dict
):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": config["host"]["required_calibration_receipt_status"],
                "pixel_unit": config["host"]["pixel_unit"],
                "flux_calibration_system": config["host"][
                    "flux_calibration_system"
                ],
            }
        )
    )
    calibration_sha = sha256_file(calibration)
    rows = []
    for index in range(4):
        cutout = np.full((101, 101, 3), index, dtype=np.float32)
        path = tmp_path / f"payload_{index}.npz"
        np.savez_compressed(path, cutout=cutout)
        rows.append(
            {
                "galaxy_id": f"h{index}",
                "local_relpath": path.name,
                "payload_sha256": sha256_file(path),
                "evaluation_spatial_group_id": f"g{index}",
                "final_role": "experiment_host",
                "psfsize_r": 1.2 if index < 2 else 1.35,
                "psfdepth_r": psfdepth_for_mag(24.0),
                "pixel_scale": 0.262,
                "ra": 10.0 + index,
                "dec": -3.0 + index,
                "pixel_unit": config["host"]["pixel_unit"],
                "flux_calibration_system": config["host"][
                    "flux_calibration_system"
                ],
                "calibration_provenance_path": str(calibration),
                "calibration_provenance_sha256": calibration_sha,
            }
        )
    hosts = prepare_final_hosts(
        pd.DataFrame(rows),
        tmp_path,
        config,
        expected_hosts=4,
        expected_cells=2,
        hosts_per_cell=2,
    )
    assert len(hosts) == 4
    assert hosts["host_parent_id"].is_unique
    for row in hosts.to_dict(orient="records"):
        with np.load(row["host_pixel_path"], allow_pickle=False) as archive:
            assert array_bytes_sha256(archive["cutout"]) == row["host_pixel_sha256"]


def test_zero_padding_does_not_wrap_flux():
    image = torch.zeros((33, 33), dtype=torch.float32)
    image[16, 0] = 1.0
    convolved = zero_padded_gaussian_convolution(
        image, sigma_pixels=1.2, truncate_sigma=8.0
    )
    assert convolved[16, 0] > 0
    assert convolved[16, -1] == 0
    assert convolved.min() >= 0


def test_parent_render_retains_raw_host_and_no_added_noise(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    parents, receipt = render_parent_bank(
        scenes, config, tmp_path / "render", limit=1
    )
    assert receipt["status"] == "design_only_grec_render_canary"
    row = parents.iloc[0]
    host = np.load(row["host_store_path"], allow_pickle=False)
    assert array_bytes_sha256(host) == row["host_pixel_sha256"]
    with np.load(row["parent_pixel_path"], allow_pickle=False) as archive:
        expectation = archive["expectation_chw"]
        arc = archive["injection_expectation_chw"]
    np.testing.assert_array_equal(expectation, host.transpose(2, 0, 1) + arc)
    assert row["added_source_photon_noise"] == False  # noqa: E712
    assert row["added_sky_noise"] == False  # noqa: E712


def test_final_render_api_requires_validation_receipt(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    with pytest.raises(ContractError, match="lacks renderer validation"):
        render_parent_bank(scenes, config, tmp_path / "final")


def test_photon_units_mean_variance_and_zero_pixels():
    expectation = np.zeros((3, 2, 2), dtype=np.float32)
    expectation[:, 0, 0] = [0.02, 0.2, 2.0]
    gain = np.array([150.0, 150.0, 150.0])
    mean, variance = photon_moments_nmgy(expectation, gain)
    np.testing.assert_allclose(mean, expectation)
    np.testing.assert_allclose(
        variance[:, 0, 0], expectation[:, 0, 0] / gain
    )
    rng = np.random.Generator(np.random.PCG64DXSM(771))
    draws = rng.poisson(expectation[:, 0, 0] * gain, size=(200_000, 3)) / gain
    np.testing.assert_allclose(draws.mean(axis=0), expectation[:, 0, 0], rtol=0.01)
    np.testing.assert_allclose(
        draws.var(axis=0), expectation[:, 0, 0] / gain, rtol=0.02
    )
    physical, _ = draw_photon_arc(
        expectation,
        gain,
        stream_namespace="test",
        scene_id="scene",
        draw_index=0,
    )
    assert np.count_nonzero(physical[:, 1:, :]) == 0


def test_photon_stream_is_keyed_and_repeatable():
    expectation = np.full((3, 4, 4), 0.2, dtype=np.float32)
    gains = np.full(3, 150.0)
    first, key_first = draw_photon_arc(
        expectation,
        gains,
        stream_namespace="test",
        scene_id="a",
        draw_index=3,
    )
    replay, key_replay = draw_photon_arc(
        expectation,
        gains,
        stream_namespace="test",
        scene_id="a",
        draw_index=3,
    )
    other, key_other = draw_photon_arc(
        expectation,
        gains,
        stream_namespace="test",
        scene_id="a",
        draw_index=4,
    )
    np.testing.assert_array_equal(first, replay)
    assert key_first == key_replay
    assert key_other != key_first
    assert not np.array_equal(first, other)


def test_photon_child_preserves_parent_ids_and_host_bytes(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    parents, _ = render_parent_bank(scenes, config, tmp_path / "render", limit=1)
    children, receipt = render_photon_children(
        parents,
        config,
        tmp_path / "photon",
        {"selected_draw_count": 2, "_receipt_sha256": "canary"},
        limit=1,
        canary_label="design_only_unit_test",
    )
    assert len(children) == 2
    assert children["parent_scene_id"].eq(parents.iloc[0]["scene_id"]).all()
    assert children["parent_grec_row_id"].eq(parents.iloc[0]["grec_row_id"]).all()
    assert children["host_pixel_sha256"].eq(
        parents.iloc[0]["host_pixel_sha256"]
    ).all()
    host = np.load(children.iloc[0]["host_store_path"], allow_pickle=False)
    assert array_bytes_sha256(host) == parents.iloc[0]["host_pixel_sha256"]
    assert receipt["existing_host_sky_detector_coadd_noise_redrawn"] is False


def test_limited_photon_api_requires_explicit_canary_label(
    tmp_path: Path, config: dict
):
    with pytest.raises(ContractError, match="explicitly labelled"):
        render_photon_children(
            pd.DataFrame(),
            config,
            tmp_path / "photon",
            {"selected_draw_count": 2},
            limit=1,
            canary_label=None,
        )


def test_production_photon_api_requires_complete_parent_authority(
    tmp_path: Path, config: dict
):
    with pytest.raises(ContractError, match="complete parent/scene"):
        render_photon_children(
            pd.DataFrame(),
            config,
            tmp_path / "photon",
            {"selected_draw_count": 2},
        )


def test_mcse_selector_uses_host_equal_weights():
    rng = np.random.default_rng(9)
    probabilities = rng.uniform(0.2, 0.8, size=2000)
    physical = (rng.random((2000, 64)) < probabilities[:, None]).astype(np.uint8)
    baseline = (rng.random(2000) < probabilities).astype(np.uint8)
    result = choose_draw_count(
        physical,
        baseline,
        [f"h{i}" for i in range(2000)],
        [1, 2, 4, 8, 16, 32, 64],
        absolute_mcse_max=0.002,
        fraction_of_sampling_se_max=0.20,
    )
    assert result["selected_draw_count"] in {1, 2, 4, 8, 16, 32, 64}
    assert result["absolute_mcse_gate_passed"]
    assert result["relative_mcse_gate_passed"]


def test_draw_count_receipt_is_required_and_score_blind(
    tmp_path: Path, config: dict
):
    bad = tmp_path / "draw.json"
    bad.write_text(
        json.dumps(
            {
                "status": "draft",
                "selected_draw_count": 16,
                "design_only": True,
                "final_outcomes_used": False,
            }
        )
    )
    with pytest.raises(ContractError, match="frozen MCSE receipt"):
        validate_draw_count_receipt(bad, config, tmp_path / "missing-threshold")


def test_scoring_refuses_absent_or_unapproved_threshold(
    tmp_path: Path, config: dict
):
    threshold = np.float32(1.25)
    receipt = tmp_path / "threshold.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "draft",
                "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
                "comparison": ">=",
                "native_logit_dtype": "float32",
                "calibration_role": "threshold_calibration",
                "pre_score_review": "approved",
                "released_for_experiment_scoring": False,
                "logit_threshold": float(threshold),
                "logit_threshold_float32_bits": int(threshold.view(np.uint32)),
                "threshold_rule": (
                    "nextafter_float32_above_s_k_plus_1; "
                    "classify_logit_greater_equal"
                ),
            }
        )
    )
    with pytest.raises(ContractError, match="does not authorize"):
        validate_scoring_authority(config, receipt)


def test_approved_threshold_preserves_float32_bits(tmp_path: Path, config: dict):
    receipt = make_threshold_receipt(tmp_path, config)
    expected = json.loads(receipt.read_text())
    _, restored = validate_scoring_authority(config, receipt)
    assert int(restored.view(np.uint32)) == expected[
        "logit_threshold_float32_bits"
    ]


def test_cells_are_derived_with_lower_midpoint_ties(config: dict):
    nodes = np.array(config["cells"]["psf_nodes_arcsec"])
    assigned = nearest_node_lower_midpoint(
        np.array([0.975, 1.125, 1.126]), nodes
    )
    np.testing.assert_array_equal(assigned, [0.90, 1.05, 1.20])
    psf, depth, raw_depth = derive_condition_cells(
        np.array([0.975, 1.125]),
        np.array([psfdepth_for_mag(22.75), psfdepth_for_mag(23.25)]),
        config,
    )
    np.testing.assert_array_equal(psf, [0.90, 1.05])
    np.testing.assert_array_equal(depth, [22.5, 23.0])
    np.testing.assert_allclose(raw_depth, [22.75, 23.25], atol=1e-12)


def test_invalid_raw_conditions_are_rejected(config: dict):
    with pytest.raises(ContractError, match="psfsize"):
        derive_condition_cells(np.array([99.0]), np.array([100.0]), config)
    with pytest.raises(ContractError, match="psfdepth"):
        derive_condition_cells(np.array([1.2]), np.array([0.0]), config)


def test_host_unit_drift_is_rejected(tmp_path: Path, config: dict):
    hosts = make_hosts(tmp_path)
    hosts.loc[0, "host_pixel_unit"] = "electron"
    with pytest.raises(ContractError, match="pixel units"):
        build_scene_table(
            hosts,
            config,
            expected_hosts=4,
            expected_cells=2,
            hosts_per_cell=2,
        )


def test_duplicate_pixel_and_spatial_hosts_are_rejected(
    tmp_path: Path, config: dict
):
    hosts = make_hosts(tmp_path)
    hosts.loc[1, "host_pixel_sha256"] = hosts.loc[0, "host_pixel_sha256"]
    with pytest.raises(ContractError, match="duplicate host pixel"):
        build_scene_table(
            hosts,
            config,
            expected_hosts=4,
            expected_cells=2,
            hosts_per_cell=2,
        )
    hosts = make_hosts(tmp_path)
    hosts.loc[1, "host_spatial_coordinate_id"] = hosts.loc[
        0, "host_spatial_coordinate_id"
    ]
    with pytest.raises(ContractError, match="duplicate spatial"):
        build_scene_table(
            hosts,
            config,
            expected_hosts=4,
            expected_cells=2,
            hosts_per_cell=2,
        )


def test_scene_identity_binds_psf_and_rejects_config_drift(
    tmp_path: Path, config: dict
):
    hosts = make_hosts(tmp_path)
    first = build_scene_table(
        hosts,
        config,
        expected_hosts=4,
        expected_cells=2,
        hosts_per_cell=2,
    ).iloc[0]
    second = first.to_dict()
    second["psf_fwhm_r_arcsec"] = 1.8
    second["psf_fwhm_g_arcsec"] = 1.8 * config["render"]["psf_fwhm_scale_g"]
    second["psf_fwhm_z_arcsec"] = 1.8 * config["render"]["psf_fwhm_scale_z"]
    second = bind_scene_identity(second)
    assert first["scene_id"] != second["scene_id"]
    stale = copy.deepcopy(config)
    stale["render"]["gaussian_truncate_sigma"] = 7.0
    with pytest.raises(ContractError, match="configuration drifted"):
        adaptive_render_source_expectation(first.to_dict(), (101, 101, 3), stale)


def test_adaptive_flux_and_lensed_convergence_over_prior_corners(
    tmp_path: Path, config: dict
):
    base = make_scenes(tmp_path, config).iloc[0].to_dict()
    for re_arcsec in (0.15, 0.50):
        for n_sersic in (0.5, 2.0):
            for q in (0.3, 1.0):
                row = dict(base)
                row.update(
                    {
                        "source_re_arcsec": re_arcsec,
                        "source_n_sersic": n_sersic,
                        "source_axis_ratio": q,
                    }
                )
                row["source_nuisance_id"] = stable_id(
                    "srcn", [re_arcsec, n_sersic, q]
                )
                row["source_parameter_id"] = stable_id(
                    "src", [row["source_nuisance_id"], row["theta_e_arcsec"]]
                )
                row = bind_scene_identity(row)
                adaptive = adaptive_render_source_expectation(
                    row, (101, 101, 3), config
                )
                reference, flux_error = render_source_expectation_at_order(
                    row,
                    (101, 101, 3),
                    config,
                    order=config["render"]["logit_reference_order"],
                )
                difference = np.abs(
                    adaptive.image_chw.astype(float) - reference.astype(float)
                )
                assert flux_error <= config["render"][
                    "unlensed_flux_relative_tolerance"
                ]
                assert difference.sum() / np.abs(reference).sum() <= config[
                    "render"
                ]["lensed_relative_l1_tolerance"]
                assert difference.max() / np.abs(reference).max() <= config[
                    "render"
                ]["lensed_relative_max_tolerance"]


def test_sie_q_one_is_sis_to_float32_precision():
    x, y = _grid(21, 21, 0.1, device=torch.device("cpu"), dtype=torch.float32)
    theta_e, x0, y0 = 1.1, 0.02, -0.03
    ax, ay = _sie_deflection(
        x, y, theta_e=theta_e, q=1.0, phi=0.73, x0=x0, y0=y0
    )
    dx, dy = x - x0, y - y0
    radius = torch.sqrt(dx.square() + dy.square() + 1e-6)
    assert torch.max(torch.abs(ax - theta_e * dx / radius)) < 2e-6
    assert torch.max(torch.abs(ay - theta_e * dy / radius)) < 2e-6


def test_fabricated_threshold_is_rejected(tmp_path: Path, config: dict):
    receipt = make_threshold_receipt(tmp_path, config)
    value = json.loads(receipt.read_text())
    value["logit_threshold"] = 123.0
    receipt.write_text(json.dumps(value))
    with pytest.raises(ContractError, match="evidence drift"):
        validate_scoring_authority(config, receipt)


def test_mcse_receipt_binds_exact_threshold(tmp_path: Path, config: dict):
    threshold = make_threshold_receipt(tmp_path, config)
    receipt = tmp_path / "draw.json"
    receipt.write_text(
        json.dumps(
            {
                "status": config["pnoise"]["required_draw_receipt_status"],
                "selected_draw_count": 16,
                "design_only": True,
                "final_outcomes_used": False,
                "metric": "host_equal_paired_detection_risk_difference",
                "absolute_mcse_gate_passed": True,
                "relative_mcse_gate_passed": True,
                "threshold_receipt_sha256": sha256_file(threshold),
                "pnoise_config_sha256": scientific_section_sha256(
                    config, "pnoise", "render", "host"
                ),
                "photon_stream_namespace": config["pnoise"][
                    "photon_stream_namespace"
                ],
                "config": config_provenance(config),
            }
        )
    )
    assert validate_draw_count_receipt(receipt, config, threshold)[
        "selected_draw_count"
    ] == 16
    other = tmp_path / "other_threshold.json"
    other.write_text(threshold.read_text() + " ")
    with pytest.raises(ContractError, match="different threshold"):
        validate_draw_count_receipt(receipt, config, other)


def test_pilot_ids_must_exist_in_design_role_ledger(
    tmp_path: Path, config: dict
):
    ledger = tmp_path / "ledger.parquet"
    pd.DataFrame(
        {
            "scene_id": ["s1", "s2"],
            "host_galaxy_id": ["h1", "h2"],
            "global_role": ["design", "experiment_host"],
        }
    ).to_parquet(ledger, index=False)
    receipt = tmp_path / "ledger.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "design_role_ledger_frozen",
                "design_role_ledger_sha256": sha256_file(ledger),
                "config_contract_sha256": config_provenance(config)[
                    "config_contract_sha256"
                ],
            }
        )
    )
    assert verify_design_role_ledger(
        ["s1"], ["h1"], ledger, receipt, config
    )["status"] == "design_role_ledger_frozen"
    with pytest.raises(ContractError, match="not all in"):
        verify_design_role_ledger(["s2"], ["h2"], ledger, receipt, config)


def test_backend_profile_is_complete_and_fail_closed(config: dict):
    profile = backend_profile(config, "cpu")
    assert profile["hardware"]["name"]
    assert profile["determinism"]["deterministic_algorithms"] is True
    assert profile["determinism"]["cudnn_benchmark"] is False
    verify_backend_profile(profile, config, "cpu")
    stale = copy.deepcopy(profile)
    stale["hardware"]["name"] = "different"
    with pytest.raises(ContractError, match="profile"):
        verify_backend_profile(stale, config, "cpu")


def test_parent_and_child_manifests_retain_full_identity(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    parents, _ = render_parent_bank(scenes, config, tmp_path / "render", limit=1)
    children, _ = render_photon_children(
        parents,
        config,
        tmp_path / "photon",
        {"selected_draw_count": 2, "_receipt_sha256": "canary"},
        limit=1,
        canary_label="design_only_identity_test",
    )
    required = {
        "host_spatial_group_id",
        "source_parameter_id",
        "lens_parameter_id",
        "render_config_id",
        "theta_e_arcsec",
        "psf_bin",
        "depth_bin",
        "host_scientific_weight",
        "scene_within_host_weight",
    }
    assert required <= set(parents)
    assert required <= set(children)


def test_inference_averages_draws_then_hosts(tmp_path: Path, config: dict):
    scenes = make_scenes(tmp_path, config)
    parent_scores = pd.DataFrame(
        {
            "grec_row_id": scenes["grec_row_id"],
            "detected_at_frozen_threshold": False,
        }
    )
    grec = analyse_grec(scenes, parent_scores, config, bootstrap_draws=20)
    assert grec["estimate"] == 0.0
    child_rows = []
    child_scores = []
    for index, scene in scenes.iterrows():
        for draw in range(2):
            row = scene.to_dict()
            row["pnoise_row_id"] = f"pn{index}_{draw}"
            row["parent_grec_row_id"] = scene["grec_row_id"]
            row["draw_index"] = draw
            child_rows.append(row)
            child_scores.append(
                {
                    "pnoise_row_id": row["pnoise_row_id"],
                    "detected_at_frozen_threshold": draw == 0,
                }
            )
    pnoise = analyse_pnoise(
        scenes,
        parent_scores,
        pd.DataFrame(child_rows),
        pd.DataFrame(child_scores),
        config,
        expected_draw_count=2,
        bootstrap_draws=20,
    )
    assert pnoise["estimate"] == pytest.approx(0.5)
    assert pnoise["finite_photon_draw_mcse"] > 0
    assert pnoise["bootstrap_design"]["primary_sampling_unit"] == (
        "host_spatial_group"
    )
    assert pnoise["bootstrap_design"][
        "cell_host_counts_preserved_in_point_estimator"
    ] is True


def test_mps_profile_binds_actual_apple_hardware_and_environment(config: dict):
    profile = backend_profile(config, "mps")
    apple = profile["hardware"]["apple"]
    assert apple["chip_type"] == "Apple M5 Max"
    assert apple["machine_model"] == "Mac17,6"
    assert apple["sysctl_cpu_brand_string"] == apple["chip_type"]
    assert apple["sysctl_hw_model"] == apple["machine_model"]
    assert profile["determinism"]["mps_environment"][
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ] == "0"
    stale = copy.deepcopy(profile)
    stale["hardware"]["apple"]["chip_type"] = "Apple M4"
    with pytest.raises(ContractError, match="profile"):
        verify_backend_profile(stale, config, "mps")


def test_threshold_cannot_outlive_external_approval(
    tmp_path: Path, config: dict
):
    receipt = make_threshold_receipt(tmp_path, config)
    value = json.loads(receipt.read_text())
    approval_path = Path(value["external_approval_receipt_path"])
    approval = json.loads(approval_path.read_text())
    approval["release_authorized"] = False
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ContractError, match="approval"):
        validate_scoring_authority(config, receipt)


def _renderer_receipt(path: Path, config: dict, gates: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "status": config["render"]["required_validation_receipt_status"],
                "config": config_provenance(config),
                "render_implementation_source_sha256": package_source_sha256(
                    "cells.py",
                    "contracts.py",
                    "rendering.py",
                    "scene_table.py",
                    "validation.py",
                ),
                "full_source_prior_flux_gate_passed": gates,
                "lensed_image_convergence_gate_passed": gates,
                "logit_convergence_gate_passed": gates,
                "sie_parity_gate_passed": gates,
                "psf_gate_passed": gates,
                "production_functions_used": gates,
                "final_outcomes_used_false": gates,
            }
        )
    )


def test_render_parent_api_invokes_complete_renderer_validator(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    receipt = tmp_path / "renderer.json"
    _renderer_receipt(receipt, config, gates=False)
    with pytest.raises(ContractError, match="scientific|gates failed"):
        render_parent_bank(
            scenes,
            config,
            tmp_path / "final",
            renderer_validation_receipt_path=receipt,
        )


def test_pnoise_parent_validation_rejects_stale_config_and_scene_identity(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    parents, _ = render_parent_bank(
        scenes, config, tmp_path / "render", limit=1
    )
    parent_manifest = tmp_path / "parents.parquet"
    parents.to_parquet(parent_manifest, index=False)
    scene_table = tmp_path / "scenes.parquet"
    scenes.iloc[:1].to_parquet(scene_table, index=False)
    renderer = tmp_path / "renderer.json"
    _renderer_receipt(renderer, config, gates=True)
    scene_receipt = tmp_path / "scene_receipt.json"
    scene_receipt.write_text(
        json.dumps(
            {
                "status": "grec_scene_table_frozen_pre_score",
                "scene_table_sha256": sha256_file(scene_table),
                "rows": 1,
                "config": config_provenance(config),
            }
        )
    )
    parent_receipt = tmp_path / "parent_receipt.json"
    parent_receipt.write_text(
        json.dumps(
            {
                "status": "grec_parent_bank_frozen_pre_score",
                "parent_manifest_sha256": sha256_file(parent_manifest),
                "scene_table_sha256": sha256_file(scene_table),
                "scene_table_receipt_sha256": sha256_file(scene_receipt),
                "renderer_validation_receipt_sha256": sha256_file(renderer),
                "rows": 1,
                "config": {},
            }
        )
    )
    with pytest.raises(ContractError, match="parent receipt"):
        validate_parent_bank_for_pnoise(
            parent_manifest_path=parent_manifest,
            parent_receipt_path=parent_receipt,
            scene_table_path=scene_table,
            scene_receipt_path=scene_receipt,
            renderer_validation_receipt_path=renderer,
            config=config,
        )
    parent_value = json.loads(parent_receipt.read_text())
    parent_value["config"] = config_provenance(config)
    tampered = pd.read_parquet(parent_manifest)
    tampered.loc[0, "source_re_arcsec"] += 0.01
    tampered.to_parquet(parent_manifest, index=False)
    parent_value["parent_manifest_sha256"] = sha256_file(parent_manifest)
    parent_receipt.write_text(json.dumps(parent_value))
    with pytest.raises(ContractError, match="parameter digest|scene ID"):
        validate_parent_bank_for_pnoise(
            parent_manifest_path=parent_manifest,
            parent_receipt_path=parent_receipt,
            scene_table_path=scene_table,
            scene_receipt_path=scene_receipt,
            renderer_validation_receipt_path=renderer,
            config=config,
        )


def test_analysis_rejects_unbound_parent_manifest_receipt(
    tmp_path: Path, config: dict
):
    threshold = make_threshold_receipt(tmp_path, config)
    parent_manifest = tmp_path / "parents.parquet"
    parent_scores = tmp_path / "scores.parquet"
    pd.DataFrame({"grec_row_id": ["x"]}).to_parquet(parent_manifest, index=False)
    pd.DataFrame(
        {"grec_row_id": ["x"], "detected_at_frozen_threshold": [False]}
    ).to_parquet(parent_scores, index=False)
    render_receipt = tmp_path / "render.json"
    render_receipt.write_text(
        json.dumps(
            {
                "status": "grec_parent_bank_frozen_pre_score",
                "parent_manifest_sha256": "0" * 64,
                "config": config_provenance(config),
            }
        )
    )
    score_receipt = tmp_path / "score.json"
    score_receipt.write_text("{}")
    with pytest.raises(ContractError, match="render receipt"):
        _analyse(
            SimpleNamespace(
                threshold_receipt=str(threshold),
                device=config["scoring"]["authoritative_device"],
                parent_render_receipt=str(render_receipt),
                parent_manifest=str(parent_manifest),
                parent_scores=str(parent_scores),
                parent_score_receipt=str(score_receipt),
                child_manifest=None,
                child_render_receipt=None,
                child_scores=None,
                child_score_receipt=None,
                mcse_receipt=None,
                output=str(tmp_path / "out.json"),
            ),
            config,
        )


def _inference_children(
    scenes: pd.DataFrame, draws: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scores = []
    for index, scene in scenes.iterrows():
        for draw in range(draws):
            row = scene.to_dict()
            row["pnoise_row_id"] = f"pn_{index}_{draw}"
            row["parent_grec_row_id"] = scene["grec_row_id"]
            row["draw_index"] = draw
            rows.append(row)
            scores.append(
                {
                    "pnoise_row_id": row["pnoise_row_id"],
                    "detected_at_frozen_threshold": draw % 2 == 0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(scores)


def test_pnoise_inference_requires_bijective_parent_coverage_and_draw_count(
    tmp_path: Path, config: dict
):
    scenes = make_scenes(tmp_path, config)
    parent_scores = pd.DataFrame(
        {
            "grec_row_id": scenes["grec_row_id"],
            "detected_at_frozen_threshold": False,
        }
    )
    children, scores = _inference_children(scenes, 2)
    missing = children[children["parent_grec_row_id"] != scenes.iloc[0]["grec_row_id"]]
    missing_scores = scores[scores["pnoise_row_id"].isin(missing["pnoise_row_id"])]
    with pytest.raises(ContractError, match="both directions"):
        analyse_pnoise(
            scenes,
            parent_scores,
            missing,
            missing_scores,
            config,
            expected_draw_count=2,
            bootstrap_draws=5,
        )
    incomplete = children.drop(children.index[0])
    incomplete_scores = scores[
        scores["pnoise_row_id"].isin(incomplete["pnoise_row_id"])
    ]
    with pytest.raises(ContractError, match="draw indices|draw count"):
        analyse_pnoise(
            scenes,
            parent_scores,
            incomplete,
            incomplete_scores,
            config,
            expected_draw_count=2,
            bootstrap_draws=5,
        )


def test_one_draw_mcse_is_unavailable_not_zero(tmp_path: Path, config: dict):
    scenes = make_scenes(tmp_path, config)
    parent_scores = pd.DataFrame(
        {
            "grec_row_id": scenes["grec_row_id"],
            "detected_at_frozen_threshold": False,
        }
    )
    children, scores = _inference_children(scenes, 1)
    result = analyse_pnoise(
        scenes,
        parent_scores,
        children,
        scores,
        config,
        expected_draw_count=1,
        bootstrap_draws=5,
    )
    assert result["finite_photon_draw_mcse"] is None
    assert result["finite_photon_draw_mcse_if_unavailable"] == "infinite"
    assert result["finite_photon_draw_precision_gate_passed"] is False


def test_scorer_closure_binds_local_wrappers(config: dict):
    closure = fixed_scorer_closure(config, "cpu")
    wrappers = closure["wrapper_source_sha256"]
    assert {
        "backend.py",
        "scoring.py",
        "threshold.py",
        "inference.py",
    } <= set(wrappers)


def test_global_multiplier_bootstrap_recenters_one_vs_five_hosts():
    rows = [
        {
            "host_galaxy_id": "small_0",
            "host_spatial_group_id": "small",
            "psf_bin": 1.2,
            "depth_bin": 24.0,
            "theta_e_arcsec": 1.0,
            "value": 1.0,
        }
    ]
    rows.extend(
        {
            "host_galaxy_id": f"large_{index}",
            "host_spatial_group_id": "large",
            "psf_bin": 1.2,
            "depth_bin": 24.0,
            "theta_e_arcsec": 1.0,
            "value": 0.0,
        }
        for index in range(5)
    )
    overall, radius, _, design = _global_spatial_multiplier_bootstrap(
        pd.DataFrame(rows), draws=20_000, seed_namespace="one-vs-five"
    )
    assert overall.mean() == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert radius.mean(axis=0)[0] == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert design["draws_recentered_to_equal_host_point_estimator"] is True
    assert design["maximum_absolute_recentering_error"] < 1e-12


def test_global_multiplier_preserves_cross_cell_covariance():
    rows = []
    for cell, psf in (("a", 1.2), ("b", 1.35)):
        rows.extend(
            [
                {
                    "host_galaxy_id": f"{cell}_shared",
                    "host_spatial_group_id": "shared",
                    "psf_bin": psf,
                    "depth_bin": 24.0,
                    "theta_e_arcsec": 1.0,
                    "value": 1.0,
                },
                {
                    "host_galaxy_id": f"{cell}_unique",
                    "host_spatial_group_id": f"unique_{cell}",
                    "psf_bin": psf,
                    "depth_bin": 24.0,
                    "theta_e_arcsec": 1.0,
                    "value": 0.0,
                },
            ]
        )
    _, _, cell_draws, design = _global_spatial_multiplier_bootstrap(
        pd.DataFrame(rows), draws=10_000, seed_namespace="cross-cell"
    )
    covariance = np.cov(cell_draws[:, 0, 0], cell_draws[:, 1, 0])[0, 1]
    assert covariance > 0
    assert design["spatial_groups_crossing_cells"] == 1
    assert design["same_multiplier_applied_across_cells_and_radii"] is True


def test_analysis_device_is_explicit_and_postscore_drift_is_rejected(
    tmp_path: Path, config: dict
):
    threshold = make_threshold_receipt(tmp_path, config)
    scenes = make_scenes(tmp_path, config)
    parent_manifest = tmp_path / "parents.parquet"
    scenes.to_parquet(parent_manifest, index=False)
    scores = tmp_path / "scores.parquet"
    pd.DataFrame(
        {
            "grec_row_id": scenes["grec_row_id"],
            "detected_at_frozen_threshold": False,
        }
    ).to_parquet(scores, index=False)
    render_receipt = tmp_path / "render.json"
    render_receipt.write_text(
        json.dumps(
            {
                "status": "grec_parent_bank_frozen_pre_score",
                "parent_manifest_sha256": sha256_file(parent_manifest),
                "config": config_provenance(config),
            }
        )
    )
    closure = fixed_scorer_closure(
        config, config["scoring"]["authoritative_device"]
    )
    stale_closure = copy.deepcopy(closure)
    stale_closure["wrapper_source_sha256"]["inference.py"] = "0" * 64
    score_receipt = tmp_path / "score_receipt.json"
    score_receipt.write_text(
        json.dumps(
            {
                "status": "frozen_scores_complete",
                "bank": "grec",
                "scores_sha256": sha256_file(scores),
                "bank_manifest_sha256": sha256_file(parent_manifest),
                "bank_receipt_sha256": sha256_file(render_receipt),
                "threshold_receipt_sha256": sha256_file(threshold),
                "scorer_closure": stale_closure,
                "scorer_closure_sha256": sha256_json(stale_closure),
                "requested_device": config["scoring"]["authoritative_device"],
            }
        )
    )
    base_args = dict(
        threshold_receipt=str(threshold),
        parent_render_receipt=str(render_receipt),
        parent_manifest=str(parent_manifest),
        parent_scores=str(scores),
        parent_score_receipt=str(score_receipt),
        child_manifest=None,
        child_render_receipt=None,
        child_scores=None,
        child_score_receipt=None,
        mcse_receipt=None,
        output=str(tmp_path / "out.json"),
    )
    with pytest.raises(ContractError, match="analysis device"):
        _analyse(SimpleNamespace(**base_args, device="cpu"), config)
    with pytest.raises(ContractError, match="parent score receipt"):
        _analyse(
            SimpleNamespace(
                **base_args,
                device=config["scoring"]["authoritative_device"],
            ),
            config,
        )
