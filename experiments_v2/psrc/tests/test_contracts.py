from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psrc.authority import build_execution_lock
from psrc.contracts import (
    GREC_SCENE_RENDER_INPUT_COLUMNS,
    ContractError,
    assert_config_integrity,
    config_provenance,
    fixed_scorer_closure,
    load_config,
    sha256_file,
    sha256_json,
    source_closure,
    stable_id,
    verify_grec_scene_identity,
    verify_parent_grec_authority,
    verify_source_closure,
)
from psrc.draw_count import (
    build_draw_count_receipt,
    validate_draw_count_receipt,
)
from psrc.inference import (
    precision_status,
    validate_analysis_design_binding,
    validate_native_score_table,
)
from psrc.threshold import (
    _load_bound_calibration,
    build_threshold_receipt,
    validate_threshold_receipt,
)
from psrc.utils import sha256_file as utility_sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/psrc_v2_parent_bound.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_config_binds_governing_documents_and_exact_file(config: dict) -> None:
    assert config["schema_version"] == 2
    assert config["status"] == "frozen_pre_scoring"
    assert config_provenance(config)["config_file_sha256"] == sha256_file(CONFIG_PATH)
    assert "config_path" not in config_provenance(config)
    assert not Path(config_provenance(config)["config_relpath"]).is_absolute()
    assert_config_integrity(config)
    assert (
        config["rendering"]["primary_flux_policy"]
        == "fixed_parent_intrinsic_per_band"
    )
    assert "not latent morphology" in config["claim"]["primary"]
    assert utility_sha256_file(str(CONFIG_PATH)) == sha256_file(CONFIG_PATH)


def test_in_memory_and_on_disk_config_drift_fail(
    config: dict, tmp_path: Path
) -> None:
    changed = copy.deepcopy(config)
    changed["threshold"]["alpha"] = 0.002
    with pytest.raises(ContractError, match="in-memory"):
        assert_config_integrity(changed)

    path = tmp_path / "config.yaml"
    path.write_bytes(CONFIG_PATH.read_bytes())
    loaded = load_config(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ContractError, match="file drift"):
        assert_config_integrity(loaded)


def test_execution_freeze_requires_explicit_exact_config_hash(
    tmp_path: Path,
) -> None:
    text = CONFIG_PATH.read_text().replace(
        "status: implementation_candidate_not_execution_authority",
        "status: frozen_pre_scoring",
    )
    path = tmp_path / "frozen.yaml"
    path.write_text(text)
    frozen = load_config(path)
    with pytest.raises(ContractError, match="expected config SHA-256"):
        build_execution_lock(
            expected_config_sha256="0" * 64,
            config=frozen,
            final_host_manifest_path=tmp_path / "missing",
            final_host_receipt_path=tmp_path / "missing",
            scene_table_path=tmp_path / "missing",
            scene_receipt_path=tmp_path / "missing",
            parent_manifest_path=tmp_path / "missing",
            parent_receipt_path=tmp_path / "missing",
            source_manifest_path=tmp_path / "missing",
            source_receipt_path=tmp_path / "missing",
            design_path=tmp_path / "missing",
            design_receipt_path=tmp_path / "missing",
            scorer_receipt_path=tmp_path / "missing",
            threshold_receipt_path=tmp_path / "missing",
            draw_count_receipt_path=tmp_path / "missing",
            output_path=tmp_path / "lock.json",
        )


def test_complete_source_closure_detects_any_digest_change(config: dict) -> None:
    closure = source_closure(config)
    assert "src/psrc/inference.py" in closure["files"]
    assert "src/psrc/scoring.py" in closure["files"]
    assert "pyproject.toml" in closure["files"]
    assert "uv.lock" in closure["files"]
    verify_source_closure(closure, config)
    stale = copy.deepcopy(closure)
    stale["files"]["src/psrc/inference.py"] = "0" * 64
    with pytest.raises(ContractError, match="closure drift"):
        verify_source_closure(stale, config)


def _scene(host: str, radius: float, config: dict) -> dict:
    values = {
        "host_galaxy_id": host,
        "host_parent_id": f"host_{host}",
        "host_pixel_sha256": "1" * 64,
        "host_payload_sha256": "2" * 64,
        "host_spatial_group_id": f"group_{host}",
        "host_spatial_coordinate_id": f"coord_{host}",
        "host_ra_deg": 12.0,
        "host_dec_deg": -3.0,
        "host_pixel_unit": "nanomaggy_per_native_pixel",
        "host_flux_calibration_system": "AB_nanomaggy_zeropoint_22.5",
        "host_calibration_provenance_sha256": "a" * 64,
        "final_role": "experiment_host",
        "psfsize_r": 1.2,
        "psfdepth_r": 1000.0,
        "psf_bin": 1.2,
        "depth_bin": 24.5,
        "depth_mag": 24.5,
        "pixel_scale_arcsec": 0.262,
        "psf_fwhm_r_arcsec": 1.2,
        "psf_fwhm_g_arcsec": 1.26,
        "psf_fwhm_z_arcsec": 1.128,
        "radius_index": config["parent_grec"]["required_radii_arcsec"].index(
            radius
        ),
        "theta_e_arcsec": radius,
        "source_r_mag": 24.0,
        "source_g_minus_r": 1.15,
        "source_r_minus_z": 0.85,
        "source_re_arcsec": 0.2,
        "source_n_sersic": 1.0,
        "source_axis_ratio": 0.7,
        "source_phi_rad": 0.4,
        "source_beta_fraction": 0.2,
        "source_beta_angle_rad": 0.3,
        "source_beta_x_arcsec": radius * 0.2 * np.cos(0.3),
        "source_beta_y_arcsec": radius * 0.2 * np.sin(0.3),
        "source_flux_nmgy_g": 0.1,
        "source_flux_nmgy_r": 0.25,
        "source_flux_nmgy_z": 0.5,
        "source_nuisance_id": f"srcn_{host}",
        "source_parameter_id": f"src_{host}_{radius}",
        "lens_axis_ratio": 0.8,
        "lens_phi_rad": 0.2,
        "shear_g1": 0.01,
        "shear_g2": -0.02,
        "lens_x0_arcsec": 0.01,
        "lens_y0_arcsec": -0.01,
        "lens_nuisance_id": f"lensn_{host}",
        "lens_parameter_id": f"lens_{host}_{radius}",
        "base_draw_key": f"draw_{host}",
        "render_config_id": "render_test",
        "render_contract_sha256": "b" * 64,
        "config_file_sha256": config["parent_grec"][
            "required_grec_config_file_sha256"
        ],
        "config_contract_sha256": config["parent_grec"][
            "required_grec_config_contract_sha256"
        ],
        "added_source_photon_noise": False,
        "added_sky_noise": False,
        "host_pixel_path": "/tmp/not-read-in-contract-test.npz",
    }
    payload = {name: values[name] for name in GREC_SCENE_RENDER_INPUT_COLUMNS}
    values["parameter_row_sha256"] = sha256_json(payload)
    values["scene_id"] = stable_id("scene", payload)
    values["grec_row_id"] = stable_id(
        "grec", [values["scene_id"], values["parameter_row_sha256"]]
    )
    return values


def _parent_authority_files(tmp_path: Path, config: dict) -> dict[str, Path]:
    hosts = tmp_path / "hosts.parquet"
    pd.DataFrame(
        {
            "host_galaxy_id": ["h0"],
            "host_physical_group_id": ["physical_h0"],
            "host_spatial_group_id": ["group_h0"],
            "host_spatial_coordinate_id": ["coord_h0"],
            "host_parent_id": ["host_h0"],
            "host_pixel_sha256": ["1" * 64],
        }
    ).to_parquet(hosts, index=False)
    host_receipt = tmp_path / "host_receipt.json"
    host_receipt.write_text(
        json.dumps(
            {
                "status": config["parent_grec"]["required_final_host_status"],
                "host_manifest_sha256": sha256_file(hosts),
                "detector_scores_used": False,
                "full_tractor_eligibility": "passed",
                "physical_group_checks": "passed",
                "role_overlap_checks": "passed",
                "pre_score_review": "approved",
            }
        )
    )
    rows = [
        _scene("h0", float(radius), config)
        for radius in config["parent_grec"]["required_radii_arcsec"]
    ]
    scenes = tmp_path / "scenes.parquet"
    parents = tmp_path / "parents.parquet"
    pd.DataFrame(rows).to_parquet(scenes, index=False)
    pd.DataFrame(rows).to_parquet(parents, index=False)
    grec_config = {
        "config_file_sha256": config["parent_grec"][
            "required_grec_config_file_sha256"
        ],
        "config_contract_sha256": config["parent_grec"][
            "required_grec_config_contract_sha256"
        ],
        "governing_document_sha256": {},
        "config_path": "/tmp/grec.toml",
    }
    scene_receipt = tmp_path / "scene_receipt.json"
    scene_receipt.write_text(
        json.dumps(
            {
                "status": config["parent_grec"]["required_scene_status"],
                "scene_table_sha256": sha256_file(scenes),
                "host_manifest_sha256": sha256_file(hosts),
                "host_receipt_sha256": sha256_file(host_receipt),
                "host_receipt_status": config["parent_grec"][
                    "required_final_host_status"
                ],
                "detector_scores_used": False,
                "rows": len(rows),
                "config": grec_config,
            }
        )
    )
    parent_receipt = tmp_path / "parent_receipt.json"
    parent_receipt.write_text(
        json.dumps(
            {
                "status": config["parent_grec"]["required_parent_bank_status"],
                "parent_manifest_sha256": sha256_file(parents),
                "scene_table_sha256": sha256_file(scenes),
                "scene_table_receipt_sha256": sha256_file(scene_receipt),
                "rows": len(rows),
                "config": grec_config,
            }
        )
    )
    return {
        "final_host_manifest_path": hosts,
        "final_host_receipt_path": host_receipt,
        "scene_table_path": scenes,
        "scene_receipt_path": scene_receipt,
        "parent_manifest_path": parents,
        "parent_receipt_path": parent_receipt,
    }


def test_parent_authority_recomputes_every_grec_scene_identity(
    tmp_path: Path, config: dict
) -> None:
    paths = _parent_authority_files(tmp_path, config)
    scenes, authority = verify_parent_grec_authority(**paths, config=config)
    assert len(scenes) == 11
    assert authority["scene_table_sha256"] == sha256_file(paths["scene_table_path"])
    verify_grec_scene_identity(scenes.iloc[0].to_dict())


def test_parent_parameter_tamper_fails_even_if_outer_hashes_are_rewritten(
    tmp_path: Path, config: dict
) -> None:
    paths = _parent_authority_files(tmp_path, config)
    scenes = pd.read_parquet(paths["scene_table_path"])
    scenes.loc[0, "source_r_mag"] += 0.1
    scenes.to_parquet(paths["scene_table_path"], index=False)
    scenes.to_parquet(paths["parent_manifest_path"], index=False)
    scene_receipt = json.loads(paths["scene_receipt_path"].read_text())
    scene_receipt["scene_table_sha256"] = sha256_file(paths["scene_table_path"])
    paths["scene_receipt_path"].write_text(json.dumps(scene_receipt))
    parent_receipt = json.loads(paths["parent_receipt_path"].read_text())
    parent_receipt["parent_manifest_sha256"] = sha256_file(
        paths["parent_manifest_path"]
    )
    parent_receipt["scene_table_sha256"] = sha256_file(paths["scene_table_path"])
    parent_receipt["scene_table_receipt_sha256"] = sha256_file(
        paths["scene_receipt_path"]
    )
    paths["parent_receipt_path"].write_text(json.dumps(parent_receipt))
    with pytest.raises(ContractError, match="identity drift"):
        verify_parent_grec_authority(**paths, config=config)


def _threshold_inputs(tmp_path: Path, config: dict) -> dict[str, Path]:
    if not Path(config["scoring"]["checkpoint"]).resolve().is_file():
        pytest.skip("external primary checkpoint is not staged")
    tmp_path.mkdir(parents=True, exist_ok=True)
    n = 20_000
    manifest = tmp_path / "calibration.parquet"
    logits = tmp_path / "logits.parquet"
    ids = [f"c{index}" for index in range(n)]
    values = np.linspace(-8, 8, n, dtype=np.float32)
    pd.DataFrame(
        {
            "calibration_id": ids,
            "global_role": "threshold_calibration",
            "host_pixel_sha256": [f"{index:064x}" for index in range(n)],
            "evaluation_spatial_group_id": [f"g{index // 4}" for index in range(n)],
        }
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        {
            "calibration_id": ids,
            "raw_logit_float32": values,
            "raw_logit_float32_bits": values.view(np.uint32),
        }
    ).to_parquet(logits, index=False)
    manifest_receipt = tmp_path / "manifest_receipt.json"
    manifest_receipt.write_text(
        json.dumps(
            {
                "status": config["threshold"][
                    "required_calibration_manifest_status"
                ],
                "calibration_manifest_sha256": sha256_file(manifest),
            }
        )
    )
    closure = fixed_scorer_closure(config, verify_runtime=False)
    scorer_receipt = tmp_path / "scorer_receipt.json"
    scorer_receipt.write_text(
        json.dumps(
            {
                "status": "fixed_scorer_verified_design_only",
                "config": config_provenance(config),
                "scorer_closure": closure,
                "scorer_closure_sha256": sha256_json(closure),
            }
        )
    )
    score_receipt = tmp_path / "score_receipt.json"
    score_receipt.write_text(
        json.dumps(
            {
                "status": config["threshold"][
                    "required_calibration_score_status"
                ],
                "calibration_manifest_sha256": sha256_file(manifest),
                "calibration_logits_sha256": sha256_file(logits),
                "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
                "scorer_closure": closure,
                "scorer_closure_sha256": sha256_json(closure),
            }
        )
    )
    return {
        "manifest_path": manifest,
        "manifest_receipt_path": manifest_receipt,
        "logits_path": logits,
        "score_receipt_path": score_receipt,
        "scorer_receipt_path": scorer_receipt,
    }


def test_threshold_receipt_recomputes_all_native_float32_evidence(
    tmp_path: Path, config: dict
) -> None:
    paths = _threshold_inputs(tmp_path, config)
    output = tmp_path / "threshold.json"
    receipt = build_threshold_receipt(
        **paths, config=config, output_path=output, final_status=False
    )
    validated, threshold = validate_threshold_receipt(
        output, config, require_final=False
    )
    assert validated["n"] == 20_000
    assert validated["k_floor_alpha_n"] == 20
    assert validated["s_k_plus_1_float32_bits"] != validated[
        "logit_threshold_float32_bits"
    ]
    assert int(threshold.view(np.uint32)) == receipt[
        "logit_threshold_float32_bits"
    ]


def test_threshold_accepts_common_grec_scorer_closure_shape(
    tmp_path: Path, config: dict
) -> None:
    paths = _threshold_inputs(tmp_path, config)
    score_receipt = json.loads(paths["score_receipt_path"].read_text())
    closure = score_receipt["scorer_closure"]
    closure["source_sha256"] = closure.pop("scorer_source_sha256")
    closure["comparison"] = closure.pop("comparator")
    closure["runtime_versions"] = {
        key: closure["runtime_versions"][key]
        for key in ("python", "numpy", "pandas", "pyarrow", "torch", "torchvision")
    }
    score_receipt["scorer_closure"] = closure
    score_receipt["scorer_closure_sha256"] = sha256_json(closure)
    paths["score_receipt_path"].write_text(json.dumps(score_receipt))
    output = tmp_path / "threshold_grec.json"
    build_threshold_receipt(
        **paths, config=config, output_path=output, final_status=False
    )
    validate_threshold_receipt(output, config, require_final=False)


def test_real_draw_count_producer_binds_pilot_design_scorer_threshold_and_target(
    tmp_path: Path, config: dict
) -> None:
    threshold_inputs = _threshold_inputs(tmp_path / "threshold", config)
    threshold_path = tmp_path / "threshold.json"
    build_threshold_receipt(
        **threshold_inputs,
        config=config,
        output_path=threshold_path,
        final_status=False,
    )
    common = {
        "host_cell_id": ["c0", "c0", "c1", "c1"],
        "theta_e_arcsec": [0.5, 1.0, 0.5, 1.0],
        "source_stratum": ["s0", "s1", "s0", "s1"],
        "parent_scene_analysis_weight": 1.0,
    }
    final_design = pd.DataFrame(
        {
            **common,
            "pair_id": [f"final-p{index}" for index in range(4)],
            "parent_grec_scene_id": [f"final-scene-{index}" for index in range(4)],
            "host_galaxy_id": [f"final-host-{index}" for index in range(4)],
            "host_spatial_group_id": [f"final-group-{index}" for index in range(4)],
            "source_id": [f"source-{index}" for index in range(4)],
            "host_pixel_sha256": [f"{100 + index:064x}" for index in range(4)],
            "host_physical_group_id": [f"final-physical-{index}" for index in range(4)],
            "host_spatial_coordinate_id": [f"final-coordinate-{index}" for index in range(4)],
            "host_parent_id": [f"final-parent-{index}" for index in range(4)],
            "source_image_sha256": [f"{200 + index:064x}" for index in range(4)],
            "source_map_sha256": [f"{300 + index:064x}" for index in range(4)],
            "derivative_ancestry_sha256": [f"{400 + index:064x}" for index in range(4)],
        }
    )
    pilot_design = pd.DataFrame(
        {
            **common,
            "pair_id": [f"pilot-p{index}" for index in range(4)],
            "parent_grec_scene_id": [f"pilot-scene-{index}" for index in range(4)],
            "host_galaxy_id": [f"pilot-host-{index}" for index in range(4)],
            "host_spatial_group_id": [f"pilot-group-{index}" for index in range(4)],
            "source_id": [f"pilot-source-{index}" for index in range(4)],
            "host_pixel_sha256": [f"{500 + index:064x}" for index in range(4)],
            "host_physical_group_id": [f"pilot-physical-{index}" for index in range(4)],
            "host_spatial_coordinate_id": [f"pilot-coordinate-{index}" for index in range(4)],
            "host_parent_id": [f"pilot-parent-{index}" for index in range(4)],
            "source_image_sha256": [f"{600 + index:064x}" for index in range(4)],
            "source_map_sha256": [f"{700 + index:064x}" for index in range(4)],
            "derivative_ancestry_sha256": [f"{800 + index:064x}" for index in range(4)],
            "global_role": "design",
        }
    )

    def write_design(
        frame: pd.DataFrame, name: str, status: str
    ) -> tuple[Path, Path]:
        path = tmp_path / f"{name}.parquet"
        receipt_path = tmp_path / f"{name}_receipt.json"
        frame.to_parquet(path, index=False)
        receipt_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "pair_design_sha256": sha256_file(path),
                    "config": config_provenance(config),
                    "source_closure": source_closure(config),
                }
            )
        )
        return path, receipt_path

    final_design_path, final_design_receipt_path = write_design(
        final_design, "final_design", "psrc_parent_child_design_complete_design_only"
    )
    pilot_design_path, pilot_design_receipt_path = write_design(
        pilot_design,
        "pilot_design",
        config["photon_noise"]["required_pilot_design_receipt_status"],
    )
    identity_columns = config["photon_noise"][
        "pilot_anti_join_identity_columns"
    ]
    role_ledger = pd.concat(
        [
            pilot_design[identity_columns].assign(global_role="design"),
            final_design[identity_columns].assign(global_role="experiment"),
        ],
        ignore_index=True,
    )
    role_ledger_path = tmp_path / "role_ledger.parquet"
    role_ledger.to_parquet(role_ledger_path, index=False)
    role_ledger_receipt_path = tmp_path / "ROLE_LEDGER_RECEIPT.json"
    role_ledger_receipt_path.write_text(
        json.dumps(
            {
                "status": config["photon_noise"][
                    "required_role_ledger_receipt_status"
                ],
                "role_ledger_sha256": sha256_file(role_ledger_path),
                "detector_scores_used": False,
                "final_outcomes_used": False,
                "config": config_provenance(config),
                "source_closure": source_closure(config),
            }
        )
    )
    pilot_path = tmp_path / "pilot_scores.parquet"
    pilot = pd.DataFrame(
        [
            {
                "pair_id": pair,
                "draw_index": draw,
                "structured_detected": False,
                "sersic_detected": False,
            }
            for pair in pilot_design["pair_id"]
            for draw in range(64)
        ]
    )
    pilot["structured_raw_logit_float32"] = np.float32(-1.0)
    pilot["structured_raw_logit_float32_bits"] = (
        pilot["structured_raw_logit_float32"]
        .to_numpy(dtype=np.float32)
        .view(np.uint32)
    )
    pilot["sersic_raw_logit_float32"] = np.float32(-1.0)
    pilot["sersic_raw_logit_float32_bits"] = (
        pilot["sersic_raw_logit_float32"]
        .to_numpy(dtype=np.float32)
        .view(np.uint32)
    )
    pilot.to_parquet(pilot_path, index=False)
    pilot_receipt_path = tmp_path / "pilot_receipt.json"
    pilot_receipt_path.write_text(
        json.dumps(
            {
                "status": config["photon_noise"][
                    "required_pilot_score_receipt_status"
                ],
                "global_role": "design",
                "pilot_scores_sha256": sha256_file(pilot_path),
                "pilot_design_sha256": sha256_file(pilot_design_path),
                "pilot_design_receipt_sha256": sha256_file(
                    pilot_design_receipt_path
                ),
                "final_design_sha256": sha256_file(final_design_path),
                "final_design_receipt_sha256": sha256_file(
                    final_design_receipt_path
                ),
                "role_ledger_sha256": sha256_file(role_ledger_path),
                "role_ledger_receipt_sha256": sha256_file(
                    role_ledger_receipt_path
                ),
                "scorer_receipt_sha256": sha256_file(
                    threshold_inputs["scorer_receipt_path"]
                ),
                "threshold_receipt_sha256": sha256_file(threshold_path),
                "candidate_draw_counts": config["photon_noise"][
                    "candidate_draw_counts"
                ],
                "precision_target": config["photon_noise"][
                    "paired_mcse_target_absolute_risk"
                ],
                "rows": len(pilot),
                "pairs": len(pilot_design),
                "draw_count": 64,
                "threshold_float32_bits": json.loads(
                    threshold_path.read_text()
                )["logit_threshold_float32_bits"],
                "comparison": ">=",
                "scorer_closure_sha256": json.loads(
                    threshold_inputs["scorer_receipt_path"].read_text()
                )["scorer_closure_sha256"],
                "final_outcomes_used": False,
                "config": config_provenance(config),
                "source_closure": source_closure(config),
            }
        )
    )
    output = tmp_path / "draw_count.json"
    receipt = build_draw_count_receipt(
        pilot_scores_path=pilot_path,
        pilot_score_receipt_path=pilot_receipt_path,
        pilot_design_path=pilot_design_path,
        pilot_design_receipt_path=pilot_design_receipt_path,
        final_design_path=final_design_path,
        final_design_receipt_path=final_design_receipt_path,
        role_ledger_path=role_ledger_path,
        role_ledger_receipt_path=role_ledger_receipt_path,
        scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
        threshold_receipt_path=threshold_path,
        config=config,
        output_path=output,
        canary_mode=True,
    )
    original_pilot_receipt = json.loads(pilot_receipt_path.read_text())
    contaminated_receipt = dict(original_pilot_receipt)
    contaminated_receipt["final_outcomes_used"] = True
    pilot_receipt_path.write_text(json.dumps(contaminated_receipt))
    with pytest.raises(ContractError, match="pilot score receipt"):
        build_draw_count_receipt(
            pilot_scores_path=pilot_path,
            pilot_score_receipt_path=pilot_receipt_path,
            pilot_design_path=pilot_design_path,
            pilot_design_receipt_path=pilot_design_receipt_path,
            final_design_path=final_design_path,
            final_design_receipt_path=final_design_receipt_path,
            role_ledger_path=role_ledger_path,
            role_ledger_receipt_path=role_ledger_receipt_path,
            scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
            threshold_receipt_path=threshold_path,
            config=config,
            output_path=tmp_path / "contaminated.json",
            canary_mode=True,
        )
    pilot_receipt_path.write_text(json.dumps(original_pilot_receipt))
    original_pilot_design_bytes = pilot_design_path.read_bytes()
    original_pilot_design_receipt = pilot_design_receipt_path.read_text()
    overlapping = pilot_design.copy()
    overlapping.loc[0, "host_galaxy_id"] = final_design.loc[
        0, "host_galaxy_id"
    ]
    overlapping.to_parquet(pilot_design_path, index=False)
    changed_design_receipt = json.loads(original_pilot_design_receipt)
    changed_design_receipt["pair_design_sha256"] = sha256_file(
        pilot_design_path
    )
    pilot_design_receipt_path.write_text(json.dumps(changed_design_receipt))
    changed_pilot_receipt = dict(original_pilot_receipt)
    changed_pilot_receipt["pilot_design_sha256"] = sha256_file(
        pilot_design_path
    )
    changed_pilot_receipt["pilot_design_receipt_sha256"] = sha256_file(
        pilot_design_receipt_path
    )
    pilot_receipt_path.write_text(json.dumps(changed_pilot_receipt))
    with pytest.raises(ContractError, match="overlaps final design"):
        build_draw_count_receipt(
            pilot_scores_path=pilot_path,
            pilot_score_receipt_path=pilot_receipt_path,
            pilot_design_path=pilot_design_path,
            pilot_design_receipt_path=pilot_design_receipt_path,
            final_design_path=final_design_path,
            final_design_receipt_path=final_design_receipt_path,
            role_ledger_path=role_ledger_path,
            role_ledger_receipt_path=role_ledger_receipt_path,
            scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
            threshold_receipt_path=threshold_path,
            config=config,
            output_path=tmp_path / "overlap.json",
            canary_mode=True,
        )
    pilot_design_path.write_bytes(original_pilot_design_bytes)
    pilot_design_receipt_path.write_text(original_pilot_design_receipt)
    pilot_receipt_path.write_text(json.dumps(original_pilot_receipt))
    rebound = pilot_design.copy()
    rebound.loc[0, "source_map_sha256"] = "f" * 64
    rebound.loc[0, "derivative_ancestry_sha256"] = "e" * 64
    rebound.to_parquet(pilot_design_path, index=False)
    rebound_design_receipt = json.loads(original_pilot_design_receipt)
    rebound_design_receipt["pair_design_sha256"] = sha256_file(
        pilot_design_path
    )
    pilot_design_receipt_path.write_text(json.dumps(rebound_design_receipt))
    rebound_pilot_receipt = dict(original_pilot_receipt)
    rebound_pilot_receipt["pilot_design_sha256"] = sha256_file(
        pilot_design_path
    )
    rebound_pilot_receipt["pilot_design_receipt_sha256"] = sha256_file(
        pilot_design_receipt_path
    )
    pilot_receipt_path.write_text(json.dumps(rebound_pilot_receipt))
    with pytest.raises(ContractError, match="role-ledger identity drift"):
        build_draw_count_receipt(
            pilot_scores_path=pilot_path,
            pilot_score_receipt_path=pilot_receipt_path,
            pilot_design_path=pilot_design_path,
            pilot_design_receipt_path=pilot_design_receipt_path,
            final_design_path=final_design_path,
            final_design_receipt_path=final_design_receipt_path,
            role_ledger_path=role_ledger_path,
            role_ledger_receipt_path=role_ledger_receipt_path,
            scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
            threshold_receipt_path=threshold_path,
            config=config,
            output_path=tmp_path / "rebound.json",
            canary_mode=True,
        )
    pilot_design_path.write_bytes(original_pilot_design_bytes)
    pilot_design_receipt_path.write_text(original_pilot_design_receipt)
    pilot_receipt_path.write_text(json.dumps(original_pilot_receipt))
    assert receipt["draw_count"] == 8
    assert receipt["candidate_draw_counts"] == [8, 16, 32, 64]
    assert receipt["precision_target"] == 0.01
    assert receipt["pilot_is_separate_design_role"] is True
    assert receipt["distribution"]["anti_join_passed"] is True
    validate_draw_count_receipt(
        output,
        config,
        expected_final_design_path=final_design_path,
        expected_final_design_receipt_path=final_design_receipt_path,
        expected_scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
        expected_threshold_receipt_path=threshold_path,
        production=False,
    )
    with pytest.raises(ContractError, match="production-authorized"):
        validate_draw_count_receipt(
            output,
            config,
            expected_final_design_path=final_design_path,
            expected_final_design_receipt_path=final_design_receipt_path,
            expected_scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
            expected_threshold_receipt_path=threshold_path,
            production=True,
        )
    pilot.loc[0, "structured_detected"] = True
    pilot.to_parquet(pilot_path, index=False)
    with pytest.raises(ContractError, match="artifact drift"):
        validate_draw_count_receipt(
            output,
            config,
            expected_final_design_path=final_design_path,
            expected_final_design_receipt_path=final_design_receipt_path,
            expected_scorer_receipt_path=threshold_inputs["scorer_receipt_path"],
            expected_threshold_receipt_path=threshold_path,
            production=False,
        )


def test_analysis_precision_cannot_report_complete_above_target() -> None:
    assert precision_status(0.009, 0.01) == ("complete", True)
    assert precision_status(0.011, 0.01) == (
        "insufficient_photon_mcse_not_complete",
        False,
    )
    assert precision_status(float("inf"), 0.01)[1] is False


def test_threshold_rejects_float64_and_bit_tamper(
    tmp_path: Path, config: dict
) -> None:
    paths = _threshold_inputs(tmp_path, config)
    table = pd.read_parquet(paths["logits_path"])
    table["raw_logit_float32"] = table["raw_logit_float32"].astype(np.float64)
    table.to_parquet(paths["logits_path"], index=False)
    with pytest.raises(ContractError, match="native float32"):
        _load_bound_calibration(
            paths["manifest_path"], paths["logits_path"], config
        )
    paths = _threshold_inputs(tmp_path / "bits", config)
    table = pd.read_parquet(paths["logits_path"])
    table.loc[0, "raw_logit_float32_bits"] ^= np.uint32(1)
    table.to_parquet(paths["logits_path"], index=False)
    with pytest.raises(ContractError, match="bits disagree"):
        _load_bound_calibration(
            paths["manifest_path"], paths["logits_path"], config
        )


def _score_rows(draws: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rendered = pd.DataFrame(
        [
            {
                "pair_id": pair,
                "rendered_pair_row_id": f"render-{pair}",
                "draw_index": draw,
            }
            for pair in ("p0", "p1")
            for draw in range(draws)
        ]
    )
    structured = np.array([-1, 0, 1, 2, 3, 4], dtype=np.float32)
    sersic = structured + np.float32(0.5)
    scores = rendered.copy()
    scores["structured_raw_logit_float32"] = structured
    scores["structured_raw_logit_float32_bits"] = structured.view(np.uint32)
    scores["sersic_raw_logit_float32"] = sersic
    scores["sersic_raw_logit_float32_bits"] = sersic.view(np.uint32)
    scores["structured_detected"] = structured >= np.float32(1)
    scores["sersic_detected"] = sersic >= np.float32(1)
    return rendered, scores


def test_score_table_requires_native_bits_complete_draws_and_rendered_ids() -> None:
    rendered, scores = _score_rows()
    validate_native_score_table(
        scores, rendered, draw_count=3, threshold=np.float32(1)
    )
    missing = scores.drop(index=0)
    with pytest.raises(ContractError, match="incomplete draw"):
        validate_native_score_table(
            missing, rendered, draw_count=3, threshold=np.float32(1)
        )
    bad_bits = scores.copy()
    bad_bits.loc[0, "structured_raw_logit_float32_bits"] ^= np.uint32(1)
    with pytest.raises(ContractError, match="bits disagree"):
        validate_native_score_table(
            bad_bits, rendered, draw_count=3, threshold=np.float32(1)
        )
    cast = scores.copy()
    cast["structured_raw_logit_float32"] = cast[
        "structured_raw_logit_float32"
    ].astype(np.float64)
    with pytest.raises(ContractError, match="native float32"):
        validate_native_score_table(
            cast, rendered, draw_count=3, threshold=np.float32(1)
        )
    wrong_render = scores.copy()
    wrong_render.loc[0, "rendered_pair_row_id"] = "another"
    with pytest.raises(ContractError, match="rendered pair"):
        validate_native_score_table(
            wrong_render, rendered, draw_count=3, threshold=np.float32(1)
        )


def test_analysis_design_and_receipt_must_match_execution_lock(
    tmp_path: Path,
) -> None:
    design = tmp_path / "design"
    receipt = tmp_path / "receipt"
    design.write_text("design")
    receipt.write_text("receipt")
    lock = {
        "bindings": {
            "pair_design": {"sha256": sha256_file(design)},
            "pair_design_receipt": {"sha256": sha256_file(receipt)},
        }
    }
    validate_analysis_design_binding(
        design_path=design,
        design_receipt_path=receipt,
        execution_lock=lock,
    )
    design.write_text("drift")
    with pytest.raises(ContractError, match="execution lock"):
        validate_analysis_design_binding(
            design_path=design,
            design_receipt_path=receipt,
            execution_lock=lock,
        )
