from __future__ import annotations

import argparse
import copy
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backend import backend_profile
from .contracts import (
    ContractError,
    atomic_write_json,
    config_provenance,
    load_config,
    runtime_record,
    sha256_file,
    sha256_json,
)
from .hosts import prepare_final_hosts
from .inference import (
    _global_spatial_multiplier_bootstrap,
    analyse_grec,
    analyse_pnoise,
)
from .pnoise import (
    choose_draw_count,
    draw_photon_arc,
    gains_from_config,
    render_photon_children,
    verify_design_role_ledger,
)
from .rendering import adaptive_render_source_expectation, render_parent_bank
from .scene_table import bind_scene_identity, build_scene_table
from .scoring import fixed_scorer_closure
from .threshold import build_threshold_receipt
from .validation import validate_renderer_design


def _psfdepth_for_mag(depth_mag: float) -> float:
    return float((5.0 * 10.0 ** ((depth_mag - 22.5) / 2.5)) ** 2)


def _synthetic_hosts(root: Path, config: dict) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(419)
    calibration = root / "host_calibration.json"
    atomic_write_json(
        calibration,
        {
            "status": config["host"]["required_calibration_receipt_status"],
            "pixel_unit": config["host"]["pixel_unit"],
            "flux_calibration_system": config["host"]["flux_calibration_system"],
            "authority": "synthetic_design_canary",
        },
    )
    calibration_sha = sha256_file(calibration)
    for index in range(4):
        host = rng.normal(0.0, 0.03, size=(101, 101, 3)).astype(np.float32)
        path = root / f"host_{index}.npz"
        np.savez_compressed(path, cutout=host)
        rows.append(
            {
                "galaxy_id": f"design_host_{index}",
                "local_relpath": path.name,
                "payload_sha256": sha256_file(path),
                "evaluation_spatial_group_id": f"design_group_{index}",
                "final_role": "experiment_host",
                "psfsize_r": 1.20 if index < 2 else 1.35,
                "psfdepth_r": _psfdepth_for_mag(24.0),
                "pixel_scale": 0.262,
                "ra": 10.0 + index * 0.1,
                "dec": -5.0 + index * 0.1,
                "pixel_unit": config["host"]["pixel_unit"],
                "flux_calibration_system": config["host"][
                    "flux_calibration_system"
                ],
                "calibration_provenance_path": str(calibration),
                "calibration_provenance_sha256": calibration_sha,
            }
        )
    return prepare_final_hosts(
        pd.DataFrame(rows),
        root,
        config,
        expected_hosts=4,
        expected_cells=2,
        hosts_per_cell=2,
    )


def run(config_path: str | Path, output_path: str | Path) -> dict:
    config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="grec_pnoise_canary_") as temporary:
        root = Path(temporary)
        hosts = _synthetic_hosts(root, config)
        hosts["final_role"] = "design"
        scenes = build_scene_table(
            hosts,
            config,
            expected_hosts=4,
            expected_cells=2,
            hosts_per_cell=2,
            required_role="design",
        )
        original_scene = scenes.iloc[0].to_dict()
        psf_changed_scene = dict(original_scene)
        psf_changed_scene["psf_fwhm_r_arcsec"] += 0.1
        psf_changed_scene["psf_fwhm_g_arcsec"] = (
            psf_changed_scene["psf_fwhm_r_arcsec"]
            * config["render"]["psf_fwhm_scale_g"]
        )
        psf_changed_scene["psf_fwhm_z_arcsec"] = (
            psf_changed_scene["psf_fwhm_r_arcsec"]
            * config["render"]["psf_fwhm_scale_z"]
        )
        psf_changed_scene = bind_scene_identity(psf_changed_scene)
        psf_identity_bound = (
            psf_changed_scene["scene_id"] != original_scene["scene_id"]
        )
        design_ledger = root / "design_role_ledger.parquet"
        scenes[["scene_id", "host_galaxy_id"]].assign(
            global_role="design"
        ).to_parquet(design_ledger, index=False)
        design_ledger_receipt = root / "DESIGN_ROLE_LEDGER_RECEIPT.json"
        atomic_write_json(
            design_ledger_receipt,
            {
                "status": "design_role_ledger_frozen",
                "design_role_ledger_sha256": sha256_file(design_ledger),
                "config_contract_sha256": config_provenance(config)[
                    "config_contract_sha256"
                ],
                "authority": "synthetic_canary_only",
            },
        )
        renderer_validation = validate_renderer_design(
            scenes,
            design_role_ledger_path=design_ledger,
            design_role_receipt_path=design_ledger_receipt,
            config=config,
            render_device=config["render"]["authoritative_device"],
            logit_device=config["scoring"]["authoritative_device"],
            authoritative_design=False,
        )
        start = time.perf_counter()
        parents, render_receipt = render_parent_bank(
            scenes, config, root / "rendered", limit=len(scenes)
        )
        render_seconds = time.perf_counter() - start
        draw_receipt = {
            "selected_draw_count": 2,
            "_receipt_sha256": "design-canary-not-frozen",
            "threshold_receipt_sha256": "synthetic-canary",
        }
        photon_start = time.perf_counter()
        children, photon_receipt = render_photon_children(
            parents,
            config,
            root / "photon",
            draw_receipt,
            limit=len(parents),
            canary_label="design_only_synthetic_full_path",
        )
        photon_seconds = time.perf_counter() - photon_start

        gains = gains_from_config(config)
        expected = np.broadcast_to(
            np.array([0.02, 0.2, 2.0], dtype=np.float32)[:, None, None],
            (3, 1, 200_000),
        ).copy()
        sample, _ = draw_photon_arc(
            expected,
            gains,
            stream_namespace=config["pnoise"]["photon_stream_namespace"],
            scene_id="synthetic_photon_law",
            draw_index=0,
        )
        observed_mean = sample.mean(axis=(1, 2), dtype=np.float64)
        observed_variance = sample.var(axis=(1, 2), ddof=1, dtype=np.float64)
        expected_vector = expected[:, 0, 0].astype(np.float64)
        theoretical_variance = expected_vector / gains
        mean_z = np.abs(observed_mean - expected_vector) / np.sqrt(
            theoretical_variance / sample.shape[-1]
        )
        variance_relative_error = np.abs(
            observed_variance / theoretical_variance - 1.0
        )
        law_passed = bool(
            np.all(mean_z < 6.0) and np.all(variance_relative_error < 0.02)
        )

        pilot_rng = np.random.default_rng(77)
        scene_probability = pilot_rng.uniform(0.1, 0.9, size=1000)
        physical = (
            pilot_rng.random((1000, 64)) < scene_probability[:, None]
        ).astype(np.uint8)
        baseline = (pilot_rng.random(1000) < scene_probability).astype(np.uint8)
        pilot = choose_draw_count(
            physical,
            baseline,
            [f"design_{index}" for index in range(1000)],
            [1, 2, 4, 8, 16, 32, 64],
            absolute_mcse_max=0.002,
            fraction_of_sampling_se_max=0.20,
        )
        host_coupling_passed = bool(
            children["parent_scene_id"].isin(parents["scene_id"]).all()
            and children["host_pixel_sha256"].isin(
                parents["host_pixel_sha256"]
            ).all()
            and children["source_parameter_id"].notna().all()
            and children["lens_parameter_id"].notna().all()
        )

        # Recompute a complete threshold receipt from bound synthetic calibration
        # artifacts. No final or catalogue data are scored.
        n_calibration = int(config["threshold"]["minimum_calibration_rows"])
        calibration_manifest = root / "calibration_manifest.parquet"
        calibration_ids = np.array(
            [f"cal_{index:05d}" for index in range(n_calibration)]
        )
        pd.DataFrame(
            {
                "calibration_id": calibration_ids,
                "global_role": "threshold_calibration",
                "host_pixel_sha256": [
                    f"{index:064x}" for index in range(n_calibration)
                ],
                "evaluation_spatial_group_id": [
                    f"cal_group_{index}" for index in range(n_calibration)
                ],
            }
        ).to_parquet(calibration_manifest, index=False)
        calibration_manifest_receipt = root / "calibration_manifest_receipt.json"
        atomic_write_json(
            calibration_manifest_receipt,
            {
                "status": config["threshold"][
                    "required_calibration_manifest_status"
                ],
                "calibration_manifest_sha256": sha256_file(
                    calibration_manifest
                ),
                "config": config_provenance(config),
            },
        )
        calibration_values = np.linspace(
            -10.0, 10.0, n_calibration, dtype=np.float32
        )
        calibration_logits = root / "calibration_logits.parquet"
        pd.DataFrame(
            {
                "calibration_id": calibration_ids,
                "raw_logit_float32": calibration_values,
                "raw_logit_float32_bits": calibration_values.view(np.uint32),
            }
        ).to_parquet(calibration_logits, index=False)
        scoring_device = config["scoring"]["authoritative_device"]
        closure = fixed_scorer_closure(config, scoring_device)
        calibration_score_receipt = root / "calibration_score_receipt.json"
        atomic_write_json(
            calibration_score_receipt,
            {
                "status": config["threshold"][
                    "required_calibration_score_status"
                ],
                "calibration_manifest_sha256": sha256_file(
                    calibration_manifest
                ),
                "calibration_logits_sha256": sha256_file(calibration_logits),
                "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
                "scorer_closure": closure,
                "scorer_closure_sha256": sha256_json(closure),
                "backend_profile": backend_profile(config, scoring_device),
                "config": config_provenance(config),
            },
        )
        approval_receipt = root / "external_canary_approval.json"
        atomic_write_json(
            approval_receipt,
            {
                "status": config["approval"]["canary_status"],
                "scope": config["approval"]["required_scope"],
                "calibration_manifest_sha256": sha256_file(
                    calibration_manifest
                ),
                "calibration_logits_sha256": sha256_file(calibration_logits),
                "calibration_score_receipt_sha256": sha256_file(
                    calibration_score_receipt
                ),
                "config_contract_sha256": config_provenance(config)[
                    "config_contract_sha256"
                ],
                "canary_only": True,
                "release_authorized": False,
                "approved_by": "synthetic-canary-custodian",
                "approved_at_utc": "2026-09-16T00:00:00Z",
            },
        )
        threshold_receipt_path = root / "threshold_receipt.json"
        threshold_receipt = build_threshold_receipt(
            manifest_path=calibration_manifest,
            manifest_receipt_path=calibration_manifest_receipt,
            logits_path=calibration_logits,
            score_receipt_path=calibration_score_receipt,
            approval_receipt_path=approval_receipt,
            config=config,
            output_path=threshold_receipt_path,
            canary_mode=True,
        )
        threshold = np.array(
            [threshold_receipt["logit_threshold_float32_bits"]],
            dtype=np.uint32,
        ).view(np.float32)[0]
        role_receipt = verify_design_role_ledger(
            scenes["scene_id"],
            scenes["host_galaxy_id"],
            design_ledger,
            design_ledger_receipt,
            config,
        )
        stale_config_rejected = False
        stale = copy.deepcopy(config)
        stale["render"]["gaussian_truncate_sigma"] = 7.0
        try:
            adaptive_render_source_expectation(
                scenes.iloc[0].to_dict(), (101, 101, 3), stale
            )
        except ContractError:
            stale_config_rejected = True

        parent_inference_manifest = scenes.copy()
        parent_decisions = (
            np.arange(len(scenes)) % 5 == 0
        )
        parent_scores = pd.DataFrame(
            {
                "grec_row_id": scenes["grec_row_id"],
                "detected_at_frozen_threshold": parent_decisions,
            }
        )
        child_inference_rows = []
        child_score_rows = []
        for scene_index, scene in scenes.iterrows():
            for draw_index in range(2):
                row = scene.to_dict()
                row["pnoise_row_id"] = f"pn_{scene_index}_{draw_index}"
                row["parent_grec_row_id"] = scene["grec_row_id"]
                row["draw_index"] = draw_index
                child_inference_rows.append(row)
                child_score_rows.append(
                    {
                        "pnoise_row_id": row["pnoise_row_id"],
                        "detected_at_frozen_threshold": bool(
                            parent_decisions[scene_index]
                            if draw_index == 0
                            else not parent_decisions[scene_index]
                        ),
                    }
                )
        grec_inference = analyse_grec(
            parent_inference_manifest,
            parent_scores,
            config,
            bootstrap_draws=100,
        )
        pnoise_inference = analyse_pnoise(
            parent_inference_manifest,
            parent_scores,
            pd.DataFrame(child_inference_rows),
            pd.DataFrame(child_score_rows),
            config,
            expected_draw_count=2,
            bootstrap_draws=100,
        )
        counterexample_rows = [
            {
                "host_galaxy_id": "small_0",
                "host_spatial_group_id": "small",
                "psf_bin": 1.2,
                "depth_bin": 24.0,
                "theta_e_arcsec": 1.0,
                "value": 1.0,
            },
            *[
                {
                    "host_galaxy_id": f"large_{index}",
                    "host_spatial_group_id": "large",
                    "psf_bin": 1.2,
                    "depth_bin": 24.0,
                    "theta_e_arcsec": 1.0,
                    "value": 0.0,
                }
                for index in range(5)
            ],
        ]
        counter_draws, _, _, counter_design = (
            _global_spatial_multiplier_bootstrap(
                pd.DataFrame(counterexample_rows),
                draws=10_000,
                seed_namespace="canary-one-vs-five",
            )
        )
        covariance_rows = []
        for label, psf in (("a", 1.2), ("b", 1.35)):
            covariance_rows.extend(
                [
                    {
                        "host_galaxy_id": f"{label}_shared",
                        "host_spatial_group_id": "shared",
                        "psf_bin": psf,
                        "depth_bin": 24.0,
                        "theta_e_arcsec": 1.0,
                        "value": 1.0,
                    },
                    {
                        "host_galaxy_id": f"{label}_unique",
                        "host_spatial_group_id": f"unique_{label}",
                        "psf_bin": psf,
                        "depth_bin": 24.0,
                        "theta_e_arcsec": 1.0,
                        "value": 0.0,
                    },
                ]
            )
        _, _, covariance_draws, covariance_design = (
            _global_spatial_multiplier_bootstrap(
                pd.DataFrame(covariance_rows),
                draws=10_000,
                seed_namespace="canary-cross-cell",
            )
        )
        cross_cell_covariance = float(
            np.cov(covariance_draws[:, 0, 0], covariance_draws[:, 1, 0])[0, 1]
        )
        report = {
            "schema_version": 1,
            "status": "design_only_canary_complete_not_scientific_outcome",
            "final_outcome_scoring_performed": False,
            "synthetic_raw_logit_convergence_scoring": True,
            "synthetic_hosts": 4,
            "synthetic_scene_rows": len(scenes),
            "all_eleven_radii_per_host": bool(
                scenes.groupby("host_galaxy_id")["theta_e_arcsec"]
                .nunique()
                .eq(11)
                .all()
            ),
            "shared_nuisance_passed": bool(
                scenes.groupby("host_galaxy_id")["base_draw_key"].nunique().eq(1).all()
            ),
            "host_units_verified": bool(
                hosts["host_pixel_unit"].eq(config["host"]["pixel_unit"]).all()
                and hosts["host_flux_calibration_system"]
                .eq(config["host"]["flux_calibration_system"])
                .all()
            ),
            "scene_identity_binds_psf": psf_identity_bound,
            "rendered_parent_canaries": len(parents),
            "render_seconds_for_parent_canaries": render_seconds,
            "seconds_per_parent_scene_cpu": render_seconds / len(parents),
            "parent_status": render_receipt["status"],
            "physical_child_canaries": len(children),
            "photon_status": photon_receipt["status"],
            "photon_seconds_for_children": photon_seconds,
            "seconds_per_photon_child_cpu_io": photon_seconds / len(children),
            "host_coupling_passed": host_coupling_passed,
            "photon_law_passed": law_passed,
            "photon_mean_z_max": float(mean_z.max()),
            "photon_variance_relative_error_max": float(
                variance_relative_error.max()
            ),
            "synthetic_mcse_selector": pilot,
            "renderer_validation": renderer_validation,
            "threshold_recomputed_from_bound_artifacts": True,
            "threshold_canary_not_released": (
                threshold_receipt["released_for_experiment_scoring"] is False
            ),
            "threshold_n": threshold_receipt["n"],
            "threshold_k": threshold_receipt["k_floor_alpha_n"],
            "threshold_s_k_plus_1": threshold_receipt["s_k_plus_1"],
            "threshold_float32_bits": int(threshold.view(np.uint32)),
            "design_role_ledger_verified": role_receipt["status"]
            == "design_role_ledger_frozen",
            "stale_config_rejected": stale_config_rejected,
            "grec_inference_canary_estimate": grec_inference["estimate"],
            "pnoise_inference_canary_estimate": pnoise_inference["estimate"],
            "global_spatial_multiplier_bootstrap_passed": bool(
                abs(counter_draws.mean() - 1.0 / 6.0) < 1.0e-12
                and counter_design["maximum_absolute_recentering_error"]
                < 1.0e-12
                and cross_cell_covariance > 0.0
                and covariance_design["spatial_groups_crossing_cells"] == 1
            ),
            "bootstrap_counterexamples": {
                "one_vs_five_equal_host_point": 1.0 / 6.0,
                "one_vs_five_recentered_draw_mean": float(
                    counter_draws.mean()
                ),
                "one_vs_five_absolute_bias": float(
                    abs(counter_draws.mean() - 1.0 / 6.0)
                ),
                "maximum_recentering_error": counter_design[
                    "maximum_absolute_recentering_error"
                ],
                "cross_cell_covariance": cross_cell_covariance,
                "shared_group_crosses_cells": covariance_design[
                    "spatial_groups_crossing_cells"
                ],
            },
            "runtime": runtime_record(),
        }
        if not all(
            [
                report["all_eleven_radii_per_host"],
                report["shared_nuisance_passed"],
                report["host_coupling_passed"],
                report["photon_law_passed"],
                report["threshold_recomputed_from_bound_artifacts"],
                report["threshold_canary_not_released"],
                report["design_role_ledger_verified"],
                report["stale_config_rejected"],
                report["host_units_verified"],
                report["scene_identity_binds_psf"],
                report["global_spatial_multiplier_bootstrap_passed"],
            ]
        ):
            raise RuntimeError(f"design canary failed: {report}")
    atomic_write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
