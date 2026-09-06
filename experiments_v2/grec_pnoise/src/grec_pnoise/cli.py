from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backend import backend_profile
from .contracts import (
    ContractError,
    atomic_write_json,
    config_provenance,
    load_config,
    load_json,
    require_receipt,
    scientific_section_sha256,
    sha256_file,
    sha256_json,
    verify_final_host_receipt,
)
from .hosts import prepare_and_write_final_hosts
from .inference import analyse_grec, analyse_pnoise
from .pnoise import (
    choose_draw_count,
    render_photon_children,
    validate_draw_count_receipt,
    verify_design_role_ledger,
    write_photon_outputs,
)
from .rendering import (
    render_parent_bank,
    validate_renderer_validation_receipt,
    write_render_outputs,
)
from .scene_table import build_scene_table
from .scoring import (
    fixed_scorer_closure,
    score_calibration_bank,
    score_frozen_bank,
    validate_scoring_authority,
    write_calibration_scores,
    write_score_outputs,
)
from .threshold import build_threshold_receipt
from .validation import validate_renderer_design


def _prepare_hosts(args: argparse.Namespace, config: dict) -> None:
    prepare_and_write_final_hosts(
        args.selection,
        args.selection_receipt,
        args.payload_root,
        config,
        args.output_dir,
    )


def _build_scenes(args: argparse.Namespace, config: dict) -> None:
    required_status = config["grec"]["required_host_receipt_status"]
    host_receipt = verify_final_host_receipt(
        args.hosts, args.host_receipt, required_status
    )
    hosts = pd.read_parquet(args.hosts)
    scenes = build_scene_table(hosts, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "grec_scene_table.parquet"
    partial = output / "grec_scene_table.partial.parquet"
    scenes.to_parquet(partial, index=False)
    partial.replace(table_path)
    receipt = {
        "schema_version": 1,
        "status": "grec_scene_table_frozen_pre_score",
        "scene_table_sha256": sha256_file(table_path),
        "host_manifest_sha256": sha256_file(args.hosts),
        "host_receipt_sha256": sha256_file(args.host_receipt),
        "host_receipt_status": host_receipt["status"],
        "rows": len(scenes),
        "distinct_hosts": int(scenes["host_galaxy_id"].nunique()),
        "distinct_cells": int(
            scenes[["psf_bin", "depth_bin"]].drop_duplicates().shape[0]
        ),
        "radii_arcsec": sorted(scenes["theta_e_arcsec"].unique().tolist()),
        "nuisance_draws_shared_across_radii": True,
        "detector_scores_used": False,
        "config": config_provenance(config),
    }
    atomic_write_json(output / "GREC_SCENE_TABLE_RECEIPT.json", receipt)


def _render_grec(args: argparse.Namespace, config: dict) -> None:
    renderer_validation = validate_renderer_validation_receipt(
        args.renderer_validation_receipt, config
    )
    scene_receipt = require_receipt(
        args.scene_receipt,
        status="grec_scene_table_frozen_pre_score",
        artifact_path=args.scenes,
        artifact_digest_field="scene_table_sha256",
    )
    if scene_receipt.get("config") != config_provenance(config):
        raise ContractError("scene table receipt was created under another config")
    scenes = pd.read_parquet(args.scenes)
    if len(scenes) != scene_receipt["rows"]:
        raise ContractError("scene receipt row count does not match its table")
    manifest, receipt = render_parent_bank(
        scenes,
        config,
        args.output_dir,
        device=args.device,
        renderer_validation_receipt_path=args.renderer_validation_receipt,
    )
    receipt["scene_table_sha256"] = sha256_file(args.scenes)
    receipt["scene_table_receipt_sha256"] = sha256_file(args.scene_receipt)
    receipt["renderer_validation_receipt_sha256"] = sha256_file(
        args.renderer_validation_receipt
    )
    receipt["renderer_validation_status"] = renderer_validation["status"]
    write_render_outputs(manifest, receipt, args.output_dir)


def _pilot_draw_count(args: argparse.Namespace, config: dict) -> None:
    validate_scoring_authority(config, args.threshold_receipt)
    pilot_receipt = load_json(args.pilot_score_receipt)
    expected = {
        "status": "design_only_pnoise_pilot_scores_complete",
        "global_role": "design",
        "final_outcomes_used": False,
        "pilot_scores_sha256": sha256_file(args.pilot_scores),
        "threshold_receipt_sha256": sha256_file(args.threshold_receipt),
        "design_role_ledger_sha256": sha256_file(args.design_role_ledger),
        "config_contract_sha256": config_provenance(config)[
            "config_contract_sha256"
        ],
    }
    failures = {
        field: (pilot_receipt.get(field), value)
        for field, value in expected.items()
        if pilot_receipt.get(field) != value
    }
    if failures:
        raise ContractError(f"invalid design-only MCSE pilot inputs: {failures}")
    with np.load(args.pilot_scores, allow_pickle=False) as archive:
        ledger_receipt = verify_design_role_ledger(
            archive["scene_ids"],
            archive["host_ids"],
            args.design_role_ledger,
            args.design_role_receipt,
            config,
        )
        result = choose_draw_count(
            archive["physical_decisions"],
            archive["expectation_decisions"],
            archive["host_ids"],
            config["pnoise"]["candidate_draw_counts"],
            absolute_mcse_max=float(config["pnoise"]["mcse_absolute_max"]),
            fraction_of_sampling_se_max=float(
                config["pnoise"]["mcse_fraction_of_sampling_se_max"]
            ),
        )
    receipt = {
        "schema_version": 1,
        "status": config["pnoise"]["required_draw_receipt_status"],
        "design_only": True,
        "final_outcomes_used": False,
        "metric": "host_equal_paired_detection_risk_difference",
        "pilot_scores_sha256": sha256_file(args.pilot_scores),
        "pilot_score_receipt_sha256": sha256_file(args.pilot_score_receipt),
        "threshold_receipt_sha256": sha256_file(args.threshold_receipt),
        "design_role_ledger_sha256": sha256_file(args.design_role_ledger),
        "design_role_ledger_receipt_sha256": sha256_file(
            args.design_role_receipt
        ),
        "design_role_ledger_status": ledger_receipt["status"],
        "pnoise_config_sha256": scientific_section_sha256(
            config, "pnoise", "render", "host"
        ),
        "photon_stream_namespace": config["pnoise"]["photon_stream_namespace"],
        "config": config_provenance(config),
        **result,
    }
    atomic_write_json(args.output_receipt, receipt)


def _render_pnoise(args: argparse.Namespace, config: dict) -> None:
    manifest, receipt = render_photon_children(
        None,
        config,
        args.output_dir,
        None,
        parent_manifest_path=args.parents,
        parent_receipt_path=args.parent_receipt,
        scene_table_path=args.scenes,
        scene_receipt_path=args.scene_receipt,
        renderer_validation_receipt_path=args.renderer_validation_receipt,
        draw_receipt_path=args.draw_receipt,
        threshold_receipt_path=args.threshold_receipt,
    )
    receipt["parent_manifest_sha256"] = sha256_file(args.parents)
    receipt["parent_receipt_sha256"] = sha256_file(args.parent_receipt)
    receipt["threshold_receipt_sha256"] = sha256_file(args.threshold_receipt)
    write_photon_outputs(manifest, receipt, args.output_dir)


def _build_threshold(args: argparse.Namespace, config: dict) -> None:
    build_threshold_receipt(
        manifest_path=args.calibration_manifest,
        manifest_receipt_path=args.calibration_manifest_receipt,
        logits_path=args.calibration_logits,
        score_receipt_path=args.calibration_score_receipt,
        approval_receipt_path=args.external_approval_receipt,
        config=config,
        output_path=args.output_receipt,
    )


def _analyse(args: argparse.Namespace, config: dict) -> None:
    if args.device != config["scoring"]["authoritative_device"]:
        raise ContractError("analysis device must equal the authoritative scoring device")
    validate_scoring_authority(
        config, args.threshold_receipt, device=args.device
    )
    wrapper_closure = fixed_scorer_closure(config, args.device)
    wrapper_closure_sha256 = sha256_json(wrapper_closure)
    parent_render_receipt = load_json(args.parent_render_receipt)
    parent_manifest_sha = sha256_file(args.parent_manifest)
    if (
        parent_render_receipt.get("status")
        != "grec_parent_bank_frozen_pre_score"
        or parent_render_receipt.get("parent_manifest_sha256")
        != parent_manifest_sha
        or parent_render_receipt.get("config") != config_provenance(config)
    ):
        raise ContractError("parent render receipt is invalid for inference")
    parent_receipt = load_json(args.parent_score_receipt)
    if (
        parent_receipt.get("status") != "frozen_scores_complete"
        or parent_receipt.get("bank") != "grec"
        or parent_receipt.get("scores_sha256") != sha256_file(args.parent_scores)
        or parent_receipt.get("bank_manifest_sha256") != parent_manifest_sha
        or parent_receipt.get("bank_receipt_sha256")
        != sha256_file(args.parent_render_receipt)
        or parent_receipt.get("threshold_receipt_sha256")
        != sha256_file(args.threshold_receipt)
        or parent_receipt.get("scorer_closure_sha256")
        != wrapper_closure_sha256
        or parent_receipt.get("scorer_closure") != wrapper_closure
        or parent_receipt.get("requested_device") != args.device
    ):
        raise ContractError("parent score receipt is invalid for inference")
    parents = pd.read_parquet(args.parent_manifest)
    parent_scores = pd.read_parquet(args.parent_scores)
    result = {
        "schema_version": 1,
        "status": "frozen_inference_complete",
        "grec": analyse_grec(parents, parent_scores, config),
    }
    child_arguments = [
        args.child_manifest,
        args.child_render_receipt,
        args.child_scores,
        args.child_score_receipt,
        args.mcse_receipt,
    ]
    if any(child_arguments) and not all(child_arguments):
        raise ContractError("all P_noise inference inputs must be supplied together")
    if args.child_manifest:
        mcse_receipt = validate_draw_count_receipt(
            args.mcse_receipt, config, args.threshold_receipt
        )
        child_render_receipt = load_json(args.child_render_receipt)
        child_manifest_sha = sha256_file(args.child_manifest)
        if (
            child_render_receipt.get("status")
            != "pnoise_physical_bank_frozen_pre_score"
            or child_render_receipt.get("child_manifest_sha256")
            != child_manifest_sha
            or child_render_receipt.get("parent_manifest_sha256")
            != parent_manifest_sha
            or child_render_receipt.get("parent_receipt_sha256")
            != sha256_file(args.parent_render_receipt)
            or child_render_receipt.get("draw_count_receipt_sha256")
            != sha256_file(args.mcse_receipt)
            or child_render_receipt.get("threshold_receipt_sha256")
            != sha256_file(args.threshold_receipt)
            or child_render_receipt.get("config") != config_provenance(config)
        ):
            raise ContractError("child render receipt is invalid for inference")
        child_receipt = load_json(args.child_score_receipt)
        if (
            child_receipt.get("status") != "frozen_scores_complete"
            or child_receipt.get("bank") != "pnoise"
            or child_receipt.get("scores_sha256") != sha256_file(args.child_scores)
            or child_receipt.get("bank_manifest_sha256") != child_manifest_sha
            or child_receipt.get("bank_receipt_sha256")
            != sha256_file(args.child_render_receipt)
            or child_receipt.get("threshold_receipt_sha256")
            != sha256_file(args.threshold_receipt)
            or child_receipt.get("scorer_closure_sha256")
            != wrapper_closure_sha256
            or child_receipt.get("scorer_closure") != wrapper_closure
            or child_receipt.get("requested_device") != args.device
        ):
            raise ContractError("child score receipt is invalid for inference")
        result["pnoise"] = analyse_pnoise(
            parents,
            parent_scores,
            pd.read_parquet(args.child_manifest),
            pd.read_parquet(args.child_scores),
            config,
            expected_draw_count=int(mcse_receipt["selected_draw_count"]),
        )
    result["inputs"] = {
        "parent_manifest_sha256": sha256_file(args.parent_manifest),
        "parent_render_receipt_sha256": sha256_file(
            args.parent_render_receipt
        ),
        "parent_scores_sha256": sha256_file(args.parent_scores),
        "parent_score_receipt_sha256": sha256_file(args.parent_score_receipt),
        "threshold_receipt_sha256": sha256_file(args.threshold_receipt),
        "analysis_device": args.device,
        "analysis_backend_profile": backend_profile(config, args.device),
        "wrapper_source_closure": wrapper_closure,
        "wrapper_source_closure_sha256": wrapper_closure_sha256,
        "config": config_provenance(config),
    }
    if args.child_manifest:
        result["inputs"].update(
            {
                "child_manifest_sha256": sha256_file(args.child_manifest),
                "child_render_receipt_sha256": sha256_file(
                    args.child_render_receipt
                ),
                "child_scores_sha256": sha256_file(args.child_scores),
                "child_score_receipt_sha256": sha256_file(
                    args.child_score_receipt
                ),
                "mcse_receipt_sha256": sha256_file(args.mcse_receipt),
            }
        )
    atomic_write_json(args.output, result)


def _validate_renderer(args: argparse.Namespace, config: dict) -> None:
    receipt = validate_renderer_design(
        pd.read_parquet(args.design_scenes),
        design_role_ledger_path=args.design_role_ledger,
        design_role_receipt_path=args.design_role_receipt,
        config=config,
        render_device=args.render_device,
        logit_device=args.logit_device,
        authoritative_design=True,
    )
    atomic_write_json(args.output_receipt, receipt)


def _score(args: argparse.Namespace, config: dict) -> None:
    scores, receipt = score_frozen_bank(
        args.manifest,
        args.bank_receipt,
        args.threshold_receipt,
        config,
        bank=args.bank,
        device=args.device,
    )
    write_score_outputs(scores, receipt, args.output_dir)


def _score_calibration(args: argparse.Namespace, config: dict) -> None:
    scores, receipt = score_calibration_bank(
        args.manifest,
        args.manifest_receipt,
        config,
        device=args.device,
    )
    write_calibration_scores(scores, receipt, args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receipt-gated G_rec/P_noise preparation and scoring"
    )
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    hosts = commands.add_parser("prepare-hosts")
    hosts.add_argument("--selection", required=True)
    hosts.add_argument("--selection-receipt", required=True)
    hosts.add_argument("--payload-root", required=True)
    hosts.add_argument("--output-dir", required=True)

    scenes = commands.add_parser("build-scenes")
    scenes.add_argument("--hosts", required=True)
    scenes.add_argument("--host-receipt", required=True)
    scenes.add_argument("--output-dir", required=True)

    render = commands.add_parser("render-grec")
    render.add_argument("--scenes", required=True)
    render.add_argument("--scene-receipt", required=True)
    render.add_argument("--renderer-validation-receipt", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--device", default="cpu")

    pilot = commands.add_parser("pilot-draw-count")
    pilot.add_argument("--pilot-scores", required=True)
    pilot.add_argument("--pilot-score-receipt", required=True)
    pilot.add_argument("--threshold-receipt", required=True)
    pilot.add_argument("--design-role-ledger", required=True)
    pilot.add_argument("--design-role-receipt", required=True)
    pilot.add_argument("--output-receipt", required=True)

    photon = commands.add_parser("render-pnoise")
    photon.add_argument("--parents", required=True)
    photon.add_argument("--parent-receipt", required=True)
    photon.add_argument("--scenes", required=True)
    photon.add_argument("--scene-receipt", required=True)
    photon.add_argument("--renderer-validation-receipt", required=True)
    photon.add_argument("--draw-receipt", required=True)
    photon.add_argument("--threshold-receipt", required=True)
    photon.add_argument("--output-dir", required=True)

    threshold = commands.add_parser("build-threshold")
    threshold.add_argument("--calibration-manifest", required=True)
    threshold.add_argument("--calibration-manifest-receipt", required=True)
    threshold.add_argument("--calibration-logits", required=True)
    threshold.add_argument("--calibration-score-receipt", required=True)
    threshold.add_argument("--external-approval-receipt", required=True)
    threshold.add_argument("--output-receipt", required=True)

    score = commands.add_parser("score")
    score.add_argument("--bank", required=True, choices=["grec", "pnoise"])
    score.add_argument("--manifest", required=True)
    score.add_argument("--bank-receipt", required=True)
    score.add_argument("--threshold-receipt", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--device", required=True)

    calibration = commands.add_parser("score-calibration")
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--manifest-receipt", required=True)
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--device", required=True)

    analyse = commands.add_parser("analyse")
    analyse.add_argument("--parent-manifest", required=True)
    analyse.add_argument("--parent-render-receipt", required=True)
    analyse.add_argument("--parent-scores", required=True)
    analyse.add_argument("--parent-score-receipt", required=True)
    analyse.add_argument("--threshold-receipt", required=True)
    analyse.add_argument("--device", required=True)
    analyse.add_argument("--child-manifest")
    analyse.add_argument("--child-render-receipt")
    analyse.add_argument("--child-scores")
    analyse.add_argument("--child-score-receipt")
    analyse.add_argument("--mcse-receipt")
    analyse.add_argument("--output", required=True)

    validation = commands.add_parser("validate-renderer")
    validation.add_argument("--design-scenes", required=True)
    validation.add_argument("--design-role-ledger", required=True)
    validation.add_argument("--design-role-receipt", required=True)
    validation.add_argument("--render-device", required=True)
    validation.add_argument("--logit-device", required=True)
    validation.add_argument("--output-receipt", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    actions = {
        "prepare-hosts": _prepare_hosts,
        "build-scenes": _build_scenes,
        "render-grec": _render_grec,
        "pilot-draw-count": _pilot_draw_count,
        "render-pnoise": _render_pnoise,
        "build-threshold": _build_threshold,
        "score": _score,
        "score-calibration": _score_calibration,
        "analyse": _analyse,
        "validate-renderer": _validate_renderer,
    }
    actions[args.command](args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
