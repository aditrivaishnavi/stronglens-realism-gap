from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .authority import build_execution_lock
from .canaries import (
    actual_photon_scoring_canary,
    run_canaries,
    write_photonized_scoring_canary_images,
)
from .contracts import load_config, sha256_file, write_scorer_receipt
from .design import prepare_parent_child_design
from .draw_count import build_draw_count_receipt
from .execution import materialize_photon_bank, render_expectation_bank
from .inference import analyze_bound_scores
from .pilot import (
    build_role_ledger,
    prepare_pilot_design,
    run_draw_pilot,
)
from .report import build_status_report, validate_status_report
from .scoring import score_frozen_photon_bank
from .source import (
    construct_source_maps,
    select_sources,
    source_eligibility_ledger,
    write_selection,
)
from .threshold import build_threshold_receipt
from .validation import validate_source_determinism, validate_source_package
from .verification import independently_verify_final

DEFAULT_CONFIG = Path("config/psrc_v2_parent_bound.yaml")


def _prepare_sources(args: argparse.Namespace, config: dict) -> None:
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ledger = source_eligibility_ledger(config)
    ledger, selected = select_sources(ledger, config, source_role=args.source_role)
    write_selection(ledger, selected, output / "selection", config)
    construct_source_maps(
        selected,
        config,
        output / "sources",
        final_status=args.final_status,
    )


def _validate_sources(args: argparse.Namespace, config: dict) -> None:
    validate_source_package(
        selection_root=args.prepared_root / "selection",
        source_root=args.prepared_root / "sources",
        config=config,
        output=args.output,
    )


def _validate_source_determinism(args: argparse.Namespace, config: dict) -> None:
    validate_source_determinism(
        primary_source_root=args.primary_source_root,
        repeated_source_root=args.repeated_source_root,
        config=config,
        output=args.output,
    )


def _build_child(args: argparse.Namespace, config: dict) -> None:
    prepare_parent_child_design(
        final_host_manifest_path=args.final_hosts,
        final_host_receipt_path=args.final_host_receipt,
        scene_table_path=args.grec_scenes,
        scene_receipt_path=args.grec_scene_receipt,
        parent_manifest_path=args.grec_parent_manifest,
        parent_receipt_path=args.grec_parent_receipt,
        source_manifest_path=args.source_manifest,
        source_receipt_path=args.source_receipt,
        config=config,
        output_dir=args.output,
        final_status=args.final_status,
    )


def _scorer_receipt(args: argparse.Namespace, config: dict) -> None:
    write_scorer_receipt(config, args.output, final_status=args.final_status)


def _threshold(args: argparse.Namespace, config: dict) -> None:
    build_threshold_receipt(
        manifest_path=args.calibration_manifest,
        manifest_receipt_path=args.calibration_manifest_receipt,
        logits_path=args.calibration_logits,
        score_receipt_path=args.calibration_score_receipt,
        scorer_receipt_path=args.scorer_receipt,
        config=config,
        output_path=args.output,
        final_status=args.final_status,
    )


def _freeze(args: argparse.Namespace, config: dict) -> None:
    build_execution_lock(
        expected_config_sha256=args.expected_config_sha256,
        config=config,
        final_host_manifest_path=args.final_hosts,
        final_host_receipt_path=args.final_host_receipt,
        scene_table_path=args.grec_scenes,
        scene_receipt_path=args.grec_scene_receipt,
        parent_manifest_path=args.grec_parent_manifest,
        parent_receipt_path=args.grec_parent_receipt,
        source_manifest_path=args.source_manifest,
        source_receipt_path=args.source_receipt,
        design_path=args.design,
        design_receipt_path=args.design_receipt,
        scorer_receipt_path=args.scorer_receipt,
        threshold_receipt_path=args.threshold_receipt,
        draw_count_receipt_path=args.draw_count_receipt,
        output_path=args.output,
    )


def _canary(args: argparse.Namespace, config: dict) -> None:
    run_canaries(config, args.output)


def _score_canary(args: argparse.Namespace, config: dict) -> None:
    with np.load(args.images, allow_pickle=False) as payload:
        images = np.array(payload[args.array_key], copy=True)
    result = actual_photon_scoring_canary(
        images,
        config,
        scorer_receipt_path=args.scorer_receipt,
        device=args.device,
    )
    result["input_artifact_sha256"] = sha256_file(args.images)
    from .utils import atomic_json

    atomic_json(args.output, result)


def _make_score_canary_images(args: argparse.Namespace, config: dict) -> None:
    write_photonized_scoring_canary_images(config, args.output)


def _render(args: argparse.Namespace, config: dict) -> None:
    render_expectation_bank(
        design_path=args.design,
        design_receipt_path=args.design_receipt,
        source_root=args.source_root,
        config=config,
        output_dir=args.output,
        execution_lock_path=args.execution_lock,
        design_only_pairs=args.design_only_pairs,
    )


def _photons(args: argparse.Namespace, config: dict) -> None:
    materialize_photon_bank(
        design_path=args.design,
        expectation_root=args.expectation_root,
        expectation_receipt_path=args.expectation_receipt,
        draw_count_receipt_path=args.draw_count_receipt,
        execution_lock_path=args.execution_lock,
        config=config,
        output_dir=args.output,
    )


def _draw_count(args: argparse.Namespace, config: dict) -> None:
    build_draw_count_receipt(
        pilot_scores_path=args.pilot_scores,
        pilot_score_receipt_path=args.pilot_score_receipt,
        pilot_design_path=args.pilot_design,
        pilot_design_receipt_path=args.pilot_design_receipt,
        final_design_path=args.final_design,
        final_design_receipt_path=args.final_design_receipt,
        role_ledger_path=args.role_ledger,
        role_ledger_receipt_path=args.role_ledger_receipt,
        scorer_receipt_path=args.scorer_receipt,
        threshold_receipt_path=args.threshold_receipt,
        config=config,
        output_path=args.output,
        canary_mode=args.canary_mode,
    )


def _build_pilot_design(args: argparse.Namespace, config: dict) -> None:
    prepare_pilot_design(
        source_manifest_path=args.source_manifest,
        source_receipt_path=args.source_receipt,
        config=config,
        output_dir=args.output,
    )


def _build_role_ledger(args: argparse.Namespace, config: dict) -> None:
    build_role_ledger(
        pilot_design_path=args.pilot_design,
        pilot_design_receipt_path=args.pilot_design_receipt,
        final_design_path=args.final_design,
        final_design_receipt_path=args.final_design_receipt,
        config=config,
        output_dir=args.output,
    )


def _run_draw_pilot(args: argparse.Namespace, config: dict) -> None:
    run_draw_pilot(
        pilot_design_path=args.pilot_design,
        pilot_design_receipt_path=args.pilot_design_receipt,
        final_design_path=args.final_design,
        final_design_receipt_path=args.final_design_receipt,
        role_ledger_path=args.role_ledger,
        role_ledger_receipt_path=args.role_ledger_receipt,
        source_root=args.source_root,
        scorer_receipt_path=args.scorer_receipt,
        threshold_receipt_path=args.threshold_receipt,
        config=config,
        output_dir=args.output,
        device=args.device,
    )


def _score(args: argparse.Namespace, config: dict) -> None:
    score_frozen_photon_bank(
        photon_manifest_path=args.photon_manifest,
        photon_receipt_path=args.photon_receipt,
        execution_lock_path=args.execution_lock,
        config=config,
        output_dir=args.output,
        device=args.device,
    )


def _analyze(args: argparse.Namespace, config: dict) -> None:
    analyze_bound_scores(
        design_path=args.design,
        design_receipt_path=args.design_receipt,
        photon_manifest_path=args.photon_manifest,
        photon_receipt_path=args.photon_receipt,
        scores_path=args.scores,
        score_receipt_path=args.score_receipt,
        draw_count_receipt_path=args.draw_count_receipt,
        execution_lock_path=args.execution_lock,
        config=config,
        output_dir=args.output,
        repeated_threshold_draws=args.repeated_threshold_draws,
    )


def _report(args: argparse.Namespace, config: dict) -> None:
    build_status_report(
        config=config,
        canary_path=args.canary,
        source_validation_path=args.source_validation,
        scorer_receipt_path=args.scorer_receipt,
        actual_scoring_canary_path=args.actual_scoring_canary,
        actual_scoring_input_path=args.actual_scoring_input,
        source_determinism_path=args.source_determinism,
        draw_count_canary_path=args.draw_count_canary,
        source_selection_root=args.source_selection_root,
        source_root=args.source_root,
        repeat_source_root=args.repeat_source_root,
        output_dir=args.output,
    )


def _validate_report(args: argparse.Namespace, config: dict) -> None:
    validate_status_report(args.report_root, config)


def _verify_final(args: argparse.Namespace, config: dict) -> None:
    independently_verify_final(
        design_path=args.design,
        design_receipt_path=args.design_receipt,
        photon_manifest_path=args.photon_manifest,
        photon_receipt_path=args.photon_receipt,
        scores_path=args.scores,
        score_receipt_path=args.score_receipt,
        draw_count_receipt_path=args.draw_count_receipt,
        execution_lock_path=args.execution_lock,
        result_path=args.result,
        config=config,
        output_path=args.output,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Parent-bound restricted P_src pipeline; no final command runs without receipts."
    )
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subs = result.add_subparsers(dest="command", required=True)

    command = subs.add_parser("prepare-sources")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--source-role", choices=("experiment", "pilot"), required=True
    )
    command.add_argument("--final-status", action="store_true")
    command.set_defaults(function=_prepare_sources)

    command = subs.add_parser("validate-sources")
    command.add_argument("--prepared-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_validate_sources)

    command = subs.add_parser("validate-source-determinism")
    command.add_argument("--primary-source-root", type=Path, required=True)
    command.add_argument("--repeated-source-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_validate_source_determinism)

    command = subs.add_parser("build-child-design")
    for name in (
        "final-hosts",
        "final-host-receipt",
        "grec-scenes",
        "grec-scene-receipt",
        "grec-parent-manifest",
        "grec-parent-receipt",
        "source-manifest",
        "source-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--final-status", action="store_true")
    command.set_defaults(function=_build_child)

    command = subs.add_parser("build-scorer-receipt")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--final-status", action="store_true")
    command.set_defaults(function=_scorer_receipt)

    command = subs.add_parser("build-threshold")
    for name in (
        "calibration-manifest",
        "calibration-manifest-receipt",
        "calibration-logits",
        "calibration-score-receipt",
        "scorer-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--final-status", action="store_true")
    command.set_defaults(function=_threshold)

    command = subs.add_parser("freeze-execution")
    command.add_argument("--expected-config-sha256", required=True)
    for name in (
        "final-hosts",
        "final-host-receipt",
        "grec-scenes",
        "grec-scene-receipt",
        "grec-parent-manifest",
        "grec-parent-receipt",
        "source-manifest",
        "source-receipt",
        "design",
        "design-receipt",
        "scorer-receipt",
        "threshold-receipt",
        "draw-count-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_freeze)

    command = subs.add_parser("canary")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_canary)

    command = subs.add_parser("score-design-canary")
    command.add_argument("--images", type=Path, required=True)
    command.add_argument("--array-key", default="images_nchw")
    command.add_argument("--scorer-receipt", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_score_canary)

    command = subs.add_parser("make-scoring-canary-images")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_make_score_canary_images)

    command = subs.add_parser("render-expectations")
    command.add_argument("--design", type=Path, required=True)
    command.add_argument("--design-receipt", type=Path, required=True)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--execution-lock", type=Path)
    command.add_argument("--design-only-pairs", type=int)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_render)

    command = subs.add_parser("materialize-photons")
    command.add_argument("--design", type=Path, required=True)
    command.add_argument("--expectation-root", type=Path, required=True)
    command.add_argument("--expectation-receipt", type=Path, required=True)
    command.add_argument("--draw-count-receipt", type=Path, required=True)
    command.add_argument("--execution-lock", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_photons)

    command = subs.add_parser("build-draw-count")
    for name in (
        "pilot-scores",
        "pilot-score-receipt",
        "pilot-design",
        "pilot-design-receipt",
        "final-design",
        "final-design-receipt",
        "role-ledger",
        "role-ledger-receipt",
        "scorer-receipt",
        "threshold-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--canary-mode", action="store_true")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_draw_count)

    command = subs.add_parser("build-pilot-design")
    command.add_argument("--source-manifest", type=Path, required=True)
    command.add_argument("--source-receipt", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_build_pilot_design)

    command = subs.add_parser("build-role-ledger")
    for name in (
        "pilot-design",
        "pilot-design-receipt",
        "final-design",
        "final-design-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_build_role_ledger)

    command = subs.add_parser("run-draw-pilot")
    for name in (
        "pilot-design",
        "pilot-design-receipt",
        "final-design",
        "final-design-receipt",
        "role-ledger",
        "role-ledger-receipt",
        "source-root",
        "scorer-receipt",
        "threshold-receipt",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_run_draw_pilot)

    command = subs.add_parser("score-frozen")
    command.add_argument("--photon-manifest", type=Path, required=True)
    command.add_argument("--photon-receipt", type=Path, required=True)
    command.add_argument("--execution-lock", type=Path, required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_score)

    command = subs.add_parser("analyze")
    for name in (
        "design",
        "design-receipt",
        "photon-manifest",
        "photon-receipt",
        "scores",
        "score-receipt",
        "draw-count-receipt",
        "execution-lock",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--repeated-threshold-draws", type=int, default=0)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_analyze)

    command = subs.add_parser("build-status-report")
    command.add_argument("--canary", type=Path, required=True)
    command.add_argument("--source-validation", type=Path, required=True)
    command.add_argument("--scorer-receipt", type=Path, required=True)
    command.add_argument("--actual-scoring-canary", type=Path, required=True)
    command.add_argument("--actual-scoring-input", type=Path, required=True)
    command.add_argument("--source-determinism", type=Path, required=True)
    command.add_argument("--draw-count-canary", type=Path, required=True)
    command.add_argument("--source-selection-root", type=Path, required=True)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--repeat-source-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_report)

    command = subs.add_parser("validate-status-report")
    command.add_argument("--report-root", type=Path, required=True)
    command.set_defaults(function=_validate_report)

    command = subs.add_parser("verify-final")
    for name in (
        "design",
        "design-receipt",
        "photon-manifest",
        "photon-receipt",
        "scores",
        "score-receipt",
        "draw-count-receipt",
        "execution-lock",
        "result",
    ):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=_verify_final)
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    args.function(args, config)


if __name__ == "__main__":
    main()
