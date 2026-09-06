from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    config_provenance,
    load_json,
    sha256_file,
    source_closure,
    validate_scorer_receipt,
    verify_source_closure,
)
from .draw_count import resolve_draw_binding, validate_draw_count_receipt
from .utils import atomic_json
from .validation import validate_source_determinism, validate_source_package


def _binding(path: str | Path, base: Path) -> dict[str, str]:
    artifact = Path(path).resolve()
    return {
        "relative_path": os.path.relpath(artifact, base),
        "sha256": sha256_file(artifact),
    }


def _resolve(report_root: Path, binding: dict[str, Any]) -> Path:
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute():
        raise ContractError("status-report binding is not portable")
    return (report_root.resolve() / relative).resolve()


def _root_relative(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), base)


def _validate_semantics(
    *,
    config: dict[str, Any],
    canary_path: Path,
    source_validation_path: Path,
    scorer_receipt_path: Path,
    actual_scoring_canary_path: Path,
    actual_scoring_input_path: Path,
    source_determinism_path: Path,
    draw_count_canary_path: Path,
    source_selection_root: Path,
    source_root: Path,
    repeat_source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canary = load_json(canary_path)
    recorded_source = load_json(source_validation_path)
    actual_scoring = load_json(actual_scoring_canary_path)
    recorded_determinism = load_json(source_determinism_path)
    draw_count_canary = load_json(draw_count_canary_path)
    scorer = validate_scorer_receipt(scorer_receipt_path, config, require_final=False)
    recomputed_source = validate_source_package(
        selection_root=source_selection_root,
        source_root=source_root,
        config=config,
        output=None,
    )
    recomputed_determinism = validate_source_determinism(
        primary_source_root=source_root,
        repeated_source_root=repeat_source_root,
        config=config,
        output=None,
    )
    draw_bindings = draw_count_canary.get("bindings", {})
    required_draw_bindings = {
        "final_design",
        "final_design_receipt",
        "scorer_receipt",
        "threshold_receipt",
    }
    if not required_draw_bindings.issubset(draw_bindings):
        raise ContractError("draw-count canary bindings are incomplete")
    validate_draw_count_receipt(
        draw_count_canary_path,
        config,
        expected_final_design_path=resolve_draw_binding(
            draw_count_canary_path, draw_bindings["final_design"]
        ),
        expected_final_design_receipt_path=resolve_draw_binding(
            draw_count_canary_path, draw_bindings["final_design_receipt"]
        ),
        expected_scorer_receipt_path=resolve_draw_binding(
            draw_count_canary_path, draw_bindings["scorer_receipt"]
        ),
        expected_threshold_receipt_path=resolve_draw_binding(
            draw_count_canary_path, draw_bindings["threshold_receipt"]
        ),
        production=False,
    )
    comparable_source = {
        key: value
        for key, value in recomputed_source.items()
        if key not in {"config", "source_closure"}
    }
    comparable_recorded_source = {
        key: value
        for key, value in recorded_source.items()
        if key not in {"config", "source_closure"}
    }
    comparable_determinism = {
        key: value
        for key, value in recomputed_determinism.items()
        if key not in {"config", "source_closure"}
    }
    comparable_recorded_determinism = {
        key: value
        for key, value in recorded_determinism.items()
        if key not in {"config", "source_closure"}
    }
    if (
        canary.get("status") != "pass"
        or canary.get("detector_outcomes_used") is not False
        or recorded_source.get("config") != config_provenance(config)
        or recorded_source.get("source_closure") != source_closure(config)
        or comparable_source != comparable_recorded_source
        or actual_scoring.get("status") != "actual_design_only_photon_raw_logit_canary"
        or actual_scoring.get("final_outcomes_used") is not False
        or actual_scoring.get("threshold_applied") is not False
        or actual_scoring.get("input_artifact_sha256")
        != sha256_file(actual_scoring_input_path)
        or actual_scoring.get("scoring", {}).get("exact_float32_roundtrip_verified")
        is not True
        or actual_scoring.get("scoring", {}).get("worker_order_invariant") is not True
        or recorded_determinism.get("config") != config_provenance(config)
        or recorded_determinism.get("source_closure") != source_closure(config)
        or comparable_determinism != comparable_recorded_determinism
        or draw_count_canary.get("canary_mode") is not True
        or draw_count_canary.get("final_outcomes_used") is not False
    ):
        raise ContractError("status-report inputs are stale or failed")
    return (
        canary,
        recomputed_source,
        {
            "scorer": scorer,
            "actual_scoring": actual_scoring,
            "draw_count": draw_count_canary,
        },
    )


def build_status_report(
    *,
    config: dict[str, Any],
    canary_path: str | Path,
    source_validation_path: str | Path,
    scorer_receipt_path: str | Path,
    actual_scoring_canary_path: str | Path,
    actual_scoring_input_path: str | Path,
    source_determinism_path: str | Path,
    draw_count_canary_path: str | Path,
    source_selection_root: str | Path,
    source_root: str | Path,
    repeat_source_root: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a report whose validator reopens every listed artifact."""
    paths = {
        "canary": Path(canary_path),
        "source_validation": Path(source_validation_path),
        "scorer_receipt": Path(scorer_receipt_path),
        "actual_scoring_canary": Path(actual_scoring_canary_path),
        "actual_scoring_input": Path(actual_scoring_input_path),
        "source_determinism": Path(source_determinism_path),
        "draw_count_canary": Path(draw_count_canary_path),
    }
    canary, source, other = _validate_semantics(
        config=config,
        canary_path=paths["canary"],
        source_validation_path=paths["source_validation"],
        scorer_receipt_path=paths["scorer_receipt"],
        actual_scoring_canary_path=paths["actual_scoring_canary"],
        actual_scoring_input_path=paths["actual_scoring_input"],
        source_determinism_path=paths["source_determinism"],
        draw_count_canary_path=paths["draw_count_canary"],
        source_selection_root=Path(source_selection_root),
        source_root=Path(source_root),
        repeat_source_root=Path(repeat_source_root),
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    blockers = [
        "No final_pre_score_grec_hosts_frozen manifest/receipt exists.",
        "No grec_scene_table_frozen_pre_score table/receipt exists.",
        "No grec_parent_bank_frozen_pre_score manifest/receipt exists.",
        "No fixed_scorer_frozen_pre_score approval receipt exists.",
        "No common_threshold_frozen receipt exists.",
        "No separate parent-bound design-role pilot, authoritative design/experiment role ledger, or production draw-count receipt exists; the synthetic producer canary is not authority.",
        "No P_src execution lock can exist until all upstream authorities exist.",
        "Empirical DR10 PSFs and calibrated per-band gains remain unavailable.",
        "The F814W empirical construction retains measurement uncertainty and is not latent or noise-free.",
    ]
    pairs = int(config["parent_grec"]["subset"]["expected_parent_scenes"])
    minimum_draws = int(config["photon_noise"]["minimum_draws"])
    maximum_draws = int(config["photon_noise"]["maximum_draws"])
    raw_bytes_minimum = pairs * minimum_draws * 2 * 101 * 101 * 3 * 4
    raw_bytes_maximum = pairs * maximum_draws * 2 * 101 * 101 * 3 * 4
    actual = other["actual_scoring"]
    draw_count = other["draw_count"]
    markdown = [
        "# Parent-bound P_src implementation status",
        "",
        "No final outcomes were accessed or scored.",
        "",
        "## Exact claim",
        "",
        config["claim"]["primary"],
        "",
        "## Validated",
        "",
        f"- Source maps: {source['selected_sources']}",
        f"- Eligible source pool: {source['eligible_sources']}",
        (
            "- Minimum retained positive flux: "
            f"{source['minimum_structured_positive_flux_retained']:.6f} "
            "(structured), "
            f"{source['minimum_sersic_positive_flux_retained']:.6f} (Sersic)"
        ),
        (
            "- Maximum pre-projection negative/positive flux: "
            f"{source['maximum_structured_negative_to_positive_flux']:.6f} "
            "(structured), "
            f"{source['maximum_sersic_negative_to_positive_flux']:.6f} (Sersic)"
        ),
        (
            "- Conservative coverage diagnostic (never a pass gate): "
            f"{canary['inference_coverage']['coverage']:.3f} over "
            f"{int(canary['inference_coverage']['replications'])} exact-design "
            "binary replications; no interval inflation."
        ),
        f"- Inference limitation: {config['inference']['method_limitation']}",
        (
            "- Actual fixed-checkpoint raw-logit canary: "
            f"{actual['images']} synthetic photonized images; input artifact "
            "and array hashes, float32 round-trip, and worker-order invariance pass."
        ),
        "- Both full source executions (manifests and every map) reproduce byte-for-byte.",
        (
            "- Separate-pilot draw-count producer canary evaluated "
            f"{draw_count['candidate_draw_counts']} and selected "
            f"{draw_count['draw_count']}; production rejects this canary receipt."
        ),
        "",
        "## Blockers",
        "",
        *[f"{index}. {value}" for index, value in enumerate(blockers, start=1)],
        "",
        "## Compute/storage",
        "",
        (
            f"The design has {pairs} pairs. Raw float32 photon pixels are "
            f"{raw_bytes_minimum / 2**30:.2f} GiB at {minimum_draws} draws and "
            f"{raw_bytes_maximum / 2**30:.2f} GiB at {maximum_draws} draws."
        ),
        (
            "Expect about 17 minutes single-process for deterministic paired "
            "expectations and 10--40 minutes for photon materialization/I/O. "
            "The 64-image MPS raw-logit canary passed; frozen-bank scoring should "
            "be re-benchmarked after the scientific draw count is available."
        ),
    ]
    report_path = output_dir / "STATUS_REPORT.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    base = output_dir.resolve()
    bindings = {name: _binding(path, base) for name, path in paths.items()}
    # List every compact source artifact directly; every map is rehashed by the
    # semantic validator from the roots below.
    compact = {
        "source_selection_receipt": Path(source_selection_root)
        / "selection_receipt.json",
        "source_eligibility": Path(source_selection_root)
        / "source_eligibility.parquet",
        "selected_sources": Path(source_selection_root) / "selected_sources.parquet",
        "source_receipt": Path(source_root) / "receipt.json",
        "source_manifest": Path(source_root) / "source_manifest.parquet",
        "repeat_source_receipt": Path(repeat_source_root) / "receipt.json",
        "repeat_source_manifest": Path(repeat_source_root) / "source_manifest.parquet",
    }
    bindings.update({name: _binding(path, base) for name, path in compact.items()})
    receipt = {
        "schema_version": 3,
        "status": "parent_bound_implementation_report_current",
        "config": config_provenance(config),
        "source_closure": source_closure(config),
        "bindings": bindings,
        "roots": {
            "source_selection_root": _root_relative(source_selection_root, base),
            "source_root": _root_relative(source_root, base),
            "repeat_source_root": _root_relative(repeat_source_root, base),
        },
        "report_sha256": sha256_file(report_path),
        "blockers": blockers,
        "raw_float32_bytes": {
            str(minimum_draws): raw_bytes_minimum,
            str(maximum_draws): raw_bytes_maximum,
        },
    }
    atomic_json(output_dir / "REPORT_RECEIPT.json", receipt)
    return receipt


def validate_status_report(
    report_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    report_root = Path(report_root)
    receipt = load_json(report_root / "REPORT_RECEIPT.json")
    if (
        receipt.get("status") != "parent_bound_implementation_report_current"
        or receipt.get("config") != config_provenance(config)
        or receipt.get("report_sha256") != sha256_file(report_root / "STATUS_REPORT.md")
    ):
        raise ContractError("status report is stale")
    verify_source_closure(receipt.get("source_closure", {}), config)
    bindings = receipt.get("bindings", {})
    for name, binding in bindings.items():
        path = _resolve(report_root, binding)
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ContractError(f"status-report listed artifact drift: {name}")
    roots = receipt.get("roots", {})

    def root(name: str) -> Path:
        relative = Path(str(roots.get(name, "")))
        if relative.is_absolute():
            raise ContractError("status-report source root is not portable")
        return (report_root.resolve() / relative).resolve()

    _validate_semantics(
        config=config,
        canary_path=_resolve(report_root, bindings["canary"]),
        source_validation_path=_resolve(report_root, bindings["source_validation"]),
        scorer_receipt_path=_resolve(report_root, bindings["scorer_receipt"]),
        actual_scoring_canary_path=_resolve(
            report_root, bindings["actual_scoring_canary"]
        ),
        actual_scoring_input_path=_resolve(
            report_root, bindings["actual_scoring_input"]
        ),
        source_determinism_path=_resolve(report_root, bindings["source_determinism"]),
        draw_count_canary_path=_resolve(report_root, bindings["draw_count_canary"]),
        source_selection_root=root("source_selection_root"),
        source_root=root("source_root"),
        repeat_source_root=root("repeat_source_root"),
    )
    return receipt
