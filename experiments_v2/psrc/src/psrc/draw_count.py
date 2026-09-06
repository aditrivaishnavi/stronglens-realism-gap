from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    config_provenance,
    load_json,
    sha256_file,
    sha256_json,
    source_closure,
    validate_scorer_receipt,
)
from .threshold import validate_threshold_receipt
from .utils import atomic_json


def _binding(path: str | Path, base: Path) -> dict[str, str]:
    artifact = Path(path).resolve()
    return {
        "relative_path": os.path.relpath(artifact, base),
        "sha256": sha256_file(artifact),
    }


def resolve_draw_binding(receipt_path: str | Path, binding: dict[str, Any]) -> Path:
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute():
        raise ContractError("draw-count artifact binding must be relative")
    return (Path(receipt_path).resolve().parent / relative).resolve()


def _distribution(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise ContractError(f"design distribution lacks columns: {missing}")
    counts = (
        frame.groupby(columns, dropna=False, observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(columns, kind="mergesort")
    )
    records = []
    for row in counts.to_dict(orient="records"):
        records.append(
            {
                **{column: row[column] for column in columns},
                "row_fraction": float(row["rows"] / len(frame)),
            }
        )
    return {
        "columns": columns,
        "rows": len(frame),
        "cells": len(counts),
        "fractions": records,
        "sha256": sha256_json(records),
    }


def _validate_authoritative_role_ledger(
    *,
    role_ledger_path: str | Path,
    role_ledger_receipt_path: str | Path,
    pilot: pd.DataFrame,
    final: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    receipt = load_json(role_ledger_receipt_path)
    ledger_sha = sha256_file(role_ledger_path)
    if (
        receipt.get("status")
        != config["photon_noise"]["required_role_ledger_receipt_status"]
        or receipt.get("role_ledger_sha256") != ledger_sha
        or receipt.get("detector_scores_used") is not False
        or receipt.get("final_outcomes_used") is not False
        or receipt.get("config") != config_provenance(config)
        or receipt.get("source_closure") != source_closure(config)
    ):
        raise ContractError("authoritative pilot/final role-ledger receipt is invalid")
    ledger = pd.read_parquet(role_ledger_path)
    identity_columns = [
        str(value)
        for value in config["photon_noise"]["pilot_anti_join_identity_columns"]
    ]
    comparison_columns = [column for column in identity_columns if column != "pair_id"]
    required = {"pair_id", "global_role", *comparison_columns}
    if required - set(ledger) or ledger["pair_id"].astype(str).duplicated().any():
        raise ContractError("authoritative role ledger lacks unique identity fields")
    expected_pair_ids = set(pilot["pair_id"].astype(str)) | set(
        final["pair_id"].astype(str)
    )
    if set(ledger["pair_id"].astype(str)) != expected_pair_ids:
        raise ContractError("authoritative role ledger does not exactly cover designs")

    def check(frame: pd.DataFrame, role: str) -> None:
        joined = frame[["pair_id", *comparison_columns]].merge(
            ledger[["pair_id", "global_role", *comparison_columns]],
            on="pair_id",
            validate="one_to_one",
            suffixes=("_design", "_ledger"),
        )
        if len(joined) != len(frame) or not joined["global_role"].eq(role).all():
            raise ContractError(
                f"authoritative role ledger does not prove {role} membership"
            )
        for column in comparison_columns:
            if (
                not joined[f"{column}_design"]
                .astype(str)
                .equals(joined[f"{column}_ledger"].astype(str))
            ):
                raise ContractError(
                    f"authoritative role-ledger identity drift: {column}"
                )

    check(pilot, "design")
    check(final, "experiment")
    return {
        "status": receipt["status"],
        "role_ledger_sha256": ledger_sha,
        "role_ledger_receipt_sha256": sha256_file(role_ledger_receipt_path),
        "pilot_rows_proven_design": len(pilot),
        "final_rows_proven_experiment": len(final),
        "identity_columns": identity_columns,
    }


def _validate_pilot_and_final_designs(
    *,
    pilot_design_path: str | Path,
    pilot_design_receipt_path: str | Path,
    final_design_path: str | Path,
    final_design_receipt_path: str | Path,
    role_ledger_path: str | Path,
    role_ledger_receipt_path: str | Path,
    config: dict[str, Any],
    canary_mode: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pilot_receipt = load_json(pilot_design_receipt_path)
    final_receipt = load_json(final_design_receipt_path)
    expected_final_status = (
        "psrc_parent_child_design_complete_design_only"
        if canary_mode
        else config["execution"]["required_design_receipt_status"]
    )
    expected = (
        (
            pilot_receipt,
            config["photon_noise"]["required_pilot_design_receipt_status"],
            pilot_design_path,
        ),
        (final_receipt, expected_final_status, final_design_path),
    )
    for receipt, status, path in expected:
        if (
            receipt.get("status") != status
            or receipt.get("pair_design_sha256") != sha256_file(path)
            or receipt.get("config") != config_provenance(config)
            or receipt.get("source_closure") != source_closure(config)
        ):
            raise ContractError("pilot/final design receipt is stale or wrong role")
    pilot = pd.read_parquet(pilot_design_path)
    final = pd.read_parquet(final_design_path)
    required = {
        "pair_id",
        "parent_grec_scene_id",
        "host_galaxy_id",
        "host_spatial_group_id",
        "source_id",
        "source_stratum",
        "host_cell_id",
        "theta_e_arcsec",
        "parent_scene_analysis_weight",
        *[
            str(value)
            for value in config["photon_noise"]["pilot_anti_join_identity_columns"]
        ],
    }
    if required - set(pilot) or required - set(final):
        raise ContractError("pilot/final design lacks role, identity, or weight fields")
    if "global_role" not in pilot or not pilot["global_role"].eq("design").all():
        raise ContractError(
            "draw-count pilot must independently verify global_role=design"
        )
    for frame, name in ((pilot, "pilot"), (final, "final")):
        if (
            frame["pair_id"].astype(str).duplicated().any()
            or frame["parent_grec_scene_id"].astype(str).duplicated().any()
            or not frame["parent_scene_analysis_weight"].eq(1.0).all()
        ):
            raise ContractError(f"{name} design IDs/weights are invalid")
    if not canary_mode:
        expected_pairs = int(config["crossing"]["expected_pairs"])
        expected_sources = int(config["selection"]["source"]["selected_sources"])
        expected_uses = int(config["crossing"]["expected_source_uses"])
        if (
            len(pilot) != expected_pairs
            or len(final) != expected_pairs
            or pilot["source_id"].nunique() != expected_sources
            or final["source_id"].nunique() != expected_sources
            or not pilot.groupby("source_id").size().eq(expected_uses).all()
            or not final.groupby("source_id").size().eq(expected_uses).all()
            or pilot["host_galaxy_id"].nunique()
            != int(config["pilot_parent"]["expected_distinct_hosts"])
            or final["host_galaxy_id"].nunique()
            != int(
                config["parent_grec"]["subset"]["expected_distinct_hosts"]
            )
        ):
            raise ContractError(
                "production pilot/final design cardinality differs"
            )
    anti_join = [
        str(value)
        for value in config["photon_noise"]["pilot_anti_join_identity_columns"]
    ]
    overlaps = {
        column: sorted(set(pilot[column].astype(str)) & set(final[column].astype(str)))
        for column in anti_join
    }
    overlaps = {column: values for column, values in overlaps.items() if values}
    if overlaps:
        raise ContractError(
            f"design-only pilot overlaps final design identities: "
            f"{ {key: len(value) for key, value in overlaps.items()} }"
        )
    columns = [
        str(value) for value in config["photon_noise"]["pilot_distribution_columns"]
    ]
    pilot_distribution = _distribution(pilot, columns)
    final_distribution = _distribution(final, columns)
    if pilot_distribution["fractions"] != final_distribution["fractions"]:
        raise ContractError("pilot and final design distributions differ")
    role_authority = _validate_authoritative_role_ledger(
        role_ledger_path=role_ledger_path,
        role_ledger_receipt_path=role_ledger_receipt_path,
        pilot=pilot,
        final=final,
        config=config,
    )
    return (
        pilot,
        final,
        {
            "anti_join_columns": anti_join,
            "anti_join_passed": True,
            "pilot": pilot_distribution,
            "final": final_distribution,
            "distribution_fractions_identical": True,
            "role_authority": role_authority,
        },
    )


def measured_paired_mcse(
    pilot_scores: pd.DataFrame,
    pilot_design: pd.DataFrame,
    *,
    draw_count: int,
) -> float:
    """MCSE of equal-parent-scene/equal-cell risk on separate design rows."""
    required = {
        "pair_id",
        "draw_index",
        "structured_detected",
        "sersic_detected",
    }
    if required - set(pilot_scores):
        raise ContractError("pilot score table lacks paired decision fields")
    selected = pilot_scores[pilot_scores["draw_index"].lt(draw_count)].copy()
    expected_indices = set(range(draw_count))
    draw_sets = selected.groupby("pair_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if (
        selected.duplicated(["pair_id", "draw_index"]).any()
        or set(selected["pair_id"].astype(str))
        != set(pilot_design["pair_id"].astype(str))
        or not draw_sets.map(lambda values: values == expected_indices).all()
    ):
        raise ContractError("pilot scores do not cover pilot design × candidate draws")
    if not selected["structured_detected"].isin([0, 1, False, True]).all() or not (
        selected["sersic_detected"].isin([0, 1, False, True]).all()
    ):
        raise ContractError("pilot threshold decisions are not binary")
    joined = selected.merge(
        pilot_design[["pair_id", "host_cell_id", "parent_scene_analysis_weight"]],
        on="pair_id",
        validate="many_to_one",
    )
    joined["difference"] = joined["sersic_detected"].astype(np.float64) - joined[
        "structured_detected"
    ].astype(np.float64)
    estimates = []
    for _, draw in joined.groupby("draw_index"):
        cells = [
            float(
                np.average(
                    cell["difference"].to_numpy(dtype=np.float64),
                    weights=cell["parent_scene_analysis_weight"].to_numpy(
                        dtype=np.float64
                    ),
                )
            )
            for _, cell in draw.groupby("host_cell_id")
        ]
        estimates.append(float(np.mean(cells)))
    values = np.asarray(estimates, dtype=np.float64)
    return (
        float(np.std(values, ddof=1) / np.sqrt(draw_count))
        if draw_count >= 2
        else float("inf")
    )


def _candidate_results(
    pilot_scores: pd.DataFrame,
    pilot_design: pd.DataFrame,
    candidates: list[int],
    target: float,
) -> tuple[list[dict[str, Any]], int | None]:
    results = []
    selected_count = None
    for candidate in candidates:
        mcse = measured_paired_mcse(pilot_scores, pilot_design, draw_count=candidate)
        passed = bool(np.isfinite(mcse) and mcse <= target)
        results.append(
            {
                "draw_count": candidate,
                "measured_mcse": mcse,
                "target": target,
                "passed": passed,
            }
        )
        if passed and selected_count is None:
            selected_count = candidate
    return results, selected_count


def _validate_pilot_score_receipt(
    *,
    receipt: dict[str, Any],
    pilot_scores_path: str | Path,
    pilot_design_path: str | Path,
    pilot_design_receipt_path: str | Path,
    final_design_path: str | Path,
    final_design_receipt_path: str | Path,
    role_ledger_path: str | Path,
    role_ledger_receipt_path: str | Path,
    scorer_receipt_path: str | Path,
    threshold_receipt_path: str | Path,
    config: dict[str, Any],
    require_final: bool,
) -> None:
    scorer_receipt = validate_scorer_receipt(
        scorer_receipt_path, config, require_final=require_final
    )
    _, threshold = validate_threshold_receipt(
        threshold_receipt_path, config, require_final=require_final
    )
    pilot_design = pd.read_parquet(pilot_design_path)
    candidates = [
        int(value) for value in config["photon_noise"]["candidate_draw_counts"]
    ]
    draw_count = max(candidates)
    expected = {
        "status": config["photon_noise"]["required_pilot_score_receipt_status"],
        "global_role": "design",
        "pilot_scores_sha256": sha256_file(pilot_scores_path),
        "pilot_design_sha256": sha256_file(pilot_design_path),
        "pilot_design_receipt_sha256": sha256_file(pilot_design_receipt_path),
        "final_design_sha256": sha256_file(final_design_path),
        "final_design_receipt_sha256": sha256_file(final_design_receipt_path),
        "role_ledger_sha256": sha256_file(role_ledger_path),
        "role_ledger_receipt_sha256": sha256_file(role_ledger_receipt_path),
        "scorer_receipt_sha256": sha256_file(scorer_receipt_path),
        "threshold_receipt_sha256": sha256_file(threshold_receipt_path),
        "candidate_draw_counts": candidates,
        "precision_target": config["photon_noise"]["paired_mcse_target_absolute_risk"],
        "rows": len(pilot_design) * draw_count,
        "pairs": len(pilot_design),
        "draw_count": draw_count,
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "comparison": ">=",
        "scorer_closure_sha256": scorer_receipt["scorer_closure_sha256"],
        "final_outcomes_used": False,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError("pilot score receipt is stale or incompletely bound")
    scores = pd.read_parquet(pilot_scores_path)
    required = {
        "pair_id",
        "draw_index",
        "structured_raw_logit_float32",
        "structured_raw_logit_float32_bits",
        "sersic_raw_logit_float32",
        "sersic_raw_logit_float32_bits",
        "structured_detected",
        "sersic_detected",
    }
    if (
        required - set(scores)
        or len(scores) != len(pilot_design) * draw_count
        or scores.duplicated(["pair_id", "draw_index"]).any()
        or set(scores["pair_id"].astype(str))
        != set(pilot_design["pair_id"].astype(str))
    ):
        raise ContractError("pilot score table support is incomplete")
    expected_draws = set(range(draw_count))
    draw_sets = scores.groupby("pair_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if not draw_sets.map(lambda values: values == expected_draws).all():
        raise ContractError("pilot score draw indices are incomplete")
    for value_column, bit_column, decision_column in (
        (
            "structured_raw_logit_float32",
            "structured_raw_logit_float32_bits",
            "structured_detected",
        ),
        (
            "sersic_raw_logit_float32",
            "sersic_raw_logit_float32_bits",
            "sersic_detected",
        ),
    ):
        values = scores[value_column].to_numpy(copy=True)
        bits = scores[bit_column].to_numpy(copy=True)
        decisions = scores[decision_column].to_numpy(dtype=bool)
        if (
            values.dtype != np.float32
            or bits.dtype != np.uint32
            or not np.isfinite(values).all()
            or not np.array_equal(values.view(np.uint32), bits)
            or not np.array_equal(values >= threshold, decisions)
        ):
            raise ContractError(
                "pilot native logits, bits, or threshold decisions differ"
            )


def build_draw_count_receipt(
    *,
    pilot_scores_path: str | Path,
    pilot_score_receipt_path: str | Path,
    pilot_design_path: str | Path,
    pilot_design_receipt_path: str | Path,
    final_design_path: str | Path,
    final_design_receipt_path: str | Path,
    role_ledger_path: str | Path,
    role_ledger_receipt_path: str | Path,
    scorer_receipt_path: str | Path,
    threshold_receipt_path: str | Path,
    config: dict[str, Any],
    output_path: str | Path,
    canary_mode: bool = False,
) -> dict[str, Any]:
    """Produce a receipt from a separate design-role pilot, never final pairs."""
    scorer_receipt = validate_scorer_receipt(
        scorer_receipt_path, config, require_final=not canary_mode
    )
    threshold_receipt, _ = validate_threshold_receipt(
        threshold_receipt_path, config, require_final=not canary_mode
    )
    pilot, _, distribution = _validate_pilot_and_final_designs(
        pilot_design_path=pilot_design_path,
        pilot_design_receipt_path=pilot_design_receipt_path,
        final_design_path=final_design_path,
        final_design_receipt_path=final_design_receipt_path,
        role_ledger_path=role_ledger_path,
        role_ledger_receipt_path=role_ledger_receipt_path,
        config=config,
        canary_mode=canary_mode,
    )
    pilot_score_receipt = load_json(pilot_score_receipt_path)
    _validate_pilot_score_receipt(
        receipt=pilot_score_receipt,
        pilot_scores_path=pilot_scores_path,
        pilot_design_path=pilot_design_path,
        pilot_design_receipt_path=pilot_design_receipt_path,
        final_design_path=final_design_path,
        final_design_receipt_path=final_design_receipt_path,
        role_ledger_path=role_ledger_path,
        role_ledger_receipt_path=role_ledger_receipt_path,
        scorer_receipt_path=scorer_receipt_path,
        threshold_receipt_path=threshold_receipt_path,
        config=config,
        require_final=not canary_mode,
    )
    scores = pd.read_parquet(pilot_scores_path)
    candidates = [
        int(value) for value in config["photon_noise"]["candidate_draw_counts"]
    ]
    target = float(config["photon_noise"]["paired_mcse_target_absolute_risk"])
    results, selected_count = _candidate_results(scores, pilot, candidates, target)
    status = (
        config["photon_noise"]["required_draw_count_receipt_status"]
        if selected_count is not None
        else "psrc_photon_draw_count_insufficient_design_only"
    )
    base = Path(output_path).resolve().parent
    receipt = {
        "schema_version": 3,
        "status": status,
        "draw_count": selected_count,
        "candidate_results": results,
        "candidate_draw_counts": candidates,
        "precision_target": target,
        "selection_rule": "smallest_candidate_with_measured_mcse_at_or_below_target",
        "pilot_is_separate_design_role": True,
        "distribution": distribution,
        "bindings": {
            "pilot_scores": _binding(pilot_scores_path, base),
            "pilot_score_receipt": _binding(pilot_score_receipt_path, base),
            "pilot_design": _binding(pilot_design_path, base),
            "pilot_design_receipt": _binding(pilot_design_receipt_path, base),
            "final_design": _binding(final_design_path, base),
            "final_design_receipt": _binding(final_design_receipt_path, base),
            "role_ledger": _binding(role_ledger_path, base),
            "role_ledger_receipt": _binding(role_ledger_receipt_path, base),
            "scorer_receipt": _binding(scorer_receipt_path, base),
            "threshold_receipt": _binding(threshold_receipt_path, base),
        },
        "scorer_receipt_status": scorer_receipt["status"],
        "threshold_receipt_status": threshold_receipt["status"],
        "final_outcomes_used": False,
        "canary_mode": canary_mode,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(Path(output_path), receipt)
    return receipt


def validate_draw_count_receipt(
    receipt_path: str | Path,
    config: dict[str, Any],
    *,
    expected_final_design_path: str | Path,
    expected_final_design_receipt_path: str | Path,
    expected_scorer_receipt_path: str | Path,
    expected_threshold_receipt_path: str | Path,
    production: bool,
) -> tuple[dict[str, Any], int]:
    """Reopen all pilot artifacts and independently recompute the selection."""
    receipt_path = Path(receipt_path)
    receipt = load_json(receipt_path)
    if (
        receipt.get("status")
        != config["photon_noise"]["required_draw_count_receipt_status"]
        or receipt.get("config") != config_provenance(config)
        or receipt.get("source_closure") != source_closure(config)
        or receipt.get("final_outcomes_used") is not False
        or (production and receipt.get("canary_mode") is not False)
    ):
        raise ContractError("draw-count receipt is not production-authorized")
    bindings = receipt.get("bindings", {})
    required_bindings = {
        "pilot_scores",
        "pilot_score_receipt",
        "pilot_design",
        "pilot_design_receipt",
        "final_design",
        "final_design_receipt",
        "role_ledger",
        "role_ledger_receipt",
        "scorer_receipt",
        "threshold_receipt",
    }
    if set(bindings) != required_bindings:
        raise ContractError("draw-count receipt artifact bindings are incomplete")
    paths = {}
    for name, binding in bindings.items():
        path = resolve_draw_binding(receipt_path, binding)
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ContractError(f"draw-count bound artifact drift: {name}")
        paths[name] = path
    expected_paths = {
        "final_design": Path(expected_final_design_path).resolve(),
        "final_design_receipt": Path(expected_final_design_receipt_path).resolve(),
        "scorer_receipt": Path(expected_scorer_receipt_path).resolve(),
        "threshold_receipt": Path(expected_threshold_receipt_path).resolve(),
    }
    if any(paths[name] != value for name, value in expected_paths.items()):
        raise ContractError("draw-count receipt binds another downstream authority")
    validate_scorer_receipt(paths["scorer_receipt"], config, require_final=production)
    validate_threshold_receipt(
        paths["threshold_receipt"], config, require_final=production
    )
    pilot, _, distribution = _validate_pilot_and_final_designs(
        pilot_design_path=paths["pilot_design"],
        pilot_design_receipt_path=paths["pilot_design_receipt"],
        final_design_path=paths["final_design"],
        final_design_receipt_path=paths["final_design_receipt"],
        role_ledger_path=paths["role_ledger"],
        role_ledger_receipt_path=paths["role_ledger_receipt"],
        config=config,
        canary_mode=bool(receipt.get("canary_mode")),
    )
    pilot_score_receipt = load_json(paths["pilot_score_receipt"])
    _validate_pilot_score_receipt(
        receipt=pilot_score_receipt,
        pilot_scores_path=paths["pilot_scores"],
        pilot_design_path=paths["pilot_design"],
        pilot_design_receipt_path=paths["pilot_design_receipt"],
        final_design_path=paths["final_design"],
        final_design_receipt_path=paths["final_design_receipt"],
        role_ledger_path=paths["role_ledger"],
        role_ledger_receipt_path=paths["role_ledger_receipt"],
        scorer_receipt_path=paths["scorer_receipt"],
        threshold_receipt_path=paths["threshold_receipt"],
        config=config,
        require_final=production,
    )
    candidates = [
        int(value) for value in config["photon_noise"]["candidate_draw_counts"]
    ]
    target = float(config["photon_noise"]["paired_mcse_target_absolute_risk"])
    results, selected = _candidate_results(
        pd.read_parquet(paths["pilot_scores"]),
        pilot,
        candidates,
        target,
    )
    if (
        receipt.get("candidate_draw_counts") != candidates
        or receipt.get("precision_target") != target
        or receipt.get("candidate_results") != results
        or receipt.get("draw_count") != selected
        or receipt.get("distribution") != distribution
        or selected is None
    ):
        raise ContractError("draw-count MCSE/grid/selection recomputation drift")
    return receipt, int(selected)
