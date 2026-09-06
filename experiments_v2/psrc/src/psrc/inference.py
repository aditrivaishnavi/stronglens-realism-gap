from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authority import resolve_binding, validate_execution_lock
from .contracts import (
    ContractError,
    config_provenance,
    load_json,
    sha256_file,
    source_closure,
)
from .threshold import (
    _load_bound_calibration,
    threshold_evidence,
    validate_threshold_receipt,
)
from .utils import atomic_json, stable_seed


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum(dtype=np.float64))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(values * weights, dtype=np.float64) / denominator)


def validate_native_score_table(
    scores: pd.DataFrame,
    rendered_rows: pd.DataFrame,
    *,
    draw_count: int,
    threshold: np.float32,
) -> None:
    required_scores = {
        "pair_id",
        "rendered_pair_row_id",
        "draw_index",
        "structured_raw_logit_float32",
        "structured_raw_logit_float32_bits",
        "sersic_raw_logit_float32",
        "sersic_raw_logit_float32_bits",
        "structured_detected",
        "sersic_detected",
    }
    if required_scores - set(scores):
        raise ContractError("score table lacks native values/bits or pair identities")
    for value_column, bits_column in (
        ("structured_raw_logit_float32", "structured_raw_logit_float32_bits"),
        ("sersic_raw_logit_float32", "sersic_raw_logit_float32_bits"),
    ):
        values = scores[value_column].to_numpy(copy=True)
        bits = scores[bits_column].to_numpy(copy=True)
        if values.dtype != np.float32 or bits.dtype != np.uint32:
            raise ContractError("score logits/bits are not native float32/uint32")
        if not np.isfinite(values).all() or not np.array_equal(
            values.view(np.uint32), bits
        ):
            raise ContractError("score values and exact native bits disagree")
    expected_indices = set(range(draw_count))
    if scores.duplicated(["pair_id", "draw_index"]).any():
        raise ContractError("duplicate pair/draw in score table")
    draw_sets = scores.groupby("pair_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if not draw_sets.map(lambda value: value == expected_indices).all():
        raise ContractError("score table has incomplete draw indices")
    identity = ["pair_id", "rendered_pair_row_id", "draw_index"]
    if set(map(tuple, scores[identity].to_numpy())) != set(
        map(tuple, rendered_rows[identity].to_numpy())
    ):
        raise ContractError("score rows do not exactly bind rendered pair rows")
    structured = scores["structured_raw_logit_float32"].to_numpy(copy=True)
    sersic = scores["sersic_raw_logit_float32"].to_numpy(copy=True)
    if not np.array_equal(
        scores["structured_detected"].to_numpy(dtype=bool),
        structured >= threshold,
    ) or not np.array_equal(
        scores["sersic_detected"].to_numpy(dtype=bool),
        sersic >= threshold,
    ):
        raise ContractError("stored decisions do not equal native logit >= threshold")


def validate_analysis_design_binding(
    *,
    design_path: str | Path,
    design_receipt_path: str | Path,
    execution_lock: dict[str, Any],
) -> None:
    if (
        sha256_file(design_path) != execution_lock["bindings"]["pair_design"]["sha256"]
        or sha256_file(design_receipt_path)
        != execution_lock["bindings"]["pair_design_receipt"]["sha256"]
    ):
        raise ContractError("analysis design path/receipt differ from execution lock")


def validate_and_join_scores(
    *,
    design_path: str | Path,
    design_receipt_path: str | Path,
    photon_manifest_path: str | Path,
    photon_receipt_path: str | Path,
    scores_path: str | Path,
    score_receipt_path: str | Path,
    draw_count_receipt_path: str | Path,
    execution_lock_path: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.float32, dict[str, Any], dict[str, Any]]:
    """Bind scores to exact rendered pairs, complete draws, scorer and threshold."""
    lock = validate_execution_lock(execution_lock_path, config)
    if (
        sha256_file(draw_count_receipt_path)
        != lock["bindings"]["draw_count_receipt"]["sha256"]
    ):
        raise ContractError(
            "analysis draw-count receipt differs from execution lock"
        )
    threshold_path = resolve_binding(
        execution_lock_path, lock["bindings"]["threshold_receipt"]
    )
    threshold_receipt, threshold = validate_threshold_receipt(
        threshold_path, config, require_final=True
    )
    validate_analysis_design_binding(
        design_path=design_path,
        design_receipt_path=design_receipt_path,
        execution_lock=lock,
    )
    from .draw_count import validate_draw_count_receipt

    draw_receipt, frozen_draw_count = validate_draw_count_receipt(
        draw_count_receipt_path,
        config,
        expected_final_design_path=design_path,
        expected_final_design_receipt_path=design_receipt_path,
        expected_scorer_receipt_path=resolve_binding(
            execution_lock_path, lock["bindings"]["scorer_receipt"]
        ),
        expected_threshold_receipt_path=threshold_path,
        production=True,
    )
    design_receipt = load_json(design_receipt_path)
    photon_receipt = load_json(photon_receipt_path)
    score_receipt = load_json(score_receipt_path)
    expected_score = {
        "status": config["execution"]["required_score_receipt_status"],
        "scores_sha256": sha256_file(scores_path),
        "photon_manifest_sha256": sha256_file(photon_manifest_path),
        "photon_receipt_sha256": sha256_file(photon_receipt_path),
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "threshold_receipt_sha256": sha256_file(threshold_path),
        "scorer_receipt_sha256": lock["bindings"]["scorer_receipt"]["sha256"],
        "comparison": ">=",
        "config": config_provenance(config),
        "source_closure": source_closure(config),
        "draw_count": frozen_draw_count,
    }
    if any(score_receipt.get(key) != value for key, value in expected_score.items()):
        raise ContractError("score receipt is stale or binds another rendered bank")
    if (
        design_receipt.get("pair_design_sha256") != sha256_file(design_path)
        or design_receipt.get("status")
        != config["execution"]["required_design_receipt_status"]
        or photon_receipt.get("manifest_sha256") != sha256_file(photon_manifest_path)
        or photon_receipt.get("status")
        != config["execution"]["required_photon_receipt_status"]
        or photon_receipt.get("draw_count") != score_receipt.get("draw_count")
        or photon_receipt.get("draw_count") != frozen_draw_count
        or photon_receipt.get("draw_count_receipt_sha256")
        != sha256_file(draw_count_receipt_path)
    ):
        raise ContractError("design/photon/score receipt chain is inconsistent")
    design = pd.read_parquet(design_path)
    photon = pd.read_parquet(photon_manifest_path)
    scores = pd.read_parquet(scores_path)
    draws = int(photon_receipt["draw_count"])
    expected_indices = set(range(draws))
    for frame, label in ((photon, "photon"),):
        if frame.duplicated(["pair_id", "draw_index"]).any():
            raise ContractError(f"duplicate pair/draw in {label} table")
        draw_sets = frame.groupby("pair_id")["draw_index"].apply(
            lambda values: set(map(int, values))
        )
        if not draw_sets.map(lambda value: value == expected_indices).all():
            raise ContractError(f"{label} table has incomplete draw indices")
    if len(photon) != len(design) * draws or len(scores) != len(photon):
        raise ContractError("pair/draw row count differs from frozen cardinality")
    identity = ["pair_id", "rendered_pair_row_id", "draw_index"]
    validate_native_score_table(scores, photon, draw_count=draws, threshold=threshold)
    joined = photon.merge(scores, on=identity, validate="one_to_one").merge(
        design[
            [
                "pair_id",
                "parent_grec_scene_id",
                "host_galaxy_id",
                "host_spatial_group_id",
                "host_cell_id",
                "theta_e_arcsec",
                "source_r_mag",
                "source_id",
                "source_stratum",
                "source_analysis_weight",
            ]
        ],
        on=[
            "pair_id",
            "host_galaxy_id",
            "host_spatial_group_id",
            "host_cell_id",
            "source_id",
            "source_stratum",
        ],
        validate="many_to_one",
    )
    return joined, threshold, threshold_receipt, draw_receipt


def photon_average(joined: pd.DataFrame, expected_draws: int) -> pd.DataFrame:
    averaged = (
        joined.assign(
            draw_difference=(
                joined["sersic_detected"].astype(np.float64)
                - joined["structured_detected"].astype(np.float64)
            )
        )
        .groupby(
            [
                "pair_id",
                "parent_grec_scene_id",
                "host_galaxy_id",
                "host_spatial_group_id",
                "host_cell_id",
                "theta_e_arcsec",
                "source_r_mag",
                "source_id",
                "source_stratum",
                "source_analysis_weight",
            ],
            as_index=False,
        )
        .agg(
            photon_draws=("draw_index", "size"),
            structured_detection_risk=("structured_detected", "mean"),
            sersic_detection_risk=("sersic_detected", "mean"),
            paired_detection_risk_difference=("draw_difference", "mean"),
        )
    )
    if not averaged["photon_draws"].eq(expected_draws).all():
        raise ContractError("photon averaging did not retain frozen draw count")
    return averaged


def point_estimate(frame: pd.DataFrame) -> float:
    cell_values = []
    for _, cell in frame.groupby("host_cell_id"):
        cell_values.append(
            _weighted_mean(
                cell["paired_detection_risk_difference"].to_numpy(dtype=np.float64),
                cell["source_analysis_weight"].to_numpy(dtype=np.float64),
            )
        )
    return float(np.mean(cell_values))


def global_spatial_source_bootstrap(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed_namespace: str,
    interval_deviation_inflation: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use one global spatial multiplier and one source multiplier per draw."""
    groups = np.sort(frame["host_spatial_group_id"].astype(str).unique())
    sources = np.sort(frame["source_id"].astype(str).unique())
    cells = np.sort(frame["host_cell_id"].astype(str).unique())
    group_index = pd.Index(groups).get_indexer(
        frame["host_spatial_group_id"].astype(str)
    )
    source_index = pd.Index(sources).get_indexer(frame["source_id"].astype(str))
    source_stratum = (
        frame[["source_id", "source_stratum"]]
        .drop_duplicates()
        .set_index("source_id")
        .loc[sources, "source_stratum"]
        .astype(str)
        .to_numpy()
    )
    cell_index = pd.Index(cells).get_indexer(frame["host_cell_id"].astype(str))
    base_weight = frame["source_analysis_weight"].to_numpy(dtype=np.float64)
    outcome = frame["paired_detection_risk_difference"].to_numpy(dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(stable_seed(seed_namespace, draws)))
    raw = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        group_multiplier = rng.exponential(size=len(groups))
        source_multiplier = rng.exponential(size=len(sources))
        # Preserve the source-stratum target rather than randomly changing its mass.
        for stratum in np.unique(source_stratum):
            members = source_stratum == stratum
            source_multiplier[members] /= source_multiplier[members].mean()
        weights = (
            base_weight
            * group_multiplier[group_index]
            * source_multiplier[source_index]
        )
        cell_draws = [
            _weighted_mean(outcome[cell_index == cell], weights[cell_index == cell])
            for cell in range(len(cells))
        ]
        raw[draw] = np.mean(cell_draws)
    point = point_estimate(frame)
    finite = raw[np.isfinite(raw)]
    recentered = finite - finite.mean() + point
    if interval_deviation_inflation < 1.0:
        raise ContractError("bootstrap interval inflation cannot be below one")
    recentered = point + interval_deviation_inflation * (recentered - point)
    group_cells = (
        frame[["host_spatial_group_id", "host_cell_id"]]
        .drop_duplicates()
        .groupby("host_spatial_group_id")
        .size()
    )
    return recentered, {
        "host_sampling_unit": "host_spatial_group_id",
        "one_global_multiplier_per_spatial_group": True,
        "same_spatial_multiplier_across_all_host_cells": True,
        "source_sampling_unit": "source_id",
        "one_global_multiplier_per_source": True,
        "source_multiplier_normalized_within_source_stratum": True,
        "host_cell_weighting": "equal_fixed_cells",
        "source_strata_fixed": True,
        "draws_recentered_to_point_estimate": True,
        "interval_deviation_inflation": float(interval_deviation_inflation),
        "spatial_groups_crossing_host_cells": int((group_cells > 1).sum()),
        "finite_draws": len(recentered),
    }


def photon_mcse(joined: pd.DataFrame) -> float:
    draw_estimates = []
    for _, draw in joined.groupby("draw_index"):
        frame = draw[
            [
                "host_cell_id",
                "source_analysis_weight",
                "structured_detected",
                "sersic_detected",
            ]
        ].copy()
        frame["paired_detection_risk_difference"] = frame["sersic_detected"].astype(
            float
        ) - frame["structured_detected"].astype(float)
        draw_estimates.append(point_estimate(frame))
    values = np.asarray(draw_estimates, dtype=np.float64)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def precision_status(mcse: float, target: float) -> tuple[str, bool]:
    passed = bool(np.isfinite(mcse) and mcse <= target)
    return (
        "complete" if passed else "insufficient_photon_mcse_not_complete",
        passed,
    )


def repeated_threshold_sensitivity(
    *,
    calibration_manifest_path: str | Path,
    calibration_logits_path: str | Path,
    joined: pd.DataFrame,
    config: dict[str, Any],
    draws: int,
) -> dict[str, Any]:
    """Separately vary the calibration procedure using whole spatial groups."""
    manifest, values = _load_bound_calibration(
        calibration_manifest_path, calibration_logits_path, config
    )
    calibration = manifest[["calibration_id", "evaluation_spatial_group_id"]].merge(
        pd.read_parquet(calibration_logits_path),
        on="calibration_id",
        validate="one_to_one",
    )
    group_members = [
        indices.to_numpy()
        for _, indices in calibration.groupby(
            "evaluation_spatial_group_id"
        ).groups.items()
    ]
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            stable_seed(
                config["inference"]["bootstrap_seed_namespace"],
                "repeated-threshold",
                draws,
            )
        )
    )
    structured = joined["structured_raw_logit_float32"].to_numpy(copy=True)
    sersic = joined["sersic_raw_logit_float32"].to_numpy(copy=True)
    if structured.dtype != np.float32 or sersic.dtype != np.float32:
        raise ContractError("repeated-threshold sensitivity requires native logits")
    thresholds = np.empty(draws, dtype=np.float32)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(group_members), size=len(group_members))
        sampled_values = np.concatenate(
            [values[group_members[index]] for index in sampled]
        )
        _, threshold = threshold_evidence(
            sampled_values.astype(np.float32, copy=False),
            float(config["threshold"]["alpha"]),
        )
        thresholds[draw] = threshold
        pair = joined.copy()
        pair["structured_detected"] = structured >= threshold
        pair["sersic_detected"] = sersic >= threshold
        averaged = photon_average(
            pair,
            expected_draws=int(pair["draw_index"].nunique()),
        )
        estimates[draw] = point_estimate(averaged)
    return {
        "status": "separate_repeated_calibration_sensitivity",
        "not_the_primary_fixed_threshold_interval": True,
        "calibration_resampling_unit": "evaluation_spatial_group_id",
        "draws": draws,
        "threshold_bits_quantiles": np.quantile(
            thresholds.view(np.uint32), [0.025, 0.5, 0.975]
        ).tolist(),
        "risk_difference_quantiles": np.quantile(
            estimates, [0.025, 0.5, 0.975]
        ).tolist(),
        "limitation": (
            "This varies the repeated calibration procedure and does not replace "
            "the primary interval conditional on the once-frozen threshold."
        ),
    }


def analyze_bound_scores(
    *,
    design_path: str | Path,
    design_receipt_path: str | Path,
    photon_manifest_path: str | Path,
    photon_receipt_path: str | Path,
    scores_path: str | Path,
    score_receipt_path: str | Path,
    draw_count_receipt_path: str | Path,
    execution_lock_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
    repeated_threshold_draws: int = 0,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    joined, threshold, threshold_receipt, draw_receipt = validate_and_join_scores(
        design_path=design_path,
        design_receipt_path=design_receipt_path,
        photon_manifest_path=photon_manifest_path,
        photon_receipt_path=photon_receipt_path,
        scores_path=scores_path,
        score_receipt_path=score_receipt_path,
        draw_count_receipt_path=draw_count_receipt_path,
        execution_lock_path=execution_lock_path,
        config=config,
    )
    photon_receipt = load_json(photon_receipt_path)
    averaged = photon_average(joined, int(photon_receipt["draw_count"]))
    point = point_estimate(averaged)
    bootstrap, bootstrap_design = global_spatial_source_bootstrap(
        averaged,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed_namespace=str(config["inference"]["bootstrap_seed_namespace"]),
        interval_deviation_inflation=float(
            config["inference"]["interval_deviation_inflation"]
        ),
    )
    confidence = float(config["inference"]["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(bootstrap, [tail, 1.0 - tail])
    mcse = photon_mcse(joined)
    mcse_target = float(config["photon_noise"]["paired_mcse_target_absolute_risk"])
    status, precision_passed = precision_status(mcse, mcse_target)
    repeated = None
    if repeated_threshold_draws:
        repeated = repeated_threshold_sensitivity(
            calibration_manifest_path=threshold_receipt["calibration_manifest_path"],
            calibration_logits_path=threshold_receipt["calibration_logits_path"],
            joined=joined,
            config=config,
            draws=repeated_threshold_draws,
        )
    averages_path = output_dir / "pair_photon_averages.parquet"
    temporary_averages = output_dir / ".pair_photon_averages.parquet.partial"
    averaged.to_parquet(temporary_averages, index=False)
    temporary_averages.replace(averages_path)
    result = {
        "schema_version": 2,
        "status": status,
        "primary_inference_complete": precision_passed,
        "primary_estimand": config["inference"]["primary_estimand"],
        "risk_difference_sersic_minus_structured": point,
        "threshold_conditional_95_interval": [
            float(interval[0]),
            float(interval[1]),
        ],
        "bootstrap_design": bootstrap_design,
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "threshold_conditioning": (
            "Primary point/interval condition on the once-frozen common "
            "threshold; they do not include repeated calibration-sample variation."
        ),
        "repeated_threshold_sensitivity": repeated,
        "photon_mcse": mcse,
        "photon_mcse_target": mcse_target,
        "photon_mcse_passed": precision_passed,
        "draw_count_receipt_sha256": sha256_file(draw_count_receipt_path),
        "frozen_draw_count": draw_receipt["draw_count"],
        "equivalence_claim_permitted": False,
        "method_limit": config["inference"]["method_limitation"],
        "scores_sha256": sha256_file(scores_path),
        "score_receipt_sha256": sha256_file(score_receipt_path),
        "design_sha256": sha256_file(design_path),
        "design_receipt_sha256": sha256_file(design_receipt_path),
        "photon_manifest_sha256": sha256_file(photon_manifest_path),
        "photon_receipt_sha256": sha256_file(photon_receipt_path),
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "pair_photon_averages_sha256": sha256_file(averages_path),
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "RESULT.json", result)
    return result
