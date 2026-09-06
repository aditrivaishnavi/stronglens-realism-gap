from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError, load_json
from .inference import validate_and_join_scores
from .utils import atomic_json, safe_relative, sha256_file, stable_seed


def _independent_averages(joined: pd.DataFrame, draw_count: int) -> pd.DataFrame:
    frame = joined.copy()
    frame["draw_difference"] = frame["sersic_detected"].astype(np.float64) - frame[
        "structured_detected"
    ].astype(np.float64)
    columns = [
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
    averaged = (
        frame.groupby(columns, as_index=False)
        .agg(
            photon_draws=("draw_index", "size"),
            structured_detection_risk=("structured_detected", "mean"),
            sersic_detection_risk=("sersic_detected", "mean"),
            paired_detection_risk_difference=("draw_difference", "mean"),
        )
        .sort_values("pair_id", ignore_index=True)
    )
    if not averaged["photon_draws"].eq(draw_count).all():
        raise ContractError("independent photon averaging lost draw coverage")
    return averaged


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _independent_point(frame: pd.DataFrame) -> float:
    values = []
    for _, cell in frame.groupby("host_cell_id"):
        values.append(
            _weighted(
                cell["paired_detection_risk_difference"].to_numpy(float),
                cell["source_analysis_weight"].to_numpy(float),
            )
        )
    return float(np.mean(values))


def _independent_bootstrap(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    groups = np.sort(frame["host_spatial_group_id"].astype(str).unique())
    sources = np.sort(frame["source_id"].astype(str).unique())
    cells = np.sort(frame["host_cell_id"].astype(str).unique())
    group_index = pd.Index(groups).get_indexer(
        frame["host_spatial_group_id"].astype(str)
    )
    source_index = pd.Index(sources).get_indexer(frame["source_id"].astype(str))
    cell_index = pd.Index(cells).get_indexer(frame["host_cell_id"].astype(str))
    source_strata = (
        frame[["source_id", "source_stratum"]]
        .drop_duplicates()
        .set_index("source_id")
        .loc[sources, "source_stratum"]
        .astype(str)
        .to_numpy()
    )
    base = frame["source_analysis_weight"].to_numpy(float)
    outcome = frame["paired_detection_risk_difference"].to_numpy(float)
    draws = int(config["inference"]["bootstrap_draws"])
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            stable_seed(str(config["inference"]["bootstrap_seed_namespace"]), draws)
        )
    )
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        group_multiplier = rng.exponential(size=len(groups))
        source_multiplier = rng.exponential(size=len(sources))
        for stratum in np.unique(source_strata):
            members = source_strata == stratum
            source_multiplier[members] /= source_multiplier[members].mean()
        weights = base * group_multiplier[group_index] * source_multiplier[source_index]
        estimates[draw] = np.mean(
            [
                _weighted(outcome[cell_index == cell], weights[cell_index == cell])
                for cell in range(len(cells))
            ]
        )
    point = _independent_point(frame)
    estimates = estimates - estimates.mean() + point
    inflation = float(config["inference"]["interval_deviation_inflation"])
    return point + inflation * (estimates - point)


def _independent_mcse(joined: pd.DataFrame) -> float:
    estimates = []
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
        estimates.append(_independent_point(frame))
    return float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))


def independently_verify_final(
    *,
    design_path: str | Path,
    design_receipt_path: str | Path,
    photon_manifest_path: str | Path,
    photon_receipt_path: str | Path,
    scores_path: str | Path,
    score_receipt_path: str | Path,
    draw_count_receipt_path: str | Path,
    execution_lock_path: str | Path,
    result_path: str | Path,
    config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    joined, _, _, draw_receipt = validate_and_join_scores(
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
    photon = pd.read_parquet(photon_manifest_path)
    photon_root = Path(photon_manifest_path).resolve().parent
    files = photon[["photon_pair_relpath", "photon_pair_sha256"]].drop_duplicates()
    for row in files.itertuples(index=False):
        path = safe_relative(photon_root, str(row.photon_pair_relpath))
        if sha256_file(path) != str(row.photon_pair_sha256):
            raise ContractError("independent photon artifact hash differs")
        with np.load(path, allow_pickle=False) as payload:
            structured = np.asarray(payload["structured_cutouts"])
            sersic = np.asarray(payload["sersic_cutouts"])
        if (
            structured.dtype != np.float32
            or sersic.dtype != np.float32
            or structured.shape != (int(draw_receipt["draw_count"]), 101, 101, 3)
            or sersic.shape != structured.shape
            or not np.isfinite(structured).all()
            or not np.isfinite(sersic).all()
        ):
            raise ContractError("independent photon artifact content differs")
    averages = _independent_averages(joined, int(draw_receipt["draw_count"]))
    saved_averages_path = Path(result_path).resolve().parent / (
        "pair_photon_averages.parquet"
    )
    saved_averages = pd.read_parquet(saved_averages_path).sort_values(
        "pair_id", ignore_index=True
    )
    try:
        pd.testing.assert_frame_equal(averages, saved_averages, check_exact=True)
    except AssertionError as error:
        raise ContractError("saved pair photon averages differ") from error
    estimate = _independent_point(averages)
    bootstrap = _independent_bootstrap(averages, config)
    confidence = float(config["inference"]["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(bootstrap, [tail, 1.0 - tail])
    mcse = _independent_mcse(joined)
    result = load_json(result_path)
    expected = {
        "risk_difference_sersic_minus_structured": estimate,
        "threshold_conditional_95_interval": [
            float(interval[0]),
            float(interval[1]),
        ],
        "photon_mcse": mcse,
        "frozen_draw_count": draw_receipt["draw_count"],
        "design_sha256": sha256_file(design_path),
        "photon_manifest_sha256": sha256_file(photon_manifest_path),
        "scores_sha256": sha256_file(scores_path),
        "execution_lock_sha256": sha256_file(execution_lock_path),
    }
    differences = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if differences:
        raise ContractError(f"independent final result differs: {differences}")
    if (
        result.get("status") != "complete"
        or result.get("primary_inference_complete") is not True
        or result.get("photon_mcse_passed") is not True
        or result.get("pair_photon_averages_sha256") != sha256_file(saved_averages_path)
    ):
        raise ContractError("final analysis is incomplete or unbound")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    verification = {
        "schema_version": 1,
        "status": config["execution"]["required_analysis_receipt_status"],
        "analysis": "P_src",
        "pairs": int(joined["pair_id"].nunique()),
        "draw_count": int(draw_receipt["draw_count"]),
        "photon_artifacts_rehashed": len(files),
        "photon_rows": int(photon_receipt["rows"]),
        "native_score_bits_and_decisions_revalidated": True,
        "point_interval_and_mcse_recomputed": True,
        "risk_difference_sersic_minus_structured": estimate,
        "threshold_conditional_95_interval": [
            float(interval[0]),
            float(interval[1]),
        ],
        "photon_mcse": mcse,
        "result_sha256": sha256_file(result_path),
    }
    atomic_json(output_path, verification)
    return verification
