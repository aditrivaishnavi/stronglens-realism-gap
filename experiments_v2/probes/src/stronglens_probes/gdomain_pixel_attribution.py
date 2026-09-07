"""Leakage-safe attribution of the final G_domain pixel-statistic probe."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .contracts import ContractError

PIXEL_FEATURE_NAMES = (
    "g_mean",
    "g_std",
    "g_median",
    "g_p99",
    "g_min",
    "g_max",
    "r_mean",
    "r_std",
    "r_median",
    "r_p99",
    "r_min",
    "r_max",
    "z_mean",
    "z_std",
    "z_median",
    "z_p99",
    "z_min",
    "z_max",
)

BAND_INDICES = {
    "g": tuple(range(0, 6)),
    "r": tuple(range(6, 12)),
    "z": tuple(range(12, 18)),
}


@dataclass(frozen=True)
class CrossfitResult:
    auc: float
    fold_results: list[dict[str, Any]]
    oof: pd.DataFrame
    coefficients: pd.DataFrame


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("pixel-attribution config must be a mapping")
    return payload


def validate_input_bindings(root: Path, config: Mapping[str, Any]) -> None:
    for label, specification in config["inputs"].items():
        path = root / str(specification["path"])
        if not path.exists() or sha256_file(path) != str(specification["sha256"]):
            raise ContractError(f"pixel-attribution input differs: {label}")


def validate_specification(config: Mapping[str, Any]) -> None:
    probe = config["probe"]
    if probe["weighting"] != "class_balanced_dependency_component_equal":
        raise ContractError("pixel-attribution weighting differs")
    if probe["scaler"] != "fold_local_standard_scaler":
        raise ContractError("pixel-attribution scaler differs")
    if tuple(config["pixel_attribution"]["feature_order"]) != PIXEL_FEATURE_NAMES:
        raise ContractError("pixel feature order differs")
    expected_analyses = {
        "full_model_reproduction",
        "univariate_crossfit",
        "leave_one_feature_out",
        "single_band",
        "leave_one_band_out",
        "fold_standardized_coefficients",
    }
    if set(config["pixel_attribution"]["analyses"]) != expected_analyses:
        raise ContractError("pixel-attribution analysis set differs")
    expected_forbidden = {
        "global_scaling",
        "outcome_dependent_folds",
        "row_bootstrap",
        "permutation_p_values",
    }
    if set(config["pixel_attribution"]["forbidden"]) != expected_forbidden:
        raise ContractError("pixel-attribution forbidden set differs")


def build_row_plan(
    manifest: pd.DataFrame,
    authoritative_oof: pd.DataFrame,
    fold_map: pd.DataFrame,
) -> pd.DataFrame:
    manifest_columns = [
        "row_id",
        "population",
        "reference_id",
        "host_galaxy_id",
    ]
    oof_columns = [
        "row_id",
        "dependency_component_id",
        "fold",
        "label",
    ]
    if not set(manifest_columns) <= set(manifest):
        raise ContractError("G_domain manifest lacks attribution columns")
    if not set(oof_columns) <= set(authoritative_oof):
        raise ContractError("authoritative OOF lacks attribution columns")
    component_plan = authoritative_oof[oof_columns].drop_duplicates("row_id")
    if len(component_plan) != len(authoritative_oof):
        raise ContractError("authoritative OOF duplicates row IDs")
    plan = manifest[manifest_columns].merge(
        component_plan,
        on="row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if plan[oof_columns[1:]].isna().any().any():
        raise ContractError("authoritative OOF omits manifest rows")
    expected_labels = plan["population"].eq("observed_reference").astype(int)
    if not np.array_equal(plan["label"].to_numpy(int), expected_labels):
        raise ContractError("authoritative OOF labels differ from population")
    observed_folds = (
        plan[["dependency_component_id", "fold"]]
        .drop_duplicates()
        .sort_values("dependency_component_id", ignore_index=True)
    )
    expected_folds = (
        fold_map[["dependency_component_id", "fold"]]
        .drop_duplicates()
        .sort_values("dependency_component_id", ignore_index=True)
    )
    try:
        pd.testing.assert_frame_equal(
            observed_folds,
            expected_folds,
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("authoritative OOF differs from sealed folds") from error
    return plan


def component_equal_class_weights(plan: pd.DataFrame) -> np.ndarray:
    labels = plan["label"].to_numpy(int)
    components = plan["dependency_component_id"].astype(str).to_numpy()
    weights = np.zeros(len(plan), dtype=float)
    for label in (0, 1):
        selected = labels == label
        active_components = sorted(set(components[selected]))
        if not active_components:
            raise ContractError(f"pixel-attribution cohort lacks class {label}")
        for component in active_components:
            rows = selected & (components == component)
            weights[rows] = 0.5 / len(active_components) / rows.sum()
    return weights * len(plan)


def crossfit_matrix(
    plan: pd.DataFrame,
    features: np.ndarray,
    *,
    feature_names: Sequence[str],
    probe_config: Mapping[str, Any],
) -> CrossfitResult:
    values = np.asarray(features)
    if (
        values.ndim != 2
        or values.shape != (len(plan), len(feature_names))
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ContractError("pixel-attribution feature matrix is invalid")
    labels = plan["label"].to_numpy(int)
    weights = component_equal_class_weights(plan)
    scores = np.full(len(plan), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for fold in range(int(probe_config["folds"])):
        evaluate = plan["fold"].to_numpy(int) == fold
        train = ~evaluate
        scaler = StandardScaler().fit(
            values[train],
            sample_weight=weights[train],
        )
        estimator = LogisticRegression(
            C=float(probe_config["C"]),
            solver=str(probe_config["solver"]),
            max_iter=int(probe_config["max_iter"]),
            random_state=int(probe_config["estimator_seed"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(
                scaler.transform(values[train]),
                labels[train],
                sample_weight=weights[train],
            )
        if int(estimator.n_iter_[0]) >= estimator.max_iter:
            raise ContractError("pixel-attribution probe reached max_iter")
        scores[evaluate] = estimator.decision_function(
            scaler.transform(values[evaluate])
        )
        folds.append(
            {
                "fold": fold,
                "rows": int(evaluate.sum()),
                "auc": float(
                    roc_auc_score(
                        labels[evaluate],
                        scores[evaluate],
                        sample_weight=weights[evaluate],
                    )
                ),
            }
        )
        for index, feature in enumerate(feature_names):
            coefficients.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "standardized_coefficient": float(estimator.coef_[0, index]),
                }
            )
    if not np.isfinite(scores).all():
        raise ContractError("pixel-attribution cross-fit produced missing scores")
    auc = float(roc_auc_score(labels, scores, sample_weight=weights))
    oof = plan[
        [
            "row_id",
            "population",
            "reference_id",
            "host_galaxy_id",
            "dependency_component_id",
            "fold",
            "label",
        ]
    ].copy()
    oof["analysis_weight"] = weights
    oof["oof_score"] = scores
    return CrossfitResult(
        auc=auc,
        fold_results=folds,
        oof=oof,
        coefficients=pd.DataFrame(coefficients),
    )


def attribution_suite(
    plan: pd.DataFrame,
    features: np.ndarray,
    *,
    probe_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if features.shape != (len(plan), 18):
        raise ContractError("pixel-statistic matrix has the wrong dimensions")
    analyses: list[tuple[str, str, tuple[int, ...]]] = [
        ("full_model", "all", tuple(range(18)))
    ]
    for index, feature in enumerate(PIXEL_FEATURE_NAMES):
        analyses.append(("univariate", feature, (index,)))
        analyses.append(
            (
                "leave_one_feature_out",
                feature,
                tuple(candidate for candidate in range(18) if candidate != index),
            )
        )
    for band, indices in BAND_INDICES.items():
        analyses.append(("single_band", band, indices))
        analyses.append(
            (
                "leave_one_band_out",
                band,
                tuple(index for index in range(18) if index not in indices),
            )
        )

    result_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    full_coefficients: pd.DataFrame | None = None
    full_auc: float | None = None
    for analysis, feature, indices in analyses:
        names = [PIXEL_FEATURE_NAMES[index] for index in indices]
        result = crossfit_matrix(
            plan,
            np.asarray(features[:, indices], dtype=np.float32),
            feature_names=names,
            probe_config=probe_config,
        )
        if analysis == "full_model":
            full_auc = result.auc
            full_coefficients = result.coefficients
        result_rows.append(
            {
                "analysis": analysis,
                "feature_or_band": feature,
                "feature_count": len(indices),
                "auc": result.auc,
            }
        )
        oof_frames.append(
            result.oof.assign(analysis=analysis, feature_or_band=feature)
        )
    if full_auc is None or full_coefficients is None:
        raise ContractError("pixel-attribution suite omitted full model")
    results = pd.DataFrame(result_rows)
    results["delta_auc_from_full"] = results["auc"] - full_auc
    coefficient_summary = (
        full_coefficients.groupby("feature", as_index=False)
        .agg(
            median_standardized_coefficient=("standardized_coefficient", "median"),
            minimum_standardized_coefficient=("standardized_coefficient", "min"),
            maximum_standardized_coefficient=("standardized_coefficient", "max"),
            positive_folds=(
                "standardized_coefficient",
                lambda values: int((values > 0).sum()),
            ),
            negative_folds=(
                "standardized_coefficient",
                lambda values: int((values < 0).sum()),
            ),
        )
        .sort_values("feature", ignore_index=True)
    )
    return results, pd.concat(oof_frames, ignore_index=True), coefficient_summary
