"""Frozen-representation, dependency-grouped cross-fitted linear probes."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .authorities import AuthorityContext
from .contracts import ContractError, representation_label, require_columns
from .features import (
    FeatureArtifact,
    FeatureSuiteAuthority,
)


def _validate_features(manifest: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    if (
        values.ndim != 2
        or values.shape[0] != len(manifest)
        or values.shape[1] < 1
        or not np.isfinite(values).all()
    ):
        raise ContractError("feature matrix is non-finite or misaligned")
    if values.dtype != np.float32:
        raise ContractError("authenticated features must remain float32")
    return values


def _attach_folds(
    manifest: pd.DataFrame,
    fold_map: pd.DataFrame,
    *,
    component_column: str,
    n_splits: int,
) -> pd.DataFrame:
    require_columns(fold_map, {component_column, "fold"}, context="probe fold map")
    if fold_map[component_column].duplicated().any():
        raise ContractError("probe fold map duplicates dependency components")
    joined = manifest.merge(
        fold_map, on=component_column, how="left", validate="many_to_one"
    )
    if joined["fold"].isna().any():
        raise ContractError("probe fold map omits rows")
    joined["fold"] = joined["fold"].astype(int)
    if set(joined["fold"]) != set(range(n_splits)):
        raise ContractError("probe fold map does not contain every registered fold")
    if joined.groupby(component_column)["fold"].nunique().max() != 1:
        raise ContractError("a dependency component crosses probe folds")
    return joined


def _multipliers(
    components: pd.Series,
    supplied: Mapping[str, float] | None,
) -> np.ndarray:
    if supplied is None:
        return np.ones(len(components), dtype=np.float64)
    values = np.asarray([supplied.get(str(value), 0) for value in components], float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("bootstrap component multipliers must be non-negative")
    return values


def _component_equal_class_weights(
    labels: np.ndarray,
    components: pd.Series,
    multipliers: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    component_values = components.astype(str).to_numpy()
    for label in (0, 1):
        selected = labels == label
        active_components = sorted(
            {
                component_values[index]
                for index in np.flatnonzero(selected & (multipliers > 0))
            }
        )
        if not active_components:
            raise ContractError(f"resample contains no active class-{label} component")
        raw_total = 0.0
        for component in active_components:
            rows = selected & (component_values == component)
            multiplicity = float(multipliers[np.flatnonzero(rows)[0]])
            if not np.all(multipliers[rows] == multiplicity):
                raise ContractError("one component has inconsistent multipliers")
            weights[rows] = multiplicity / rows.sum()
            raw_total += multiplicity
        weights[selected] *= 0.5 / raw_total
    return weights * len(labels)


def _new_estimator(config: Mapping[str, Any]) -> LogisticRegression:
    probe = config["probe"]
    return LogisticRegression(
        C=float(probe["C"]),
        solver=str(probe["solver"]),
        max_iter=int(probe["max_iter"]),
        random_state=int(probe["estimator_seed"]),
    )


def _fit_axis(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[StandardScaler, LogisticRegression]:
    active = train_weights > 0
    if set(np.unique(train_labels[active])) != {0, 1}:
        raise ContractError("probe training fold lacks an active class")
    scaler = StandardScaler().fit(
        train_features[active], sample_weight=train_weights[active]
    )
    transformed = scaler.transform(train_features[active])
    estimator = _new_estimator(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(
            transformed,
            train_labels[active],
            sample_weight=train_weights[active],
        )
    if int(estimator.n_iter_[0]) >= estimator.max_iter:
        raise ContractError("linear probe reached max_iter")
    return scaler, estimator


def crossfit_domain(
    artifact: FeatureArtifact,
    fold_map: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
    component_multipliers: Mapping[str, float] | None = None,
    suite_authority: FeatureSuiteAuthority | None = None,
    _prevalidated: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the G_domain probe with fold-local preprocessing and dependency weights."""
    if not _prevalidated:
        if suite_authority is None:
            raise ContractError("G_domain requires external feature-suite authority")
        suite_authority.revalidate(
            config=dict(config),
            authority=authority,
            stage="probe_analysis",
        )
        if artifact is not suite_authority.artifacts.get(artifact.representation_id):
            raise ContractError("feature artifact is not in external suite authority")
    if artifact.authority_digest != authority.authority_digest:
        raise ContractError("G_domain features have the wrong authority")
    manifest = pd.read_parquet(artifact.manifest_path)
    features = np.load(artifact.features_path, mmap_mode="r", allow_pickle=False)
    representation_id = artifact.representation_id
    primary_id = config["representations"]["primary"]["id"]
    representation_role = "primary" if representation_id == primary_id else "diagnostic"
    if sklearn.__version__.split(".")[:2] != ["1", "9"]:
        raise ContractError("the frozen probe environment requires scikit-learn 1.9.x")
    required = {
        "row_id",
        "population",
        "dependency_component_id",
        "reference_id",
        "host_galaxy_id",
        "physical_group_id",
        "spatial_group_id",
        "psf_id",
        "render_parameter_id",
        "pixel_sha256",
    }
    require_columns(manifest, required, context="G_domain analysis manifest")
    if manifest["row_id"].astype(str).duplicated().any():
        raise ContractError("G_domain analysis row IDs are not unique")
    if set(manifest["population"]) != {"observed_reference", "simulated"}:
        raise ContractError("unexpected G_domain populations")
    observed = manifest["population"].eq("observed_reference")
    if (
        observed.sum() != 43
        or manifest.loc[observed, "reference_id"].astype(str).nunique() != 43
    ):
        raise ContractError(
            "G_domain must include every one of the shared 43 references"
        )
    authority.assert_reference_payloads(
        manifest.loc[observed],
        id_column="reference_id",
        payload_column="pixel_sha256",
        exact=True,
        context="G_domain feature manifest references",
    )
    authority.assert_reference_metadata(
        manifest.loc[observed],
        id_column="reference_id",
        physical_column="physical_group_id",
        spatial_column="spatial_group_id",
        exact=True,
        context="G_domain feature manifest references",
    )
    authority.assert_reference_render_metadata(
        manifest.loc[observed],
        id_column="reference_id",
        psf_column="psf_id",
        render_column="render_parameter_id",
        exact=True,
        context="G_domain feature manifest references",
    )
    values = _validate_features(manifest, features)
    n_splits = int(config["probe"]["folds"])
    attached = _attach_folds(
        manifest,
        fold_map,
        component_column="dependency_component_id",
        n_splits=n_splits,
    )
    labels = observed.astype(np.int8).to_numpy()
    multipliers = _multipliers(
        attached["dependency_component_id"], component_multipliers
    )
    weights = _component_equal_class_weights(
        labels, attached["dependency_component_id"], multipliers
    )
    scores = np.full(len(attached), np.nan, dtype=np.float64)
    fold_results: list[dict[str, Any]] = []
    for fold in range(n_splits):
        evaluate = attached["fold"].to_numpy() == fold
        train = ~evaluate
        scaler, estimator = _fit_axis(
            values[train], labels[train], weights[train], config
        )
        fold_scores = estimator.decision_function(scaler.transform(values[evaluate]))
        scores[evaluate] = fold_scores
        active_evaluate = evaluate & (weights > 0)
        if set(np.unique(labels[active_evaluate])) != {0, 1}:
            raise ContractError(f"G_domain evaluation fold {fold} lacks a class")
        fold_results.append(
            {
                "fold": fold,
                "rows": int(evaluate.sum()),
                "dependency_components": int(
                    attached.loc[evaluate, "dependency_component_id"].nunique()
                ),
                "weighted_auc": float(
                    roc_auc_score(
                        labels[active_evaluate],
                        scores[active_evaluate],
                        sample_weight=weights[active_evaluate],
                    )
                ),
            }
        )
    if not np.isfinite(scores).all():
        raise ContractError("G_domain has missing OOF scores")
    active = weights > 0
    primary_auc = float(
        roc_auc_score(labels[active], scores[active], sample_weight=weights[active])
    )
    row_weights = multipliers.copy()
    row_auc = float(
        roc_auc_score(
            labels[row_weights > 0],
            scores[row_weights > 0],
            sample_weight=row_weights[row_weights > 0],
        )
    )
    label = representation_label(
        config, representation_id, requested_role=representation_role
    )
    predictions = attached[
        [
            "row_id",
            "population",
            "reference_id",
            "host_galaxy_id",
            "dependency_component_id",
            "fold",
        ]
    ].copy()
    predictions["label"] = labels
    predictions["analysis_weight"] = weights
    predictions["oof_score"] = scores
    result = {
        "schema_version": 1,
        "name": "G_domain",
        "representation": representation_id,
        "representation_result_label": label,
        "feature_receipt_sha256": artifact.receipt_sha256,
        "primary_metric": "dependency_component_equal_oof_roc_auc",
        "primary_oof_auc": primary_auc,
        "row_weighted_oof_auc_diagnostic": row_auc,
        "observed_references": 43,
        "simulated_rows": int((~observed).sum()),
        "simulated_hosts": int(
            manifest.loc[~observed, "host_galaxy_id"].astype(str).nunique()
        ),
        "folds": fold_results,
        "oof_score_distribution": {
            "minimum": float(scores.min()),
            "median": float(np.median(scores)),
            "maximum": float(scores.max()),
        },
        "fold_local_standardisation": True,
        "dependency_grouped": True,
        "uncertainty_scope": "finite_cohort_model_conditional_descriptive",
        "population_inference": False,
        "significance_test": None,
        "interpretation": (
            "Finite-cohort, model-conditional separability in one frozen "
            "representation; not population inference, significance evidence, a "
            "causal rendering effect, or an independent replication."
        ),
    }
    return result, predictions


def _weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(weights.sum())
    if total <= 0:
        raise ContractError("weighted scale has no support")
    mean = float(np.sum(weights * values) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    return mean, float(np.sqrt(max(variance, 0.0)))


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if float(weights.sum()) <= 0:
        raise ContractError("weighted estimate has no support")
    return float(np.average(values, weights=weights))


def crossfit_gmatch_axis(
    artifact: FeatureArtifact,
    fold_map: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
    component_multipliers: Mapping[str, float] | None = None,
    suite_authority: FeatureSuiteAuthority | None = None,
    _prevalidated: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Fit observed-vs-bare axes and evaluate held-out within-target movement."""
    if not _prevalidated:
        if suite_authority is None:
            raise ContractError("G_match requires external feature-suite authority")
        suite_authority.revalidate(
            config=dict(config),
            authority=authority,
            stage="probe_analysis",
        )
        if artifact is not suite_authority.artifacts.get(artifact.representation_id):
            raise ContractError("feature artifact is not in external suite authority")
    if artifact.authority_digest != authority.authority_digest:
        raise ContractError("G_match features have the wrong authority")
    manifest = pd.read_parquet(artifact.manifest_path)
    features = np.load(artifact.features_path, mmap_mode="r", allow_pickle=False)
    representation_id = artifact.representation_id
    primary_id = config["representations"]["primary"]["id"]
    representation_role = "primary" if representation_id == primary_id else "diagnostic"
    required = {
        "row_id",
        "pair_id",
        "population",
        "reference_id",
        "target_galaxy_id",
        "dependency_component_id",
        "physical_group_id",
        "spatial_group_id",
        "psf_id",
        "render_parameter_id",
        "pixel_sha256",
    }
    require_columns(manifest, required, context="G_match analysis manifest")
    if manifest["row_id"].astype(str).duplicated().any():
        raise ContractError("G_match analysis row IDs are not unique")
    if set(manifest["population"]) != {
        "observed_reference",
        "bare",
        "injected",
    }:
        raise ContractError("unexpected G_match populations")
    matched_reference = manifest[manifest["population"].eq("observed_reference")]
    authority.assert_reference_payloads(
        matched_reference,
        id_column="reference_id",
        payload_column="pixel_sha256",
        exact=False,
        context="G_match feature manifest references",
    )
    authority.assert_reference_metadata(
        matched_reference,
        id_column="reference_id",
        physical_column="physical_group_id",
        spatial_column="spatial_group_id",
        exact=False,
        context="G_match feature manifest references",
    )
    authority.assert_reference_render_metadata(
        matched_reference,
        id_column="reference_id",
        psf_column="psf_id",
        render_column="render_parameter_id",
        exact=False,
        context="G_match feature manifest references",
    )
    minimum = int(
        config["G_match"]["matching"][
            "minimum_matched_references_for_five_fold_analysis"
        ]
    )
    if matched_reference["reference_id"].nunique() < minimum:
        raise ContractError("G_match feature manifest has fewer than 10 references")
    for pair_id, frame in manifest.groupby("pair_id", sort=False):
        if len(frame) != 3 or set(frame["population"]) != {
            "observed_reference",
            "bare",
            "injected",
        }:
            raise ContractError(f"{pair_id} does not contain one analysis trio")
        if frame["reference_id"].astype(str).nunique() != 1:
            raise ContractError(f"{pair_id} contains multiple references")
        generated = frame[frame["population"].isin(["bare", "injected"])]
        if generated["target_galaxy_id"].astype(str).nunique() != 1:
            raise ContractError(f"{pair_id} bare/injected target IDs differ")
        if frame["dependency_component_id"].nunique() != 1:
            raise ContractError(f"{pair_id} crosses dependency components")
    values = _validate_features(manifest, features)
    n_splits = int(config["probe"]["folds"])
    attached = _attach_folds(
        manifest,
        fold_map,
        component_column="dependency_component_id",
        n_splits=n_splits,
    )
    population = attached["population"].to_numpy()
    fit_rows = np.isin(population, ["observed_reference", "bare"])
    fit_labels = (population == "observed_reference").astype(np.int8)
    multipliers = _multipliers(
        attached["dependency_component_id"], component_multipliers
    )
    fit_weights = np.zeros(len(attached), dtype=float)
    fit_weights[fit_rows] = _component_equal_class_weights(
        fit_labels[fit_rows],
        attached.loc[fit_rows, "dependency_component_id"],
        multipliers[fit_rows],
    )
    scores = np.full(len(attached), np.nan, dtype=float)
    fold_scales: list[dict[str, Any]] = []
    for fold in range(n_splits):
        evaluate = attached["fold"].to_numpy() == fold
        train = (~evaluate) & fit_rows
        scaler, estimator = _fit_axis(
            values[train],
            fit_labels[train],
            fit_weights[train],
            config,
        )
        raw_train = estimator.decision_function(scaler.transform(values[train]))
        training_bare = train & (population == "bare") & (fit_weights > 0)
        mean, sd = _weighted_mean_sd(
            raw_train[(population[train] == "bare") & (fit_weights[train] > 0)],
            fit_weights[training_bare],
        )
        if not np.isfinite(sd) or sd <= 1e-12:
            raise ContractError(f"G_match fold {fold} has degenerate bare-axis scale")
        raw_evaluate = estimator.decision_function(scaler.transform(values[evaluate]))
        scores[evaluate] = (raw_evaluate - mean) / sd
        fold_scales.append(
            {
                "fold": fold,
                "training_rows": int(train.sum()),
                "training_injected_rows": 0,
                "training_bare_score_mean": mean,
                "training_bare_score_sd": sd,
            }
        )
    if not np.isfinite(scores).all():
        raise ContractError("G_match has missing cross-fitted axis scores")

    predictions = attached[
        [
            "row_id",
            "pair_id",
            "population",
            "reference_id",
            "target_galaxy_id",
            "dependency_component_id",
            "fold",
        ]
    ].copy()
    predictions["axis_score"] = scores
    wide = predictions.pivot(
        index=[
            "pair_id",
            "reference_id",
            "target_galaxy_id",
            "dependency_component_id",
        ],
        columns="population",
        values="axis_score",
    ).reset_index()
    wide.columns.name = None
    if wide[["observed_reference", "bare", "injected"]].isna().any().any():
        raise ContractError("G_match analysis trios did not pivot one-to-one")
    component_multiplier = {
        str(component): float(multiplier)
        for component, multiplier in zip(
            attached["dependency_component_id"], multipliers, strict=True
        )
    }
    pair_weights = np.asarray(
        [
            component_multiplier[str(component)]
            for component in wide["dependency_component_id"]
        ],
        dtype=float,
    )
    active_pairs = pair_weights > 0
    movement = wide["injected"].to_numpy() - wide["bare"].to_numpy()
    primary = _weighted_average(movement[active_pairs], pair_weights[active_pairs])

    def pooled_auc(positive: str, negative: str) -> float:
        labels = np.r_[np.ones(active_pairs.sum()), np.zeros(active_pairs.sum())]
        pooled_scores = np.r_[
            wide.loc[active_pairs, positive].to_numpy(),
            wide.loc[active_pairs, negative].to_numpy(),
        ]
        pooled_weights = np.r_[pair_weights[active_pairs], pair_weights[active_pairs]]
        return float(roc_auc_score(labels, pooled_scores, sample_weight=pooled_weights))

    concordance = np.where(movement > 0.0, 1.0, np.where(movement < 0.0, 0.0, 0.5))
    label = representation_label(
        config, representation_id, requested_role=representation_role
    )
    pair_results = wide.copy()
    pair_results["within_target_movement"] = movement
    pair_results["analysis_weight"] = pair_weights
    result = {
        "schema_version": 1,
        "name": "G_match",
        "representation": representation_id,
        "representation_result_label": label,
        "feature_receipt_sha256": artifact.receipt_sha256,
        "primary_metric": "equal_target_mean_within_target_axis_movement",
        "primary_within_target_movement": primary,
        "matched_targets": len(wide),
        "reference_axis_fit_rows": ["observed_reference", "bare"],
        "injected_rows_used_to_fit_axis_or_scale": False,
        "fold_scales": fold_scales,
        "diagnostics": {
            "pooled_observed_vs_bare_auc": pooled_auc("observed_reference", "bare"),
            "pooled_observed_vs_injected_auc": pooled_auc(
                "observed_reference", "injected"
            ),
            "pooled_injected_vs_bare_auc": pooled_auc("injected", "bare"),
            "paired_injected_above_bare_concordance": _weighted_average(
                concordance[active_pairs], pair_weights[active_pairs]
            ),
        },
        "movement_distribution": {
            "minimum": float(movement.min()),
            "median": float(np.median(movement)),
            "maximum": float(movement.max()),
        },
        "pooled_auc_is_primary": False,
        "uncertainty_scope": "finite_cohort_model_conditional_descriptive",
        "population_inference": False,
        "significance_test": None,
        "matching_uncertainty": "conditional_on_frozen_score_blind_match",
        "interpretation": (
            "Within-target movement toward the observed-reference side of a "
            "cross-fitted axis in this finite cohort; pooled AUCs are non-paired "
            "descriptive diagnostics, not population or significance claims."
        ),
    }
    return result, predictions, pair_results


def crossfit_domain_suite(
    suite_authority: FeatureSuiteAuthority,
    fold_map: pd.DataFrame,
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
) -> dict[str, tuple[dict[str, Any], pd.DataFrame]]:
    """Operational domain entry point requiring both diagnostics."""
    suite_authority.revalidate(
        config=config,
        authority=authority,
        stage="probe_analysis",
    )
    artifacts = suite_authority.artifacts
    return {
        representation: crossfit_domain(
            artifact,
            fold_map,
            config=config,
            authority=authority,
            _prevalidated=True,
        )
        for representation, artifact in artifacts.items()
    }


def crossfit_gmatch_suite(
    suite_authority: FeatureSuiteAuthority,
    fold_map: pd.DataFrame,
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
) -> dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    """Operational match entry point requiring both diagnostics."""
    suite_authority.revalidate(
        config=config,
        authority=authority,
        stage="probe_analysis",
    )
    artifacts = suite_authority.artifacts
    return {
        representation: crossfit_gmatch_axis(
            artifact,
            fold_map,
            config=config,
            authority=authority,
            _prevalidated=True,
        )
        for representation, artifact in artifacts.items()
    }
