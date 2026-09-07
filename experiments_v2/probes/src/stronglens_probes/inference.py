"""Dependency-aware resampling with common draws for the shared references."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authorities import AuthorityContext
from .authorization import ExecutionAuthorization
from .contracts import (
    ContractError,
    canonical_json_sha256,
    require_columns,
    require_nonempty_columns,
    sha256_file,
)
from .features import FeatureArtifact, FeatureSuiteAuthority
from .probes import crossfit_domain, crossfit_gmatch_axis


@dataclass(frozen=True)
class JointResample:
    draw: int
    domain_component_multipliers: dict[str, float]
    gmatch_component_multipliers: dict[str, float]
    shared_reference_component_multipliers: dict[str, float]


def evaluate_joint_resample(
    resample: JointResample,
    *,
    domain_artifact: FeatureArtifact,
    domain_folds: pd.DataFrame,
    gmatch_artifact: FeatureArtifact,
    gmatch_folds: pd.DataFrame,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> dict[str, Any]:
    """Evaluate one draw while sharing memory-mapped feature bases."""
    domain, _ = crossfit_domain(
        domain_artifact,
        domain_folds,
        config=config,
        authority=authority,
        component_multipliers=resample.domain_component_multipliers,
        _prevalidated=True,
    )
    gmatch, _, _ = crossfit_gmatch_axis(
        gmatch_artifact,
        gmatch_folds,
        config=config,
        authority=authority,
        component_multipliers=resample.gmatch_component_multipliers,
        _prevalidated=True,
    )
    return {
        "draw": resample.draw,
        "gdomain_oof_auc": domain["primary_oof_auc"],
        "gmatch_within_target_movement": gmatch["primary_within_target_movement"],
        "minimum_shared_reference_multiplier": min(
            value
            for value in resample.shared_reference_component_multipliers.values()
            if value > 0
        ),
    }


def _one_to_one_mapping(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    context: str,
) -> dict[str, str]:
    require_columns(frame, {left, right}, context=context)
    grouped = frame.groupby(left)[right].nunique()
    if (grouped != 1).any():
        raise ContractError(f"{context} does not map {left} to one {right}")
    return (
        frame[[left, right]]
        .drop_duplicates(left)
        .set_index(left)[right]
        .astype(str)
        .to_dict()
    )


def iter_joint_resamples(
    domain_manifest: pd.DataFrame,
    gmatch_manifest: pd.DataFrame,
    domain_folds: pd.DataFrame,
    gmatch_folds: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> Iterator[JointResample]:
    """Yield common reference-component draws and experiment-specific multipliers."""
    authority.assert_config(config)
    if draws < 1:
        raise ContractError("bootstrap draws must be positive")
    required = {
        "population",
        "reference_id",
        "dependency_component_id",
        "joint_dependency_id",
    }
    require_columns(domain_manifest, required, context="G_domain bootstrap manifest")
    require_columns(gmatch_manifest, required, context="G_match bootstrap manifest")
    require_nonempty_columns(
        domain_manifest, required, context="G_domain bootstrap manifest"
    )
    require_nonempty_columns(
        gmatch_manifest, required, context="G_match bootstrap manifest"
    )
    if not domain_manifest["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("G_domain bootstrap manifest has wrong authority")
    if not gmatch_manifest["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("G_match bootstrap manifest has wrong authority")
    domain_reference = domain_manifest[
        domain_manifest["population"].eq("observed_reference")
    ].copy()
    authority.assert_exact_references(
        domain_reference,
        id_column="reference_id",
        context="joint-inference domain references",
    )
    domain_reference_map = _one_to_one_mapping(
        domain_reference,
        left="reference_id",
        right="joint_dependency_id",
        context="domain shared-reference map",
    )
    match_reference = gmatch_manifest[
        gmatch_manifest["population"].eq("observed_reference")
    ].copy()
    if match_reference["reference_id"].nunique() < 10:
        raise ContractError("joint inference requires at least 10 matched references")
    match_reference_map = _one_to_one_mapping(
        match_reference,
        left="reference_id",
        right="joint_dependency_id",
        context="match shared-reference map",
    )
    for reference_id, joint_id in match_reference_map.items():
        if domain_reference_map.get(reference_id) != joint_id:
            raise ContractError(
                f"shared reference {reference_id} has inconsistent joint dependency"
            )

    domain_observed_component = _one_to_one_mapping(
        domain_reference,
        left="dependency_component_id",
        right="joint_dependency_id",
        context="domain observed component map",
    )
    match_component = _one_to_one_mapping(
        gmatch_manifest,
        left="dependency_component_id",
        right="joint_dependency_id",
        context="match component map",
    )
    all_domain_reference_components = sorted(set(domain_reference_map.values()))
    shared_reference_components = sorted(set(match_reference_map.values()))
    domain_only_reference_components = sorted(
        set(all_domain_reference_components).difference(shared_reference_components)
    )
    all_domain_simulated_components = set(
        domain_manifest.loc[
            domain_manifest["population"].eq("simulated"),
            "dependency_component_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    if not all_domain_simulated_components:
        raise ContractError("joint inference has no G_domain simulation components")
    domain_only_components = sorted(
        all_domain_simulated_components.difference(all_domain_reference_components)
    )
    domain_fold_map = (
        domain_folds.set_index("dependency_component_id")["fold"].astype(int).to_dict()
    )
    gmatch_fold_map = (
        gmatch_folds.set_index("dependency_component_id")["fold"].astype(int).to_dict()
    )
    shared_strata: dict[tuple[int, int], list[str]] = {}
    for component in shared_reference_components:
        if component not in domain_fold_map or component not in gmatch_fold_map:
            raise ContractError("shared reference component lacks a fold")
        shared_strata.setdefault(
            (domain_fold_map[component], gmatch_fold_map[component]), []
        ).append(component)
    domain_reference_strata: dict[int, list[str]] = {}
    for component in domain_only_reference_components:
        if component not in domain_fold_map:
            raise ContractError("domain-only reference component lacks a fold")
        domain_reference_strata.setdefault(domain_fold_map[component], []).append(
            component
        )
    simulation_strata: dict[int, list[str]] = {}
    for component in domain_only_components:
        if component not in domain_fold_map:
            raise ContractError("domain simulation component lacks a fold")
        simulation_strata.setdefault(domain_fold_map[component], []).append(component)

    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
        shared_count = {component: 0.0 for component in shared_reference_components}
        for components in shared_strata.values():
            for component in rng.choice(components, size=len(components), replace=True):
                shared_count[str(component)] += 1.0
        domain_reference_count = {
            component: 0.0 for component in domain_only_reference_components
        }
        for components in domain_reference_strata.values():
            for component in rng.choice(components, size=len(components), replace=True):
                domain_reference_count[str(component)] += 1.0
        simulated_count = {component: 0.0 for component in domain_only_components}
        for components in simulation_strata.values():
            for component in rng.choice(components, size=len(components), replace=True):
                simulated_count[str(component)] += 1.0
        for fold in range(int(config["probe"]["folds"])):
            if not (
                any(
                    shared_count[component] > 0
                    for component in shared_reference_components
                    if domain_fold_map[component] == fold
                )
                or any(
                    domain_reference_count[component] > 0
                    for component in domain_only_reference_components
                    if domain_fold_map[component] == fold
                )
            ):
                raise ContractError("stratified bootstrap lost reference class")
            if not any(
                simulated_count[component] > 0
                for component in domain_only_components
                if domain_fold_map[component] == fold
            ):
                raise ContractError("stratified bootstrap lost simulation class")
        domain_multipliers = {}
        for component, joint_id in domain_observed_component.items():
            domain_multipliers[component] = float(
                shared_count[joint_id]
                if joint_id in shared_count
                else domain_reference_count[joint_id]
            )
        domain_multipliers.update(
            {
                component: float(simulated_count[component])
                for component in domain_only_components
            }
        )
        match_multipliers = {
            component: float(shared_count[joint_id])
            for component, joint_id in match_component.items()
        }
        yield JointResample(
            draw=draw,
            domain_component_multipliers=domain_multipliers,
            gmatch_component_multipliers=match_multipliers,
            shared_reference_component_multipliers={
                component: float(shared_count[component])
                for component in shared_reference_components
            },
        )


def joint_refit_bootstrap(
    domain_suite: FeatureSuiteAuthority,
    domain_folds: pd.DataFrame,
    gmatch_suite: FeatureSuiteAuthority,
    gmatch_folds: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
    draws: int | None = None,
    seed: int | None = None,
    design_only: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Refit both probes under common shared-reference dependency draws."""
    authority.assert_config(config)
    domain_suite.revalidate(
        config=dict(config), authority=authority, stage="joint_inference"
    )
    gmatch_suite.revalidate(
        config=dict(config), authority=authority, stage="joint_inference"
    )
    domain_artifacts = domain_suite.artifacts
    gmatch_artifacts = gmatch_suite.artifacts
    primary_id = config["representations"]["primary"]["id"]
    domain_artifact = domain_artifacts[primary_id]
    gmatch_artifact = gmatch_artifacts[primary_id]
    feature_suite_receipt_sha256 = canonical_json_sha256(
        {"domain": domain_suite.sha256, "gmatch": gmatch_suite.sha256}
    )
    domain_manifest = pd.read_parquet(domain_artifact.manifest_path)
    gmatch_manifest = pd.read_parquet(gmatch_artifact.manifest_path)
    inference = config["inference"]
    total_draws = int(inference["bootstrap_draws"] if draws is None else draws)
    if not design_only and total_draws != int(inference["bootstrap_draws"]):
        raise ContractError("production joint inference must use exactly 10,000 draws")
    bootstrap_seed = int(inference["seed"] if seed is None else seed)
    records: list[dict[str, Any]] = []
    for resample in iter_joint_resamples(
        domain_manifest,
        gmatch_manifest,
        domain_folds,
        gmatch_folds,
        draws=total_draws,
        seed=bootstrap_seed,
        config=config,
        authority=authority,
    ):
        records.append(
            evaluate_joint_resample(
                resample,
                domain_artifact=domain_artifact,
                domain_folds=domain_folds,
                gmatch_artifact=gmatch_artifact,
                gmatch_folds=gmatch_folds,
                config=config,
                authority=authority,
            )
        )
    values = pd.DataFrame(records)
    covariance = (
        float(
            np.cov(
                values["gdomain_oof_auc"],
                values["gmatch_within_target_movement"],
                ddof=1,
            )[0, 1]
        )
        if total_draws > 1
        else None
    )
    correlation = (
        float(
            values[["gdomain_oof_auc", "gmatch_within_target_movement"]]
            .corr()
            .iloc[0, 1]
        )
        if total_draws > 1
        else None
    )
    summary = {
        "schema_version": 1,
        "draws": total_draws,
        "seed": bootstrap_seed,
        "probe_refit_each_draw": True,
        "matching_refit_each_draw": False,
        "matching_condition": "conditional_on_frozen_score_blind_match",
        "multiplier_scheme": "stratified_dependency_bootstrap",
        "feature_suite_receipt_sha256": feature_suite_receipt_sha256,
        "authority_digest": authority.authority_digest,
        "shared_reference_component_resamples": True,
        "independent_replication_claim": False,
        "gdomain_auc_descriptive_quantiles_2.5_97.5": np.quantile(
            values["gdomain_oof_auc"], [0.025, 0.975]
        ).tolist(),
        "gmatch_movement_descriptive_quantiles_2.5_97.5": np.quantile(
            values["gmatch_within_target_movement"], [0.025, 0.975]
        ).tolist(),
        "cross_probe_covariance": covariance,
        "cross_probe_correlation": correlation,
        "resampling_distribution": {
            "gdomain_auc": {
                "minimum": float(values["gdomain_oof_auc"].min()),
                "median": float(values["gdomain_oof_auc"].median()),
                "maximum": float(values["gdomain_oof_auc"].max()),
            },
            "gmatch_movement": {
                "minimum": float(values["gmatch_within_target_movement"].min()),
                "median": float(values["gmatch_within_target_movement"].median()),
                "maximum": float(values["gmatch_within_target_movement"].max()),
            },
        },
        "formal_confidence_intervals": None,
        "coverage_validated_for_final_estimators": False,
        "uncertainty_label": (
            "finite_cohort_model_conditional_resampling_stability_quantiles"
        ),
        "population_inference": False,
        "significance_test": None,
        "conservative_parameter_bounds": {
            "gdomain_auc": [0.0, 1.0],
            "gmatch_movement": None,
        },
        "interval_note": (
            "No 95% CI is claimed until actual-estimator design simulation attains "
            "at least 0.90 coverage for both estimands. No finite model-free bound "
            "exists for standardized movement."
        ),
    }
    return summary, values


def bootstrap_recovery_cross_reference(
    scored_parent_path: Path,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
    draws: int,
    seed: int,
    design_only: bool = False,
    authorization: ExecutionAuthorization,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Resample shared host/spatial dependencies for full and subset recovery."""
    authority.assert_config(config)
    scored_parent_manifest_sha256 = sha256_file(scored_parent_path)
    authorization.revalidate(
        config=dict(config),
        authority=authority,
        stage="recovery_inference",
        manifest_sha256=scored_parent_manifest_sha256,
    )
    scored_parent = pd.read_parquet(scored_parent_path)
    if draws < 2:
        raise ContractError("recovery bootstrap requires at least two draws")
    if not design_only and draws != int(config["inference"]["bootstrap_draws"]):
        raise ContractError("production recovery inference requires 10,000 draws")
    require_columns(
        scored_parent,
        {"host_spatial_group_id", "detected", "in_gdomain_subset", "data_role"},
        context="scored G_rec parent",
    )
    if not scored_parent["data_role"].eq(authorization.data_role).all():
        raise ContractError("recovery inputs have the wrong authorized data role")
    if design_only and authorization.data_role != "design":
        raise ContractError("design-only recovery received final-role inputs")
    if not design_only and authorization.data_role != "final":
        raise ContractError("final recovery requires final-role authorization")
    if not scored_parent["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("scored recovery parent has the wrong authority")
    for column in ("detected", "in_gdomain_subset"):
        if not pd.api.types.is_bool_dtype(scored_parent[column].dtype):
            raise ContractError(f"recovery input {column} must be strict boolean")
    require_nonempty_columns(
        scored_parent,
        {"host_spatial_group_id"},
        context="scored G_rec parent",
    )
    groups = scored_parent["host_spatial_group_id"].astype(str)
    unique_groups = sorted(groups.unique())
    if len(unique_groups) < 2:
        raise ContractError("recovery inference needs at least two dependency groups")
    group_code = {group: index for index, group in enumerate(unique_groups)}
    inverse = np.asarray([group_code[group] for group in groups], dtype=int)
    detected = scored_parent["detected"].to_numpy()
    subset = scored_parent["in_gdomain_subset"].to_numpy()
    if not subset.any() or subset.all():
        raise ContractError("recovery subset must have nonempty subset and complement")
    group_values = groups.to_numpy()
    group_subset = {
        group: (
            "mixed"
            if subset[group_values == group].any()
            and not subset[group_values == group].all()
            else "subset"
            if subset[group_values == group].all()
            else "complement"
        )
        for group in unique_groups
    }
    strata: dict[str, list[int]] = {}
    for group, index in group_code.items():
        strata.setdefault(group_subset[group], []).append(index)
    records: list[dict[str, Any]] = []
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
        component_weights = np.zeros(len(unique_groups), dtype=float)
        for indices in strata.values():
            for index in rng.choice(indices, size=len(indices), replace=True):
                component_weights[int(index)] += 1.0
        weights = component_weights[inverse]
        subset_weights = weights * subset
        if weights.sum() <= 0 or subset_weights.sum() <= 0:
            raise ContractError(f"recovery draw {draw} has no subset support")
        records.append(
            {
                "draw": draw,
                "full_recovery": float(np.average(detected, weights=weights)),
                "subset_recovery": float(np.average(detected, weights=subset_weights)),
            }
        )
    values = pd.DataFrame(records)
    summary = {
        "schema_version": 1,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "host_spatial_group_id",
        "multiplier_scheme": "stratified_dependency_bootstrap",
        "authority_digest": authority.authority_digest,
        "common_draws_for_full_and_subset": True,
        "full_descriptive_quantiles_2.5_97.5": np.quantile(
            values["full_recovery"], [0.025, 0.975]
        ).tolist(),
        "subset_descriptive_quantiles_2.5_97.5": np.quantile(
            values["subset_recovery"], [0.025, 0.975]
        ).tolist(),
        "full_subset_covariance": float(
            np.cov(values["full_recovery"], values["subset_recovery"], ddof=1)[0, 1]
        )
        if draws > 1
        else None,
        "independent_replication_claim": False,
        "formal_confidence_intervals": None,
        "coverage_validated_for_final_recovery": False,
        "uncertainty_label": (
            "finite_cohort_model_conditional_resampling_stability_quantiles"
        ),
        "population_inference": False,
        "significance_test": None,
        "conservative_probability_bounds": [0.0, 1.0],
    }
    return summary, values


def stratified_dependency_support_canary(
    component_folds: np.ndarray,
    component_classes: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Prove every fixed fold/class stratum survives every bootstrap draw."""
    folds = np.asarray(component_folds, dtype=int)
    classes = np.asarray(component_classes, dtype=int)
    if draws < 1 or len(folds) < 2 or len(folds) != len(classes):
        raise ContractError("invalid stratified support-canary inputs")
    unique_folds = np.unique(folds)
    unique_classes = np.unique(classes)
    if len(unique_folds) < 2 or len(unique_classes) != 2:
        raise ContractError("support canary requires at least two folds")
    strata = {
        (fold, label): np.flatnonzero((folds == fold) & (classes == label))
        for fold in unique_folds
        for label in unique_classes
    }
    if any(len(indices) == 0 for indices in strata.values()):
        raise ContractError("support canary has an empty fold/class stratum")
    minimum_support = float("inf")
    maximum_multiplier = 0
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
        weights = np.zeros(len(folds), dtype=int)
        for indices in strata.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            np.add.at(weights, sampled, 1)
        for indices in strata.values():
            support = int(weights[indices].sum())
            minimum_support = min(minimum_support, support)
            if support <= 0:
                raise ContractError(f"stratum lost support at draw {draw}")
        maximum_multiplier = max(maximum_multiplier, int(weights.max()))
    return {
        "schema_version": 1,
        "scheme": "stratified_dependency_bootstrap",
        "draws": draws,
        "components": len(folds),
        "folds": len(unique_folds),
        "minimum_stratum_support": int(minimum_support),
        "maximum_component_multiplier": maximum_multiplier,
        "all_fold_class_strata_supported": True,
    }
