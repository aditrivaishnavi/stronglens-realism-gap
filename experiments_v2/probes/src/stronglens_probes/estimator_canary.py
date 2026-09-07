"""Synthetic coverage and full-refit benchmark for the actual probe estimators."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .contracts import ContractError


def _fit_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x, sample_weight=train_w)
    estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(
            scaler.transform(train_x),
            train_y,
            sample_weight=train_w,
        )
    return estimator.decision_function(scaler.transform(test_x))


def _study(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    references = rng.normal(0.8, 0.7, size=(43, 6))
    bare = rng.normal(-0.5, 0.7, size=(43, 6))
    injected = bare.copy()
    injected[:, 0] += 0.45
    component_sizes = rng.integers(1, 6, size=60)
    simulated = np.concatenate(
        [rng.normal(-0.4, 0.8, size=(size, 6)) for size in component_sizes]
    )
    simulated_component = np.repeat(np.arange(60), component_sizes)
    reference_fold = np.arange(43) % 5
    simulation_fold = np.arange(60) % 5
    return {
        "references": references,
        "bare": bare,
        "injected": injected,
        "simulated": simulated,
        "simulated_component": simulated_component,
        "reference_fold": reference_fold,
        "simulation_fold": simulation_fold,
        "component_sizes": component_sizes,
    }


def _metrics(
    study: dict[str, Any],
    reference_weight: np.ndarray,
    simulation_weight: np.ndarray,
) -> tuple[float, float]:
    references = study["references"]
    simulated = study["simulated"]
    component = study["simulated_component"]
    ref_fold = study["reference_fold"]
    sim_fold = study["simulation_fold"][component]
    x = np.vstack([references, simulated])
    y = np.r_[np.ones(len(references), dtype=int), np.zeros(len(simulated), dtype=int)]
    folds = np.r_[ref_fold, sim_fold]
    row_simulation_weight = (
        simulation_weight[component] / np.bincount(component)[component]
    )
    weights = np.r_[reference_weight, row_simulation_weight]
    domain_score = np.empty(len(x), dtype=float)
    for fold in range(5):
        test = folds == fold
        train = ~test
        domain_score[test] = _fit_score(
            x[train],
            y[train],
            weights[train],
            x[test],
        )
    auc = float(roc_auc_score(y, domain_score, sample_weight=weights))

    movement = np.empty(len(references), dtype=float)
    for fold in range(5):
        test = ref_fold == fold
        train = ~test
        train_x = np.vstack([references[train], study["bare"][train]])
        train_y = np.r_[
            np.ones(train.sum(), dtype=int),
            np.zeros(train.sum(), dtype=int),
        ]
        train_w = np.r_[reference_weight[train], reference_weight[train]]
        scaler = StandardScaler().fit(train_x, sample_weight=train_w)
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500).fit(
            scaler.transform(train_x),
            train_y,
            sample_weight=train_w,
        )
        raw_bare_train = estimator.decision_function(
            scaler.transform(study["bare"][train])
        )
        bare_mean = np.average(raw_bare_train, weights=reference_weight[train])
        bare_sd = np.sqrt(
            np.average(
                np.square(raw_bare_train - bare_mean),
                weights=reference_weight[train],
            )
        )
        if bare_sd <= 1e-12:
            raise ContractError("synthetic match axis is degenerate")
        raw_bare = estimator.decision_function(scaler.transform(study["bare"][test]))
        raw_injected = estimator.decision_function(
            scaler.transform(study["injected"][test])
        )
        movement[test] = (raw_injected - raw_bare) / bare_sd
    return auc, float(np.average(movement, weights=reference_weight))


def _intervals(
    study: dict[str, Any],
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.empty((draws, 2), dtype=float)
    for draw in range(draws):
        reference_weight = np.zeros(43, dtype=float)
        simulation_weight = np.zeros(60, dtype=float)
        for fold in range(5):
            reference_indices = np.flatnonzero(study["reference_fold"] == fold)
            simulation_indices = np.flatnonzero(study["simulation_fold"] == fold)
            np.add.at(
                reference_weight,
                rng.choice(
                    reference_indices,
                    size=len(reference_indices),
                    replace=True,
                ),
                1,
            )
            np.add.at(
                simulation_weight,
                rng.choice(
                    simulation_indices,
                    size=len(simulation_indices),
                    replace=True,
                ),
                1,
            )
        values[draw] = _metrics(
            study,
            reference_weight,
            simulation_weight,
        )
    return values, np.quantile(values, [0.025, 0.975], axis=0)


def _matching_failure_canary() -> bool:
    references = 43
    candidates = 60
    invalid = np.full((references, candidates), np.inf)
    cost = np.full((references, candidates + references), np.inf)
    cost[:, :candidates] = invalid
    for index in range(references):
        cost[index, candidates + index] = 1000.0
    _, selected = linear_sum_assignment(cost)
    return bool(np.all(selected >= candidates))


def run_estimator_coverage_canary(
    *,
    benchmark_draws: int,
    oracle_studies: int,
    coverage_studies: int,
    posterior_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Run grouped refits; this is a synthetic software canary, not final coverage."""
    if benchmark_draws != 50:
        raise ContractError("registered full-refit benchmark requires exactly 50 draws")
    if oracle_studies < 20 or coverage_studies < 10 or posterior_draws < 30:
        raise ContractError("estimator coverage canary is undersized")
    oracle_values = np.asarray(
        [
            _metrics(
                _study(seed + index),
                np.ones(43),
                np.ones(60),
            )
            for index in range(oracle_studies)
        ]
    )
    oracle = oracle_values.mean(axis=0)
    covered = np.zeros(2, dtype=int)
    for index in range(coverage_studies):
        study = _study(seed + 10_000 + index)
        _, interval = _intervals(
            study,
            draws=posterior_draws,
            seed=seed + 20_000 + index,
        )
        covered += (interval[0] <= oracle) & (oracle <= interval[1])

    benchmark_study = _study(seed + 30_000)
    started = time.perf_counter()
    benchmark_values, _ = _intervals(
        benchmark_study,
        draws=benchmark_draws,
        seed=seed + 40_000,
    )
    elapsed = time.perf_counter() - started
    covariance = float(np.cov(benchmark_values[:, 0], benchmark_values[:, 1])[0, 1])
    result = {
        "schema_version": 1,
        "status": "complete_synthetic_actual_estimator_canary",
        "estimators": [
            "dependency_grouped_cross_fitted_auc",
            "paired_cross_fitted_reference_axis_movement",
        ],
        "unequal_component_sizes": [1, 5],
        "matching_no_common_support_rejected": _matching_failure_canary(),
        "oracle": {
            "studies": oracle_studies,
            "gdomain_auc": float(oracle[0]),
            "gmatch_movement": float(oracle[1]),
        },
        "coverage": {
            "studies": coverage_studies,
            "posterior_draws_per_study": posterior_draws,
            "gdomain_auc": float(covered[0] / coverage_studies),
            "gmatch_movement": float(covered[1] / coverage_studies),
            "minimum_required_coverage": 0.9,
            "passed": bool(
                (covered[0] / coverage_studies) >= 0.9
                and (covered[1] / coverage_studies) >= 0.9
            ),
        },
        "full_refit_benchmark": {
            "draws": benchmark_draws,
            "fold_fits": benchmark_draws * 10,
            "elapsed_seconds": elapsed,
            "seconds_per_draw": elapsed / benchmark_draws,
            "joint_covariance": covariance,
            "finite": bool(np.isfinite(benchmark_values).all()),
            "auc_bounds_valid": bool(
                ((benchmark_values[:, 0] >= 0) & (benchmark_values[:, 0] <= 1)).all()
            ),
        },
        "interpretation": (
            "Synthetic Gaussian software/coverage canary for the implemented "
            "estimators. It does not establish coverage for final astronomical data, "
            "matching uncertainty, renderer uncertainty, or population transport."
        ),
    }
    result["interval_decision"] = (
        "eligible_for_further_scientific_review"
        if result["coverage"]["passed"]
        else "decline_95pct_confidence_intervals_use_conservative_bounds"
    )
    if (
        not result["matching_no_common_support_rejected"]
        or not result["full_refit_benchmark"]["finite"]
        or not result["full_refit_benchmark"]["auc_bounds_valid"]
        or not np.isfinite(covariance)
    ):
        raise ContractError("actual-estimator canary failed")
    return result
