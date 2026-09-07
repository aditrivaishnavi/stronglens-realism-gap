from __future__ import annotations

import numpy as np
import pandas as pd

from stronglens_probes.gdomain_pixel_attribution import (
    PIXEL_FEATURE_NAMES,
    attribution_suite,
    build_row_plan,
    crossfit_matrix,
)


def probe_config() -> dict[str, object]:
    return {
        "folds": 5,
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 5000,
        "estimator_seed": 42,
    }


def synthetic_plan_features(
    dimension: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(91)
    rows = []
    features = []
    for label, population in ((0, "simulated"), (1, "observed_reference")):
        for index in range(15):
            rows.append(
                {
                    "row_id": f"row-{label}-{index}",
                    "population": population,
                    "reference_id": (
                        f"reference-{index}" if label == 1 else "not_applicable"
                    ),
                    "host_galaxy_id": (
                        f"host-{index}" if label == 0 else "not_applicable"
                    ),
                    "dependency_component_id": f"component-{label}-{index}",
                    "fold": index % 5,
                    "label": label,
                }
            )
            values = rng.normal(0.0, 0.25, size=dimension)
            values[0] += 1.5 if label == 1 else -1.5
            features.append(values)
    return pd.DataFrame(rows), np.asarray(features, dtype=np.float32)


def test_row_plan_requires_sealed_fold_agreement() -> None:
    plan, _ = synthetic_plan_features(2)
    manifest = plan[
        ["row_id", "population", "reference_id", "host_galaxy_id"]
    ]
    oof = plan[["row_id", "dependency_component_id", "fold", "label"]]
    folds = plan[["dependency_component_id", "fold"]].drop_duplicates()
    observed = build_row_plan(manifest, oof, folds)
    assert observed["row_id"].tolist() == manifest["row_id"].tolist()


def test_crossfit_matrix_is_dependency_closed() -> None:
    plan, features = synthetic_plan_features(3)
    result = crossfit_matrix(
        plan,
        features,
        feature_names=["signal", "noise_a", "noise_b"],
        probe_config=probe_config(),
    )
    assert result.auc > 0.99
    assert len(result.fold_results) == 5
    assert result.oof["oof_score"].notna().all()
    assert len(result.coefficients) == 15


def test_attribution_suite_runs_all_registered_models() -> None:
    plan, features = synthetic_plan_features(18)
    results, oof, coefficients = attribution_suite(
        plan,
        features,
        probe_config=probe_config(),
    )
    assert len(results) == 43
    assert set(results["analysis"]) == {
        "full_model",
        "univariate",
        "leave_one_feature_out",
        "single_band",
        "leave_one_band_out",
    }
    assert set(coefficients["feature"]) == set(PIXEL_FEATURE_NAMES)
    assert oof.groupby(["analysis", "feature_or_band"]).size().eq(len(plan)).all()
