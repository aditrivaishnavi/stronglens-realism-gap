#!/usr/bin/env python3
"""Independently refit and verify all persisted pixel-attribution models."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_indices(analysis: str, feature_or_band: str) -> tuple[int, ...]:
    if analysis == "full_model":
        return tuple(range(18))
    if feature_or_band in {"g", "r", "z"}:
        start = {"g": 0, "r": 6, "z": 12}[feature_or_band]
        band = tuple(range(start, start + 6))
        if analysis == "single_band":
            return band
        if analysis == "leave_one_band_out":
            return tuple(index for index in range(18) if index not in band)
    index = FEATURE_NAMES.index(feature_or_band)
    if analysis == "univariate":
        return (index,)
    if analysis == "leave_one_feature_out":
        return tuple(candidate for candidate in range(18) if candidate != index)
    raise RuntimeError(f"unknown attribution analysis: {analysis}")


def component_weights(plan: pd.DataFrame) -> np.ndarray:
    labels = plan["label"].to_numpy(int)
    components = plan["dependency_component_id"].astype(str).to_numpy()
    weights = np.zeros(len(plan))
    for label in (0, 1):
        selected = labels == label
        active = sorted(set(components[selected]))
        for component in active:
            rows = selected & (components == component)
            weights[rows] = 0.5 / len(active) / rows.sum()
    return weights * len(plan)


def refit(
    plan: pd.DataFrame,
    features: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = plan["label"].to_numpy(int)
    weights = component_weights(plan)
    scores = np.full(len(plan), np.nan)
    coefficients = []
    for fold in range(int(config["probe"]["folds"])):
        evaluate = plan["fold"].to_numpy(int) == fold
        train = ~evaluate
        scaler = StandardScaler().fit(
            features[train],
            sample_weight=weights[train],
        )
        estimator = LogisticRegression(
            C=float(config["probe"]["C"]),
            solver=str(config["probe"]["solver"]),
            max_iter=int(config["probe"]["max_iter"]),
            random_state=int(config["probe"]["estimator_seed"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(
                scaler.transform(features[train]),
                labels[train],
                sample_weight=weights[train],
            )
        scores[evaluate] = estimator.decision_function(
            scaler.transform(features[evaluate])
        )
        coefficients.append(estimator.coef_[0].copy())
    return scores, weights, np.asarray(coefficients)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def verify(root: Path) -> dict[str, Any]:
    output = root / "results_v2/probes/gdomain_pixel_attribution_v1"
    receipt_path = output / "GDOMAIN_PIXEL_ATTRIBUTION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    protocol_path = root / receipt["protocol_path"]
    config = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(protocol_path) != receipt["protocol_sha256"]:
        raise RuntimeError("pixel-attribution protocol hash differs")

    verified_inputs = {}
    for label, specification in config["inputs"].items():
        path = root / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise RuntimeError(f"pixel-attribution input differs: {label}")
        verified_inputs[label] = digest
    source = (
        root
        / "experiments_v2/probes/src/stronglens_probes/"
        "gdomain_pixel_attribution.py"
    )
    contracts = root / "experiments_v2/probes/src/stronglens_probes/contracts.py"
    runner = (
        root / "experiments_v2/probes/scripts/run_gdomain_pixel_attribution.py"
    )
    if sha256_file(source) != receipt["source_code_sha256"]:
        raise RuntimeError("pixel-attribution source hash differs")
    if sha256_file(contracts) != receipt["contracts_code_sha256"]:
        raise RuntimeError("contracts source hash differs")
    if sha256_file(runner) != receipt["runner_sha256"]:
        raise RuntimeError("pixel-attribution runner hash differs")

    verified_outputs = {}
    for name, specification in receipt["output_files"].items():
        path = output / name
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise RuntimeError(f"pixel-attribution output differs: {name}")
        verified_outputs[name] = digest

    manifest = pd.read_parquet(root / config["inputs"]["manifest"]["path"])
    authoritative = pd.read_parquet(
        root / config["inputs"]["authoritative_oof"]["path"]
    )
    folds = pd.read_parquet(root / config["inputs"]["folds"]["path"])
    plan = manifest[
        ["row_id", "population", "reference_id", "host_galaxy_id"]
    ].merge(
        authoritative[
            ["row_id", "dependency_component_id", "fold", "label"]
        ],
        on="row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    observed_folds = (
        plan[["dependency_component_id", "fold"]]
        .drop_duplicates()
        .sort_values("dependency_component_id", ignore_index=True)
    )
    expected_folds = folds.sort_values(
        "dependency_component_id", ignore_index=True
    )
    pd.testing.assert_frame_equal(observed_folds, expected_folds, check_exact=True)
    features = np.load(
        root / config["inputs"]["feature_matrix"]["path"],
        allow_pickle=False,
    )
    persisted_results = pd.read_parquet(output / "pixel_attribution.parquet")
    persisted_oof = pd.read_parquet(output / "pixel_attribution_oof.parquet")
    expected_keys = {("full_model", "all")}
    expected_keys.update(("univariate", feature) for feature in FEATURE_NAMES)
    expected_keys.update(
        ("leave_one_feature_out", feature) for feature in FEATURE_NAMES
    )
    expected_keys.update(("single_band", band) for band in ("g", "r", "z"))
    expected_keys.update(
        ("leave_one_band_out", band) for band in ("g", "r", "z")
    )
    observed_keys = set(
        zip(
            persisted_results["analysis"].astype(str),
            persisted_results["feature_or_band"].astype(str),
            strict=True,
        )
    )
    if observed_keys != expected_keys or len(persisted_results) != 43:
        raise RuntimeError("pixel-attribution result key set differs")
    if persisted_results[
        ["analysis", "feature_or_band"]
    ].duplicated().any():
        raise RuntimeError("pixel-attribution result keys are duplicated")
    full_auc = float(
        persisted_results.loc[
            persisted_results["analysis"].eq("full_model"), "auc"
        ].iloc[0]
    )
    if not np.allclose(
        persisted_results["delta_auc_from_full"],
        persisted_results["auc"] - full_auc,
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("pixel-attribution AUC deltas differ")
    expected_feature_counts = {
        "full_model": 18,
        "univariate": 1,
        "leave_one_feature_out": 17,
        "single_band": 6,
        "leave_one_band_out": 12,
    }
    for row in persisted_results.itertuples(index=False):
        if int(row.feature_count) != expected_feature_counts[str(row.analysis)]:
            raise RuntimeError("pixel-attribution feature count differs")
    summary = json.loads(
        (output / "GDOMAIN_PIXEL_ATTRIBUTION_RESULTS.json").read_text(
            encoding="utf-8"
        )
    )
    if tuple(summary["feature_order"]) != FEATURE_NAMES:
        raise RuntimeError("pixel-attribution summary feature order differs")
    if (
        int(summary.get("schema_version", 0)) != 1
        or summary.get("status") != "complete"
        or summary.get("diagnostic_id") != config["diagnostic_id"]
        or summary.get("scope") != config["scientific_scope"]
    ):
        raise RuntimeError("pixel-attribution summary scientific metadata differs")
    if not np.isclose(
        float(summary["full_auc"]), full_auc, rtol=0.0, atol=1e-14
    ):
        raise RuntimeError("pixel-attribution summary full AUC differs")
    summary_rows = pd.DataFrame(summary["rows"]).sort_values(
        ["analysis", "feature_or_band"], ignore_index=True
    )
    persisted_sorted = persisted_results.sort_values(
        ["analysis", "feature_or_band"], ignore_index=True
    )
    pd.testing.assert_frame_equal(
        summary_rows[persisted_sorted.columns],
        persisted_sorted,
        check_exact=False,
        rtol=0.0,
        atol=1e-14,
    )
    oof_keys = set(
        zip(
            persisted_oof["analysis"].astype(str),
            persisted_oof["feature_or_band"].astype(str),
            strict=True,
        )
    )
    if oof_keys != expected_keys or len(persisted_oof) != 43 * len(plan):
        raise RuntimeError("pixel-attribution OOF model set differs")
    if persisted_oof[
        ["analysis", "feature_or_band", "row_id"]
    ].duplicated().any():
        raise RuntimeError("pixel-attribution OOF keys are duplicated")

    checks = []
    full_coefficients: np.ndarray | None = None
    for row in persisted_results.itertuples(index=False):
        indices = feature_indices(str(row.analysis), str(row.feature_or_band))
        scores, weights, coefficients = refit(
            plan,
            np.asarray(features[:, indices], dtype=np.float32),
            config,
        )
        selected = persisted_oof[
            persisted_oof["analysis"].eq(row.analysis)
            & persisted_oof["feature_or_band"].eq(row.feature_or_band)
        ]
        metadata_columns = [
            "row_id",
            "population",
            "reference_id",
            "host_galaxy_id",
            "dependency_component_id",
            "fold",
            "label",
        ]
        pd.testing.assert_frame_equal(
            selected[metadata_columns].reset_index(drop=True),
            plan[metadata_columns].reset_index(drop=True),
            check_exact=True,
        )
        if not np.allclose(
            scores,
            selected["oof_score"].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(f"refit scores differ: {row.analysis}")
        if not np.allclose(
            weights,
            selected["analysis_weight"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"refit weights differ: {row.analysis}")
        auc = float(
            roc_auc_score(
                plan["label"],
                scores,
                sample_weight=weights,
            )
        )
        if not np.isclose(auc, float(row.auc), rtol=0.0, atol=1e-14):
            raise RuntimeError(f"refit AUC differs: {row.analysis}")
        if row.analysis == "full_model":
            full_coefficients = coefficients
        checks.append(
            {
                "analysis": str(row.analysis),
                "feature_or_band": str(row.feature_or_band),
                "auc": auc,
            }
        )
    if len(checks) != 43 or full_coefficients is None:
        raise RuntimeError("pixel-attribution model set is incomplete")

    persisted_coefficients = pd.read_parquet(
        output / "pixel_coefficient_summary.parquet"
    ).sort_values("feature", ignore_index=True)
    rows = []
    for index, feature in enumerate(FEATURE_NAMES):
        values = full_coefficients[:, index]
        rows.append(
            {
                "feature": feature,
                "median_standardized_coefficient": float(np.median(values)),
                "minimum_standardized_coefficient": float(values.min()),
                "maximum_standardized_coefficient": float(values.max()),
                "positive_folds": int((values > 0).sum()),
                "negative_folds": int((values < 0).sum()),
            }
        )
    pd.testing.assert_frame_equal(
        persisted_coefficients,
        pd.DataFrame(rows).sort_values("feature", ignore_index=True),
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )

    verification = {
        "schema_version": 1,
        "status": "pass",
        "receipt_sha256": sha256_file(receipt_path),
        "verifier_sha256": sha256_file(Path(__file__)),
        "verified_inputs": verified_inputs,
        "verified_outputs": verified_outputs,
        "model_refits": checks,
        "coefficient_summary_recomputed": True,
    }
    destination = output / "INDEPENDENT_VERIFICATION.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(
        json.dumps(
            verification,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
