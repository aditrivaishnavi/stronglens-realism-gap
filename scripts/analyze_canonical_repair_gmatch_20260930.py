#!/usr/bin/env python3
"""Recompute four G_match axes for the canonical-host repair."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from stronglens_probes.contracts import load_config
from stronglens_probes.dependencies import (
    assign_dependency_folds,
    build_dependency_components,
)
from stronglens_probes.feature_extraction import _load_encoder, pixel_statistics
from stronglens_probes.probes import (
    _component_equal_class_weights,
    _fit_axis,
    _weighted_mean_sd,
)

ROOT = Path(__file__).resolve().parents[1]


def domain_helpers() -> Any:
    path = ROOT / "scripts/analyze_canonical_repair_domain_20260930.py"
    specification = importlib.util.spec_from_file_location("domain_helpers", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load repair domain helpers")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def feature_matrix(
    representation: str,
    manifest: pd.DataFrame,
    device: str,
    helpers: Any,
) -> np.ndarray:
    old_manifest = pd.read_parquet(
        ROOT / "results_v2/gmatch/preparation/gmatch_analysis_manifest.parquet"
    )
    old_features = np.load(
        ROOT / f"results_v2/gmatch/features/{representation}_features.npy",
        allow_pickle=False,
    )
    dimension = old_features.shape[1]
    old_by_hash = {
        value: index
        for index, value in enumerate(old_manifest["pixel_sha256"].astype(str))
    }
    result = np.empty((len(manifest), dimension), dtype=np.float32)
    missing = []
    for index, value in enumerate(manifest["pixel_sha256"].astype(str)):
        old_index = old_by_hash.get(value)
        if old_index is None:
            missing.append(index)
        else:
            result[index] = old_features[old_index]
    if representation == "pixel_statistics":
        for start in range(0, len(missing), 64):
            indices = missing[start : start + 64]
            result[indices] = pixel_statistics(
                helpers.load_images(manifest.iloc[indices])
            )
        return result
    checkpoint = {
        "stein_ssl_resnet50": ROOT
        / "checkpoints/frozen_external_encoders/gmatch_v2/"
        "stein_ssl_resnet50_d42798a4.ckpt",
        "imagenet_effnet_v2_s": ROOT
        / "checkpoints/frozen_external_encoders/gmatch_v2/"
        "imagenet_effnet_v2_s_dd5fe13b.pth",
    }[representation]
    model, preprocess = _load_encoder(representation, checkpoint)
    model = model.to(device).eval()
    for start in range(0, len(missing), 64):
        indices = missing[start : start + 64]
        images = helpers.load_images(manifest.iloc[indices])
        tensor = preprocess(images).to(device)
        with torch.no_grad():
            result[indices] = (
                model(tensor).detach().cpu().numpy().astype(np.float32)
            )
    return result


def fit_axis(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fold_map: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    attached = manifest.merge(
        fold_map, on="dependency_component_id", how="left", validate="many_to_one"
    )
    population = attached["population"].to_numpy()
    fit_rows = np.isin(population, ["observed_reference", "bare"])
    fit_labels = (population == "observed_reference").astype(np.int8)
    weights = np.zeros(len(attached), dtype=float)
    weights[fit_rows] = _component_equal_class_weights(
        fit_labels[fit_rows],
        attached.loc[fit_rows, "dependency_component_id"],
        np.ones(fit_rows.sum()),
    )
    scores = np.empty(len(attached), dtype=float)
    for fold in range(5):
        evaluate = attached["fold"].to_numpy() == fold
        train = (~evaluate) & fit_rows
        scaler, estimator = _fit_axis(
            features[train], fit_labels[train], weights[train], config
        )
        raw_train = estimator.decision_function(scaler.transform(features[train]))
        bare_train = population[train] == "bare"
        active = weights[train] > 0
        mean, sd = _weighted_mean_sd(
            raw_train[bare_train & active],
            weights[train][bare_train & active],
        )
        scores[evaluate] = (
            estimator.decision_function(scaler.transform(features[evaluate])) - mean
        ) / sd
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
    movement = wide["injected"].to_numpy() - wide["bare"].to_numpy()

    def auc(positive: str, negative: str) -> float:
        labels = np.r_[np.ones(len(wide)), np.zeros(len(wide))]
        values = np.r_[wide[positive].to_numpy(), wide[negative].to_numpy()]
        return float(roc_auc_score(labels, values))

    result = {
        "matched_targets": len(wide),
        "mean_movement": float(movement.mean()),
        "median_movement": float(np.median(movement)),
        "minimum_movement": float(movement.min()),
        "maximum_movement": float(movement.max()),
        "positive_movements": int((movement > 0).sum()),
        "negative_movements": int((movement < 0).sum()),
        "observed_vs_bare_auc": auc("observed_reference", "bare"),
        "observed_vs_injected_auc": auc("observed_reference", "injected"),
        "injected_vs_bare_auc": auc("injected", "bare"),
    }
    wide["within_target_movement"] = movement
    return result, wide


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/gmatch/preparation/"
        "gmatch_analysis_manifest.parquet"
    )
    config = load_config(ROOT / "experiments_v2/probes/configs/probes.yaml")
    manifest = build_dependency_components(
        manifest,
        row_id_column="row_id",
        dependency_columns=config["probe"]["dependency_columns"],
    )
    labels = manifest["population"].eq("observed_reference").astype(np.int8)
    fold_map = assign_dependency_folds(
        manifest.assign(_label=labels),
        component_column="dependency_component_id",
        label_column="_label",
        n_splits=5,
        seed=42,
    )
    manifest.to_parquet(args.output_dir / "gmatch_analysis_manifest.parquet", index=False)
    fold_map.to_parquet(args.output_dir / "gmatch_dependency_folds.parquet", index=False)
    helpers = domain_helpers()
    results: dict[str, Any] = {}
    operational, _ = helpers.operational_features(manifest, args.device)
    for representation in (
        "stein_ssl_resnet50",
        "imagenet_effnet_v2_s",
        "pixel_statistics",
        "operational_penultimate",
    ):
        features = (
            operational
            if representation == "operational_penultimate"
            else feature_matrix(representation, manifest, args.device, helpers)
        )
        result, pairs = fit_axis(features, manifest, fold_map, config)
        results[representation] = result
        pairs.to_parquet(
            args.output_dir / f"{representation}_pair_movements.parquet", index=False
        )
    payload = {
        "schema_version": 1,
        "status": "canonical_host_repair_gmatch_complete",
        "rows": len(manifest),
        "dependency_components": int(manifest["dependency_component_id"].nunique()),
        "results": results,
    }
    (args.output_dir / "GMATCH_REPAIR_RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir / "GMATCH_REPAIR_RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
