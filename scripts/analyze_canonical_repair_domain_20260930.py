#!/usr/bin/env python3
"""Recompute the four G_domain representations for the canonical-host repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from stronglens_probes.contracts import load_config, stable_digest
from stronglens_probes.dependencies import (
    assign_dependency_folds,
    build_dependency_components,
)
from stronglens_probes.feature_extraction import (
    _load_encoder,
    _load_image,
    pixel_statistics,
)
from stronglens_probes.preparation import _float_identity
from stronglens_probes.probes import (
    _component_equal_class_weights,
    _fit_axis,
)

from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = (
    "stein_ssl_resnet50",
    "imagenet_effnet_v2_s",
    "pixel_statistics",
    "operational_penultimate",
)
FEATURE_DIMENSIONS = {
    "stein_ssl_resnet50": 2048,
    "imagenet_effnet_v2_s": 1280,
    "pixel_statistics": 18,
    "operational_penultimate": 1280,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_pixel(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_parquet(
        ROOT / "results_v2/probes/gdomain_v2/gdomain_analysis_manifest.parquet"
    )
    observed = old[old["population"].eq("observed_reference")].copy()
    authority_digest = str(observed["authority_digest"].iloc[0])
    parents = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/grec/parents/"
        "grec_parent_manifest.parquet"
    )
    selection = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/selection/"
        "final_grec_selection.parquet"
    )
    flux_z = selection["flux_z"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        zmag = np.where(flux_z > 0, 22.5 - 2.5 * np.log10(flux_z), np.nan)
    ellipticity = np.hypot(
        selection["shape_e1"].to_numpy(float),
        selection["shape_e2"].to_numpy(float),
    )
    axis_ratio = np.clip((1.0 - ellipticity) / (1.0 + ellipticity), 0.0, 1.0)
    selected_hosts = set(
        selection.loc[
            np.isfinite(zmag)
            & (zmag <= 19.5)
            & selection["sersic"].ge(4.0)
            & (axis_ratio >= 0.6),
            "galaxy_id",
        ].astype(str)
    )
    subset = parents[
        parents["host_galaxy_id"].astype(str).isin(selected_hosts)
    ].copy()
    lookup = selection.set_index(selection["galaxy_id"].astype(str))
    simulated = pd.DataFrame(
        {
            "row_id": "gdomain:simulated:" + subset["grec_row_id"].astype(str),
            "population": "simulated",
            "reference_id": "not_applicable:simulated",
            "host_galaxy_id": subset["host_galaxy_id"].astype(str),
            "host_parent_id": subset["host_parent_id"].astype(str),
            "physical_group_id": subset["host_galaxy_id"]
            .astype(str)
            .map(lookup["physical_group_id"].astype(str)),
            "geographic_spatial_key": "healpix128:"
            + subset["host_galaxy_id"]
            .astype(str)
            .map(lookup["healpix_128"].astype(int).astype(str)),
            "spatial_group_id": subset["host_spatial_group_id"].astype(str),
            "source_parameter_id": subset["source_nuisance_id"].astype(str),
            "lens_parameter_id": subset["lens_nuisance_id"].astype(str),
            "psf_id": [
                "grec_psf_"
                + stable_digest(
                    "grec_psf_v1",
                    row.host_galaxy_id,
                    _float_identity(row.psf_fwhm_g_arcsec),
                    _float_identity(row.psf_fwhm_r_arcsec),
                    _float_identity(row.psf_fwhm_z_arcsec),
                )[:40]
                for row in subset.itertuples(index=False)
            ],
            "rendering_id": subset["base_draw_key"].astype(str),
            "render_parameter_id": subset["base_draw_key"].astype(str),
            "random_draw_id": subset["base_draw_key"].astype(str),
            "base_pixel_sha256": subset["host_pixel_sha256"].astype(str),
            "pixel_sha256": subset["expectation_pixel_sha256"].astype(str),
            "pixel_file_sha256": subset["parent_artifact_sha256"].astype(str),
            "pixel_relpath": subset["parent_pixel_path"].astype(str),
            "pixel_array_key": "expectation_chw",
            "pixel_layout": "CHW",
            "pair_id": "",
            "target_galaxy_id": subset["host_galaxy_id"].astype(str),
            "authority_digest": authority_digest,
            "data_role": "final",
        }
    )
    manifest = pd.concat([observed, simulated], ignore_index=True).sort_values(
        "row_id", ignore_index=True
    )
    config = load_config(
        ROOT / "experiments_v2/probes/configs/probes.yaml"
    )
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
    return manifest, fold_map


def load_images(rows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            _load_image(
                resolve_pixel(str(row.pixel_relpath)),
                expected_file_sha256=str(row.pixel_file_sha256),
                expected_pixel_sha256=str(row.pixel_sha256),
                array_key=str(row.pixel_array_key),
                layout=str(row.pixel_layout),
            )
            for row in rows.itertuples(index=False)
        ]
    )


def external_features(
    representation: str,
    manifest: pd.DataFrame,
    device: str,
) -> np.ndarray:
    old_manifest = pd.read_parquet(
        ROOT / "results_v2/probes/gdomain_v2/gdomain_analysis_manifest.parquet"
    )
    old_path = (
        ROOT
        / "results_v2/probes/gdomain_v2"
        / f"{representation}_features.npy"
    )
    old_values = np.load(old_path, allow_pickle=False)
    by_hash = {
        value: index
        for index, value in enumerate(old_manifest["pixel_sha256"].astype(str))
    }
    output = np.empty(
        (len(manifest), FEATURE_DIMENSIONS[representation]), dtype=np.float32
    )
    missing: list[int] = []
    for index, value in enumerate(manifest["pixel_sha256"].astype(str)):
        old_index = by_hash.get(value)
        if old_index is None:
            missing.append(index)
        else:
            output[index] = old_values[old_index]
    if representation == "pixel_statistics":
        for start in range(0, len(missing), 64):
            indices = missing[start : start + 64]
            output[indices] = pixel_statistics(load_images(manifest.iloc[indices]))
        return output
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
        images = load_images(manifest.iloc[indices])
        tensor = preprocess(images).to(device)
        with torch.no_grad():
            output[indices] = (
                model(tensor).detach().cpu().numpy().astype(np.float32)
            )
    return output


def operational_features(
    manifest: pd.DataFrame, device: str
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = ROOT / "data/downloads/models/primary_seed42_epoch160.pt"
    model, metadata = load_model_and_spec(str(checkpoint), device)
    if metadata != {"mode": "raw_robust", "crop": False, "clip_range": 10.0}:
        raise ValueError("operational preprocessing metadata differs")
    model = model.eval()
    features = np.empty((len(manifest), 1280), dtype=np.float32)
    logits = np.empty(len(manifest), dtype=np.float32)
    for start in range(0, len(manifest), 64):
        stop = min(start + 64, len(manifest))
        images_hwc = load_images(manifest.iloc[start:stop])
        images_chw = np.transpose(images_hwc, (0, 3, 1, 2))
        prepared = np.stack(
            [
                preprocess_stack(
                    image,
                    mode="raw_robust",
                    crop=False,
                    crop_size=0,
                    clip_range=10.0,
                    annulus_r_in=20.0,
                    annulus_r_out=32.0,
                )
                for image in images_chw
            ]
        )
        tensor = torch.from_numpy(prepared).to(device)
        present = len(tensor)
        if present < 64:
            tensor = torch.cat(
                [tensor, tensor[-1:].expand(64 - present, -1, -1, -1)]
            )
        with torch.no_grad():
            encoded = torch.flatten(model.avgpool(model.features(tensor)), 1)
            intact = model(tensor).reshape(-1)
        features[start:stop] = encoded[:present].cpu().numpy().astype(np.float32)
        logits[start:stop] = intact[:present].cpu().numpy().astype(np.float32)
    return features, logits


def fit_probe(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fold_map: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    attached = manifest.merge(
        fold_map, on="dependency_component_id", how="left", validate="many_to_one"
    )
    labels = attached["population"].eq("observed_reference").astype(np.int8).to_numpy()
    weights = _component_equal_class_weights(
        labels, attached["dependency_component_id"], np.ones(len(attached))
    )
    scores = np.empty(len(attached), dtype=float)
    folds = []
    for fold in range(5):
        evaluate = attached["fold"].to_numpy() == fold
        scaler, estimator = _fit_axis(
            features[~evaluate], labels[~evaluate], weights[~evaluate], config
        )
        scores[evaluate] = estimator.decision_function(
            scaler.transform(features[evaluate])
        )
        folds.append(
            {
                "fold": fold,
                "rows": int(evaluate.sum()),
                "observed": int(labels[evaluate].sum()),
                "auc": float(
                    roc_auc_score(
                        labels[evaluate],
                        scores[evaluate],
                        sample_weight=weights[evaluate],
                    )
                ),
            }
        )
    result = {
        "component_equal_oof_auc": float(
            roc_auc_score(labels, scores, sample_weight=weights)
        ),
        "row_weighted_oof_auc": float(roc_auc_score(labels, scores)),
        "folds": folds,
    }
    rows = attached[["row_id", "population", "dependency_component_id", "fold"]]
    rows = rows.assign(score=scores, weight=weights)
    return result, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, fold_map = build_manifest()
    manifest.to_parquet(args.output_dir / "gdomain_analysis_manifest.parquet", index=False)
    fold_map.to_parquet(args.output_dir / "gdomain_dependency_folds.parquet", index=False)
    config = load_config(ROOT / "experiments_v2/probes/configs/probes.yaml")
    results: dict[str, Any] = {}
    operational, logits = operational_features(manifest, args.device)
    for representation in REPRESENTATIONS:
        if representation == "operational_penultimate":
            features = operational
        else:
            features = external_features(representation, manifest, args.device)
        np.save(args.output_dir / f"{representation}_features.npy", features)
        result, rows = fit_probe(features, manifest, fold_map, config)
        results[representation] = result
        rows.to_parquet(args.output_dir / f"{representation}_oof.parquet", index=False)
    labels = manifest["population"].eq("observed_reference").astype(np.int8).to_numpy()
    weights = _component_equal_class_weights(
        labels, manifest["dependency_component_id"], np.ones(len(manifest))
    )
    results["fixed_operational_head"] = {
        "component_equal_auc": float(
            roc_auc_score(labels, logits, sample_weight=weights)
        ),
        "row_weighted_auc": float(roc_auc_score(labels, logits)),
    }
    payload = {
        "schema_version": 1,
        "status": "canonical_host_repair_domain_complete",
        "rows": len(manifest),
        "observed_rows": int(labels.sum()),
        "simulated_rows": int((labels == 0).sum()),
        "simulated_hosts": int(
            manifest.loc[
                manifest["population"].eq("simulated"), "host_galaxy_id"
            ].nunique()
        ),
        "dependency_components": int(
            manifest["dependency_component_id"].nunique()
        ),
        "results": results,
        "manifest_sha256": sha256_file(
            args.output_dir / "gdomain_analysis_manifest.parquet"
        ),
    }
    (args.output_dir / "GDOMAIN_REPAIR_RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir / "GDOMAIN_REPAIR_RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
