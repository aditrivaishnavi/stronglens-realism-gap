#!/usr/bin/env python3
"""Execute bounded existing-score, annulus, and brightness controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import torch

from dhs.fixed_width_scoring import (
    array_batches,
    assert_no_batch_coupling,
    fixed_width_logits,
)
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.utils import radial_mask, robust_median_mad


class ControlError(RuntimeError):
    """A frozen control-contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def seed_from_key(namespace: str, *parts: object) -> int:
    material = canonical_json_bytes([namespace, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def stable_id(prefix: str, value: Any, length: int = 40) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"{prefix}_{digest[:length]}"


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def root_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_payload(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ControlError(f"unsafe payload path {relative!r}")
    path = root.joinpath(*value.parts)
    if not path.is_file():
        raise ControlError(f"missing payload {relative}")
    return path


def outputs(root: Path) -> dict[str, Path]:
    base = root / "results_v2/canonical_test_host_repair_v1/revision_controls"
    return {
        "base": base,
        "seal": base / "PROTOCOL_SEAL.json",
        "threshold": base / "THRESHOLD_INDEPENDENT_VERIFICATION.json",
        "cell_table": base / "grec_cell_radius_response.parquet",
        "cell_csv": base / "grec_cell_radius_response.csv",
        "cell_figure": base / "grec_cell_radius_response.pdf",
        "cell_receipt": base / "GREC_CELL_RADIUS_RECEIPT.json",
        "annulus_rows": base / "annulus_scores.parquet",
        "annulus_stats": base / "annulus_normalization_stats.parquet",
        "annulus_pairs": base / "annulus_triplet_changes.parquet",
        "annulus_result": base / "ANNULUS_SENSITIVITY_RESULT.json",
        "bright_design": base / "brightness_design.parquet",
        "bright_images": base / "brightness_derived_images.parquet",
        "bright_scores": base / "brightness_scores.parquet",
        "bright_cells": base / "brightness_cell_summary.parquet",
        "bright_targets": base / "brightness_target_summary.parquet",
        "bright_result": base / "BRIGHTNESS_STRESS_RESULT.json",
        "bright_figure": base / "brightness_response.pdf",
    }


def load_config(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "experiments_v2/revision_controls_canonical_repair_v1/config/protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "frozen_before_new_control_outcomes"
    ):
        raise ControlError("revision-controls protocol is not frozen")
    return payload, path


def require_file(path: Path, expected_sha256: str, name: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ControlError(f"{name} is missing or has the wrong hash: {path}")


def validate_authorities(root: Path, config: dict[str, Any]) -> None:
    require_file(
        root_path(root, config["checkpoint"]["path"]),
        config["checkpoint"]["sha256"],
        "checkpoint",
    )
    require_file(
        root_path(root, config["threshold"]["calibration_logits_path"]),
        config["threshold"]["calibration_logits_sha256"],
        "calibration logits",
    )
    require_file(
        root_path(root, config["threshold"]["receipt_path"]),
        config["threshold"]["receipt_sha256"],
        "threshold receipt",
    )
    require_file(
        root_path(root, config["annulus"]["manifest_path"]),
        config["annulus"]["manifest_sha256"],
        "G_match manifest",
    )


def seal(root: Path) -> dict[str, Any]:
    config, config_path = load_config(root)
    validate_authorities(root, config)
    runner = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "status": "revision_controls_protocol_sealed_pre_outcome",
        "protocol_path": str(config_path.relative_to(root)),
        "protocol_sha256": sha256_file(config_path),
        "runner_path": str(runner.relative_to(root)),
        "runner_sha256": sha256_file(runner),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "new_control_outcomes_opened": False,
    }
    atomic_json(payload, outputs(root)["seal"])
    return payload


def validate_seal(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = outputs(root)["seal"]
    if not path.is_file():
        raise ControlError("control protocol must be sealed before execution")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config_path = root / "experiments_v2/revision_controls_canonical_repair_v1/config/protocol.json"
    if (
        payload.get("status") != "revision_controls_protocol_sealed_pre_outcome"
        or payload.get("protocol_sha256") != sha256_file(config_path)
        or payload.get("runner_sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ControlError("control protocol seal differs")
    validate_authorities(root, config)
    return payload


def verify_threshold(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    seal_payload = validate_seal(root, config)
    score_path = root_path(root, config["threshold"]["calibration_logits_path"])
    scores = pd.read_parquet(score_path)
    required = {
        "calibration_id",
        "raw_logit_float32",
        "raw_logit_float32_bits",
    }
    if set(scores.columns) != required or len(scores) != 20002:
        raise ControlError("calibration-score schema or support differs")
    values = scores["raw_logit_float32"].to_numpy(dtype=np.float32)
    if not np.array_equal(
        values.view(np.uint32),
        scores["raw_logit_float32_bits"].to_numpy(dtype=np.uint32),
    ):
        raise ControlError("stored calibration float32 bits differ")
    ordered = np.sort(values)[::-1]
    k = int(math.floor(float(config["threshold"]["alpha"]) * len(values)))
    s_k = ordered[k - 1]
    s_k_plus_1 = ordered[k]
    threshold = np.nextafter(
        s_k_plus_1, np.float32(np.inf), dtype=np.float32
    )
    admitted = int(np.sum(values >= threshold))
    expected = config["threshold"]
    if (
        k != 20
        or int(threshold.view(np.uint32)) != int(expected["float32_bits"])
        or float(threshold) != float(expected["float32_value"])
        or admitted != 20
    ):
        raise ControlError("direct threshold recomputation differs")
    payload = {
        "schema_version": 1,
        "status": "threshold_independently_recomputed_from_stored_logits",
        "protocol_seal_sha256": sha256_file(outputs(root)["seal"]),
        "protocol_seal_status": seal_payload["status"],
        "calibration_logits_sha256": sha256_file(score_path),
        "rows": len(values),
        "alpha": float(expected["alpha"]),
        "k_floor_alpha_n": k,
        "s_k": float(s_k),
        "s_k_float32_bits": int(s_k.view(np.uint32)),
        "s_k_plus_1": float(s_k_plus_1),
        "s_k_plus_1_float32_bits": int(s_k_plus_1.view(np.uint32)),
        "threshold": float(threshold),
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "comparison": ">=",
        "negatives_at_or_above": admitted,
    }
    atomic_json(payload, outputs(root)["threshold"])
    return payload


def cell_map(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    validate_seal(root, config)
    scene_path = root / "results_v2/canonical_test_host_repair_v1/grec/scenes/grec_scene_table.parquet"
    score_path = root / "results_v2/canonical_test_host_repair_v1/grec/scores/grec_scores.parquet"
    analysis_path = root / "results_v2/grec/GREC_ANALYSIS.json"
    scenes = pd.read_parquet(scene_path)
    scores = pd.read_parquet(score_path)
    joined = scenes.merge(scores, on="grec_row_id", validate="one_to_one")
    if len(joined) != 49500:
        raise ControlError("G_rec score join support differs")
    table = (
        joined.groupby(
            ["psf_bin", "depth_bin", "theta_e_arcsec"], as_index=False
        )
        .agg(
            rows=("grec_row_id", "size"),
            hosts=("host_galaxy_id", "nunique"),
            detections=("detected_at_frozen_threshold", "sum"),
            recovery=("detected_at_frozen_threshold", "mean"),
        )
        .sort_values(["psf_bin", "depth_bin", "theta_e_arcsec"])
        .reset_index(drop=True)
    )
    if (
        len(table) != 99
        or not table["rows"].eq(500).all()
        or not table["hosts"].eq(500).all()
    ):
        raise ControlError("cell/radius support differs")
    authority = json.loads(analysis_path.read_text(encoding="utf-8"))["grec"][
        "cell_radius_records"
    ]
    authority_frame = (
        pd.DataFrame(authority)
        .sort_values(["psf_bin", "depth_bin", "theta_e_arcsec"])
        .reset_index(drop=True)
    )
    if not np.allclose(
        table["recovery"], authority_frame["recovery"], rtol=0, atol=0
    ):
        raise ControlError("cell/radius recovery differs from final authority")
    atomic_parquet(table, outputs(root)["cell_table"])
    outputs(root)["cell_csv"].parent.mkdir(parents=True, exist_ok=True)
    if outputs(root)["cell_csv"].exists():
        raise FileExistsError(f"refusing to overwrite {outputs(root)['cell_csv']}")
    table.to_csv(outputs(root)["cell_csv"], index=False)
    _plot_cell_map(table, outputs(root)["cell_figure"])
    occupied = (
        table[["psf_bin", "depth_bin"]]
        .drop_duplicates()
        .sort_values(["psf_bin", "depth_bin"])
    )
    all_cells = pd.MultiIndex.from_product(
        [
            sorted(table["psf_bin"].unique()),
            sorted(table["depth_bin"].unique()),
        ],
        names=["psf_bin", "depth_bin"],
    )
    missing = all_cells.difference(
        pd.MultiIndex.from_frame(occupied[["psf_bin", "depth_bin"]])
    )
    payload = {
        "schema_version": 1,
        "status": "grec_cell_radius_response_reproduced_from_final_scores",
        "rows_joined": len(joined),
        "cell_radius_records": len(table),
        "rows_per_cell_radius": sorted(table["rows"].unique().tolist()),
        "occupied_cells": len(occupied),
        "possible_psf_depth_combinations": len(all_cells),
        "empty_cells": [
            {"psf_bin": float(psf), "depth_bin": float(depth)}
            for psf, depth in missing
        ],
        "overall_recovery": float(
            joined["detected_at_frozen_threshold"].mean()
        ),
        "table_sha256": sha256_file(outputs(root)["cell_table"]),
        "csv_sha256": sha256_file(outputs(root)["cell_csv"]),
        "figure_sha256": sha256_file(outputs(root)["cell_figure"]),
        "scene_table_sha256": sha256_file(scene_path),
        "scores_sha256": sha256_file(score_path),
        "analysis_authority_sha256": sha256_file(analysis_path),
        "interpretation": "conditional response on occupied design support, not completeness",
    }
    atomic_json(payload, outputs(root)["cell_receipt"])
    return payload


def _plot_cell_map(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = (
        table[["psf_bin", "depth_bin"]]
        .drop_duplicates()
        .sort_values(["psf_bin", "depth_bin"])
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.0), sharex=True, sharey=True)
    for axis, row in zip(axes.flat, cells.itertuples(index=False), strict=True):
        frame = table[
            table["psf_bin"].eq(row.psf_bin)
            & table["depth_bin"].eq(row.depth_bin)
        ]
        axis.plot(frame["theta_e_arcsec"], 100 * frame["recovery"], marker="o")
        axis.set_title(f"PSF {row.psf_bin:.2f}, depth {row.depth_bin:.1f}")
        axis.grid(alpha=0.25)
    for index, axis in enumerate(axes.flat):
        if index // 3 == 2:
            axis.set_xlabel(r"$\theta_{\rm E}$ (arcsec)")
        if index % 3 == 0:
            axis.set_ylabel("Recovery (%)")
    fig.suptitle("Conditional response in the nine occupied PSF–depth cells")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_chw(root: Path, row: Any) -> np.ndarray:
    path = safe_payload(root, str(row.pixel_relpath))
    if sha256_file(path) != str(row.pixel_file_sha256):
        raise ControlError(f"payload file hash differs: {row.pixel_relpath}")
    with np.load(path, allow_pickle=False) as payload:
        key = str(row.pixel_array_key)
        if key not in payload:
            raise ControlError(f"{row.pixel_relpath} lacks {key}")
        native = np.asarray(payload[key])
    if array_sha256(native) != str(row.pixel_sha256):
        raise ControlError(f"payload array hash differs: {row.pixel_relpath}")
    if str(row.pixel_layout) == "HWC":
        image = np.transpose(native, (2, 0, 1))
    elif str(row.pixel_layout) == "CHW":
        image = native
    else:
        raise ControlError(f"unsupported layout {row.pixel_layout}")
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (3, 101, 101) or not np.isfinite(image).all():
        raise ControlError(f"invalid image {row.pixel_relpath}")
    return image


def _load_model(root: Path, config: dict[str, Any]) -> tuple[Any, str, dict]:
    device = str(config["inference"]["authoritative_device"])
    if device == "mps" and not torch.backends.mps.is_available():
        raise ControlError("authoritative MPS device is unavailable")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model, metadata = load_model_and_spec(
        str(root_path(root, config["checkpoint"]["path"])), device
    )
    if metadata != {"mode": "raw_robust", "crop": False, "clip_range": 10.0}:
        raise ControlError(f"checkpoint preprocessing differs: {metadata}")
    contract = assert_no_batch_coupling(model)
    return model, device, contract


def _score_images(
    model: Any,
    device: str,
    images: Iterator[np.ndarray],
    *,
    clip_range: float,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    processed = (
        preprocess_stack(
            image,
            mode="raw_robust",
            crop=False,
            crop_size=0,
            clip_range=clip_range,
            annulus_r_in=20.0,
            annulus_r_out=32.0,
        )
        for image in images
    )
    values, record = fixed_width_logits(
        model,
        array_batches(processed, batch_size),
        batch_size=batch_size,
        device=device,
        filler="last",
    )
    result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ControlError("scoring produced non-finite logits")
    return result, record


def _annulus_stats(
    image: np.ndarray, r_in: float, r_out: float, clip: float
) -> list[dict[str, float]]:
    records = []
    mask = radial_mask(101, 101, r_in, r_out)
    for band in range(3):
        median, mad = robust_median_mad(image[band][mask])
        normalized = (image[band] - median) / mad
        records.append(
            {
                "band_index": band,
                "median": median,
                "mad": mad,
                "floor_used": bool(mad <= 1.0000001e-8),
                "clipped_fraction": float(np.mean(np.abs(normalized) > clip)),
            }
        )
    return records


def annulus(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    validate_seal(root, config)
    manifest = pd.read_parquet(root_path(root, config["annulus"]["manifest_path"]))
    if len(manifest) != 129:
        raise ControlError("annulus manifest support differs")
    raw_images = [_load_chw(root, row) for row in manifest.itertuples(index=False)]
    model, device, batch_contract = _load_model(root, config)
    mappings = {
        "training_20_32": (20.0, 32.0),
        "alternative_32p5_45": (32.5, 45.0),
    }
    score_frames = []
    statistic_records = []
    inference_records = {}
    threshold = np.asarray(
        config["threshold"]["float32_value"], dtype=np.float32
    )
    for name, (r_in, r_out) in mappings.items():
        processed = [
            preprocess_stack(
                image,
                mode="raw_robust",
                crop=False,
                crop_size=0,
                clip_range=10.0,
                annulus_r_in=r_in,
                annulus_r_out=r_out,
            )
            for image in raw_images
        ]
        values, inference = fixed_width_logits(
            model,
            array_batches(processed, 64),
            batch_size=64,
            device=device,
            filler="last",
        )
        logits = values.astype(np.float32)
        frame = manifest[
            [
                "row_id",
                "pair_id",
                "population",
                "reference_id",
                "target_galaxy_id",
                "pixel_sha256",
            ]
        ].copy()
        frame["mapping"] = name
        frame["annulus_r_in"] = r_in
        frame["annulus_r_out"] = r_out
        frame["clip_range"] = 10.0
        frame["raw_logit_float32"] = logits
        frame["raw_logit_float32_bits"] = logits.view(np.uint32)
        frame["detected_at_fixed_threshold"] = logits >= threshold
        score_frames.append(frame)
        inference_records[name] = inference
        for row_index, image in enumerate(raw_images):
            for item in _annulus_stats(image, r_in, r_out, 10.0):
                statistic_records.append(
                    {
                        "row_id": str(manifest.iloc[row_index]["row_id"]),
                        "pair_id": str(manifest.iloc[row_index]["pair_id"]),
                        "population": str(manifest.iloc[row_index]["population"]),
                        "mapping": name,
                        **item,
                    }
                )
    scores = pd.concat(score_frames, ignore_index=True)
    statistics = pd.DataFrame(statistic_records)
    wide = scores.pivot(
        index=["row_id", "pair_id", "population"],
        columns="mapping",
        values=["raw_logit_float32", "detected_at_fixed_threshold"],
    )
    wide.columns = [f"{first}_{second}" for first, second in wide.columns]
    wide = wide.reset_index()
    wide["alternative_minus_training_logit"] = (
        wide["raw_logit_float32_alternative_32p5_45"]
        - wide["raw_logit_float32_training_20_32"]
    )
    wide["decision_changed"] = (
        wide["detected_at_fixed_threshold_alternative_32p5_45"]
        != wide["detected_at_fixed_threshold_training_20_32"]
    )
    pair_changes = wide.pivot(
        index="pair_id",
        columns="population",
        values="alternative_minus_training_logit",
    ).reset_index()
    pair_changes.columns.name = None
    atomic_parquet(scores, outputs(root)["annulus_rows"])
    atomic_parquet(statistics, outputs(root)["annulus_stats"])
    atomic_parquet(pair_changes, outputs(root)["annulus_pairs"])
    population_summary = {}
    for population, frame in wide.groupby("population"):
        population_summary[str(population)] = {
            "rows": len(frame),
            "mean_logit_change": float(
                frame["alternative_minus_training_logit"].mean()
            ),
            "median_logit_change": float(
                frame["alternative_minus_training_logit"].median()
            ),
            "minimum_logit_change": float(
                frame["alternative_minus_training_logit"].min()
            ),
            "maximum_logit_change": float(
                frame["alternative_minus_training_logit"].max()
            ),
            "decision_changes": int(frame["decision_changed"].sum()),
            "training_detections": int(
                frame["detected_at_fixed_threshold_training_20_32"].sum()
            ),
            "alternative_detections": int(
                frame[
                    "detected_at_fixed_threshold_alternative_32p5_45"
                ].sum()
            ),
        }
    _verify_annulus_training_bits(root, manifest, wide)
    payload = {
        "schema_version": 1,
        "status": "bounded_annulus_sensitivity_complete",
        "rows": len(manifest),
        "triplets": int(manifest["pair_id"].nunique()),
        "mappings": mappings,
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "threshold_recalibrated": False,
        "population_summary": population_summary,
        "inference_records": inference_records,
        "batch_contract": batch_contract,
        "scores_sha256": sha256_file(outputs(root)["annulus_rows"]),
        "normalization_stats_sha256": sha256_file(outputs(root)["annulus_stats"]),
        "triplet_changes_sha256": sha256_file(outputs(root)["annulus_pairs"]),
        "interpretation": (
            "fixed-cohort preprocessing sensitivity; not an optimal annulus or "
            "equal-FPR comparison"
        ),
    }
    atomic_json(payload, outputs(root)["annulus_result"])
    return payload


def _verify_annulus_training_bits(
    root: Path, manifest: pd.DataFrame, wide: pd.DataFrame
) -> None:
    authority = np.full(len(manifest), np.uint32(0), dtype=np.uint32)
    present = np.zeros(len(manifest), dtype=bool)
    endpoint = pd.read_parquet(
        root
        / "results_v2/evaluation_v2/terminal_confirmed_scores/endpoint_scores.parquet"
    )
    grec = pd.read_parquet(root / "results_v2/canonical_test_host_repair_v1/grec/scores/grec_scores.parquet")
    for index, row in manifest.reset_index(drop=True).iterrows():
        if row["population"] == "observed_reference":
            match = endpoint[endpoint["galaxy_id"].eq(row["reference_id"])]
            authority[index] = np.uint32(match.iloc[0]["raw_logit_float32_bits"])
            present[index] = True
        elif row["population"] == "injected":
            match = grec[grec["grec_row_id"].eq(row["grec_row_id"])]
            authority[index] = np.uint32(match.iloc[0]["raw_logit_float32_bits"])
            present[index] = True
    new_by_row = wide.set_index("row_id")[
        "raw_logit_float32_training_20_32"
    ]
    ordered = new_by_row.loc[manifest["row_id"]].to_numpy(dtype=np.float32)
    if not np.array_equal(ordered[present].view(np.uint32), authority[present]):
        raise ControlError("training-annulus logits differ from score authorities")


def _matched_source_rows(
    root: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest = pd.read_parquet(root_path(root, config["brightness"]["manifest_path"]))
    scenes = pd.read_parquet(
        root / "results_v2/canonical_test_host_repair_v1/grec/scenes/grec_scene_table.parquet"
    )[["grec_row_id", "source_r_mag"]]
    records = []
    for pair_id, frame in manifest.groupby("pair_id", sort=True):
        if set(frame["population"]) != {"observed_reference", "bare", "injected"}:
            raise ControlError(f"{pair_id} is not a complete triplet")
        bare_row = frame[frame["population"].eq("bare")].iloc[0]
        injected_row = frame[frame["population"].eq("injected")].iloc[0]
        bare = _load_chw(root, bare_row)
        injected_path = safe_payload(root, str(injected_row["pixel_relpath"]))
        with np.load(injected_path, allow_pickle=False) as payload:
            expectation = np.asarray(payload["expectation_chw"], dtype=np.float32)
            source = np.asarray(
                payload["injection_expectation_chw"], dtype=np.float32
            )
        if (
            source.shape != (3, 101, 101)
            or (source < 0).any()
            or not np.array_equal(
                (bare + source).astype(np.float32), expectation
            )
        ):
            raise ControlError(f"{pair_id} fails isolated-source reconstruction")
        scene = scenes[scenes["grec_row_id"].eq(injected_row["grec_row_id"])]
        if len(scene) != 1:
            raise ControlError(f"{pair_id} scene brightness join differs")
        records.append(
            {
                "pair_id": str(pair_id),
                "reference_id": str(injected_row["reference_id"]),
                "target_galaxy_id": str(injected_row["target_galaxy_id"]),
                "grec_row_id": str(injected_row["grec_row_id"]),
                "bare_row": bare_row,
                "injected_row": injected_row,
                "bare": bare,
                "source": source,
                "expectation": expectation,
                "original_source_r_mag": float(scene.iloc[0]["source_r_mag"]),
                "bare_sha256": array_sha256(bare),
                "source_sha256": array_sha256(source),
                "expectation_sha256": array_sha256(expectation),
            }
        )
    if len(records) != 43:
        raise ControlError("brightness parent support differs")
    return records


def _draw_source(
    source: np.ndarray,
    *,
    gain: float,
    namespace: str,
    target: str,
    magnitude: float,
    draw_index: int,
) -> tuple[np.ndarray, str, int]:
    seed = seed_from_key(namespace, target, f"{magnitude:.1f}", draw_index)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    electrons = rng.poisson(np.asarray(source, dtype=np.float64) * gain)
    physical = (electrons / gain).astype(np.float32)
    draw_key = stable_id(
        "photon", [namespace, target, f"{magnitude:.1f}", draw_index]
    )
    return physical, draw_key, seed


def _make_brightness_design(
    parents: list[dict[str, Any]], config: dict[str, Any]
) -> pd.DataFrame:
    records = []
    for parent in parents:
        for magnitude in config["brightness"]["magnitudes"]:
            scale = 10.0 ** (
                -0.4 * (float(magnitude) - parent["original_source_r_mag"])
            )
            records.append(
                {
                    "cell_id": stable_id(
                        "brightness_cell",
                        [parent["pair_id"], f"{float(magnitude):.1f}"],
                    ),
                    "pair_id": parent["pair_id"],
                    "reference_id": parent["reference_id"],
                    "target_galaxy_id": parent["target_galaxy_id"],
                    "grec_row_id": parent["grec_row_id"],
                    "original_source_r_mag": parent["original_source_r_mag"],
                    "target_source_r_mag": float(magnitude),
                    "source_scale": scale,
                    "bare_sha256": parent["bare_sha256"],
                    "source_parent_sha256": parent["source_sha256"],
                    "expectation_parent_sha256": parent["expectation_sha256"],
                }
            )
    design = pd.DataFrame(records)
    if len(design) != 344 or design["cell_id"].duplicated().any():
        raise ControlError("brightness design support differs")
    return design


def _iter_derived(
    parents_by_pair: dict[str, dict[str, Any]],
    design: pd.DataFrame,
    config: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    gain = float(config["brightness"]["gain_electron_per_nmgy"])
    namespace = str(config["brightness"]["stream_namespace"])
    for cell in design.itertuples(index=False):
        parent = parents_by_pair[str(cell.pair_id)]
        scaled = (parent["source"] * np.float32(cell.source_scale)).astype(
            np.float32
        )
        expectation = (parent["bare"] + scaled).astype(np.float32)
        yield (
            {
                "derived_image_id": stable_id(
                    "bright_image", [cell.cell_id, "expectation"]
                ),
                "cell_id": str(cell.cell_id),
                "pair_id": str(cell.pair_id),
                "reference_id": str(cell.reference_id),
                "target_galaxy_id": str(cell.target_galaxy_id),
                "target_source_r_mag": float(cell.target_source_r_mag),
                "arm": "expectation",
                "draw_index": -1,
                "draw_key": "not_applicable_expectation",
                "draw_seed": -1,
                "source_image_sha256": array_sha256(scaled),
                "derived_image_sha256": array_sha256(expectation),
                "source_flux_sum": float(
                    scaled.sum(dtype=np.float64)
                ),
                "source_peak": float(scaled.max()),
            },
            expectation,
        )
        for draw_index in range(int(config["brightness"]["photon_draws"])):
            photon, draw_key, seed = _draw_source(
                scaled,
                gain=gain,
                namespace=namespace,
                target=str(cell.target_galaxy_id),
                magnitude=float(cell.target_source_r_mag),
                draw_index=draw_index,
            )
            image = (parent["bare"] + photon).astype(np.float32)
            yield (
                {
                    "derived_image_id": stable_id(
                        "bright_image",
                        [cell.cell_id, "photon", draw_index],
                    ),
                    "cell_id": str(cell.cell_id),
                    "pair_id": str(cell.pair_id),
                    "reference_id": str(cell.reference_id),
                    "target_galaxy_id": str(cell.target_galaxy_id),
                    "target_source_r_mag": float(cell.target_source_r_mag),
                    "arm": "photon",
                    "draw_index": draw_index,
                    "draw_key": draw_key,
                    "draw_seed": seed,
                    "source_image_sha256": array_sha256(photon),
                    "derived_image_sha256": array_sha256(image),
                    "source_flux_sum": float(
                        photon.sum(dtype=np.float64)
                    ),
                    "source_peak": float(photon.max()),
                },
                image,
            )


def brightness(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    validate_seal(root, config)
    parents = _matched_source_rows(root, config)
    parents_by_pair = {item["pair_id"]: item for item in parents}
    design = _make_brightness_design(parents, config)
    derived = list(_iter_derived(parents_by_pair, design, config))
    image_manifest = pd.DataFrame([record for record, _ in derived])
    if (
        len(image_manifest) != 3096
        or image_manifest["derived_image_id"].duplicated().any()
        or image_manifest["derived_image_sha256"].duplicated().any()
    ):
        raise ControlError("derived brightness-image support differs")
    atomic_parquet(design, outputs(root)["bright_design"])
    atomic_parquet(image_manifest, outputs(root)["bright_images"])

    model, device, batch_contract = _load_model(root, config)
    threshold = np.asarray(
        config["threshold"]["float32_value"], dtype=np.float32
    )
    score_frames = []
    inference_records = {}
    for clip in config["brightness"]["clip_ranges"]:
        logits, record = _score_images(
            model,
            device,
            (image for _, image in derived),
            clip_range=float(clip),
            batch_size=int(config["inference"]["batch_size"]),
        )
        frame = image_manifest.copy()
        frame["clip_range"] = float(clip)
        frame["raw_logit_float32"] = logits
        frame["raw_logit_float32_bits"] = logits.view(np.uint32)
        frame["detected_at_fixed_threshold"] = logits >= threshold
        score_frames.append(frame)
        inference_records[f"clip_{float(clip):g}"] = record
    scores = pd.concat(score_frames, ignore_index=True)
    if len(scores) != 6192:
        raise ControlError("brightness score support differs")
    atomic_parquet(scores, outputs(root)["bright_scores"])
    cell_summary, target_summary, result = _analyze_brightness(
        scores, config, threshold
    )
    atomic_parquet(cell_summary, outputs(root)["bright_cells"])
    atomic_parquet(target_summary, outputs(root)["bright_targets"])
    _plot_brightness(cell_summary, outputs(root)["bright_figure"])
    result.update(
        {
            "schema_version": 1,
            "status": "bounded_brightness_photon_clipping_stress_complete",
            "protocol_seal_sha256": sha256_file(outputs(root)["seal"]),
            "design_sha256": sha256_file(outputs(root)["bright_design"]),
            "derived_manifest_sha256": sha256_file(
                outputs(root)["bright_images"]
            ),
            "scores_sha256": sha256_file(outputs(root)["bright_scores"]),
            "cell_summary_sha256": sha256_file(outputs(root)["bright_cells"]),
            "target_summary_sha256": sha256_file(
                outputs(root)["bright_targets"]
            ),
            "figure_sha256": sha256_file(outputs(root)["bright_figure"]),
            "batch_contract": batch_contract,
            "inference_records": inference_records,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "device": device,
            },
        }
    )
    atomic_json(result, outputs(root)["bright_result"])
    return result


def _analyze_brightness(
    scores: pd.DataFrame,
    config: dict[str, Any],
    threshold: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_columns = [
        "pair_id",
        "target_galaxy_id",
        "target_source_r_mag",
        "clip_range",
        "arm",
    ]
    target = (
        scores.groupby(group_columns, as_index=False)
        .agg(
            mean_logit=("raw_logit_float32", "mean"),
            detection_rate=("detected_at_fixed_threshold", "mean"),
            draws=("derived_image_id", "size"),
            mean_source_flux=("source_flux_sum", "mean"),
            mean_source_peak=("source_peak", "mean"),
        )
        .sort_values(group_columns)
    )
    cells = (
        target.groupby(
            ["target_source_r_mag", "clip_range", "arm"], as_index=False
        )
        .agg(
            targets=("target_galaxy_id", "nunique"),
            mean_logit=("mean_logit", "mean"),
            mean_detection_rate=("detection_rate", "mean"),
            target_sd_detection_rate=("detection_rate", "std"),
            mean_source_flux=("mean_source_flux", "mean"),
            mean_source_peak=("mean_source_peak", "mean"),
        )
        .sort_values(["target_source_r_mag", "clip_range", "arm"])
    )
    expectation = target[target["arm"].eq("expectation")].rename(
        columns={
            "mean_logit": "expectation_logit",
            "detection_rate": "expectation_detection_rate",
        }
    )
    photon = target[target["arm"].eq("photon")].rename(
        columns={
            "mean_logit": "photon_mean_logit",
            "detection_rate": "photon_detection_rate",
        }
    )
    keys = ["pair_id", "target_galaxy_id", "target_source_r_mag", "clip_range"]
    paired = expectation[
        keys + ["expectation_logit", "expectation_detection_rate"]
    ].merge(
        photon[keys + ["photon_mean_logit", "photon_detection_rate"]],
        on=keys,
        validate="one_to_one",
    )
    paired["photon_minus_expectation_logit"] = (
        paired["photon_mean_logit"] - paired["expectation_logit"]
    )
    paired["photon_minus_expectation_detection_rate"] = (
        paired["photon_detection_rate"]
        - paired["expectation_detection_rate"]
    )
    contrasts = (
        paired.groupby(["target_source_r_mag", "clip_range"], as_index=False)
        .agg(
            targets=("target_galaxy_id", "nunique"),
            photon_minus_expectation_logit=(
                "photon_minus_expectation_logit",
                "mean",
            ),
            photon_minus_expectation_detection_rate=(
                "photon_minus_expectation_detection_rate",
                "mean",
            ),
        )
        .sort_values(["target_source_r_mag", "clip_range"])
    )
    # Finite-photon MCSE: draw-specific cohort means, then SD/sqrt(8).
    photon_rows = scores[scores["arm"].eq("photon")].copy()
    draw_means = (
        photon_rows.groupby(
            ["target_source_r_mag", "clip_range", "draw_index"], as_index=False
        )["detected_at_fixed_threshold"]
        .mean()
        .rename(columns={"detected_at_fixed_threshold": "cohort_rate"})
    )
    mcse = (
        draw_means.groupby(["target_source_r_mag", "clip_range"], as_index=False)
        .agg(draw_sd=("cohort_rate", "std"), draws=("draw_index", "nunique"))
    )
    mcse["finite_photon_mcse"] = mcse["draw_sd"] / np.sqrt(mcse["draws"])
    result = {
        "targets": int(scores["target_galaxy_id"].nunique()),
        "magnitudes": sorted(scores["target_source_r_mag"].unique().tolist()),
        "unique_derived_images": int(scores["derived_image_id"].nunique()),
        "score_rows": len(scores),
        "expectation_images": int(
            scores[scores["arm"].eq("expectation")][
                "derived_image_id"
            ].nunique()
        ),
        "photon_images": int(
            scores[scores["arm"].eq("photon")]["derived_image_id"].nunique()
        ),
        "photon_draws_per_target_magnitude": int(
            config["brightness"]["photon_draws"]
        ),
        "gain_electron_per_nmgy": float(
            config["brightness"]["gain_electron_per_nmgy"]
        ),
        "clip_ranges": config["brightness"]["clip_ranges"],
        "threshold": float(threshold),
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "threshold_recalibrated": False,
        "photon_minus_expectation_contrasts": contrasts.to_dict("records"),
        "finite_photon_mcse": mcse.to_dict("records"),
        "interpretation": (
            "fixed-target brightness/photon/clipping response; not calibrated "
            "visibility, equal-FPR clipping, or completeness"
        ),
    }
    return cells, target, result


def _plot_brightness(cells: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for clip in sorted(cells["clip_range"].unique()):
        for arm, linestyle in (("expectation", "-"), ("photon", "--")):
            frame = cells[
                cells["clip_range"].eq(clip) & cells["arm"].eq(arm)
            ]
            label = f"{arm}, clip {clip:g}"
            axes[0].plot(
                frame["target_source_r_mag"],
                100 * frame["mean_detection_rate"],
                marker="o",
                linestyle=linestyle,
                label=label,
            )
            axes[1].plot(
                frame["target_source_r_mag"],
                frame["mean_logit"],
                marker="o",
                linestyle=linestyle,
                label=label,
            )
    axes[0].set_ylabel("Detection rate (%)")
    axes[1].set_ylabel("Mean native logit")
    for axis in axes:
        axis.set_xlabel("Intrinsic source r magnitude")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("seal", "threshold", "cell-map", "annulus", "brightness"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "seal":
        result = seal(root)
    elif args.command == "threshold":
        result = verify_threshold(root)
    elif args.command == "cell-map":
        result = cell_map(root)
    elif args.command == "annulus":
        result = annulus(root)
    else:
        result = brightness(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
