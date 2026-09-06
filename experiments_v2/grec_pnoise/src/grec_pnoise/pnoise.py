from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    array_bytes_sha256,
    assert_config_integrity,
    assert_score_blind_columns,
    atomic_write_json,
    config_provenance,
    load_json,
    package_source_sha256,
    scientific_section_sha256,
    seed_from_key,
    sha256_file,
    stable_id,
)


def validate_parent_bank_for_pnoise(
    *,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    scene_table_path: str | Path,
    scene_receipt_path: str | Path,
    renderer_validation_receipt_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .rendering import validate_renderer_validation_receipt
    from .scene_table import verify_scene_identity

    parent_receipt = load_json(parent_receipt_path)
    parent_sha = sha256_file(parent_manifest_path)
    scene_sha = sha256_file(scene_table_path)
    checks = {
        "status": "grec_parent_bank_frozen_pre_score",
        "parent_manifest_sha256": parent_sha,
        "scene_table_sha256": scene_sha,
        "scene_table_receipt_sha256": sha256_file(scene_receipt_path),
        "renderer_validation_receipt_sha256": sha256_file(
            renderer_validation_receipt_path
        ),
        "config": config_provenance(config),
    }
    failures = {
        field: (parent_receipt.get(field), value)
        for field, value in checks.items()
        if parent_receipt.get(field) != value
    }
    if failures:
        raise ContractError(f"invalid G_rec parent receipt for P_noise: {failures}")
    scene_receipt = load_json(scene_receipt_path)
    if (
        scene_receipt.get("status") != "grec_scene_table_frozen_pre_score"
        or scene_receipt.get("scene_table_sha256") != scene_sha
        or scene_receipt.get("config") != config_provenance(config)
    ):
        raise ContractError("scene table receipt is invalid for P_noise")
    validate_renderer_validation_receipt(
        renderer_validation_receipt_path, config
    )
    parents = pd.read_parquet(parent_manifest_path)
    scenes = pd.read_parquet(scene_table_path)
    if len(parents) != parent_receipt.get("rows") or len(scenes) != scene_receipt.get(
        "rows"
    ):
        raise ContractError("parent/scene receipt row count mismatch")
    if set(parents["scene_id"].astype(str)) != set(scenes["scene_id"].astype(str)):
        raise ContractError("parent bank does not exactly cover the frozen scene table")
    for row in parents.to_dict(orient="records"):
        verify_scene_identity(row, config)
    return parents, parent_receipt


def gains_from_config(config: Mapping[str, Any]) -> np.ndarray:
    section = config["pnoise"]
    gains = np.array(
        [
            section["gain_electron_per_nanomaggy_g"],
            section["gain_electron_per_nanomaggy_r"],
            section["gain_electron_per_nanomaggy_z"],
        ],
        dtype=np.float64,
    )
    if gains.shape != (3,) or not np.isfinite(gains).all() or (gains <= 0).any():
        raise ContractError("all three gains must be finite positive electron/nanomaggy")
    return gains


def photon_moments_nmgy(
    expectation_nmgy: np.ndarray, gains_electron_per_nmgy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expectation = np.asarray(expectation_nmgy, dtype=np.float64)
    if expectation.ndim < 1 or expectation.shape[0] != 3:
        raise ContractError("source expectation must be CHW with three bands")
    if not np.isfinite(expectation).all() or (expectation < 0).any():
        raise ContractError("source expectation must be finite and non-negative")
    gains = np.asarray(gains_electron_per_nmgy, dtype=np.float64)
    reshape = (3,) + (1,) * (expectation.ndim - 1)
    gain_field = gains.reshape(reshape)
    return expectation, expectation / gain_field


def draw_photon_arc(
    expectation_nmgy_chw: np.ndarray,
    gains_electron_per_nmgy: np.ndarray,
    *,
    stream_namespace: str,
    scene_id: str,
    draw_index: int,
) -> tuple[np.ndarray, str]:
    """Draw source counts only: Poisson(I*gain)/gain, independently keyed."""
    if draw_index < 0:
        raise ContractError("draw_index must be non-negative")
    expectation, _ = photon_moments_nmgy(
        expectation_nmgy_chw, gains_electron_per_nmgy
    )
    gains = np.asarray(gains_electron_per_nmgy, dtype=np.float64)[:, None, None]
    expected_electrons = expectation * gains
    noise_key = stable_id(
        "photon",
        [stream_namespace, scene_id, int(draw_index)],
        length=40,
    )
    seed = seed_from_key(stream_namespace, scene_id, draw_index)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    electrons = rng.poisson(expected_electrons)
    physical = (electrons / gains).astype(np.float32)
    if not np.isfinite(physical).all() or (physical < 0).any():
        raise ContractError("photon draw produced invalid arc pixels")
    return physical, noise_key


def validate_draw_count_receipt(
    receipt_path: str | Path,
    config: Mapping[str, Any],
    threshold_receipt_path: str | Path,
) -> dict[str, Any]:
    assert_config_integrity(config)
    receipt = load_json(receipt_path)
    expected_status = config["pnoise"]["required_draw_receipt_status"]
    if receipt.get("status") != expected_status:
        raise ContractError("P_noise final rendering requires a frozen MCSE receipt")
    required = {
        "design_only": True,
        "final_outcomes_used": False,
        "metric": "host_equal_paired_detection_risk_difference",
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise ContractError(f"invalid draw-count receipt field {field!r}")
    candidates = {int(x) for x in config["pnoise"]["candidate_draw_counts"]}
    draw_count = int(receipt.get("selected_draw_count", 0))
    if draw_count not in candidates:
        raise ContractError("selected draw count is not a configured candidate")
    if not receipt.get("absolute_mcse_gate_passed") or not receipt.get(
        "relative_mcse_gate_passed"
    ):
        raise ContractError("draw-count receipt did not pass both MCSE gates")
    threshold_digest = sha256_file(threshold_receipt_path)
    if receipt.get("threshold_receipt_sha256") != threshold_digest:
        raise ContractError("draw-count receipt binds a different threshold receipt")
    from .scoring import validate_scoring_authority

    validate_scoring_authority(config, threshold_receipt_path)
    expected_pnoise = scientific_section_sha256(
        config, "pnoise", "render", "host"
    )
    if receipt.get("pnoise_config_sha256") != expected_pnoise:
        raise ContractError("draw-count receipt was selected under another P_noise config")
    if receipt.get("photon_stream_namespace") != config["pnoise"][
        "photon_stream_namespace"
    ]:
        raise ContractError("draw-count receipt has another photon stream namespace")
    if receipt.get("config") != config_provenance(config):
        raise ContractError("draw-count receipt was produced under another config")
    return receipt


def verify_design_role_ledger(
    scene_ids: Sequence[str],
    host_ids: Sequence[str],
    ledger_path: str | Path,
    ledger_receipt_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_receipt = load_json(ledger_receipt_path)
    ledger_sha = sha256_file(ledger_path)
    checks = {
        "status": "design_role_ledger_frozen",
        "design_role_ledger_sha256": ledger_sha,
        "config_contract_sha256": config_provenance(config)[
            "config_contract_sha256"
        ],
    }
    if any(ledger_receipt.get(field) != value for field, value in checks.items()):
        raise ContractError("design role ledger receipt is invalid or stale")
    ledger = pd.read_parquet(ledger_path)
    required = {"scene_id", "host_galaxy_id", "global_role"}
    if required - set(ledger):
        raise ContractError("design role ledger lacks scene/host/role columns")
    if ledger["scene_id"].astype(str).duplicated().any():
        raise ContractError("design role ledger contains duplicate scene IDs")
    requested = pd.DataFrame(
        {"scene_id": np.asarray(scene_ids, dtype=str), "host_galaxy_id": np.asarray(host_ids, dtype=str)}
    )
    if requested["scene_id"].duplicated().any():
        raise ContractError("pilot contains duplicate scene IDs")
    joined = requested.merge(
        ledger[list(required)].astype(str),
        on=["scene_id", "host_galaxy_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["global_role"].isna().any() or not joined["global_role"].eq("design").all():
        raise ContractError("pilot scene/host IDs are not all in the frozen design role")
    return ledger_receipt


def choose_draw_count(
    physical_decisions: np.ndarray,
    expectation_decisions: np.ndarray,
    host_ids: Sequence[str],
    candidate_draw_counts: Sequence[int],
    *,
    absolute_mcse_max: float,
    fraction_of_sampling_se_max: float,
) -> dict[str, Any]:
    """Choose repeats from design-only outcomes without looking at final scenes."""
    physical = np.asarray(physical_decisions)
    baseline = np.asarray(expectation_decisions)
    hosts = np.asarray(host_ids, dtype=str)
    if physical.ndim != 2 or baseline.shape != (physical.shape[0],):
        raise ContractError("pilot decisions must have shapes (scene,draw) and (scene,)")
    if hosts.shape != baseline.shape:
        raise ContractError("one host ID is required per design scene")
    if physical.shape[1] < 2:
        raise ContractError("at least two pilot draws are needed to estimate MC variance")
    if not np.isin(physical, [0, 1]).all() or not np.isin(baseline, [0, 1]).all():
        raise ContractError("MCSE pilot consumes binary threshold decisions")
    unique_hosts, inverse, scene_counts = np.unique(
        hosts, return_inverse=True, return_counts=True
    )
    if unique_hosts.size < 2:
        raise ContractError("MCSE pilot needs at least two design hosts")
    weights = 1.0 / (unique_hosts.size * scene_counts[inverse])
    per_scene_variance = physical.astype(np.float64).var(axis=1, ddof=1)
    per_scene_effect = physical.astype(np.float64).mean(axis=1) - baseline
    host_effect = np.bincount(
        inverse, weights=per_scene_effect / scene_counts[inverse], minlength=unique_hosts.size
    )
    sampling_se = float(host_effect.std(ddof=1) / math.sqrt(unique_hosts.size))

    candidates = sorted({int(value) for value in candidate_draw_counts})
    records: list[dict[str, Any]] = []
    selected: int | None = None
    for count in candidates:
        if count < 1 or count > physical.shape[1]:
            continue
        mcse = float(
            math.sqrt(np.sum(np.square(weights) * per_scene_variance) / count)
        )
        relative_limit = fraction_of_sampling_se_max * sampling_se
        absolute_pass = mcse <= absolute_mcse_max
        relative_pass = mcse <= relative_limit
        records.append(
            {
                "draw_count": count,
                "estimated_mcse": mcse,
                "absolute_limit": float(absolute_mcse_max),
                "relative_limit": float(relative_limit),
                "absolute_pass": absolute_pass,
                "relative_pass": relative_pass,
            }
        )
        if selected is None and absolute_pass and relative_pass:
            selected = count
    if selected is None:
        raise ContractError(
            "no candidate draw count passes the design-only MCSE gates; "
            "increase the predeclared pilot maximum before final outcomes"
        )
    selected_record = next(x for x in records if x["draw_count"] == selected)
    return {
        "selected_draw_count": selected,
        "estimated_mcse": selected_record["estimated_mcse"],
        "estimated_sampling_se": sampling_se,
        "absolute_mcse_gate_passed": selected_record["absolute_pass"],
        "relative_mcse_gate_passed": selected_record["relative_pass"],
        "candidate_results": records,
        "design_hosts": int(unique_hosts.size),
        "design_scenes": int(physical.shape[0]),
        "pilot_draws_per_scene": int(physical.shape[1]),
    }


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def render_photon_children(
    parents: pd.DataFrame | None,
    config: Mapping[str, Any],
    output_dir: str | Path,
    draw_receipt: Mapping[str, Any] | None,
    *,
    limit: int | None = None,
    canary_label: str | None = None,
    parent_manifest_path: str | Path | None = None,
    parent_receipt_path: str | Path | None = None,
    scene_table_path: str | Path | None = None,
    scene_receipt_path: str | Path | None = None,
    renderer_validation_receipt_path: str | Path | None = None,
    draw_receipt_path: str | Path | None = None,
    threshold_receipt_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive repeated physical children from immutable expectation parents."""
    assert_config_integrity(config)
    if limit is None:
        authority_paths = (
            parent_manifest_path,
            parent_receipt_path,
            scene_table_path,
            scene_receipt_path,
            renderer_validation_receipt_path,
            draw_receipt_path,
            threshold_receipt_path,
        )
        if any(path is None for path in authority_paths):
            raise ContractError(
                "production P_noise API requires complete parent/scene/renderer/"
                "draw/threshold authority paths"
            )
        parents, _ = validate_parent_bank_for_pnoise(
            parent_manifest_path=parent_manifest_path,
            parent_receipt_path=parent_receipt_path,
            scene_table_path=scene_table_path,
            scene_receipt_path=scene_receipt_path,
            renderer_validation_receipt_path=renderer_validation_receipt_path,
            config=config,
        )
        draw_receipt = validate_draw_count_receipt(
            draw_receipt_path, config, threshold_receipt_path
        )
        draw_receipt = dict(draw_receipt)
        draw_receipt["_receipt_sha256"] = sha256_file(draw_receipt_path)
    else:
        if not canary_label or not canary_label.startswith("design_only_"):
            raise ContractError(
                "limited P_noise rendering must be explicitly labelled design-only canary"
            )
        if parents is None or draw_receipt is None:
            raise ContractError("design-only canary requires parent rows and draw settings")
    assert parents is not None and draw_receipt is not None
    assert_score_blind_columns(list(parents.columns))
    required = {
        "grec_row_id",
        "scene_id",
        "parameter_row_sha256",
        "host_galaxy_id",
        "host_parent_id",
        "host_pixel_sha256",
        "host_store_path",
        "host_store_file_sha256",
        "parent_pixel_path",
        "parent_artifact_sha256",
        "expectation_pixel_sha256",
        "injection_expectation_sha256",
    }
    missing = sorted(required - set(parents.columns))
    if missing:
        raise ContractError(f"parent manifest lacks P_noise fields: {missing}")
    draw_count = int(draw_receipt["selected_draw_count"])
    gains = gains_from_config(config)
    stream_namespace = str(config["pnoise"]["photon_stream_namespace"])
    output = Path(output_dir)
    table = parents if limit is None else parents.iloc[:limit]
    rows: list[dict[str, Any]] = []

    for parent in table.to_dict(orient="records"):
        parent_path = Path(str(parent["parent_pixel_path"]))
        host_path = Path(str(parent["host_store_path"]))
        if sha256_file(parent_path) != str(parent["parent_artifact_sha256"]):
            raise ContractError("parent artifact bytes changed after parent freeze")
        if sha256_file(host_path) != str(parent["host_store_file_sha256"]):
            raise ContractError("raw host store bytes changed after parent freeze")
        host_hwc = np.load(host_path, allow_pickle=False)
        if array_bytes_sha256(host_hwc) != str(parent["host_pixel_sha256"]):
            raise ContractError("raw host element bytes changed after parent freeze")
        with np.load(parent_path, allow_pickle=False) as archive:
            expectation = np.array(archive["expectation_chw"], copy=True)
            arc_expectation = np.array(
                archive["injection_expectation_chw"], copy=True
            )
        if array_bytes_sha256(expectation) != str(parent["expectation_pixel_sha256"]):
            raise ContractError("parent expectation pixels changed")
        if array_bytes_sha256(arc_expectation) != str(
            parent["injection_expectation_sha256"]
        ):
            raise ContractError("parent arc expectation pixels changed")
        host_chw = host_hwc.transpose(2, 0, 1)
        if not np.array_equal(
            host_chw + arc_expectation, expectation
        ):
            raise ContractError("parent is not exact host-plus-expectation construction")

        for draw_index in range(draw_count):
            physical_arc, noise_key = draw_photon_arc(
                arc_expectation,
                gains,
                stream_namespace=stream_namespace,
                scene_id=str(parent["scene_id"]),
                draw_index=draw_index,
            )
            physical = (host_chw + physical_arc).astype(np.float32, copy=False)
            child_id = stable_id(
                "pnoise",
                [parent["scene_id"], noise_key, draw_index],
                length=40,
            )
            artifact = (
                output
                / "pnoise_children"
                / child_id[:2]
                / f"{child_id}.npz"
            )
            _atomic_save_npz(
                artifact,
                physical_chw=physical,
                photon_arc_chw=physical_arc,
            )
            rows.append(
                {
                    **parent,
                    "pnoise_row_id": child_id,
                    "draw_index": draw_index,
                    "photon_draw_key": noise_key,
                    "parent_grec_row_id": str(parent["grec_row_id"]),
                    "parent_scene_id": str(parent["scene_id"]),
                    "parent_parameter_row_sha256": str(
                        parent["parameter_row_sha256"]
                    ),
                    "parent_expectation_pixel_sha256": str(
                        parent["expectation_pixel_sha256"]
                    ),
                    "host_galaxy_id": str(parent["host_galaxy_id"]),
                    "host_parent_id": str(parent["host_parent_id"]),
                    "host_pixel_sha256": str(parent["host_pixel_sha256"]),
                    "host_store_path": str(host_path),
                    "host_store_file_sha256": str(
                        parent["host_store_file_sha256"]
                    ),
                    "child_pixel_path": str(artifact),
                    "child_artifact_sha256": sha256_file(artifact),
                    "physical_pixel_sha256": array_bytes_sha256(physical),
                    "photon_arc_sha256": array_bytes_sha256(physical_arc),
                    "gain_electron_per_nanomaggy_g": float(gains[0]),
                    "gain_electron_per_nanomaggy_r": float(gains[1]),
                    "gain_electron_per_nanomaggy_z": float(gains[2]),
                }
            )
    manifest = pd.DataFrame(rows)
    assert_score_blind_columns(list(manifest.columns))
    receipt = {
        "schema_version": 1,
        "status": (
            "design_only_pnoise_render_canary"
            if limit is not None
            else "pnoise_physical_bank_frozen_pre_score"
        ),
        "canary_label": canary_label if limit is not None else None,
        "rows": len(manifest),
        "parent_scenes": int(manifest["parent_scene_id"].nunique()),
        "draw_count": draw_count,
        "draw_count_receipt_sha256": str(
            draw_receipt.get("_receipt_sha256", "not_supplied_to_function")
        ),
        "threshold_receipt_sha256": str(
            draw_receipt.get("threshold_receipt_sha256", "")
        ),
        "host_background_source": "exact_parent_raw_host_element_bytes",
        "existing_host_sky_detector_coadd_noise_redrawn": False,
        "source_photon_law": "Poisson(expectation_nmgy * electron_per_nmgy) / electron_per_nmgy",
        "source_flux_unit": "nanomaggy_per_native_pixel",
        "count_unit": "electron",
        "gain_status": config["pnoise"]["gain_status"],
        "photon_stream_namespace": stream_namespace,
        "pnoise_config": dict(config["pnoise"]),
        "pnoise_config_sha256": scientific_section_sha256(
            config, "pnoise", "render", "host"
        ),
        "config": config_provenance(config),
        "implementation_source_sha256": package_source_sha256(
            "contracts.py", "pnoise.py", "rendering.py"
        ),
        "rng": "numpy.Generator(PCG64DXSM), one SHA256-derived stream per scene/draw",
    }
    return manifest, receipt


def write_photon_outputs(
    manifest: pd.DataFrame,
    receipt: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pnoise_child_manifest.parquet"
    temporary = output / "pnoise_child_manifest.partial.parquet"
    manifest.to_parquet(temporary, index=False)
    temporary.replace(manifest_path)
    receipt["child_manifest_sha256"] = sha256_file(manifest_path)
    receipt_path = output / "PNOISE_RENDER_RECEIPT.json"
    atomic_write_json(receipt_path, receipt)
    return manifest_path, receipt_path
