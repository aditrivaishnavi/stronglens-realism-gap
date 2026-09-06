#!/usr/bin/env python3
"""Resume the frozen G_rec render without changing hash-bound package sources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from grec_pnoise.contracts import (
    ContractError,
    array_bytes_sha256,
    assert_score_blind_columns,
    atomic_write_json,
    config_provenance,
    load_config,
    package_source_sha256,
    runtime_record,
    scientific_section_sha256,
    sha256_file,
)
from grec_pnoise.rendering import (
    _load_host,
    render_parent_bank,
    render_source_expectation_at_order,
    validate_renderer_validation_receipt,
)
from grec_pnoise.scene_table import verify_scene_identity

CHECKPOINT_DIR_NAME = "grec_parent_resume_checkpoints"
MANIFEST_NAME = "grec_parent_manifest.parquet"
RECEIPT_NAME = "GREC_RENDER_RECEIPT.json"
AUDIT_NAME = "GREC_RENDER_RESUME_AUDIT.json"
RENDER_SOURCE_FILES = (
    "cells.py",
    "contracts.py",
    "rendering.py",
    "scene_table.py",
    "validation.py",
)
EXTENDED_QUADRATURE_ORDERS = (48, 64, 96)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_checkpoint_part(
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    *,
    end_index: int,
) -> None:
    if not rows:
        return
    start_index = end_index - len(rows) + 1
    path = checkpoint_dir / f"part-{start_index:08d}-{end_index:08d}.parquet"
    _atomic_write_parquet(pd.DataFrame(rows), path)


def _artifact_path(output: Path, scene_id: str) -> Path:
    return output / "grec_parents" / scene_id[:2] / f"{scene_id}.npz"


def _active_original_renderer() -> list[int]:
    """Return live original-renderer PIDs."""
    try:
        process_table = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    active: list[int] = []
    for line in process_table.splitlines():
        if (
            "grec_pnoise.cli" not in line
            or "render-grec" not in line
        ):
            continue
        fields = line.strip().split(maxsplit=1)
        if fields and fields[0].isdigit():
            active.append(int(fields[0]))
    return active


def _load_and_validate_host(
    scene: Mapping[str, Any], output: Path
) -> tuple[np.ndarray, Path]:
    host = _load_host(scene)
    host_hash = str(scene["host_pixel_sha256"])
    if array_bytes_sha256(host) != host_hash:
        raise ContractError("host element bytes differ from frozen scene identity")
    stored_host = output / "raw_hosts" / host_hash[:2] / f"{host_hash}.npy"
    if not stored_host.exists():
        _atomic_save_npy(stored_host, host)
    restored = np.load(stored_host, allow_pickle=False)
    if (
        restored.dtype != host.dtype
        or restored.shape != host.shape
        or array_bytes_sha256(restored) != host_hash
    ):
        raise ContractError("raw-host store did not preserve exact element bytes")
    return host, stored_host


def recover_existing_row(
    scene: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any] | None:
    """Validate an atomic v1 artifact and reconstruct its manifest row."""
    verify_scene_identity(scene, config)
    scene_id = str(scene["scene_id"])
    artifact = _artifact_path(output, scene_id)
    if not artifact.exists():
        return None
    host, stored_host = _load_and_validate_host(scene, output)
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != {
                "expectation_chw",
                "injection_expectation_chw",
            }:
                raise ContractError(f"unexpected parent archive members at {artifact}")
            expectation = np.array(archive["expectation_chw"], copy=True)
            arc = np.array(archive["injection_expectation_chw"], copy=True)
    except (OSError, ValueError) as exc:
        raise ContractError(f"invalid completed parent artifact at {artifact}") from exc
    expected_shape = (3, host.shape[0], host.shape[1])
    if expectation.shape != expected_shape or arc.shape != expected_shape:
        raise ContractError(f"parent artifact shape mismatch at {artifact}")
    if expectation.dtype != np.float32 or arc.dtype != np.float32:
        raise ContractError(f"parent artifact dtype mismatch at {artifact}")
    if (
        not np.isfinite(expectation).all()
        or not np.isfinite(arc).all()
        or (arc < np.float32(0.0)).any()
    ):
        raise ContractError(f"parent artifact contains invalid values at {artifact}")
    host_chw = host.transpose(2, 0, 1)
    if not np.array_equal(
        (host_chw + arc).astype(np.float32, copy=False), expectation
    ):
        raise ContractError(f"parent artifact is not exact host-plus-arc at {artifact}")
    return {
        **scene,
        "grec_row_id": str(scene["grec_row_id"]),
        "scene_id": scene_id,
        "parameter_row_sha256": str(scene["parameter_row_sha256"]),
        "host_galaxy_id": str(scene["host_galaxy_id"]),
        "host_parent_id": str(scene["host_parent_id"]),
        "host_pixel_sha256": str(scene["host_pixel_sha256"]),
        "host_store_path": str(stored_host),
        "host_store_file_sha256": sha256_file(stored_host),
        "parent_pixel_path": str(artifact),
        "parent_artifact_sha256": sha256_file(artifact),
        "expectation_pixel_sha256": array_bytes_sha256(expectation),
        "injection_expectation_sha256": array_bytes_sha256(arc),
        "host_arithmetic_roundtrip_exact": bool(
            np.array_equal(expectation - arc, host_chw)
        ),
        "added_source_photon_noise": False,
        "added_sky_noise": False,
        # With exactly two configured orders, every successful adaptive render
        # necessarily selected the second (maximum) order.
        "quadrature_selected_order": max(config["render"]["quadrature_orders"]),
        # V1 artifacts did not persist these diagnostics. Their successful
        # atomic creation proves the frozen gates passed, but exact values
        # cannot be reconstructed without repeating the render.
        "unlensed_flux_relative_error": np.nan,
        "lensed_previous_order_relative_l1": np.nan,
        "lensed_previous_order_relative_max": np.nan,
        "resume_row_origin": "validated_existing_v1_atomic_artifact",
    }


def _render_with_extended_quadrature(
    scene: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Apply the frozen tolerances at additional, strictly higher orders."""
    verify_scene_identity(scene, config)
    host, _ = _load_and_validate_host(scene, output)
    orders = tuple(int(value) for value in config["render"]["quadrature_orders"])
    if orders != (24, 32):
        raise ContractError("numerical amendment is bound to base orders 24 and 32")
    flux_tolerance = float(config["render"]["unlensed_flux_relative_tolerance"])
    l1_tolerance = float(config["render"]["lensed_relative_l1_tolerance"])
    max_tolerance = float(config["render"]["lensed_relative_max_tolerance"])
    previous: np.ndarray | None = None
    selected: tuple[np.ndarray, int, float, float, float] | None = None
    for order in (*orders, *EXTENDED_QUADRATURE_ORDERS):
        image, flux_error = render_source_expectation_at_order(
            scene,
            host.shape,
            config,
            order=order,
            device=device,
        )
        if previous is not None:
            difference = np.abs(
                image.astype(np.float64) - previous.astype(np.float64)
            )
            relative_l1 = float(
                difference.sum()
                / max(np.abs(image.astype(np.float64)).sum(), 1.0e-30)
            )
            relative_max = float(
                difference.max()
                / max(np.abs(image.astype(np.float64)).max(), 1.0e-30)
            )
            if (
                flux_error <= flux_tolerance
                and relative_l1 <= l1_tolerance
                and relative_max <= max_tolerance
            ):
                selected = (
                    image,
                    order,
                    flux_error,
                    relative_l1,
                    relative_max,
                )
                break
        previous = image
    if selected is None:
        raise ContractError(
            "extended quadrature failed the unchanged preregistered tolerances "
            f"through order {EXTENDED_QUADRATURE_ORDERS[-1]}"
        )
    arc, order, flux_error, relative_l1, relative_max = selected
    expectation = (host.transpose(2, 0, 1) + arc).astype(np.float32, copy=False)
    artifact = _artifact_path(output, str(scene["scene_id"]))
    _atomic_save_npz(
        artifact,
        expectation_chw=expectation,
        injection_expectation_chw=arc,
    )
    row = recover_existing_row(scene, config, output)
    if row is None:
        raise ContractError("extended render was not atomically persisted")
    row.update(
        {
            "quadrature_selected_order": order,
            "unlensed_flux_relative_error": flux_error,
            "lensed_previous_order_relative_l1": relative_l1,
            "lensed_previous_order_relative_max": relative_max,
            "resume_row_origin": "rendered_under_score_blind_quadrature_amendment",
        }
    )
    return row


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    scene: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    recovered = recover_existing_row(scene, config, output)
    if recovered is None:
        raise ContractError("checkpoint references a missing parent artifact")
    exact_fields = (
        "scene_id",
        "grec_row_id",
        "parameter_row_sha256",
        "host_pixel_sha256",
        "parent_artifact_sha256",
        "expectation_pixel_sha256",
        "injection_expectation_sha256",
    )
    failures = [
        field
        for field in exact_fields
        if str(row.get(field)) != str(recovered.get(field))
    ]
    if failures:
        raise ContractError(f"checkpoint row failed artifact validation: {failures}")
    result = dict(row)
    result["resume_row_origin"] = str(
        row.get("resume_row_origin", "validated_resumable_checkpoint")
    )
    return result


def _maximum_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.max())


def resume_render(
    scenes: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    *,
    device: str,
    checkpoint_every: int,
    allow_extended_quadrature: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    assert_score_blind_columns(list(scenes.columns))
    checkpoint_dir = output / CHECKPOINT_DIR_NAME
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    checkpoint_parts = sorted(checkpoint_dir.glob("part-*.parquet"))
    if checkpoint_parts:
        checkpoint = pd.concat(
            [pd.read_parquet(path) for path in checkpoint_parts],
            ignore_index=True,
        )
        if checkpoint["scene_id"].astype(str).duplicated().any():
            raise ContractError("resume checkpoint has duplicate scene IDs")
        checkpoint_rows = {
            str(row["scene_id"]): row
            for row in checkpoint.to_dict(orient="records")
        }

    rows: list[dict[str, Any]] = []
    checkpoint_reused = 0
    v1_recovered = 0
    newly_rendered = 0
    pending_checkpoint_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes.to_dict(orient="records"), start=1):
        scene_id = str(scene["scene_id"])
        if scene_id in checkpoint_rows:
            row = _validate_checkpoint_row(
                checkpoint_rows[scene_id], scene, config, output
            )
            checkpoint_reused += 1
        else:
            row = recover_existing_row(scene, config, output)
            if row is not None:
                v1_recovered += 1
            else:
                try:
                    rendered, _ = render_parent_bank(
                        pd.DataFrame([scene]),
                        config,
                        output,
                        device=device,
                        limit=1,
                    )
                    row = rendered.iloc[0].to_dict()
                    row["resume_row_origin"] = "rendered_by_resumable_runner"
                except ContractError as exc:
                    if (
                        not allow_extended_quadrature
                        or "adaptive pixel integration failed" not in str(exc)
                    ):
                        raise
                    row = _render_with_extended_quadrature(
                        scene,
                        config,
                        output,
                        device=device,
                    )
                newly_rendered += 1
            pending_checkpoint_rows.append(row)
        rows.append(row)
        if (
            len(pending_checkpoint_rows) >= checkpoint_every
            or (index == len(scenes) and pending_checkpoint_rows)
        ):
            _write_checkpoint_part(
                pending_checkpoint_rows,
                checkpoint_dir,
                end_index=index,
            )
            pending_checkpoint_rows = []

    manifest = pd.DataFrame(rows)
    assert_score_blind_columns(list(manifest.columns))
    if len(manifest) != len(scenes):
        raise ContractError("resumed manifest does not cover the frozen scene table")
    if manifest["scene_id"].astype(str).duplicated().any():
        raise ContractError("resumed manifest has duplicate scene IDs")
    diagnostics_available = int(
        pd.to_numeric(
            manifest["unlensed_flux_relative_error"], errors="coerce"
        ).notna().sum()
    )
    receipt = {
        "schema_version": 1,
        "status": "grec_parent_bank_frozen_pre_score",
        "rows": len(manifest),
        "distinct_hosts": int(manifest["host_galaxy_id"].nunique()),
        "raw_host_element_bytes_preserved": True,
        "source_model": "single_sersic",
        "lens_model": "SIE_plus_external_shear",
        "added_source_photon_noise": False,
        "additional_sky_noise": False,
        "convolution_boundary": "explicit_zero_padding",
        "operation_order": config["render"]["operation_order"],
        "render_scientific_config_sha256": scientific_section_sha256(
            config, "source", "lens", "render", "host", "cells", "grec"
        ),
        "config": config_provenance(config),
        "quadrature_selected_order_counts": {
            str(key): int(value)
            for key, value in manifest["quadrature_selected_order"]
            .value_counts()
            .sort_index()
            .items()
        },
        "maximum_unlensed_flux_relative_error": _maximum_or_none(
            manifest["unlensed_flux_relative_error"]
        ),
        "maximum_lensed_previous_order_relative_l1": _maximum_or_none(
            manifest["lensed_previous_order_relative_l1"]
        ),
        "maximum_lensed_previous_order_relative_max": _maximum_or_none(
            manifest["lensed_previous_order_relative_max"]
        ),
        "convergence_diagnostics_available_rows": diagnostics_available,
        "convergence_diagnostics_unavailable_recovered_rows": (
            len(manifest) - diagnostics_available
        ),
        "recovery_validation": (
            "existing artifacts required valid atomic NPZ structure, exact float32 "
            "shape, finite nonnegative arc pixels, frozen host bytes, and exact "
            "float32 host-plus-arc equality"
        ),
        "render_implementation_source_sha256": package_source_sha256(
            *RENDER_SOURCE_FILES
        ),
        "resume_script_sha256": sha256_file(__file__),
        "extended_quadrature_orders": (
            list(EXTENDED_QUADRATURE_ORDERS)
            if allow_extended_quadrature
            else []
        ),
        "runtime": runtime_record(),
    }
    audit = {
        "schema_version": 1,
        "status": "grec_render_resume_complete",
        "checkpoint_rows_reused": checkpoint_reused,
        "existing_v1_artifacts_recovered": v1_recovered,
        "rows_newly_rendered": newly_rendered,
        "rows_total": len(manifest),
        "checkpoint_every_rows": checkpoint_every,
        "checkpoint_part_count": len(list(checkpoint_dir.glob("part-*.parquet"))),
        "extended_quadrature_enabled": allow_extended_quadrature,
        "resume_script_sha256": sha256_file(__file__),
    }
    return manifest, receipt, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--scene-receipt", required=True)
    parser.add_argument("--renderer-validation-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--numerical-amendment")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ContractError("--checkpoint-every must be positive")

    config = load_config(args.config)
    output = Path(args.output_dir)
    active = _active_original_renderer()
    if active:
        raise ContractError(
            f"refusing concurrent recovery while original renderer is active: {active}"
        )
    renderer_receipt = validate_renderer_validation_receipt(
        args.renderer_validation_receipt, config
    )
    scene_receipt = json.loads(Path(args.scene_receipt).read_text(encoding="utf-8"))
    if (
        scene_receipt.get("status") != "grec_scene_table_frozen_pre_score"
        or scene_receipt.get("scene_table_sha256") != sha256_file(args.scenes)
        or scene_receipt.get("config") != config_provenance(config)
    ):
        raise ContractError("scene table receipt is invalid or stale")
    scenes = pd.read_parquet(args.scenes)
    if len(scenes) != int(scene_receipt["rows"]):
        raise ContractError("scene table row count differs from its receipt")
    allow_extended_quadrature = False
    amendment_sha256: str | None = None
    if args.numerical_amendment:
        amendment = json.loads(
            Path(args.numerical_amendment).read_text(encoding="utf-8")
        )
        if (
            amendment.get("status")
            != "score_blind_quadrature_extension_authorized"
            or amendment.get("production_config_sha256")
            != sha256_file(args.config)
            or amendment.get("scene_table_sha256") != sha256_file(args.scenes)
            or amendment.get("base_orders") != [24, 32]
            or amendment.get("supplemental_orders")
            != list(EXTENDED_QUADRATURE_ORDERS)
            or amendment.get("tolerances_unchanged") is not True
        ):
            raise ContractError("numerical amendment is invalid or stale")
        allow_extended_quadrature = True
        amendment_sha256 = sha256_file(args.numerical_amendment)

    manifest, receipt, audit = resume_render(
        scenes,
        config,
        output,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        allow_extended_quadrature=allow_extended_quadrature,
    )
    receipt["numerical_amendment_sha256"] = amendment_sha256
    audit["numerical_amendment_sha256"] = amendment_sha256
    manifest_path = output / MANIFEST_NAME
    _atomic_write_parquet(manifest, manifest_path)
    receipt["parent_manifest_sha256"] = sha256_file(manifest_path)
    receipt["scene_table_sha256"] = sha256_file(args.scenes)
    receipt["scene_table_receipt_sha256"] = sha256_file(args.scene_receipt)
    receipt["renderer_validation_receipt_sha256"] = sha256_file(
        args.renderer_validation_receipt
    )
    receipt["renderer_validation_status"] = renderer_receipt["status"]
    atomic_write_json(output / RECEIPT_NAME, receipt)
    audit["parent_manifest_sha256"] = receipt["parent_manifest_sha256"]
    audit["render_receipt_sha256"] = sha256_file(output / RECEIPT_NAME)
    atomic_write_json(output / AUDIT_NAME, audit)


if __name__ == "__main__":
    main()
