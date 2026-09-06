from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    PilotContractError,
    array_bytes_sha256,
    assert_config_integrity,
    ensure_no_final_outcome_path,
    sha256_file,
)


def _reopen_design_host(row: Mapping[str, Any]) -> tuple[str, int]:
    path = ensure_no_final_outcome_path(row["host_pixel_path"])
    if sha256_file(path) != str(row["host_payload_sha256"]):
        raise PilotContractError(f"design host payload changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "cutout" not in archive.files:
            raise PilotContractError(f"design host lacks cutout: {path}")
        pixels = np.array(archive["cutout"], copy=True)
    if (
        pixels.dtype != np.float32
        or pixels.shape != (101, 101, 3)
        or not np.isfinite(pixels).all()
        or array_bytes_sha256(pixels) != str(row["host_pixel_sha256"])
    ):
        raise PilotContractError(f"design host raw-array bytes changed: {path}")
    return str(row["pilot_cell"]), path.stat().st_size


def _reopen_comparison_pixel(
    row: Mapping[str, Any],
    *,
    payload_root: Path,
) -> tuple[str, int]:
    path = ensure_no_final_outcome_path(row["source_payload_path"])
    if payload_root not in path.parents:
        raise PilotContractError(f"comparison payload escapes root: {path}")
    if sha256_file(path) != str(row["source_payload_sha256"]):
        raise PilotContractError(f"comparison payload changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if str(row["raw_array_key"]) not in archive.files:
            raise PilotContractError(f"comparison payload lacks raw array: {path}")
        pixels = np.array(archive[str(row["raw_array_key"])], copy=True)
    if (
        pixels.dtype != np.float32
        or pixels.shape != (101, 101, 3)
        or not np.isfinite(pixels).all()
        or array_bytes_sha256(pixels) != str(row["raw_array_sha256"])
    ):
        raise PilotContractError(f"comparison raw-array bytes changed: {path}")
    return str(row["global_role"]), path.stat().st_size


def deep_revalidate_frozen_pixels(
    config: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Physically reopen every frozen design and comparison NPZ payload."""
    assert_config_integrity(config)
    output_path = Path(output).resolve()
    host_manifest = pd.read_parquet(output_path / "design_host_manifest.parquet")
    comparison_manifest = pd.read_parquet(
        output_path / "comparison_role_pixel_hashes.parquet"
    )
    if len(host_manifest) != int(config["design"]["expected_hosts"]):
        raise PilotContractError("deep validation host count differs from config")
    if len(comparison_manifest) != 32_514:
        raise PilotContractError("deep validation comparison count differs from freeze")
    workers = int(config["validation"]["comparison_pixel_hash_workers"])
    if workers < 1 or workers > 32:
        raise PilotContractError("deep validation worker count is invalid")

    started = time.perf_counter()
    design_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        design_results = list(
            executor.map(
                _reopen_design_host,
                host_manifest.to_dict(orient="records"),
            )
        )
    design_seconds = time.perf_counter() - design_started

    comparison_started = time.perf_counter()
    payload_root = Path(config["paths"]["candidate_payload_root"]).resolve()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        comparison_results = list(
            executor.map(
                lambda row: _reopen_comparison_pixel(
                    row,
                    payload_root=payload_root,
                ),
                comparison_manifest.to_dict(orient="records"),
            )
        )
    comparison_seconds = time.perf_counter() - comparison_started
    elapsed = time.perf_counter() - started
    design_cells = Counter(role for role, _ in design_results)
    comparison_roles = Counter(role for role, _ in comparison_results)
    expected_roles = {
        "threshold_calibration": 20_002,
        "achieved_fpr": 12_512,
    }
    if dict(comparison_roles) != expected_roles:
        raise PilotContractError(
            f"deep comparison role counts differ: {dict(comparison_roles)}"
        )
    if len(design_cells) != int(config["design"]["expected_cells"]) or any(
        count != int(config["design"]["hosts_per_cell"])
        for count in design_cells.values()
    ):
        raise PilotContractError("deep design cell counts differ")
    return {
        "status": "pnoise_prelaunch_deep_pixel_revalidation_passed",
        "design_hosts_reopened": len(design_results),
        "design_role_counts": dict(sorted(design_cells.items())),
        "design_payload_bytes_read": int(
            sum(size for _, size in design_results)
        ),
        "comparison_payloads_reopened": len(comparison_results),
        "comparison_role_counts": dict(sorted(comparison_roles.items())),
        "comparison_payload_bytes_read": int(
            sum(size for _, size in comparison_results)
        ),
        "raw_array_key": "cutout",
        "raw_array_dtype": "float32",
        "raw_array_shape": [101, 101, 3],
        "payload_sha256_reverified": True,
        "raw_array_sha256_reverified": True,
        "workers": workers,
        "design_seconds": design_seconds,
        "comparison_seconds": comparison_seconds,
        "elapsed_seconds": elapsed,
    }
