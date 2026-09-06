from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    assert_config_integrity,
    atomic_write_json,
    config_provenance,
    fixed_scorer_closure,
    load_json,
    sha256_file,
    sha256_json,
    validate_scorer_receipt,
)


def _validate_calibration_scorer_closure(
    closure: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Accept the common G_rec scorer receipt while enforcing the same core."""
    scoring = config["scoring"]
    source_hashes = closure.get("scorer_source_sha256", closure.get("source_sha256"))
    expected = {
        "checkpoint_sha256": scoring["checkpoint_sha256"],
        "checkpoint_epoch": int(scoring["checkpoint_epoch"]),
        "checkpoint_architecture": scoring["checkpoint_architecture"],
        "batch_size": int(scoring["batch_size"]),
        "filler_policy": scoring["filler_policy"],
        "native_logit_dtype": "float32",
    }
    mismatches = {
        field: (closure.get(field), value)
        for field, value in expected.items()
        if closure.get(field) != value
    }
    comparator = closure.get("comparator", closure.get("comparison"))
    runtime = closure.get("runtime_versions", {})
    required_runtime = {
        key: str(scoring["runtime_versions"][key])
        for key in ("python", "numpy", "pandas", "pyarrow", "torch", "torchvision")
    }
    if (
        mismatches
        or comparator != ">="
        or source_hashes != dict(scoring["source_sha256"])
        or any(runtime.get(key) != value for key, value in required_runtime.items())
    ):
        raise ContractError(
            f"calibration scorer does not match fixed common scorer: {mismatches}"
        )


def _load_bound_calibration(
    manifest_path: str | Path,
    logits_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_parquet(manifest_path)
    logits = pd.read_parquet(logits_path)
    required_manifest = {
        "calibration_id",
        "global_role",
        "host_pixel_sha256",
        "evaluation_spatial_group_id",
    }
    required_logits = {
        "calibration_id",
        "raw_logit_float32",
        "raw_logit_float32_bits",
    }
    if required_manifest - set(manifest) or required_logits - set(logits):
        raise ContractError("calibration manifest/logits lack threshold fields")
    if not manifest["global_role"].eq("threshold_calibration").all():
        raise ContractError("calibration manifest contains another global role")
    if (
        manifest["calibration_id"].astype(str).duplicated().any()
        or logits["calibration_id"].astype(str).duplicated().any()
        or manifest["host_pixel_sha256"].astype(str).duplicated().any()
    ):
        raise ContractError("duplicate calibration identity or pixel")
    if set(manifest["calibration_id"].astype(str)) != set(
        logits["calibration_id"].astype(str)
    ):
        raise ContractError("calibration logits do not exactly cover manifest")
    joined = manifest[["calibration_id"]].merge(
        logits, on="calibration_id", validate="one_to_one", sort=False
    )
    values = joined["raw_logit_float32"].to_numpy(copy=True)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ContractError("calibration logits must be finite native float32")
    bits = joined["raw_logit_float32_bits"].to_numpy(copy=True)
    if bits.dtype != np.uint32:
        raise ContractError("calibration logit bits must be native uint32")
    if not np.array_equal(values.view(np.uint32), bits):
        raise ContractError("calibration logit values and exact bits disagree")
    minimum = int(config["threshold"]["minimum_calibration_rows"])
    if len(values) < minimum:
        raise ContractError(f"calibration n={len(values)} is below {minimum}")
    return manifest, values


def threshold_evidence(
    values: np.ndarray, alpha: float
) -> tuple[dict[str, Any], np.float32]:
    logits = np.asarray(values)
    if logits.dtype != np.float32 or logits.ndim != 1 or not np.isfinite(logits).all():
        raise ContractError("threshold requires finite 1-D native float32 logits")
    if not (0 < alpha < 1):
        raise ContractError("alpha must be in (0,1)")
    n = int(logits.size)
    k = int(math.floor(alpha * n))
    if k < 1 or k >= n:
        raise ContractError("calibration size cannot identify s_(k+1)")
    ordered = np.sort(logits, kind="stable")[::-1]
    s_k = np.float32(ordered[k - 1])
    s_k_plus_1 = np.float32(ordered[k])
    threshold = np.nextafter(s_k_plus_1, np.float32(np.inf), dtype=np.float32)
    admitted = int(np.count_nonzero(logits >= threshold))
    if admitted > k:
        raise ContractError("float32 threshold overspent nominal allowance")
    return (
        {
            "n": n,
            "alpha": float(alpha),
            "k_floor_alpha_n": k,
            "s_k": float(s_k),
            "s_k_float32_bits": int(s_k.view(np.uint32)),
            "s_k_plus_1": float(s_k_plus_1),
            "s_k_plus_1_float32_bits": int(s_k_plus_1.view(np.uint32)),
            "logit_threshold": float(threshold),
            "logit_threshold_float32_bits": int(threshold.view(np.uint32)),
            "negatives_at_or_above_threshold": admitted,
            "realised_calibration_fpr": admitted / n,
            "ties_at_s_k_plus_1": int(np.count_nonzero(logits == s_k_plus_1)),
        },
        threshold,
    )


def build_threshold_receipt(
    *,
    manifest_path: str | Path,
    manifest_receipt_path: str | Path,
    logits_path: str | Path,
    score_receipt_path: str | Path,
    scorer_receipt_path: str | Path,
    config: Mapping[str, Any],
    output_path: str | Path,
    final_status: bool,
) -> dict[str, Any]:
    assert_config_integrity(config)
    validate_scorer_receipt(scorer_receipt_path, config, require_final=final_status)
    manifest_sha = sha256_file(manifest_path)
    logits_sha = sha256_file(logits_path)
    manifest_receipt = load_json(manifest_receipt_path)
    score_receipt = load_json(score_receipt_path)
    if (
        manifest_receipt.get("status")
        != config["threshold"]["required_calibration_manifest_status"]
        or manifest_receipt.get("calibration_manifest_sha256") != manifest_sha
    ):
        raise ContractError("calibration manifest receipt is invalid")
    required_score = {
        "status": config["threshold"]["required_calibration_score_status"],
        "calibration_manifest_sha256": manifest_sha,
        "calibration_logits_sha256": logits_sha,
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
    }
    if any(score_receipt.get(key) != value for key, value in required_score.items()):
        raise ContractError("calibration score receipt is invalid")
    calibration_closure = score_receipt.get("scorer_closure")
    if not isinstance(calibration_closure, dict) or score_receipt.get(
        "scorer_closure_sha256"
    ) != sha256_json(calibration_closure):
        raise ContractError("calibration scorer closure digest is invalid")
    _validate_calibration_scorer_closure(calibration_closure, config)
    closure = fixed_scorer_closure(config, verify_runtime=final_status)
    _, values = _load_bound_calibration(manifest_path, logits_path, config)
    evidence, _ = threshold_evidence(values, float(config["threshold"]["alpha"]))
    receipt = {
        "schema_version": 2,
        "status": (
            config["threshold"]["required_threshold_receipt_status"]
            if final_status
            else "threshold_canary_recomputed_not_released"
        ),
        "released_for_experiment_scoring": bool(final_status),
        "calibration_manifest_path": str(Path(manifest_path).resolve()),
        "calibration_manifest_sha256": manifest_sha,
        "calibration_manifest_receipt_sha256": sha256_file(manifest_receipt_path),
        "calibration_logits_path": str(Path(logits_path).resolve()),
        "calibration_logits_sha256": logits_sha,
        "calibration_score_receipt_sha256": sha256_file(score_receipt_path),
        "scorer_receipt_path": str(Path(scorer_receipt_path).resolve()),
        "scorer_receipt_sha256": sha256_file(scorer_receipt_path),
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "scorer_closure": closure,
        "scorer_closure_sha256": sha256_json(closure),
        "calibration_scorer_closure": calibration_closure,
        "calibration_scorer_closure_sha256": sha256_json(calibration_closure),
        "native_logit_dtype": "float32",
        "comparison": ">=",
        "threshold_rule": "nextafter_float32_above_s_k_plus_1",
        "config": config_provenance(config),
        **evidence,
    }
    atomic_write_json(output_path, receipt)
    return receipt


def validate_threshold_receipt(
    receipt_path: str | Path,
    config: Mapping[str, Any],
    *,
    require_final: bool,
) -> tuple[dict[str, Any], np.float32]:
    assert_config_integrity(config)
    receipt = load_json(receipt_path)
    expected_status = (
        config["threshold"]["required_threshold_receipt_status"]
        if require_final
        else "threshold_canary_recomputed_not_released"
    )
    expected = {
        "status": expected_status,
        "released_for_experiment_scoring": require_final,
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "native_logit_dtype": "float32",
        "comparison": ">=",
        "threshold_rule": "nextafter_float32_above_s_k_plus_1",
        "config": config_provenance(config),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError("threshold receipt status/config/scorer contract drift")
    scorer_path = receipt.get("scorer_receipt_path")
    if not scorer_path or sha256_file(scorer_path) != receipt.get(
        "scorer_receipt_sha256"
    ):
        raise ContractError("bound scorer receipt drift")
    validate_scorer_receipt(scorer_path, config, require_final=require_final)
    manifest_path = receipt.get("calibration_manifest_path")
    logits_path = receipt.get("calibration_logits_path")
    if (
        not manifest_path
        or not logits_path
        or sha256_file(manifest_path) != receipt.get("calibration_manifest_sha256")
        or sha256_file(logits_path) != receipt.get("calibration_logits_sha256")
    ):
        raise ContractError("bound calibration manifest/logits drift")
    _, values = _load_bound_calibration(manifest_path, logits_path, config)
    evidence, threshold = threshold_evidence(
        values, float(config["threshold"]["alpha"])
    )
    for field, value in evidence.items():
        if receipt.get(field) != value:
            raise ContractError(f"threshold arithmetic drift: {field}")
    if receipt.get("scorer_closure") != fixed_scorer_closure(
        config, verify_runtime=require_final
    ):
        raise ContractError("threshold scorer closure drift")
    calibration_closure = receipt.get("calibration_scorer_closure")
    if not isinstance(calibration_closure, dict) or receipt.get(
        "calibration_scorer_closure_sha256"
    ) != sha256_json(calibration_closure):
        raise ContractError("bound calibration scorer closure digest drift")
    _validate_calibration_scorer_closure(calibration_closure, config)
    return receipt, threshold
