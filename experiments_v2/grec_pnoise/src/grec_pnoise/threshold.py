from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    atomic_write_json,
    config_provenance,
    load_json,
    sha256_file,
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
        raise ContractError("calibration manifest/logits lack threshold contract fields")
    if not manifest["global_role"].eq("threshold_calibration").all():
        raise ContractError("calibration manifest contains another role")
    for frame, name in ((manifest, "manifest"), (logits, "logits")):
        if frame["calibration_id"].astype(str).duplicated().any():
            raise ContractError(f"duplicate calibration IDs in {name}")
    if manifest["host_pixel_sha256"].astype(str).duplicated().any():
        raise ContractError("duplicate calibration pixels are forbidden")
    if set(manifest["calibration_id"].astype(str)) != set(
        logits["calibration_id"].astype(str)
    ):
        raise ContractError("calibration logits do not exactly cover the manifest")
    joined = manifest[["calibration_id"]].merge(
        logits, on="calibration_id", validate="one_to_one", sort=False
    )
    values = joined["raw_logit_float32"].to_numpy()
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ContractError("calibration logits must be finite native float32")
    bits = joined["raw_logit_float32_bits"].to_numpy(dtype=np.uint32)
    if not np.array_equal(values.view(np.uint32), bits):
        raise ContractError("calibration logit decimal values and bits disagree")
    minimum = int(config["threshold"]["minimum_calibration_rows"])
    if len(values) < minimum:
        raise ContractError(f"calibration n={len(values)} is below {minimum}")
    return manifest, values


def threshold_evidence(
    values: np.ndarray, alpha: float
) -> tuple[dict[str, Any], np.float32]:
    logits = np.asarray(values)
    if logits.dtype != np.float32 or logits.ndim != 1 or not np.isfinite(logits).all():
        raise ContractError("threshold construction requires finite 1-D float32 logits")
    if not (0.0 < alpha < 1.0):
        raise ContractError("alpha must lie in (0,1)")
    n = int(logits.size)
    k = int(math.floor(alpha * n))
    if k < 1 or k + 1 > n:
        raise ContractError("calibration size cannot identify s_(k+1)")
    ordered = np.sort(logits)[::-1]
    s_k_plus_1 = np.float32(ordered[k])
    threshold = np.nextafter(
        s_k_plus_1, np.float32(np.inf), dtype=np.float32
    )
    admitted = int(np.count_nonzero(logits >= threshold))
    if admitted > k:
        raise ContractError("float32 threshold overspent its false-positive allowance")
    evidence = {
        "n": n,
        "alpha": float(alpha),
        "k_floor_alpha_n": k,
        "s_k": float(np.float32(ordered[k - 1])),
        "s_k_float32_bits": int(np.float32(ordered[k - 1]).view(np.uint32)),
        "s_k_plus_1": float(s_k_plus_1),
        "s_k_plus_1_float32_bits": int(s_k_plus_1.view(np.uint32)),
        "logit_threshold": float(threshold),
        "logit_threshold_float32_bits": int(threshold.view(np.uint32)),
        "negatives_at_or_above_threshold": admitted,
        "realised_calibration_fpr": admitted / n,
        "ties_at_s_k_plus_1": int(np.count_nonzero(logits == s_k_plus_1)),
    }
    return evidence, threshold


def build_threshold_receipt(
    *,
    manifest_path: str | Path,
    manifest_receipt_path: str | Path,
    logits_path: str | Path,
    score_receipt_path: str | Path,
    approval_receipt_path: str | Path,
    config: Mapping[str, Any],
    output_path: str | Path,
    canary_mode: bool = False,
) -> dict[str, Any]:
    manifest_receipt = load_json(manifest_receipt_path)
    score_receipt = load_json(score_receipt_path)
    manifest_sha = sha256_file(manifest_path)
    logits_sha = sha256_file(logits_path)
    score_receipt_sha = sha256_file(score_receipt_path)
    if manifest_receipt.get("status") != config["threshold"][
        "required_calibration_manifest_status"
    ]:
        raise ContractError("calibration manifest is not frozen")
    if manifest_receipt.get("calibration_manifest_sha256") != manifest_sha:
        raise ContractError("calibration manifest receipt digest mismatch")
    if manifest_receipt.get("config") != config_provenance(config):
        raise ContractError("calibration manifest was frozen under another config")
    score_checks = {
        "status": config["threshold"]["required_calibration_score_status"],
        "calibration_manifest_sha256": manifest_sha,
        "calibration_logits_sha256": logits_sha,
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
    }
    failures = {
        field: (score_receipt.get(field), expected)
        for field, expected in score_checks.items()
        if score_receipt.get(field) != expected
    }
    if failures:
        raise ContractError(f"calibration score receipt mismatch: {failures}")
    if score_receipt.get("config") != config_provenance(config):
        raise ContractError("calibration scores were produced under another config")
    closure = score_receipt.get("scorer_closure")
    backend = score_receipt.get("backend_profile")
    if not isinstance(closure, dict) or not isinstance(backend, dict):
        raise ContractError("calibration scores lack scorer/backend closure")
    from .contracts import sha256_json

    if score_receipt.get("scorer_closure_sha256") != sha256_json(closure):
        raise ContractError("calibration scorer closure digest mismatch")
    if backend.get("profile_sha256") != sha256_json(
        {key: value for key, value in backend.items() if key != "profile_sha256"}
    ):
        raise ContractError("calibration backend profile digest mismatch")
    from .backend import verify_backend_profile
    from .scoring import fixed_scorer_closure

    requested_device = str(backend.get("requested_device", ""))
    if requested_device != config["scoring"]["authoritative_device"]:
        raise ContractError("calibration scores used a non-authoritative backend")
    expected_closure = fixed_scorer_closure(config, requested_device)
    if closure != expected_closure:
        raise ContractError("calibration scorer closure is not the pinned scorer")
    verify_backend_profile(backend, config, requested_device)
    _, values = _load_bound_calibration(manifest_path, logits_path, config)
    evidence, _ = threshold_evidence(values, float(config["threshold"]["alpha"]))
    approval = load_json(approval_receipt_path)
    expected_approval_status = (
        config["approval"]["canary_status"]
        if canary_mode
        else config["approval"]["required_threshold_status"]
    )
    approval_checks = {
        "status": expected_approval_status,
        "scope": config["approval"]["required_scope"],
        "calibration_manifest_sha256": manifest_sha,
        "calibration_logits_sha256": logits_sha,
        "calibration_score_receipt_sha256": score_receipt_sha,
        "config_contract_sha256": config_provenance(config)[
            "config_contract_sha256"
        ],
        "canary_only": canary_mode,
        "release_authorized": not canary_mode,
    }
    approval_failures = {
        field: (approval.get(field), expected)
        for field, expected in approval_checks.items()
        if approval.get(field) != expected
    }
    if (
        approval_failures
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at_utc", "")).endswith("Z")
    ):
        raise ContractError(
            f"external pre-score approval receipt is invalid: {approval_failures}"
        )
    receipt = {
        "schema_version": 1,
        "status": (
            "threshold_canary_recomputed_not_released"
            if canary_mode
            else config["scoring"]["required_threshold_receipt_status"]
        ),
        "calibration_role": "threshold_calibration",
        "calibration_manifest_path": str(Path(manifest_path).resolve()),
        "calibration_manifest_sha256": manifest_sha,
        "calibration_manifest_receipt_sha256": sha256_file(manifest_receipt_path),
        "calibration_logits_path": str(Path(logits_path).resolve()),
        "calibration_logits_sha256": logits_sha,
        "calibration_score_receipt_sha256": score_receipt_sha,
        "external_approval_receipt_path": str(
            Path(approval_receipt_path).resolve()
        ),
        "external_approval_receipt_sha256": sha256_file(approval_receipt_path),
        "external_approved_by": approval["approved_by"],
        "external_approved_at_utc": approval["approved_at_utc"],
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "scorer_closure": closure,
        "scorer_closure_sha256": score_receipt["scorer_closure_sha256"],
        "backend_profile": backend,
        "authoritative_backend": backend["requested_device"],
        "native_logit_dtype": "float32",
        "threshold_rule": (
            "nextafter_float32_above_s_k_plus_1; "
            "classify_logit_greater_equal"
        ),
        "comparison": ">=",
        "pre_score_review": "external_receipt_bound_not_self_asserted",
        "released_for_experiment_scoring": False,
        "external_release_authorized": not canary_mode,
        "config": config_provenance(config),
        **evidence,
    }
    atomic_write_json(output_path, receipt)
    return receipt


def validate_threshold_receipt(
    receipt_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.float32]:
    receipt = load_json(receipt_path)
    expected = {
        "status": config["scoring"]["required_threshold_receipt_status"],
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "comparison": ">=",
        "native_logit_dtype": "float32",
        "calibration_role": "threshold_calibration",
        "pre_score_review": "external_receipt_bound_not_self_asserted",
        "released_for_experiment_scoring": False,
        "external_release_authorized": True,
    }
    failures = {
        field: (receipt.get(field), value)
        for field, value in expected.items()
        if receipt.get(field) != value
    }
    if failures:
        raise ContractError(f"threshold receipt does not authorize scoring: {failures}")
    approval_path = receipt.get("external_approval_receipt_path")
    if (
        not approval_path
        or sha256_file(approval_path)
        != receipt.get("external_approval_receipt_sha256")
    ):
        raise ContractError("external threshold approval receipt is absent or changed")
    approval = load_json(approval_path)
    approval_expected = {
        "status": config["approval"]["required_threshold_status"],
        "scope": config["approval"]["required_scope"],
        "calibration_manifest_sha256": receipt.get(
            "calibration_manifest_sha256"
        ),
        "calibration_logits_sha256": receipt.get("calibration_logits_sha256"),
        "calibration_score_receipt_sha256": receipt.get(
            "calibration_score_receipt_sha256"
        ),
        "config_contract_sha256": config_provenance(config)[
            "config_contract_sha256"
        ],
        "canary_only": False,
        "release_authorized": True,
    }
    if any(approval.get(field) != value for field, value in approval_expected.items()):
        raise ContractError("external threshold approval no longer authorizes receipt")
    manifest_path = receipt.get("calibration_manifest_path")
    logits_path = receipt.get("calibration_logits_path")
    if not manifest_path or not logits_path:
        raise ContractError("threshold receipt lacks bound calibration artifacts")
    if sha256_file(manifest_path) != receipt.get("calibration_manifest_sha256"):
        raise ContractError("bound calibration manifest drift")
    if sha256_file(logits_path) != receipt.get("calibration_logits_sha256"):
        raise ContractError("bound calibration logits drift")
    _, values = _load_bound_calibration(manifest_path, logits_path, config)
    evidence, threshold = threshold_evidence(
        values, float(config["threshold"]["alpha"])
    )
    for field, value in evidence.items():
        if receipt.get(field) != value:
            raise ContractError(f"threshold receipt evidence drift at {field}")
    return receipt, threshold
