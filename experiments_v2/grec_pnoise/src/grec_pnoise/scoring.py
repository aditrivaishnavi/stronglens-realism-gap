from __future__ import annotations

import importlib
import importlib.metadata
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .backend import backend_profile, verify_backend_profile
from .contracts import (
    ContractError,
    assert_config_integrity,
    atomic_write_json,
    config_provenance,
    package_source_sha256,
    runtime_record,
    sha256_file,
    sha256_json,
)
from .threshold import validate_threshold_receipt


def float32_from_bits(bits: int) -> np.float32:
    return np.array([int(bits)], dtype=np.uint32).view(np.float32)[0]


def validate_scoring_authority(
    config: Mapping[str, Any],
    threshold_receipt_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[dict[str, Any], np.float32]:
    """Recompute threshold evidence and optionally verify the exact backend."""
    assert_config_integrity(config)
    receipt, threshold = validate_threshold_receipt(threshold_receipt_path, config)
    if receipt.get("config") != config_provenance(config):
        raise ContractError("threshold receipt was built under another configuration")
    if device is not None:
        closure = fixed_scorer_closure(config, device)
        if receipt.get("scorer_closure_sha256") != sha256_json(closure):
            raise ContractError("threshold scorer closure differs from current scorer")
        if receipt.get("scorer_closure") != closure:
            raise ContractError("threshold scorer closure body differs")
        verify_backend_profile(receipt["backend_profile"], config, device)
    return receipt, threshold


def verify_fixed_scorer(config: Mapping[str, Any]) -> None:
    scoring = config["scoring"]
    root = Path(scoring["repo_root"]).resolve()
    checkpoint = Path(scoring["checkpoint"]).resolve()
    if sha256_file(checkpoint) != scoring["checkpoint_sha256"]:
        raise ContractError("checkpoint digest differs from the fixed model receipt")
    for relative, expected in scoring["source_sha256"].items():
        path = root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise ContractError(f"fixed scorer source drift at {relative}: {observed}")
    expected_versions = scoring["runtime_versions"]
    observed_versions = {
        "python": sys.version.split()[0],
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "pyarrow", "torch", "torchvision")
        },
    }
    mismatches = {
        package: (observed_versions.get(package), expected)
        for package, expected in expected_versions.items()
        if observed_versions.get(package) != expected
    }
    if mismatches:
        raise ContractError(f"fixed scoring runtime version drift: {mismatches}")


def fixed_scorer_closure(
    config: Mapping[str, Any], requested_device: str
) -> dict[str, Any]:
    assert_config_integrity(config)
    verify_fixed_scorer(config)
    scoring = config["scoring"]
    return {
        "schema_version": 1,
        "checkpoint_path": str(Path(scoring["checkpoint"]).resolve()),
        "checkpoint_sha256": scoring["checkpoint_sha256"],
        "checkpoint_epoch": int(scoring["checkpoint_epoch"]),
        "checkpoint_architecture": str(scoring["checkpoint_architecture"]),
        "source_sha256": dict(scoring["source_sha256"]),
        "wrapper_source_sha256": package_source_sha256(
            "backend.py",
            "cli.py",
            "contracts.py",
            "inference.py",
            "scoring.py",
            "threshold.py",
        ),
        "runtime_versions": dict(scoring["runtime_versions"]),
        "batch_size": int(scoring["batch_size"]),
        "filler_policy": str(scoring["filler_policy"]),
        "native_logit_dtype": str(scoring["native_logit_dtype"]),
        "comparison": str(scoring["comparison"]),
        "backend_profile_sha256": backend_profile(
            config, requested_device
        )["profile_sha256"],
    }


def _import_fixed_modules(config: Mapping[str, Any]):
    root = Path(config["scoring"]["repo_root"]).resolve()
    loaded = sys.modules.get("dhs")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if root not in loaded_path.parents:
            raise ContractError(f"dhs was already imported from {loaded_path}, not {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    fixed = importlib.import_module("dhs.fixed_width_scoring")
    preprocess = importlib.import_module("dhs.preprocess")
    utilities = importlib.import_module("dhs.scoring_utils")
    if Path(fixed.__file__).resolve() != root / "dhs/fixed_width_scoring.py":
        raise ContractError("imported fixed-width scorer is not the pinned file")
    return fixed, preprocess, utilities


def _verify_bank_receipt(
    manifest_path: str | Path,
    bank_receipt_path: str | Path,
    bank: str,
    config: Mapping[str, Any],
    threshold_receipt_path: str | Path,
) -> dict[str, Any]:
    from .contracts import load_json

    receipt = load_json(bank_receipt_path)
    if bank == "grec":
        expected_status = "grec_parent_bank_frozen_pre_score"
        digest_field = "parent_manifest_sha256"
    elif bank == "pnoise":
        expected_status = "pnoise_physical_bank_frozen_pre_score"
        digest_field = "child_manifest_sha256"
    else:
        raise ContractError(f"unknown score bank {bank!r}")
    if receipt.get("status") != expected_status:
        raise ContractError(f"{bank} bank is not frozen for scoring")
    observed = sha256_file(manifest_path)
    if receipt.get(digest_field) != observed:
        raise ContractError(f"{bank} receipt does not bind its input manifest")
    if receipt.get("config") != config_provenance(config):
        raise ContractError(f"{bank} bank was rendered under another configuration")
    if bank == "pnoise" and receipt.get("threshold_receipt_sha256") != sha256_file(
        threshold_receipt_path
    ):
        raise ContractError("P_noise bank draw count binds another threshold")
    return receipt


def _raw_images(
    manifest: pd.DataFrame,
    bank: str,
) -> Iterable[np.ndarray]:
    if bank == "grec":
        path_column = "parent_pixel_path"
        file_hash_column = "parent_artifact_sha256"
        array_key = "expectation_chw"
        pixel_hash_column = "expectation_pixel_sha256"
    else:
        path_column = "child_pixel_path"
        file_hash_column = "child_artifact_sha256"
        array_key = "physical_chw"
        pixel_hash_column = "physical_pixel_sha256"
    from .contracts import array_bytes_sha256

    for row in manifest.to_dict(orient="records"):
        path = Path(str(row[path_column]))
        if sha256_file(path) != str(row[file_hash_column]):
            raise ContractError(f"pixel artifact drift before scoring: {path}")
        with np.load(path, allow_pickle=False) as archive:
            image = np.array(archive[array_key], copy=True)
        if (
            image.dtype != np.float32
            or image.ndim != 3
            or image.shape[0] != 3
            or not np.isfinite(image).all()
        ):
            raise ContractError(f"invalid detector input array: {path}")
        if array_bytes_sha256(image) != str(row[pixel_hash_column]):
            raise ContractError(f"pixel array drift before scoring: {path}")
        yield image


def design_only_raw_logits(
    images: Iterable[np.ndarray],
    config: Mapping[str, Any],
    *,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Raw logits for synthetic/design convergence only; no decisions or outcomes."""
    closure = fixed_scorer_closure(config, device)
    profile = backend_profile(config, device)
    fixed, preprocess, utilities = _import_fixed_modules(config)
    scoring = config["scoring"]
    model, preprocess_kwargs = utilities.load_model_and_spec(
        scoring["checkpoint"], device
    )
    model = model.to(device).eval()
    fixed.pin_deterministic_kernels()
    decoupling = fixed.assert_no_batch_coupling(model)

    def processed() -> Iterable[np.ndarray]:
        for image in images:
            value = np.asarray(image)
            if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 3:
                raise ContractError("design convergence image must be float32 CHW")
            yield preprocess.preprocess_stack(value, **preprocess_kwargs)

    width = int(scoring["batch_size"])
    logits, batches = fixed.fixed_width_logits(
        model,
        fixed.array_batches(processed(), width),
        batch_size=width,
        device=device,
        filler=str(scoring["filler_policy"]),
    )
    return logits.astype(np.float32), {
        "status": "design_only_raw_logit_convergence",
        "final_outcome_scoring": False,
        "scorer_closure": closure,
        "scorer_closure_sha256": sha256_json(closure),
        "backend_profile": profile,
        "batch": batches,
        "decoupling": decoupling,
    }


def score_calibration_bank(
    manifest_path: str | Path,
    manifest_receipt_path: str | Path,
    config: Mapping[str, Any],
    *,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score only the frozen calibration role; no threshold is constructed here."""
    from .contracts import array_bytes_sha256, load_json

    assert_config_integrity(config)
    if device != config["scoring"]["authoritative_device"]:
        raise ContractError("calibration must use the frozen authoritative backend")
    manifest_sha = sha256_file(manifest_path)
    manifest_receipt = load_json(manifest_receipt_path)
    if manifest_receipt.get("status") != config["threshold"][
        "required_calibration_manifest_status"
    ] or manifest_receipt.get("calibration_manifest_sha256") != manifest_sha:
        raise ContractError("calibration manifest receipt is invalid")
    if manifest_receipt.get("config") != config_provenance(config):
        raise ContractError("calibration manifest was frozen under another config")
    manifest = pd.read_parquet(manifest_path)
    required = {
        "calibration_id",
        "global_role",
        "input_pixel_path",
        "input_artifact_sha256",
        "input_pixel_sha256",
    }
    if required - set(manifest):
        raise ContractError("calibration manifest lacks scoring input fields")
    if not manifest["global_role"].eq("threshold_calibration").all():
        raise ContractError("calibration scoring received a non-calibration role")
    if manifest["calibration_id"].astype(str).duplicated().any():
        raise ContractError("calibration IDs are not unique")
    closure = fixed_scorer_closure(config, device)
    profile = backend_profile(config, device)
    fixed, preprocess, utilities = _import_fixed_modules(config)
    scoring = config["scoring"]
    model, preprocess_kwargs = utilities.load_model_and_spec(
        scoring["checkpoint"], device
    )
    model = model.to(device).eval()
    fixed.pin_deterministic_kernels()
    fixed.assert_no_batch_coupling(model)

    def images() -> Iterable[np.ndarray]:
        for row in manifest.to_dict(orient="records"):
            path = Path(str(row["input_pixel_path"]))
            if sha256_file(path) != str(row["input_artifact_sha256"]):
                raise ContractError("calibration pixel artifact drift")
            with np.load(path, allow_pickle=False) as archive:
                image = np.array(archive["image_chw"], copy=True)
            if (
                image.dtype != np.float32
                or image.ndim != 3
                or image.shape[0] != 3
                or not np.isfinite(image).all()
                or array_bytes_sha256(image) != str(row["input_pixel_sha256"])
            ):
                raise ContractError("invalid calibration detector input")
            yield preprocess.preprocess_stack(image, **preprocess_kwargs)

    batch_size = int(scoring["batch_size"])
    logits64, batch_record = fixed.fixed_width_logits(
        model,
        fixed.array_batches(images(), batch_size),
        batch_size=batch_size,
        device=device,
        filler=str(scoring["filler_policy"]),
    )
    logits = logits64.astype(np.float32)
    scores = pd.DataFrame(
        {
            "calibration_id": manifest["calibration_id"].astype(str),
            "raw_logit_float32": logits,
            "raw_logit_float32_bits": logits.view(np.uint32),
        }
    )
    receipt = {
        "schema_version": 1,
        "status": config["threshold"]["required_calibration_score_status"],
        "calibration_manifest_sha256": manifest_sha,
        "calibration_manifest_receipt_sha256": sha256_file(
            manifest_receipt_path
        ),
        "rows": len(scores),
        "checkpoint_sha256": scoring["checkpoint_sha256"],
        "requested_device": device,
        "scorer_closure": closure,
        "scorer_closure_sha256": sha256_json(closure),
        "backend_profile": profile,
        "fixed_width": batch_record,
        "config": config_provenance(config),
    }
    return scores, receipt


def write_calibration_scores(
    scores: pd.DataFrame,
    receipt: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / "calibration_logits.parquet"
    partial = output / "calibration_logits.partial.parquet"
    scores.to_parquet(partial, index=False)
    partial.replace(scores_path)
    receipt["calibration_logits_sha256"] = sha256_file(scores_path)
    receipt_path = output / "CALIBRATION_SCORE_RECEIPT.json"
    atomic_write_json(receipt_path, receipt)
    return scores_path, receipt_path


def score_frozen_bank(
    manifest_path: str | Path,
    bank_receipt_path: str | Path,
    threshold_receipt_path: str | Path,
    config: Mapping[str, Any],
    *,
    bank: str,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score raw frozen pixels; cannot be called without all three receipts."""
    if device != config["scoring"]["authoritative_device"]:
        raise ContractError("final scoring must use the frozen authoritative backend")
    threshold_receipt, threshold = validate_scoring_authority(
        config, threshold_receipt_path, device=device
    )
    bank_receipt = _verify_bank_receipt(
        manifest_path, bank_receipt_path, bank, config, threshold_receipt_path
    )
    closure = fixed_scorer_closure(config, device)
    profile = verify_backend_profile(
        threshold_receipt["backend_profile"], config, device
    )

    fixed, preprocess, utilities = _import_fixed_modules(config)
    scoring = config["scoring"]
    model, preprocess_kwargs = utilities.load_model_and_spec(
        scoring["checkpoint"], device
    )
    raw_checkpoint = torch.load(
        scoring["checkpoint"], map_location=device, weights_only=False
    )
    epoch = int(raw_checkpoint.get("epoch", -1))
    architecture = str((raw_checkpoint.get("train", {}) or {}).get("arch", ""))
    if epoch != int(scoring["checkpoint_epoch"]) or architecture != str(
        scoring["checkpoint_architecture"]
    ):
        raise ContractError("checkpoint identity metadata does not match configuration")
    model = model.to(device).eval()
    kernel_record = fixed.pin_deterministic_kernels()
    decoupling = fixed.assert_no_batch_coupling(model)
    manifest = pd.read_parquet(manifest_path)
    batch_size = int(scoring["batch_size"])

    def processed() -> Iterable[np.ndarray]:
        for image in _raw_images(manifest, bank):
            yield preprocess.preprocess_stack(image, **preprocess_kwargs)

    logits64, batch_record = fixed.fixed_width_logits(
        model,
        fixed.array_batches(processed(), batch_size),
        batch_size=batch_size,
        device=device,
        filler=str(scoring["filler_policy"]),
    )
    logits = logits64.astype(np.float32)
    if logits.shape != (len(manifest),):
        raise ContractError("scorer did not return exactly one logit per input row")
    decisions = logits >= threshold
    identity_column = "grec_row_id" if bank == "grec" else "pnoise_row_id"
    scores = pd.DataFrame(
        {
            identity_column: manifest[identity_column].astype(str),
            "raw_logit_float32": logits,
            "raw_logit_float32_bits": logits.view(np.uint32),
            "detected_at_frozen_threshold": decisions,
        }
    )
    receipt = {
        "schema_version": 1,
        "status": "frozen_scores_complete",
        "bank": bank,
        "rows": len(scores),
        "checkpoint_sha256": scoring["checkpoint_sha256"],
        "checkpoint_epoch": epoch,
        "checkpoint_architecture": architecture,
        "requested_device": device,
        "scorer_closure": closure,
        "scorer_closure_sha256": sha256_json(closure),
        "backend_profile": profile,
        "threshold_receipt_sha256": sha256_file(threshold_receipt_path),
        "threshold_float32_bits": int(np.asarray(threshold).view(np.uint32)),
        "comparison": ">=",
        "bank_receipt_sha256": sha256_file(bank_receipt_path),
        "bank_manifest_sha256": sha256_file(manifest_path),
        "bank_status": bank_receipt["status"],
        "fixed_width": batch_record,
        "batch_decoupling": decoupling,
        "kernels": kernel_record,
        "runtime": runtime_record(),
    }
    return scores, receipt


def write_score_outputs(
    scores: pd.DataFrame,
    receipt: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / f"{receipt['bank']}_scores.parquet"
    partial = scores_path.with_name(scores_path.stem + ".partial.parquet")
    scores.to_parquet(partial, index=False)
    partial.replace(scores_path)
    receipt["scores_sha256"] = sha256_file(scores_path)
    receipt_path = output / f"{receipt['bank'].upper()}_SCORE_RECEIPT.json"
    atomic_write_json(receipt_path, receipt)
    return scores_path, receipt_path
