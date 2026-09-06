from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .authority import resolve_binding, validate_execution_lock
from .contracts import (
    ContractError,
    config_provenance,
    fixed_scorer_closure,
    load_json,
    sha256_file,
    source_closure,
    validate_scorer_receipt,
)
from .threshold import validate_threshold_receipt
from .utils import atomic_json, safe_relative


def _import_fixed_modules(config: dict):
    root = Path(config["scoring"]["repo_root"]).resolve()
    loaded = sys.modules.get("dhs")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if root not in loaded_path.parents:
            raise ContractError(f"dhs already imported from {loaded_path}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    fixed = importlib.import_module("dhs.fixed_width_scoring")
    preprocess = importlib.import_module("dhs.preprocess")
    utilities = importlib.import_module("dhs.scoring_utils")
    if Path(fixed.__file__).resolve() != root / "dhs/fixed_width_scoring.py":
        raise ContractError("imported scorer is not the pinned repository file")
    return fixed, preprocess, utilities


def _native_float32_logits(
    images_chw: Iterable[np.ndarray],
    config: dict,
    *,
    scorer_receipt_path: str | Path,
    require_final_scorer: bool,
    device: str,
) -> tuple[np.ndarray, dict]:
    scorer_receipt = validate_scorer_receipt(
        scorer_receipt_path, config, require_final=require_final_scorer
    )
    if device != config["scoring"]["authoritative_device"]:
        raise ContractError("scoring device differs from fixed scorer receipt")
    fixed, preprocess, utilities = _import_fixed_modules(config)
    model, preprocess_kwargs = utilities.load_model_and_spec(
        config["scoring"]["checkpoint"], device
    )
    model = model.to(device).eval()
    fixed.pin_deterministic_kernels()
    decoupling = fixed.assert_no_batch_coupling(model)

    def processed() -> Iterable[np.ndarray]:
        for image in images_chw:
            value = np.asarray(image)
            if (
                value.dtype != np.float32
                or value.ndim != 3
                or value.shape[0] != 3
                or not np.isfinite(value).all()
            ):
                raise ContractError("scorer input must be finite native float32 CHW")
            yield preprocess.preprocess_stack(value, **preprocess_kwargs)

    width = int(config["scoring"]["batch_size"])
    transcribed64, batch = fixed.fixed_width_logits(
        model,
        fixed.array_batches(processed(), width),
        batch_size=width,
        device=device,
        filler=str(config["scoring"]["filler_policy"]),
    )
    if transcribed64.dtype != np.float64:
        raise ContractError("pinned scorer transcription dtype unexpectedly changed")
    logits = transcribed64.astype(np.float32)
    if not np.array_equal(transcribed64, logits.astype(np.float64)):
        raise ContractError(
            "scorer returned values not exactly representable as float32"
        )
    if not np.isfinite(logits).all():
        raise ContractError("scorer returned non-finite native logits")
    return logits, {
        "scorer_receipt_sha256": sha256_file(scorer_receipt_path),
        "scorer_closure_sha256": scorer_receipt["scorer_closure_sha256"],
        "native_model_dtype": "float32",
        "pinned_wrapper_transcription_dtype": "float64",
        "exact_float32_roundtrip_verified": True,
        "batch": batch,
        "batch_decoupling": decoupling,
    }


def score_design_only_canary(
    images_chw: np.ndarray,
    config: dict,
    *,
    scorer_receipt_path: str | Path,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Score declared design-only images without applying a threshold."""
    images = np.asarray(images_chw)
    if images.dtype != np.float32 or images.ndim != 4:
        raise ContractError("design canary must be NCHW native float32")
    logits, receipt = _native_float32_logits(
        iter(images),
        config,
        scorer_receipt_path=scorer_receipt_path,
        require_final_scorer=False,
        device=device,
    )
    reversed_logits, _ = _native_float32_logits(
        iter(images[::-1].copy()),
        config,
        scorer_receipt_path=scorer_receipt_path,
        require_final_scorer=False,
        device=device,
    )
    if not np.array_equal(logits, reversed_logits[::-1]):
        raise ContractError("raw-logit scoring changed under worker/input permutation")
    receipt.update(
        {
            "status": "design_only_raw_logit_canary",
            "final_outcomes_used": False,
            "threshold_applied": False,
            "worker_order_invariant": True,
        }
    )
    return logits, receipt


def _photon_images(
    photon_manifest: pd.DataFrame,
    photon_root: Path,
) -> Iterable[np.ndarray]:
    cache_path: str | None = None
    cache: dict[str, np.ndarray] = {}
    for row in photon_manifest.itertuples(index=False):
        path = str(safe_relative(photon_root, str(row.photon_pair_relpath)).resolve())
        if path != cache_path:
            if sha256_file(path) != str(row.photon_pair_sha256):
                raise ContractError("photon pair artifact drift")
            with np.load(path, allow_pickle=False) as payload:
                cache = {
                    "structured": np.array(payload["structured_cutouts"], copy=True),
                    "sersic": np.array(payload["sersic_cutouts"], copy=True),
                }
            if any(
                value.dtype != np.float32
                or value.ndim != 4
                or value.shape[1:] != (101, 101, 3)
                for value in cache.values()
            ):
                raise ContractError("photon bank arrays are not native float32 NHWC")
            cache_path = path
        draw = int(row.draw_index)
        yield np.transpose(cache["structured"][draw], (2, 0, 1))
        yield np.transpose(cache["sersic"][draw], (2, 0, 1))


def score_frozen_photon_bank(
    *,
    photon_manifest_path: str | Path,
    photon_receipt_path: str | Path,
    execution_lock_path: str | Path,
    config: dict,
    output_dir: Path,
    device: str,
) -> dict:
    """Final scorer implementation. This function was not executed here."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    lock = validate_execution_lock(execution_lock_path, config)
    scorer_path = resolve_binding(
        execution_lock_path, lock["bindings"]["scorer_receipt"]
    )
    threshold_path = resolve_binding(
        execution_lock_path, lock["bindings"]["threshold_receipt"]
    )
    threshold_receipt, threshold = validate_threshold_receipt(
        threshold_path, config, require_final=True
    )
    photon_receipt = load_json(photon_receipt_path)
    if (
        photon_receipt.get("status")
        != config["execution"]["required_photon_receipt_status"]
        or photon_receipt.get("manifest_sha256") != sha256_file(photon_manifest_path)
        or photon_receipt.get("execution_lock_sha256")
        != sha256_file(execution_lock_path)
        or photon_receipt.get("draw_count_receipt_sha256")
        != lock["bindings"]["draw_count_receipt"]["sha256"]
        or photon_receipt.get("config") != config_provenance(config)
        or photon_receipt.get("source_closure") != source_closure(config)
    ):
        raise ContractError("photon bank receipt is stale or unbound")
    manifest = pd.read_parquet(photon_manifest_path)
    draws = int(photon_receipt["draw_count"])
    expected_indices = set(range(draws))
    draw_sets = manifest.groupby("pair_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if (
        len(manifest) != photon_receipt["pairs"] * draws
        or manifest.duplicated(["pair_id", "draw_index"]).any()
        or not draw_sets.map(lambda value: value == expected_indices).all()
    ):
        raise ContractError("photon manifest lacks complete pair/draw coverage")
    manifest = manifest.sort_values(["pair_id", "draw_index"], kind="mergesort")
    logits, scoring = _native_float32_logits(
        _photon_images(manifest, Path(photon_manifest_path).resolve().parent),
        config,
        scorer_receipt_path=scorer_path,
        require_final_scorer=True,
        device=device,
    )
    if len(logits) != 2 * len(manifest):
        raise ContractError("scorer did not return exactly two arm logits per draw")
    structured = logits[0::2]
    sersic = logits[1::2]
    scores = manifest[["pair_id", "rendered_pair_row_id", "draw_index"]].copy()
    scores["structured_raw_logit_float32"] = structured
    scores["structured_raw_logit_float32_bits"] = structured.view(np.uint32)
    scores["sersic_raw_logit_float32"] = sersic
    scores["sersic_raw_logit_float32_bits"] = sersic.view(np.uint32)
    scores["structured_detected"] = structured >= threshold
    scores["sersic_detected"] = sersic >= threshold
    scores_path = output_dir / "scores.parquet"
    temporary_scores = output_dir / ".scores.parquet.partial"
    scores.to_parquet(temporary_scores, index=False)
    os.replace(temporary_scores, scores_path)
    receipt = {
        "schema_version": 2,
        "status": config["execution"]["required_score_receipt_status"],
        "rows": len(scores),
        "pairs": int(scores["pair_id"].nunique()),
        "draw_count": draws,
        "draw_indices": list(range(draws)),
        "scores_sha256": sha256_file(scores_path),
        "photon_manifest_sha256": sha256_file(photon_manifest_path),
        "photon_receipt_sha256": sha256_file(photon_receipt_path),
        "photon_receipt_status": photon_receipt["status"],
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "threshold_receipt_sha256": sha256_file(threshold_path),
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "comparison": ">=",
        "scorer_receipt_sha256": sha256_file(scorer_path),
        "scorer_closure": fixed_scorer_closure(config, verify_runtime=True),
        "scoring": scoring,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "SCORE_RECEIPT.json", receipt)
    return receipt
