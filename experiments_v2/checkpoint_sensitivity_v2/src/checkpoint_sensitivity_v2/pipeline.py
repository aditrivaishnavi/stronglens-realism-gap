from __future__ import annotations

import itertools
import json
import os
import platform
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from evaluation_endpoints_v2.pipeline import (
    _images as endpoint_images,
)
from evaluation_endpoints_v2.pipeline import (
    _result_summary as endpoint_result_summary,
)
from evaluation_endpoints_v2.pipeline import (
    _validate_manifest as validate_endpoint_manifest,
)
from evaluation_endpoints_v2.pipeline import (
    load_endpoint_config,
)
from evaluation_endpoints_v2.pipeline import (
    validate_seal as validate_endpoint_seal,
)
from grec_pnoise.backend import backend_profile
from grec_pnoise.contracts import (
    array_bytes_sha256,
    load_json,
    sha256_file,
)
from grec_pnoise.contracts import (
    config_provenance as grec_config_provenance,
)
from grec_pnoise.contracts import (
    load_config as load_grec_config,
)
from grec_pnoise.inference import analyse_grec, analyse_pnoise
from grec_pnoise.scoring import (
    _import_fixed_modules,
    _raw_images,
    _verify_bank_receipt,
    validate_scoring_authority,
)
from grec_pnoise.threshold import threshold_evidence
from psrc.authority import validate_execution_lock
from psrc.contracts import (
    config_provenance as psrc_config_provenance,
)
from psrc.contracts import (
    load_config as load_psrc_config,
)
from psrc.inference import (
    global_spatial_source_bootstrap,
    photon_average,
    photon_mcse,
    point_estimate,
    precision_status,
    validate_and_join_scores,
    validate_native_score_table,
)
from psrc.scoring import _photon_images

POPULATIONS = (
    "calibration",
    "achieved_fpr",
    "terminal_confirmed",
    "grec",
    "pnoise",
    "psrc",
)


class SensitivityError(RuntimeError):
    """A checkpoint-panel or frozen-input contract was violated."""


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SensitivityError(f"{path} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(
            value,
            handle,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def load_panel(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    panel = _read_json(config_path)
    if (
        panel.get("schema_version") != 1
        or panel.get("status") != "complete_final_protocol_checkpoint_panel"
        or panel.get("study_id")
        not in {
            "final_detector_checkpoint_sensitivity_v2",
            "best_validation_checkpoint_sensitivity_v2",
        }
    ):
        raise SensitivityError("checkpoint panel identity differs")
    models = panel.get("checkpoints")
    if not isinstance(models, list) or len(models) != 12:
        raise SensitivityError("checkpoint panel must contain exactly 12 models")
    model_ids = [str(item.get("model_id", "")) for item in models]
    if any(not value for value in model_ids) or len(set(model_ids)) != len(model_ids):
        raise SensitivityError("checkpoint model IDs must be unique and nonblank")
    primary = str(panel.get("primary_model_id", ""))
    if model_ids.count(primary) != 1:
        raise SensitivityError("panel must identify exactly one primary model")
    if sum(bool(item.get("primary")) for item in models) != 1:
        raise SensitivityError("panel primary flags are inconsistent")
    if not next(item for item in models if item["model_id"] == primary)["primary"]:
        raise SensitivityError("primary model ID and primary flag disagree")
    panel["__config_path__"] = str(config_path)
    panel["__config_sha256__"] = sha256_file(config_path)
    return panel, config_path


def _root(panel: Mapping[str, Any]) -> Path:
    root = Path(str(panel["repository_root"])).resolve()
    if not root.is_dir():
        raise SensitivityError(f"repository root is missing: {root}")
    return root


def _input(panel: Mapping[str, Any], name: str) -> Path:
    path = _root(panel) / str(panel["inputs"][name])
    if not path.is_file():
        raise SensitivityError(f"required input is missing: {path}")
    return path


def _source_binding_names(population: str) -> tuple[str, ...]:
    shared = ("grec_config", "common_threshold_receipt")
    names = {
        "calibration": (
            *shared,
            "calibration_manifest",
            "calibration_manifest_receipt",
            "calibration_primary_scores",
            "calibration_primary_score_receipt",
        ),
        "achieved_fpr": (
            *shared,
            "endpoint_config",
            "achieved_fpr_manifest",
            "achieved_fpr_manifest_receipt",
            "achieved_fpr_primary_scores",
            "achieved_fpr_primary_score_receipt",
            "achieved_fpr_primary_result",
        ),
        "terminal_confirmed": (
            *shared,
            "endpoint_config",
            "terminal_manifest",
            "terminal_manifest_receipt",
            "terminal_primary_scores",
            "terminal_primary_score_receipt",
            "terminal_primary_result",
        ),
        "grec": (
            *shared,
            "grec_manifest",
            "grec_receipt",
            "grec_primary_scores",
            "grec_primary_score_receipt",
            "grec_primary_result",
        ),
        "pnoise": (
            *shared,
            "grec_manifest",
            "grec_primary_scores",
            "pnoise_manifest",
            "pnoise_receipt",
            "pnoise_primary_scores",
            "pnoise_primary_score_receipt",
            "pnoise_primary_result",
        ),
        "psrc": (
            "grec_config",
            "common_threshold_receipt",
            "psrc_config",
            "psrc_design",
            "psrc_design_receipt",
            "psrc_draw_count_receipt",
            "psrc_execution_lock",
            "psrc_manifest",
            "psrc_receipt",
            "psrc_primary_scores",
            "psrc_primary_score_receipt",
            "psrc_primary_result",
        ),
    }
    return names[population]


def _source_bindings(panel: Mapping[str, Any], population: str) -> dict[str, str]:
    return {
        name: sha256_file(_input(panel, name))
        for name in _source_binding_names(population)
    }


def _checkpoint_path(panel: Mapping[str, Any], spec: Mapping[str, Any]) -> Path:
    path = Path(str(panel["checkpoint_root"])) / str(spec["relative_path"])
    if not path.is_file():
        raise SensitivityError(f"checkpoint is missing: {path}")
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise SensitivityError(
            f"checkpoint hash differs for {spec['model_id']}: {observed}"
        )
    return path


def _validate_checkpoint_metadata(
    path: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    expected_epoch = int(spec.get("checkpoint_epoch", 160))
    train = checkpoint.get("train", {}) or {}
    dataset = checkpoint.get("dataset", {}) or {}
    architecture = str(train.get("arch", ""))
    if epoch != expected_epoch or architecture != "efficientnet_v2_s":
        raise SensitivityError(
            f"{spec['model_id']} has epoch/architecture {epoch}/{architecture}"
        )
    expected_mixture = {
        "candidate_plus_tier_b": "tier_ab_historical",
        "candidate_only": "tier_a_candidates",
    }[str(spec["training_mixture"])]
    expected_balanced = str(spec["sampler"]) == "balanced_50_50"
    observed = {
        "mixture_id": str(train.get("mixture_id", "")),
        "seed": int(dataset.get("seed", -1)),
        "balanced_sampling": bool(train.get("balanced_sampling", False)),
        "schedule": str(train.get("lr_schedule", "")),
    }
    expected = {
        "mixture_id": expected_mixture,
        "seed": int(spec["seed"]),
        "balanced_sampling": expected_balanced,
        "schedule": str(spec["schedule"]),
    }
    if observed != expected:
        raise SensitivityError(
            f"{spec['model_id']} scientific metadata differs: {observed} != {expected}"
        )
    return {**observed, "epoch": epoch, "architecture": architecture}


def validate_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in panel["checkpoints"]:
        path = _checkpoint_path(panel, spec)
        observed_metadata = _validate_checkpoint_metadata(path, spec)
        rows.append(
            {
                "model_id": spec["model_id"],
                "checkpoint_path": str(path),
                "checkpoint_sha256": spec["sha256"],
                "bytes": path.stat().st_size,
                "epoch": observed_metadata["epoch"],
                "architecture": "efficientnet_v2_s",
                "scientific_metadata": observed_metadata,
            }
        )
    if int(panel["batch_size"]) != 64 or str(panel["device"]) != "mps":
        raise SensitivityError("final panel requires batch size 64 on MPS")
    return {
        "status": "checkpoint_panel_validated",
        "models": rows,
        "panel_config_sha256": panel["__config_sha256__"],
    }


def _load_models(
    panel: Mapping[str, Any],
    grec_config: Mapping[str, Any],
) -> tuple[dict[str, torch.nn.Module], dict[str, Any], Any, Any]:
    fixed, preprocess, utilities = _import_fixed_modules(grec_config)
    models: dict[str, torch.nn.Module] = {}
    common_preprocess: dict[str, Any] | None = None
    device = str(panel["device"])
    fixed.pin_deterministic_kernels()
    for spec in panel["checkpoints"]:
        path = _checkpoint_path(panel, spec)
        model, preprocess_kwargs = utilities.load_model_and_spec(str(path), device)
        if common_preprocess is None:
            common_preprocess = dict(preprocess_kwargs)
        elif preprocess_kwargs != common_preprocess:
            raise SensitivityError(
                f"preprocessing differs for checkpoint {spec['model_id']}"
            )
        model = model.to(device).eval()
        fixed.assert_no_batch_coupling(model)
        models[str(spec["model_id"])] = model
    if common_preprocess is None:
        raise SensitivityError("checkpoint panel is empty")
    return models, common_preprocess, fixed, preprocess


def _chunked(values: Iterable[np.ndarray], size: int) -> Iterable[list[np.ndarray]]:
    iterator = iter(values)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _score_images(
    images: Iterable[np.ndarray],
    *,
    rows: int,
    models: Mapping[str, torch.nn.Module],
    preprocess_kwargs: Mapping[str, Any],
    fixed: Any,
    preprocess: Any,
    device: str,
    batch_size: int,
    chunk_rows: int,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if chunk_rows < batch_size or chunk_rows % batch_size:
        raise SensitivityError("chunk rows must be a positive batch-size multiple")
    chunks: dict[str, list[np.ndarray]] = {model_id: [] for model_id in models}
    seen = 0
    started = time.monotonic()
    terminal_rows: int | None = None
    for raw_chunk in _chunked(images, chunk_rows):
        processed = [
            preprocess.preprocess_stack(np.asarray(image), **preprocess_kwargs)
            for image in raw_chunk
        ]
        cpu_batches = list(fixed.array_batches(processed, batch_size))
        device_batches = [batch.to(device) for batch in cpu_batches]
        if len(raw_chunk) % batch_size:
            terminal_rows = len(raw_chunk) % batch_size
        for model_id, model in models.items():
            logits64, _ = fixed.fixed_width_logits(
                model,
                device_batches,
                batch_size=batch_size,
                device=device,
                filler="last",
            )
            logits = logits64.astype(np.float32)
            if not np.array_equal(logits64, logits.astype(np.float64)):
                raise SensitivityError(
                    f"{model_id} produced a non-float32 transcription"
                )
            chunks[model_id].append(logits)
        seen += len(raw_chunk)
        elapsed = max(time.monotonic() - started, 1e-9)
        print(
            f"{label}: {seen}/{rows} raw images; "
            f"{seen / elapsed:.1f} images/s through {len(models)} models",
            flush=True,
        )
    if seen != rows:
        raise SensitivityError(f"{label} yielded {seen} images, expected {rows}")
    combined = {
        model_id: np.concatenate(values).astype(np.float32, copy=False)
        for model_id, values in chunks.items()
    }
    if any(values.shape != (rows,) for values in combined.values()):
        raise SensitivityError(f"{label} panel score coverage differs")
    return combined, {
        "rows": rows,
        "models": len(models),
        "batch_size": batch_size,
        "chunk_rows": chunk_rows,
        "terminal_batch_real_rows": terminal_rows,
        "elapsed_seconds": time.monotonic() - started,
        "raw_images_read_once_for_complete_panel": True,
        "preprocessing_applied_once_per_image": True,
        "model_forward_passes_per_image": len(models),
    }


def _calibration_images(
    manifest: pd.DataFrame,
) -> Iterable[np.ndarray]:
    required = {
        "calibration_id",
        "global_role",
        "input_pixel_path",
        "input_artifact_sha256",
        "input_pixel_sha256",
    }
    if (
        required - set(manifest)
        or not manifest["global_role"].eq("threshold_calibration").all()
    ):
        raise SensitivityError("calibration manifest differs")
    for row in manifest.itertuples(index=False):
        path = Path(str(row.input_pixel_path))
        if sha256_file(path) != str(row.input_artifact_sha256):
            raise SensitivityError(f"calibration artifact drift: {path}")
        with np.load(path, allow_pickle=False) as payload:
            image = np.array(payload["image_chw"], copy=True)
        if (
            image.dtype != np.float32
            or image.shape != (3, 101, 101)
            or not np.isfinite(image).all()
            or array_bytes_sha256(image) != str(row.input_pixel_sha256)
        ):
            raise SensitivityError(f"calibration pixels differ: {path}")
        yield image


def _validate_calibration(
    panel: Mapping[str, Any],
    grec_config: Mapping[str, Any],
) -> pd.DataFrame:
    manifest_path = _input(panel, "calibration_manifest")
    receipt_path = _input(panel, "calibration_manifest_receipt")
    receipt = load_json(receipt_path)
    if (
        receipt.get("status")
        != grec_config["threshold"]["required_calibration_manifest_status"]
        or receipt.get("calibration_manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("config") != grec_config_provenance(grec_config)
    ):
        raise SensitivityError("calibration manifest authority differs")
    frame = pd.read_parquet(manifest_path)
    if len(frame) != 20_002 or frame["calibration_id"].duplicated().any():
        raise SensitivityError("calibration support differs")
    return frame


def _endpoint_population(
    panel: Mapping[str, Any], role: str
) -> tuple[pd.DataFrame, Iterable[np.ndarray], dict[str, Any]]:
    config_path = _input(panel, "endpoint_config")
    endpoint_config, endpoint_root = load_endpoint_config(config_path)
    validate_endpoint_seal(config_path)
    if role == "achieved_fpr":
        manifest_path = _input(panel, "achieved_fpr_manifest")
        receipt_path = _input(panel, "achieved_fpr_manifest_receipt")
    else:
        manifest_path = _input(panel, "terminal_manifest")
        receipt_path = _input(panel, "terminal_manifest_receipt")
    frame, _ = validate_endpoint_manifest(
        endpoint_config, endpoint_root, role, manifest_path, receipt_path
    )
    return frame, endpoint_images(endpoint_root, frame), endpoint_config


def _grec_population(
    panel: Mapping[str, Any],
    grec_config: Mapping[str, Any],
    bank: str,
) -> tuple[pd.DataFrame, Iterable[np.ndarray]]:
    if bank == "grec":
        manifest_path = _input(panel, "grec_manifest")
        receipt_path = _input(panel, "grec_receipt")
    else:
        manifest_path = _input(panel, "pnoise_manifest")
        receipt_path = _input(panel, "pnoise_receipt")
    _verify_bank_receipt(
        manifest_path,
        receipt_path,
        bank,
        grec_config,
        _root(panel) / "results_v2/evaluation_v2/COMMON_THRESHOLD_RECEIPT.json",
    )
    frame = pd.read_parquet(manifest_path)
    return frame, _raw_images(frame, bank)


def _psrc_population(
    panel: Mapping[str, Any],
) -> tuple[pd.DataFrame, Iterable[np.ndarray], dict[str, Any]]:
    config = load_psrc_config(_input(panel, "psrc_config"))
    lock_path = _input(panel, "psrc_execution_lock")
    validate_execution_lock(lock_path, config)
    validate_and_join_scores(
        design_path=_input(panel, "psrc_design"),
        design_receipt_path=_input(panel, "psrc_design_receipt"),
        photon_manifest_path=_input(panel, "psrc_manifest"),
        photon_receipt_path=_input(panel, "psrc_receipt"),
        scores_path=_input(panel, "psrc_primary_scores"),
        score_receipt_path=_input(panel, "psrc_primary_score_receipt"),
        draw_count_receipt_path=_input(panel, "psrc_draw_count_receipt"),
        execution_lock_path=lock_path,
        config=config,
    )
    manifest = pd.read_parquet(_input(panel, "psrc_manifest")).sort_values(
        ["pair_id", "draw_index"], kind="mergesort", ignore_index=True
    )
    images = _photon_images(manifest, _input(panel, "psrc_manifest").parent)
    expanded = manifest.loc[manifest.index.repeat(2)].reset_index(drop=True)
    expanded["arm"] = np.tile(["structured", "sersic"], len(manifest))
    return expanded, images, config


def _population(
    panel: Mapping[str, Any],
    name: str,
    grec_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Iterable[np.ndarray], dict[str, Any]]:
    if name == "calibration":
        frame = _validate_calibration(panel, grec_config)
        return frame[["calibration_id"]].copy(), _calibration_images(frame), {}
    if name in {"achieved_fpr", "terminal_confirmed"}:
        frame, images, endpoint_config = _endpoint_population(panel, name)
        identity = frame[
            [
                "endpoint_id",
                "galaxy_id",
                "global_role",
                "spatial_group_id",
                "physical_group_id",
            ]
        ].copy()
        return identity, images, {"endpoint_config": endpoint_config}
    if name in {"grec", "pnoise"}:
        frame, images = _grec_population(panel, grec_config, name)
        identity = "grec_row_id" if name == "grec" else "pnoise_row_id"
        return frame[[identity]].copy(), images, {"manifest": frame}
    if name == "psrc":
        frame, images, psrc_config = _psrc_population(panel)
        identity = frame[
            ["pair_id", "rendered_pair_row_id", "draw_index", "arm"]
        ].copy()
        return identity, images, {"psrc_config": psrc_config}
    raise SensitivityError(f"unknown population: {name}")


def _primary_expected(
    panel: Mapping[str, Any], population: str, identities: pd.DataFrame
) -> np.ndarray:
    if population == "calibration":
        source = pd.read_parquet(_input(panel, "calibration_primary_scores"))
        joined = identities.merge(source, on="calibration_id", validate="one_to_one")
        return joined["raw_logit_float32"].to_numpy(dtype=np.float32)
    if population in {"achieved_fpr", "terminal_confirmed"}:
        key = (
            "achieved_fpr_primary_scores"
            if population == "achieved_fpr"
            else "terminal_primary_scores"
        )
        source = pd.read_parquet(_input(panel, key))
        joined = identities[["endpoint_id"]].merge(
            source[["endpoint_id", "raw_logit_float32"]],
            on="endpoint_id",
            validate="one_to_one",
        )
        return joined["raw_logit_float32"].to_numpy(dtype=np.float32)
    if population in {"grec", "pnoise"}:
        identity = "grec_row_id" if population == "grec" else "pnoise_row_id"
        source = pd.read_parquet(_input(panel, f"{population}_primary_scores"))
        joined = identities.merge(
            source[[identity, "raw_logit_float32"]],
            on=identity,
            validate="one_to_one",
        )
        return joined["raw_logit_float32"].to_numpy(dtype=np.float32)
    source = pd.read_parquet(_input(panel, "psrc_primary_scores")).sort_values(
        ["pair_id", "draw_index"], kind="mergesort", ignore_index=True
    )
    expected = np.empty(2 * len(source), dtype=np.float32)
    expected[0::2] = source["structured_raw_logit_float32"].to_numpy(dtype=np.float32)
    expected[1::2] = source["sersic_raw_logit_float32"].to_numpy(dtype=np.float32)
    return expected


def _validate_primary_reference(panel: Mapping[str, Any], population: str) -> None:
    score_name = {
        "calibration": "calibration_primary_scores",
        "achieved_fpr": "achieved_fpr_primary_scores",
        "terminal_confirmed": "terminal_primary_scores",
        "grec": "grec_primary_scores",
        "pnoise": "pnoise_primary_scores",
        "psrc": "psrc_primary_scores",
    }[population]
    receipt_name = {
        "calibration": "calibration_primary_score_receipt",
        "achieved_fpr": "achieved_fpr_primary_score_receipt",
        "terminal_confirmed": "terminal_primary_score_receipt",
        "grec": "grec_primary_score_receipt",
        "pnoise": "pnoise_primary_score_receipt",
        "psrc": "psrc_primary_score_receipt",
    }[population]
    score_path = _input(panel, score_name)
    receipt = _read_json(_input(panel, receipt_name))
    if receipt.get(
        "scores_sha256", receipt.get("calibration_logits_sha256")
    ) != sha256_file(score_path):
        raise SensitivityError(f"{population} primary score receipt hash differs")
    primary_spec = next(
        item
        for item in panel["checkpoints"]
        if item["model_id"] == panel["primary_model_id"]
    )
    checkpoint_sha256 = receipt.get(
        "checkpoint_sha256",
        (receipt.get("scorer_closure") or {}).get("checkpoint_sha256"),
    )
    if checkpoint_sha256 != primary_spec["sha256"]:
        raise SensitivityError(f"{population} primary score receipt checkpoint differs")
    manifest_name = {
        "calibration": "calibration_manifest",
        "achieved_fpr": "achieved_fpr_manifest",
        "terminal_confirmed": "terminal_manifest",
        "grec": "grec_manifest",
        "pnoise": "pnoise_manifest",
        "psrc": "psrc_manifest",
    }[population]
    observed_manifest = receipt.get(
        "calibration_manifest_sha256",
        receipt.get(
            "manifest_sha256",
            receipt.get("bank_manifest_sha256", receipt.get("photon_manifest_sha256")),
        ),
    )
    if observed_manifest != sha256_file(_input(panel, manifest_name)):
        raise SensitivityError(f"{population} primary score receipt manifest differs")


def score_population(
    config_path: str | Path,
    population: str,
    output_root: str | Path,
    *,
    chunk_rows: int = 1024,
) -> dict[str, Any]:
    if population not in POPULATIONS:
        raise SensitivityError(f"unknown population: {population}")
    panel, _ = load_panel(config_path)
    require_primary_reproduction = bool(panel.get("require_primary_reproduction", True))
    output = Path(output_root).resolve() / population
    if output.exists():
        receipt_path = output / "PANEL_SCORE_RECEIPT.json"
        scores_path = output / "panel_logits.parquet"
        if receipt_path.is_file() and scores_path.is_file():
            receipt = _read_json(receipt_path)
            if (
                receipt.get("population") == population
                and receipt.get("scores_sha256") == sha256_file(scores_path)
                and receipt.get("panel_config_sha256") == panel["__config_sha256__"]
                and (
                    not require_primary_reproduction
                    or receipt.get("primary_bit_reproduction_passed") is True
                )
            ):
                print(f"{population}: already complete; skipping", flush=True)
                return receipt
        raise FileExistsError(f"incomplete or incompatible output: {output}")
    validation = validate_panel(panel)
    root = _root(panel)
    os.chdir(root)
    grec_config = load_grec_config(_input(panel, "grec_config"))
    if int(panel["batch_size"]) != int(grec_config["scoring"]["batch_size"]) or str(
        panel["device"]
    ) != str(grec_config["scoring"]["authoritative_device"]):
        raise SensitivityError("panel scorer differs from final scoring contract")
    authority_receipt, _ = validate_scoring_authority(
        grec_config,
        _input(panel, "common_threshold_receipt"),
        device=str(panel["device"]),
    )
    if require_primary_reproduction:
        _validate_primary_reference(panel, population)
    source_bindings = _source_bindings(panel, population)
    identities, images, authority = _population(panel, population, grec_config)
    models, preprocess_kwargs, fixed, preprocess = _load_models(panel, grec_config)
    scores, batch_record = _score_images(
        images,
        rows=len(identities),
        models=models,
        preprocess_kwargs=preprocess_kwargs,
        fixed=fixed,
        preprocess=preprocess,
        device=str(panel["device"]),
        batch_size=int(panel["batch_size"]),
        chunk_rows=chunk_rows,
        label=population,
    )
    primary = str(panel["primary_model_id"])
    if require_primary_reproduction:
        expected = _primary_expected(panel, population, identities)
        if not np.array_equal(scores[primary], expected):
            differing = int(
                np.count_nonzero(
                    scores[primary].view(np.uint32) != expected.view(np.uint32)
                )
            )
            raise SensitivityError(
                f"{population} primary bit reproduction failed for {differing} rows"
            )
    frame = identities.copy()
    for model_id, values in scores.items():
        frame[f"{model_id}__raw_logit_float32"] = values
        frame[f"{model_id}__raw_logit_float32_bits"] = values.view(np.uint32)
    scores_path = output / "panel_logits.parquet"
    _atomic_parquet(scores_path, frame)
    authority_record = {
        key: (psrc_config_provenance(value) if key == "psrc_config" else value)
        for key, value in authority.items()
        if key != "manifest" and key != "endpoint_config"
    }
    receipt = {
        "schema_version": 1,
        "status": "checkpoint_panel_population_scored",
        "population": population,
        "rows": len(frame),
        "panel_config_sha256": panel["__config_sha256__"],
        "checkpoint_validation": validation,
        "checkpoints": {
            item["model_id"]: item["sha256"] for item in panel["checkpoints"]
        },
        "scores_sha256": sha256_file(scores_path),
        "primary_model_id": primary,
        "primary_reproduction_required": require_primary_reproduction,
        "primary_bit_reproduction_passed": (
            True if require_primary_reproduction else None
        ),
        "primary_expected_scores_sha256": (
            sha256_file(
                _input(
                    panel,
                    {
                        "calibration": "calibration_primary_scores",
                        "achieved_fpr": "achieved_fpr_primary_scores",
                        "terminal_confirmed": "terminal_primary_scores",
                        "grec": "grec_primary_scores",
                        "pnoise": "pnoise_primary_scores",
                        "psrc": "psrc_primary_scores",
                    }[population],
                )
            )
            if require_primary_reproduction
            else None
        ),
        "source_bindings": source_bindings,
        "source_manifest_sha256": sha256_file(
            _input(
                panel,
                {
                    "calibration": "calibration_manifest",
                    "achieved_fpr": "achieved_fpr_manifest",
                    "terminal_confirmed": "terminal_manifest",
                    "grec": "grec_manifest",
                    "pnoise": "pnoise_manifest",
                    "psrc": "psrc_manifest",
                }[population],
            )
        ),
        "batch_record": batch_record,
        "preprocessing": preprocess_kwargs,
        "device": panel["device"],
        "batch_size": panel["batch_size"],
        "primary_authoritative_backend_profile": authority_receipt["backend_profile"],
        "observed_backend_profile": backend_profile(grec_config, str(panel["device"])),
        "rerendered": False,
        "retrained": False,
        "authority": authority_record,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    _atomic_json(output / "PANEL_SCORE_RECEIPT.json", receipt)
    return receipt


def _score_column(model_id: str) -> str:
    return f"{model_id}__raw_logit_float32"


def _thresholds(
    panel: Mapping[str, Any], output_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, np.float32]]:
    calibration = pd.read_parquet(output_root / "calibration/panel_logits.parquet")
    records: dict[str, dict[str, Any]] = {}
    values: dict[str, np.float32] = {}
    for spec in panel["checkpoints"]:
        model_id = str(spec["model_id"])
        evidence, threshold = threshold_evidence(
            calibration[_score_column(model_id)].to_numpy(copy=True),
            float(panel["alpha"]),
        )
        records[model_id] = {
            **evidence,
            "checkpoint_sha256": spec["sha256"],
        }
        values[model_id] = threshold
    return records, values


def _endpoint_analysis(
    panel: Mapping[str, Any],
    output_root: Path,
    model_id: str,
    threshold: np.float32,
    role: str,
) -> dict[str, Any]:
    frame = pd.read_parquet(output_root / role / "panel_logits.parquet")
    scores = frame[
        [
            "endpoint_id",
            "galaxy_id",
            "global_role",
            "spatial_group_id",
            "physical_group_id",
        ]
    ].copy()
    values = frame[_score_column(model_id)].to_numpy(copy=True)
    scores["raw_logit_float32"] = values
    scores["raw_logit_float32_bits"] = values.view(np.uint32)
    scores["detected_at_frozen_threshold"] = values >= threshold
    endpoint_config, _ = load_endpoint_config(_input(panel, "endpoint_config"))
    return endpoint_result_summary(endpoint_config, role, scores)


def _grec_scores(
    output_root: Path,
    model_id: str,
    threshold: np.float32,
    population: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(output_root / population / "panel_logits.parquet")
    identity = "grec_row_id" if population == "grec" else "pnoise_row_id"
    values = frame[_score_column(model_id)].to_numpy(copy=True)
    return pd.DataFrame(
        {
            identity: frame[identity].astype(str),
            "raw_logit_float32": values,
            "raw_logit_float32_bits": values.view(np.uint32),
            "detected_at_frozen_threshold": values >= threshold,
        }
    )


def _psrc_result(
    panel: Mapping[str, Any],
    output_root: Path,
    model_id: str,
    threshold: np.float32,
) -> dict[str, Any]:
    wide = pd.read_parquet(output_root / "psrc/panel_logits.parquet")
    structured = wide[wide["arm"].eq("structured")].copy()
    sersic = wide[wide["arm"].eq("sersic")].copy()
    identity = ["pair_id", "rendered_pair_row_id", "draw_index"]
    if len(structured) != len(sersic):
        raise SensitivityError("P_src arm score counts differ")
    merged = structured[identity + [_score_column(model_id)]].merge(
        sersic[identity + [_score_column(model_id)]],
        on=identity,
        validate="one_to_one",
        suffixes=("_structured", "_sersic"),
    )
    structured_values = merged[f"{_score_column(model_id)}_structured"].to_numpy(
        dtype=np.float32
    )
    sersic_values = merged[f"{_score_column(model_id)}_sersic"].to_numpy(
        dtype=np.float32
    )
    scores = merged[identity].copy()
    scores["structured_raw_logit_float32"] = structured_values
    scores["structured_raw_logit_float32_bits"] = structured_values.view(np.uint32)
    scores["sersic_raw_logit_float32"] = sersic_values
    scores["sersic_raw_logit_float32_bits"] = sersic_values.view(np.uint32)
    scores["structured_detected"] = structured_values >= threshold
    scores["sersic_detected"] = sersic_values >= threshold
    photon = pd.read_parquet(_input(panel, "psrc_manifest")).sort_values(
        ["pair_id", "draw_index"], kind="mergesort", ignore_index=True
    )
    draw_receipt = _read_json(_input(panel, "psrc_draw_count_receipt"))
    draws = int(draw_receipt["draw_count"])
    validate_native_score_table(scores, photon, draw_count=draws, threshold=threshold)
    design = pd.read_parquet(_input(panel, "psrc_design"))
    joined = photon.merge(scores, on=identity, validate="one_to_one").merge(
        design[
            [
                "pair_id",
                "parent_grec_scene_id",
                "host_galaxy_id",
                "host_spatial_group_id",
                "host_cell_id",
                "theta_e_arcsec",
                "source_r_mag",
                "source_id",
                "source_stratum",
                "source_analysis_weight",
            ]
        ],
        on=[
            "pair_id",
            "host_galaxy_id",
            "host_spatial_group_id",
            "host_cell_id",
            "source_id",
            "source_stratum",
        ],
        validate="many_to_one",
    )
    averaged = photon_average(joined, draws)
    config = load_psrc_config(_input(panel, "psrc_config"))
    estimate = point_estimate(averaged)
    bootstrap, bootstrap_design = global_spatial_source_bootstrap(
        averaged,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed_namespace=str(config["inference"]["bootstrap_seed_namespace"]),
        interval_deviation_inflation=float(
            config["inference"]["interval_deviation_inflation"]
        ),
    )
    confidence = float(config["inference"]["confidence_level"])
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(bootstrap, [tail, 1.0 - tail])
    mcse = photon_mcse(joined)
    target = float(config["photon_noise"]["paired_mcse_target_absolute_risk"])
    status, passed = precision_status(mcse, target)
    return {
        "status": status,
        "primary_inference_complete": passed,
        "primary_estimand": config["inference"]["primary_estimand"],
        "risk_difference_sersic_minus_structured": estimate,
        "threshold_conditional_95_interval": list(map(float, interval)),
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "threshold_conditioning": (
            "Point and interval condition on this checkpoint's threshold "
            "calibrated from the common final calibration population."
        ),
        "photon_mcse": mcse,
        "photon_mcse_target": target,
        "photon_mcse_passed": passed,
        "structured_detections": int(scores["structured_detected"].sum()),
        "sersic_detections": int(scores["sersic_detected"].sum()),
        "draw_count": draws,
        "bootstrap_design": bootstrap_design,
        "equivalence_claim_permitted": False,
        "method_limit": config["inference"]["method_limitation"],
    }


def _validate_primary_result_parity(
    panel: Mapping[str, Any], model_result: Mapping[str, Any]
) -> None:
    comparisons = {
        "achieved_fpr": _read_json(_input(panel, "achieved_fpr_primary_result")),
        "terminal_confirmed": _read_json(_input(panel, "terminal_primary_result")),
        "grec": _read_json(_input(panel, "grec_primary_result"))["grec"],
        "pnoise": _read_json(_input(panel, "pnoise_primary_result"))["pnoise"],
    }
    for name, expected in comparisons.items():
        if model_result[name] != expected:
            raise SensitivityError(f"primary {name} result parity failed")
    expected_psrc = _read_json(_input(panel, "psrc_primary_result"))
    for field in (
        "status",
        "primary_inference_complete",
        "risk_difference_sersic_minus_structured",
        "threshold_conditional_95_interval",
        "photon_mcse",
        "photon_mcse_target",
        "photon_mcse_passed",
        "bootstrap_design",
    ):
        if model_result["psrc"].get(field) != expected_psrc.get(field):
            raise SensitivityError(f"primary P_src result parity failed at {field}")


def analyze_panel(
    config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    panel, _ = load_panel(config_path)
    validate_panel(panel)
    require_primary_reproduction = bool(panel.get("require_primary_reproduction", True))
    require_primary_reproduction = bool(panel.get("require_primary_reproduction", True))
    output = Path(output_root).resolve()
    grec_config = load_grec_config(_input(panel, "grec_config"))
    validate_scoring_authority(
        grec_config,
        _input(panel, "common_threshold_receipt"),
        device=str(panel["device"]),
    )
    for population in POPULATIONS:
        receipt = _read_json(output / population / "PANEL_SCORE_RECEIPT.json")
        scores_path = output / population / "panel_logits.parquet"
        _population(panel, population, grec_config)
        if (
            receipt.get("population") != population
            or receipt.get("scores_sha256") != sha256_file(scores_path)
            or receipt.get("panel_config_sha256") != panel["__config_sha256__"]
            or (
                require_primary_reproduction
                and receipt.get("primary_bit_reproduction_passed") is not True
            )
            or receipt.get("source_bindings") != _source_bindings(panel, population)
        ):
            raise SensitivityError(f"{population} score receipt differs")
    threshold_records, thresholds = _thresholds(panel, output)
    common_threshold = _read_json(_input(panel, "common_threshold_receipt"))
    primary_id = str(panel["primary_model_id"])
    if float(panel["alpha"]) != float(grec_config["threshold"]["alpha"]):
        raise SensitivityError("panel alpha differs from final threshold authority")
    if require_primary_reproduction and threshold_records[primary_id][
        "logit_threshold_float32_bits"
    ] != int(common_threshold["logit_threshold_float32_bits"]):
        raise SensitivityError("primary threshold does not reproduce final authority")
    _atomic_json(
        output / "MODEL_SPECIFIC_THRESHOLDS.json",
        {
            "schema_version": 1,
            "status": "model_specific_equal_calibration_fpr_thresholds",
            "panel_config_sha256": panel["__config_sha256__"],
            "alpha": panel["alpha"],
            "models": threshold_records,
        },
    )
    grec_manifest = pd.read_parquet(_input(panel, "grec_manifest"))
    pnoise_manifest = pd.read_parquet(_input(panel, "pnoise_manifest"))
    model_results = []
    for spec in panel["checkpoints"]:
        model_id = str(spec["model_id"])
        threshold = thresholds[model_id]
        grec_scores = _grec_scores(output, model_id, threshold, "grec")
        pnoise_scores = _grec_scores(output, model_id, threshold, "pnoise")
        result = {
            "model": dict(spec),
            "threshold": threshold_records[model_id],
            "achieved_fpr": _endpoint_analysis(
                panel, output, model_id, threshold, "achieved_fpr"
            ),
            "terminal_confirmed": _endpoint_analysis(
                panel, output, model_id, threshold, "terminal_confirmed"
            ),
            "grec": analyse_grec(grec_manifest, grec_scores, grec_config),
            "pnoise": analyse_pnoise(
                grec_manifest,
                grec_scores,
                pnoise_manifest,
                pnoise_scores,
                grec_config,
                expected_draw_count=8,
            ),
            "psrc": _psrc_result(panel, output, model_id, threshold),
        }
        model_results.append(result)
        print(f"analysis complete: {model_id}", flush=True)
    if require_primary_reproduction:
        primary_result = next(
            item
            for item in model_results
            if item["model"]["model_id"] == panel["primary_model_id"]
        )
        _validate_primary_result_parity(panel, primary_result)
    score_bindings = {
        population: {
            "scores_sha256": sha256_file(output / population / "panel_logits.parquet"),
            "receipt_sha256": sha256_file(
                output / population / "PANEL_SCORE_RECEIPT.json"
            ),
        }
        for population in POPULATIONS
    }
    final = {
        "schema_version": 1,
        "status": "final_checkpoint_sensitivity_complete",
        "panel_config_sha256": panel["__config_sha256__"],
        "primary_model_id": panel["primary_model_id"],
        "models": model_results,
        "score_bindings": score_bindings,
        "thresholds_sha256": sha256_file(output / "MODEL_SPECIFIC_THRESHOLDS.json"),
        "rerendered": False,
        "retrained": False,
        "gdomain_rerun": False,
        "gmatch_rerun": False,
        "primary_reproduction_required": require_primary_reproduction,
        "primary_threshold_parity_passed": (
            True if require_primary_reproduction else None
        ),
        "primary_result_parity_passed": (
            True if require_primary_reproduction else None
        ),
        "interpretation": (
            "Complete correlated checkpoint panel under the final fixed pixel "
            "populations and model-specific equal-calibration-FPR thresholds."
        ),
    }
    _atomic_json(output / "FINAL_CHECKPOINT_SENSITIVITY_RESULTS.json", final)
    return final


def verify_outputs(
    config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    panel, _ = load_panel(config_path)
    validate_panel(panel)
    require_primary_reproduction = bool(panel.get("require_primary_reproduction", True))
    output = Path(output_root).resolve()
    final_path = output / "FINAL_CHECKPOINT_SENSITIVITY_RESULTS.json"
    final = _read_json(final_path)
    if (
        final.get("status") != "final_checkpoint_sensitivity_complete"
        or final.get("panel_config_sha256") != panel["__config_sha256__"]
        or len(final.get("models", [])) != 12
        or (
            require_primary_reproduction
            and (
                final.get("primary_threshold_parity_passed") is not True
                or final.get("primary_result_parity_passed") is not True
            )
        )
    ):
        raise SensitivityError("final sensitivity result identity differs")
    checks = {}
    for population in POPULATIONS:
        scores_path = output / population / "panel_logits.parquet"
        receipt_path = output / population / "PANEL_SCORE_RECEIPT.json"
        receipt = _read_json(receipt_path)
        frame = pd.read_parquet(scores_path)
        model_columns = [
            _score_column(item["model_id"]) for item in panel["checkpoints"]
        ]
        bit_columns = [f"{column}_bits" for column in model_columns]
        if any(column not in frame for column in [*model_columns, *bit_columns]):
            raise SensitivityError(f"{population} score columns are incomplete")
        for value_column, bits_column in zip(model_columns, bit_columns, strict=True):
            values = frame[value_column].to_numpy(copy=True)
            bits = frame[bits_column].to_numpy(copy=True)
            if (
                values.dtype != np.float32
                or bits.dtype != np.uint32
                or not np.array_equal(values.view(np.uint32), bits)
            ):
                raise SensitivityError(
                    f"{population} native score bits differ for {value_column}"
                )
        if receipt.get("scores_sha256") != sha256_file(scores_path) or (
            require_primary_reproduction
            and receipt.get("primary_bit_reproduction_passed") is not True
        ):
            raise SensitivityError(f"{population} receipt differs")
        checks[population] = {
            "rows": len(frame),
            "scores_sha256": sha256_file(scores_path),
            "receipt_sha256": sha256_file(receipt_path),
            "native_score_bits_verified": True,
        }
    if final.get("score_bindings") != {
        population: {
            "scores_sha256": checks[population]["scores_sha256"],
            "receipt_sha256": checks[population]["receipt_sha256"],
        }
        for population in POPULATIONS
    }:
        raise SensitivityError("final score bindings differ")
    receipt = {
        "schema_version": 1,
        "status": "checkpoint_sensitivity_structurally_rehashed",
        "panel_config_sha256": panel["__config_sha256__"],
        "final_results_sha256": sha256_file(final_path),
        "populations": checks,
        "scope": (
            "Structural rehash and native-bit validation only; independent "
            "scientific recomputation is separate."
        ),
    }
    _atomic_json(output / "STRUCTURAL_REHASH_VERIFICATION.json", receipt)
    return receipt
