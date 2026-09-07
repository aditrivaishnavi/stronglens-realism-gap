"""Actual frozen-representation extraction from hash-bound image payloads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .authorities import AuthorityContext
from .authorization import ExecutionAuthorization
from .contracts import (
    ContractError,
    atomic_write_json,
    canonical_frame_sha256,
    require_columns,
    require_sha256_column,
    sha256_file,
)
from .features import _sequence_sha256, representation_spec

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def lupton_rgb(
    image_grz: np.ndarray, *, m: float = 0.03, q: float = 20.0
) -> np.ndarray:
    """Apply the Legacy Survey/Stein ``dr2_rgb`` channel mapping."""
    image = np.asarray(image_grz, dtype=np.float32).copy()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ContractError(f"unexpected image shape {image.shape}")
    scales_grz = np.asarray([6.0, 3.4, 2.2], dtype=np.float32)
    affine_grz = image * scales_grz + m
    intensity = np.maximum(0.0, affine_grz[:, :, 0])
    intensity += np.maximum(0.0, affine_grz[:, :, 1])
    intensity += np.maximum(0.0, affine_grz[:, :, 2])
    intensity /= 3.0
    mapped = np.arcsinh(q * intensity) / np.sqrt(q)
    safe = intensity + (intensity == 0.0) * 1e-6
    image_rgb = affine_grz[:, :, [2, 1, 0]] * (mapped / safe)[:, :, None]
    return np.clip(image_rgb, 0.0, 1.0).astype(np.float32)


def pixel_statistics(images_hwc: np.ndarray) -> np.ndarray:
    """Six fixed statistics per band, in band-major order."""
    features = []
    for image in np.asarray(images_hwc, dtype=np.float32):
        if image.shape != (101, 101, 3) or not np.isfinite(image).all():
            raise ContractError("pixel-statistic input is invalid")
        row: list[float] = []
        for band in range(3):
            values = image[:, :, band].astype(np.float64)
            row.extend(
                [
                    values.mean(),
                    values.std(),
                    float(np.median(values)),
                    float(np.percentile(values, 99)),
                    values.min(),
                    values.max(),
                ]
            )
        features.append(row)
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (len(images_hwc), 18):
        raise ContractError("pixel-statistic dimension is not 18")
    return result


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.nn import functional
        from torchvision.models import efficientnet_v2_s, resnet50
    except ImportError as error:
        raise ContractError(
            "torch and torchvision are required for encoder extraction"
        ) from error
    return torch, nn, functional, efficientnet_v2_s, resnet50


def _preprocess_stein(images_hwc: np.ndarray):
    torch, _, _, _, _ = _torch_modules()
    cropped = np.asarray(images_hwc, dtype=np.float32)[:, 2:98, 2:98, :]
    mapped = np.stack([lupton_rgb(image) for image in cropped])
    return torch.from_numpy(np.transpose(mapped, (0, 3, 1, 2))).float()


def _preprocess_imagenet(images_hwc: np.ndarray):
    torch, _, functional, _, _ = _torch_modules()
    arrays = [np.transpose(lupton_rgb(image), (2, 0, 1)) for image in images_hwc]
    tensor = torch.from_numpy(np.stack(arrays)).float()
    tensor = functional.interpolate(
        tensor,
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.from_numpy(IMAGENET_MEAN)[None, :, None, None]
    std = torch.from_numpy(IMAGENET_STD)[None, :, None, None]
    return (tensor - mean) / std


def _load_encoder(representation_id: str, checkpoint_path: Path):
    torch, nn, _, efficientnet_v2_s, resnet50 = _torch_modules()
    if representation_id == "imagenet_effnet_v2_s":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = efficientnet_v2_s(weights=None)
        model.load_state_dict(state, strict=True)
        model.classifier = nn.Identity()
        return model, _preprocess_imagenet
    if representation_id == "stein_ssl_resnet50":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        selected: dict[str, Any] = {}
        for prefix in ("encoder_q.", "model.encoder_q.", "moco.encoder_q."):
            selected = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix) and not key[len(prefix) :].startswith("fc.")
            }
            if selected:
                break
        if not selected:
            raise ContractError("Stein checkpoint has no recognized encoder weights")
        model = resnet50(weights=None)
        incompatible = model.load_state_dict(selected, strict=False)
        if set(incompatible.missing_keys) - {"fc.weight", "fc.bias"}:
            raise ContractError("Stein checkpoint is missing encoder weights")
        if incompatible.unexpected_keys:
            raise ContractError("Stein checkpoint has unexpected encoder weights")
        model.fc = nn.Identity()
        return model, _preprocess_stein
    raise ContractError(f"unsupported encoder {representation_id}")


def _safe_payload_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ContractError(f"unsafe pixel path {relative}")
    path = root.joinpath(*value.parts)
    if not path.is_file():
        raise ContractError(f"pixel payload is missing: {relative}")
    return path


def _array_bytes_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _load_image(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_pixel_sha256: str | None,
    array_key: str,
    layout: str,
) -> np.ndarray:
    if sha256_file(path) != expected_file_sha256:
        raise ContractError(f"pixel payload hash differs: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if array_key not in payload:
            raise ContractError(f"pixel payload lacks {array_key}: {path}")
        native = np.asarray(payload[array_key])
    if (
        expected_pixel_sha256 is not None
        and _array_bytes_sha256(native) != expected_pixel_sha256
    ):
        raise ContractError(f"pixel array hash differs: {path}")
    if layout == "CHW":
        image = np.transpose(native, (1, 2, 0))
    elif layout == "HWC":
        image = native
    else:
        raise ContractError(f"unsupported pixel layout {layout}")
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (101, 101, 3) or not np.isfinite(image).all():
        raise ContractError(f"invalid cutout payload: {path}")
    return image


def extract_feature_artifact(
    *,
    representation_id: str,
    manifest_path: Path,
    pixel_root: Path,
    checkpoint_path: Path | None,
    output_path: Path,
    receipt_path: Path,
    config: dict[str, Any],
    authority: AuthorityContext,
    authorization: ExecutionAuthorization,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Extract one ordered artifact; final inputs remain blocked by authorization."""
    authority.assert_config(config)
    if output_path.exists() or receipt_path.exists():
        raise ContractError("feature extraction refuses to overwrite outputs")
    if not 1 <= batch_size <= 256:
        raise ContractError("feature extraction batch size must be 1..256")
    manifest_sha256 = sha256_file(manifest_path)
    authorization.revalidate(
        config=config,
        authority=authority,
        stage="feature_extract",
        manifest_sha256=manifest_sha256,
    )
    if (
        authorization.data_role == "final"
        and config["execution"].get("feature_extraction_authorized") is not True
    ):
        raise ContractError("final feature extraction is not authorized")
    manifest = pd.read_parquet(manifest_path)
    require_columns(
        manifest,
        {"row_id", "pixel_sha256", "pixel_relpath", "authority_digest", "data_role"},
        context="feature extraction manifest",
    )
    require_sha256_column(
        manifest, "pixel_sha256", context="feature extraction manifest"
    )
    if not manifest["data_role"].eq(authorization.data_role).all():
        raise ContractError("feature extraction manifest has the wrong data role")
    if not manifest["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("feature extraction manifest has the wrong authority")
    specification = representation_spec(config, representation_id)
    if specification["checkpoint_sha256"] is None:
        if checkpoint_path is not None:
            raise ContractError("pixel statistics cannot use a checkpoint")
    else:
        if checkpoint_path is None:
            raise ContractError("encoder checkpoint is required")
        if sha256_file(checkpoint_path) != specification["checkpoint_sha256"]:
            raise ContractError("encoder checkpoint bytes have the wrong SHA-256")
    package_root = Path(__file__).resolve().parents[1]
    implementation_path = package_root / specification["extractor_code_path"]
    implementation_sha256 = sha256_file(implementation_path)
    if (
        implementation_sha256 != specification["extractor_code_sha256"]
        or implementation_sha256 != specification["preprocessing_code_sha256"]
    ):
        raise ContractError("extractor/preprocessing code differs from config")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimension = int(specification["feature_dimension"])
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.npy")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(manifest), dimension),
    )
    model = preprocess = None
    if representation_id != "pixel_statistics":
        assert checkpoint_path is not None
        model, preprocess = _load_encoder(representation_id, checkpoint_path)
        torch, _, _, _, _ = _torch_modules()
        runtime_device = torch.device(device)
        model = model.to(runtime_device).eval()
    for start in range(0, len(manifest), batch_size):
        stop = min(start + batch_size, len(manifest))
        rows = manifest.iloc[start:stop]
        images = np.stack(
            [
                _load_image(
                    _safe_payload_path(pixel_root, str(row.pixel_relpath)),
                    expected_file_sha256=str(
                        getattr(row, "pixel_file_sha256", row.pixel_sha256)
                    ),
                    expected_pixel_sha256=(
                        str(row.pixel_sha256)
                        if hasattr(row, "pixel_file_sha256")
                        else None
                    ),
                    array_key=str(getattr(row, "pixel_array_key", "cutout")),
                    layout=str(getattr(row, "pixel_layout", "HWC")),
                )
                for row in rows.itertuples(index=False)
            ]
        )
        if representation_id == "pixel_statistics":
            encoded = pixel_statistics(images)
        else:
            assert model is not None and preprocess is not None
            tensor = preprocess(images).to(runtime_device)
            with torch.no_grad():
                encoded = model(tensor).detach().cpu().numpy().astype(np.float32)
        if encoded.shape != (len(rows), dimension):
            raise ContractError("encoder produced the wrong feature dimension")
        output[start:stop] = encoded
    output.flush()
    del output
    os.replace(temporary, output_path)
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "data_role": authorization.data_role,
        "representation_id": representation_id,
        "rows": len(manifest),
        "feature_dimension": dimension,
        "dtype": "float32",
        "features_file_sha256": sha256_file(output_path),
        "analysis_manifest_sha256": manifest_sha256,
        "analysis_manifest_content_sha256": canonical_frame_sha256(
            manifest,
            columns=manifest.columns,
            sort_by=["row_id"],
        ),
        "row_ids_sha256": _sequence_sha256(manifest["row_id"]),
        "pixel_hashes_sha256": _sequence_sha256(manifest["pixel_sha256"]),
        "authority_digest": authority.authority_digest,
        "authorization_receipt_sha256": authorization.receipt_sha256,
        "checkpoint_sha256": specification["checkpoint_sha256"],
        "preprocessing_contract_sha256": specification["preprocessing_contract_sha256"],
        "extraction_contract_sha256": specification["extraction_contract_sha256"],
        "extractor_code_sha256": implementation_sha256,
        "preprocessing_code_sha256": implementation_sha256,
        "batch_size": batch_size,
        "device": device,
    }
    atomic_write_json(receipt, receipt_path)
    return receipt
