"""Feature encoders used by the grouped domain and paired-source probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision.models import efficientnet_v2_s, resnet50

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def lupton_rgb(
    image_grz: np.ndarray, *, m: float = 0.03, q: float = 20.0
) -> np.ndarray:
    """Apply the Legacy Survey/Stein ``dr2_rgb`` channel mapping."""
    image = np.asarray(image_grz, dtype=np.float32).copy()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"unexpected cutout shape: {image.shape}")
    scales_grz = np.array([6.0, 3.4, 2.2], dtype=np.float32)
    affine_grz = image * scales_grz + m
    intensity = np.maximum(0.0, affine_grz[:, :, 0])
    intensity += np.maximum(0.0, affine_grz[:, :, 1])
    intensity += np.maximum(0.0, affine_grz[:, :, 2])
    intensity /= 3.0
    mapped = np.arcsinh(q * intensity) / np.sqrt(q)
    safe = intensity + (intensity == 0.0) * 1e-6
    image_rgb = affine_grz[:, :, [2, 1, 0]] * (mapped / safe)[:, :, None]
    return np.clip(image_rgb, 0.0, 1.0).astype(np.float32)


def preprocess_imagenet(images_hwc: np.ndarray) -> torch.Tensor:
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


def preprocess_stein(images_hwc: np.ndarray) -> torch.Tensor:
    arrays = np.asarray(images_hwc, dtype=np.float32)
    if arrays.ndim != 4 or arrays.shape[1:] != (101, 101, 3):
        raise ValueError(f"unexpected image batch shape: {arrays.shape}")
    cropped = arrays[:, 2:98, 2:98, :]
    mapped = np.stack([lupton_rgb(image) for image in cropped])
    return torch.from_numpy(np.transpose(mapped, (0, 3, 1, 2))).float()


def load_imagenet_encoder(checkpoint: Path) -> tuple[nn.Module, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = efficientnet_v2_s(weights=None)
    model.load_state_dict(state, strict=True)
    model.classifier = nn.Identity()
    return model, 1280


def load_stein_encoder(checkpoint: Path) -> tuple[nn.Module, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    prefixes = ("encoder_q.", "model.encoder_q.", "moco.encoder_q.")
    selected: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix) and not key[len(prefix) :].startswith("fc.")
        }
        if selected:
            break
    if not selected:
        raise ValueError("Stein checkpoint contains no recognized encoder weights")
    model = resnet50(weights=None)
    incompatible = model.load_state_dict(selected, strict=False)
    if set(incompatible.missing_keys) - {"fc.weight", "fc.bias"}:
        raise ValueError(f"Stein encoder missing weights: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected Stein weights: {incompatible.unexpected_keys}")
    model.fc = nn.Identity()
    return model, 2048


class PrimaryFeatureEncoder(nn.Module):
    """Penultimate 1,280-component representation of EfficientNetV2-S."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.features(inputs)
        output = self.avgpool(output)
        return torch.flatten(output, 1)


def extract_batches(
    model: nn.Module,
    images: np.ndarray,
    *,
    preprocess,
    device: str,
    batch_size: int = 64,
    preprocess_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    model = model.to(device).eval()
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = preprocess(batch, **(preprocess_kwargs or {})).to(device)
            features.append(model(tensor).detach().cpu().numpy())
    return np.concatenate(features).astype(np.float32)
