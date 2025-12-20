"""Scoring utilities: load model + preprocessing config from checkpoint.

Ensures that scoring scripts always use IDENTICAL preprocessing to
what the model was trained with. Eliminates the class of bugs where
scripts forget to pass annulus_r_in/r_out, clip_range, crop_size, etc.

Created: 2026-02-13

Usage in any scoring script:
    from dhs.scoring_utils import load_model_and_spec
    model, pp_kwargs = load_model_and_spec(ckpt_path, device)
    proc = preprocess_stack(img_chw, **pp_kwargs)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .model import build_model
from .preprocess_spec import PreprocessSpec


def load_model_and_spec(
    checkpoint_path: str,
    device: torch.device | str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model + preprocessing kwargs from a checkpoint.

    Returns
    -------
    model : nn.Module
        Model in eval mode on the requested device.
    pp_kwargs : dict
        Keyword arguments for preprocess_stack() that reproduce the
        exact preprocessing the model was trained with.

    Notes
    -----
    Checkpoint format priority:
      1. ``ckpt["preprocess_spec"]`` (new format, saved by updated train.py)
      2. ``ckpt["dataset"]`` (existing v1-v4 checkpoints — fallback)

    If neither key exists, returns default preprocessing kwargs
    (raw_robust, crop=False, clip_range=10.0, no annulus override).
    """
    if isinstance(device, str):
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # --- Build model ---
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    base_ch = train_cfg.get("base_ch", 16)
    model = build_model(arch, in_ch=3, pretrained=False, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- Extract preprocessing spec ---
    pp_dict = ckpt.get("preprocess_spec", None)
    if pp_dict is not None:
        # New checkpoint format
        spec = PreprocessSpec.from_dict(pp_dict)
        pp_kwargs = spec.to_preprocess_kwargs()
    else:
        # Fallback: extract from dataset config (v1-v4 checkpoints)
        ds = ckpt.get("dataset", {})
        pp_kwargs = {
            "mode": ds.get("preprocessing", "raw_robust"),
            "crop": ds.get("crop", False),
            "clip_range": 10.0,
        }
        crop_size = ds.get("crop_size", 0)
        if crop_size > 0:
            pp_kwargs["crop_size"] = crop_size
        r_in = ds.get("annulus_r_in", 0.0)
        r_out = ds.get("annulus_r_out", 0.0)
        if r_in > 0:
            pp_kwargs["annulus_r_in"] = r_in
        if r_out > 0:
            pp_kwargs["annulus_r_out"] = r_out

    return model, pp_kwargs
