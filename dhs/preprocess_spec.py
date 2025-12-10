"""PreprocessSpec: single source of truth for preprocessing parameters.

This dataclass captures ALL parameters that affect how an image is
preprocessed before being fed to the model. It is saved in training
checkpoints so that scoring scripts can automatically use the same
preprocessing the model was trained with — eliminating an entire class
of "script forgot an argument" bugs.

Created: 2026-02-13
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class PreprocessSpec:
    """Immutable specification for image preprocessing.

    Fields
    ------
    mode : str
        Preprocessing mode ('raw_robust' or 'residual_radial_profile').
    crop : bool
        Whether to center-crop to crop_size.
    crop_size : int
        Target crop size (0 = use STAMP_SIZE default of 64).
    clip_range : float
        Symmetric clipping range after normalization (default 10.0).
    annulus_r_in : float or None
        Inner radius for normalization annulus. None = use default (20 px).
    annulus_r_out : float or None
        Outer radius for normalization annulus. None = use default (32 px).
    """
    mode: str
    crop: bool
    crop_size: int
    clip_range: float = 10.0
    annulus_r_in: Optional[float] = None
    annulus_r_out: Optional[float] = None

    def __post_init__(self):
        # Validate annulus radii: must be set together
        if (self.annulus_r_in is None) != (self.annulus_r_out is None):
            raise ValueError(
                "annulus_r_in and annulus_r_out must both be set or both be None. "
                f"Got r_in={self.annulus_r_in}, r_out={self.annulus_r_out}"
            )
        if self.annulus_r_in is not None and self.annulus_r_out is not None:
            if self.annulus_r_in >= self.annulus_r_out:
                raise ValueError(
                    f"annulus_r_in ({self.annulus_r_in}) must be < "
                    f"annulus_r_out ({self.annulus_r_out})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (safe for torch.save / JSON)."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PreprocessSpec":
        """Deserialize from dict (loaded from checkpoint)."""
        return PreprocessSpec(**d)

    def to_preprocess_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs dict suitable for preprocess_stack().

        Only includes non-None annulus radii so that preprocess_stack
        uses its own defaults when annulus is not explicitly set.
        """
        kwargs: Dict[str, Any] = {
            "mode": self.mode,
            "crop": self.crop,
            "clip_range": self.clip_range,
        }
        if self.crop_size > 0:
            kwargs["crop_size"] = self.crop_size
        if self.annulus_r_in is not None:
            kwargs["annulus_r_in"] = self.annulus_r_in
        if self.annulus_r_out is not None:
            kwargs["annulus_r_out"] = self.annulus_r_out
        return kwargs
