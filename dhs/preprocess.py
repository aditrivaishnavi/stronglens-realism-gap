from __future__ import annotations
import numpy as np
from .utils import normalize_outer_annulus, radial_profile_model
from .constants import STAMP_SIZE


def center_crop(img: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop a 2D image to target_size x target_size.
    
    Pipeline generates 101x101 cutouts, training expects 64x64.
    This extracts the central region.
    """
    h, w = img.shape[-2:]
    if h == target_size and w == target_size:
        return img
    if h < target_size or w < target_size:
        raise ValueError(f"Image {h}x{w} smaller than target {target_size}x{target_size}")
    
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    if img.ndim == 2:
        return img[start_h:start_h + target_size, start_w:start_w + target_size]
    elif img.ndim == 3:
        return img[:, start_h:start_h + target_size, start_w:start_w + target_size]
    else:
        raise ValueError(f"Unexpected ndim={img.ndim}")


def preprocess_stack(img3: np.ndarray, mode: str, crop: bool = True, 
                     crop_size: int | None = None,
                     clip_range: float = 10.0,
                     annulus_r_in: float | None = None,
                     annulus_r_out: float | None = None) -> np.ndarray:
    """Preprocess a 3-band image stack.
    
    Args:
        img3: (3, H, W) array with g, r, z bands
        mode: Preprocessing mode ('raw_robust' or 'residual_radial_profile')
        crop: If True, center crop to crop_size (default STAMP_SIZE=64).
              For Paper IV parity, set crop=False to keep 101x101.
        crop_size: Target crop size. None defaults to STAMP_SIZE (64).
                   Set to 0 or None with crop=False to skip cropping.
        clip_range: Clip normalized values to [-clip_range, clip_range]
        annulus_r_in: Inner radius for normalization annulus. If None, uses
                      normalize_outer_annulus default (20 px, locked to trained models).
                      For retraining with corrected annulus, pass the output of
                      default_annulus_radii(H, W).
        annulus_r_out: Outer radius for normalization annulus. Same as above.
        
    Returns:
        Preprocessed (3, target_size, target_size) array
    """
    assert img3.ndim == 3 and img3.shape[0] == 3
    
    # Determine target crop size
    target = crop_size if crop_size is not None else STAMP_SIZE
    
    # Apply center crop first if needed (101x101 -> target)
    if crop and target > 0 and img3.shape[1] != target:
        img3 = center_crop(img3, target)
    
    # Validate annulus radii: must be set together (Q1.6 fix)
    if (annulus_r_in is None) != (annulus_r_out is None):
        raise ValueError(
            "annulus_r_in and annulus_r_out must both be set or both be None. "
            f"Got annulus_r_in={annulus_r_in}, annulus_r_out={annulus_r_out}"
        )
    if annulus_r_in is not None and annulus_r_out is not None:
        if annulus_r_in >= annulus_r_out:
            raise ValueError(
                f"annulus_r_in ({annulus_r_in}) must be < annulus_r_out ({annulus_r_out})"
            )

    # Build kwargs for normalize_outer_annulus
    annulus_kwargs = {}
    if annulus_r_in is not None:
        annulus_kwargs["r_in"] = annulus_r_in
    if annulus_r_out is not None:
        annulus_kwargs["r_out"] = annulus_r_out
    
    out = []
    for b in range(3):
        x = img3[b].astype(np.float32)
        
        # Handle NaN values - replace with 0 before normalization
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x = np.where(nan_mask, 0.0, x)
        
        if mode == "raw_robust":
            x = normalize_outer_annulus(x, **annulus_kwargs)
        elif mode == "residual_radial_profile":
            xn = normalize_outer_annulus(x, **annulus_kwargs)
            model = radial_profile_model(xn)
            x = (xn - model).astype(np.float32)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        
        # Final NaN check - set any remaining NaN to 0
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        
        # Clip extreme values to prevent gradient explosion
        x = np.clip(x, -clip_range, clip_range)
        
        out.append(x)
    return np.stack(out, axis=0).astype(np.float32)
