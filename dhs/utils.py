from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

# The reported detector was trained with annulus (20, 32), so evaluation
# retains that preprocessing even though extended light can enter the annulus.


def decode_npz_blob(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob)) as z:
        return {k: z[k] for k in z.files}


def center_slices(H: int, W: int, box: int):
    cy, cx = H // 2, W // 2
    half = box // 2
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def radial_rmap(H: int, W: int) -> np.ndarray:
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def radial_mask(H: int, W: int, r_in: float, r_out: float) -> np.ndarray:
    r = radial_rmap(H, W)
    return (r >= r_in) & (r < r_out)


def robust_median_mad(x: np.ndarray, eps: float = 1e-8):
    # Use nanmedian to handle NaN values
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med))) + eps
    # Handle case where all values are NaN
    if np.isnan(med):
        med = 0.0
    if np.isnan(mad) or mad < eps:
        mad = eps
    return med, mad


def default_annulus_radii(H: int, W: int) -> tuple[float, float]:
    """Compute default annulus radii appropriate for the image size.

    The annulus should sit in the sky-dominated outer region of the stamp.
    For a stamp of half-size R = min(H, W) // 2:
      r_in = 0.65 * R   (inner edge at 65% of half-size)
      r_out = 0.90 * R   (outer edge at 90% of half-size)

    This helper is available for newly trained models. It is not used to
    rescore the reported detector.
    """
    R = min(H, W) // 2
    r_in = 0.65 * R
    r_out = 0.90 * R
    return r_in, r_out


# Module-level flag: emit the annulus-mismatch warning at most once.
_warned_annulus = False


def normalize_outer_annulus(
    img: np.ndarray, r_in: float = 20, r_out: float = 32
) -> np.ndarray:
    """Normalize a 2D image by the median/MAD of an outer annulus.

    The defaults (20, 32) match the reported detector. A newly trained model
    may instead pass ``default_annulus_radii(H, W)`` explicitly.

    Raises
    ------
    ValueError
        If r_in >= r_out or annulus has fewer than 100 pixels.
    """
    global _warned_annulus  # noqa: PLW0603

    # --- Validation ---
    if r_in >= r_out:
        raise ValueError(
            f"r_in ({r_in}) must be strictly less than r_out ({r_out}). "
            "Check that both annulus_r_in and annulus_r_out are set correctly."
        )

    H, W = img.shape
    m = radial_mask(H, W, r_in, r_out)
    n_pixels = int(np.sum(m))
    if n_pixels < 100:
        raise ValueError(
            f"Annulus [{r_in}, {r_out}] on {H}x{W} image has only {n_pixels} pixels "
            f"(need >= 100 for reliable median/MAD). Check radii."
        )

    # Warn once if the annulus is likely not sky-dominated
    if not _warned_annulus:
        half = min(H, W) // 2
        if half > 0 and r_out / half < 0.70 and H > 64:
            logger.warning(
                "Annulus r_out=%g is only %.0f%% of image half-size %d. "
                "For %dx%d stamps, consider using default_annulus_radii() "
                "when training a new model.",
                r_out,
                100 * r_out / half,
                half,
                H,
                W,
            )
            _warned_annulus = True

    masked_vals = img[m]
    # Filter NaN values for robust stats
    med, mad = robust_median_mad(masked_vals)
    return (img - med) / mad


def azimuthal_median_profile(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = np.zeros(r_max, dtype=np.float32)
    for ri in range(r_max):
        m = (r >= ri) & (r < ri + 1)
        prof[ri] = np.median(img[m]).astype(np.float32) if np.any(m) else 0.0
    return prof


def radial_profile_model(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = azimuthal_median_profile(img, r_max=r_max)
    rr = np.clip(np.floor(r).astype(int), 0, r_max - 1)
    return prof[rr].astype(np.float32)
