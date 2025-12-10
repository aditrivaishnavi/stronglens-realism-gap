from __future__ import annotations
import io
import logging
import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# KNOWN ISSUE (2026-02-13):
# The hardcoded annulus (20, 32) was tuned for 64x64 stamps. For 101x101
# stamps (crop=False, Paper IV parity), this annulus sits at 40-63% of
# the image half-width — inside the galaxy, not in the sky. This inflates
# MAD and suppresses arc contrast.
#
# default_annulus_radii() computes a proper outer-ring annulus for any
# image size. However, ALL EXISTING TRAINED MODELS were trained with
# (20, 32). Changing the default without retraining would make the model
# see inputs it was never trained on — making scores WRONG.
#
# PLAN: keep (20, 32) as the default until retraining is done with the
# corrected annulus. Use default_annulus_radii() in new training configs.
# -----------------------------------------------------------------------

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

    This ensures the annulus is in the outer ring regardless of stamp size:
      - 64x64:   r_in=20.8, r_out=28.8  (similar to old 20-32)
      - 101x101: r_in=32.5, r_out=45.0  (properly in outer ring)

    Previously hardcoded as (20, 32), which was appropriate for 64x64 stamps
    but NOT for 101x101 (the annulus sat at 40-63% of the half-size, well
    within galaxy light for bright extended hosts).
    """
    R = min(H, W) // 2
    r_in = 0.65 * R
    r_out = 0.90 * R
    return r_in, r_out


# Module-level flag: emit the annulus-mismatch warning at most once (Q1.4 fix).
_warned_annulus = False


def normalize_outer_annulus(img: np.ndarray, r_in: float = 20, r_out: float = 32) -> np.ndarray:
    """Normalize a 2D image by the median/MAD of an outer annulus.

    IMPORTANT: The defaults (20, 32) are LOCKED to match all existing
    trained models. Do NOT change these defaults without retraining.
    For new training runs on 101x101 stamps, pass the output of
    default_annulus_radii(H, W) explicitly.

    See KNOWN ISSUE comment at top of file for full context.

    Raises
    ------
    ValueError
        If r_in >= r_out or annulus has fewer than 100 pixels.
    """
    global _warned_annulus  # noqa: PLW0603

    # --- Validation (Q1.6 fix) ---
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

    # Warn once if the annulus is likely not sky-dominated (Q1.4 fix: warn-once)
    if not _warned_annulus:
        half = min(H, W) // 2
        if half > 0 and r_out / half < 0.70 and H > 64:
            logger.warning(
                "Annulus r_out=%g is only %.0f%% of image half-size %d. "
                "For %dx%d stamps, consider using default_annulus_radii() "
                "and retraining. See KNOWN ISSUE in dhs/utils.py.",
                r_out, 100 * r_out / half, half, H, W,
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
