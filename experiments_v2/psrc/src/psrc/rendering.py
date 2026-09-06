from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage, signal

from .utils import stable_seed


@dataclass(frozen=True)
class Lens:
    theta_e_arcsec: float
    q: float
    phi_rad: float
    shear_g1: float
    shear_g2: float
    x0_arcsec: float
    y0_arcsec: float


def _grid(
    height: int, width: int, pixel_scale_arcsec: float
) -> tuple[np.ndarray, np.ndarray]:
    y = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * pixel_scale_arcsec
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * pixel_scale_arcsec
    return np.meshgrid(x, y)


def sie_deflection(
    x: np.ndarray, y: np.ndarray, lens: Lens, eps: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Kormann-style SIE deflection with an explicit SIS limit."""
    if not (0 < lens.q <= 1):
        raise ValueError("SIE axis ratio must be in (0, 1]")
    dx = x - lens.x0_arcsec
    dy = y - lens.y0_arcsec
    cosine = math.cos(lens.phi_rad)
    sine = math.sin(lens.phi_rad)
    xp = cosine * dx + sine * dy
    yp = -sine * dx + cosine * dy
    if abs(lens.q - 1.0) < 1e-8:
        radius = np.sqrt(xp * xp + yp * yp + eps)
        axp = lens.theta_e_arcsec * xp / radius
        ayp = lens.theta_e_arcsec * yp / radius
    else:
        q = lens.q
        flattening = math.sqrt(1.0 - q * q)
        omega = np.sqrt(q * q * xp * xp + yp * yp + eps * eps)
        scale = lens.theta_e_arcsec * math.sqrt(q) / flattening
        axp = scale * np.arctan(flattening * xp / np.maximum(omega, eps))
        argument = np.clip(
            flattening * yp / np.maximum(omega, eps), -0.999999, 0.999999
        )
        ayp = scale * np.arctanh(argument)
    return (
        cosine * axp - sine * ayp,
        sine * axp + cosine * ayp,
    )


def gaussian_kernel(
    fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    *,
    truncate_sigma: float,
) -> np.ndarray:
    if fwhm_arcsec <= 0 or pixel_scale_arcsec <= 0:
        raise ValueError("PSF FWHM and pixel scale must be positive")
    sigma_pixels = fwhm_arcsec / (2.354820045 * pixel_scale_arcsec)
    radius = max(1, int(math.ceil(truncate_sigma * sigma_pixels)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma_pixels**2))
    kernel /= kernel.sum(dtype=np.float64)
    return kernel


def zero_padded_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear FFT convolution with an explicit zero margin and exact crop."""
    values = np.asarray(image, dtype=np.float64)
    psf = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 2 or psf.ndim != 2:
        raise ValueError("image and PSF kernel must be 2D")
    if not np.isfinite(values).all() or not np.isfinite(psf).all():
        raise ValueError("image or PSF kernel contains non-finite values")
    if np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("PSF kernel must be nonnegative with positive sum")
    psf = psf / psf.sum(dtype=np.float64)
    pad_y = psf.shape[0] // 2
    pad_x = psf.shape[1] // 2
    padded = np.pad(values, ((pad_y, pad_y), (pad_x, pad_x)))
    convolved = signal.fftconvolve(padded, psf, mode="same")
    return convolved[
        pad_y : pad_y + values.shape[0],
        pad_x : pad_x + values.shape[1],
    ]


def _normalized_source(source: np.ndarray, pixel_scale: float) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("source must be a finite nonnegative 2D map")
    integral = float(values.sum(dtype=np.float64) * pixel_scale**2)
    if integral <= 0:
        raise ValueError("source has zero integral")
    return values / integral


def raytrace_source(
    source: np.ndarray,
    *,
    source_pixel_scale_arcsec: float,
    lens: Lens,
    source_center_x_arcsec: float,
    source_center_y_arcsec: float,
    target_shape: tuple[int, int],
    target_pixel_scale_arcsec: float,
    oversample: int,
) -> np.ndarray:
    """Ray-trace unit-integral source surface brightness on a fine image grid."""
    if oversample < 1:
        raise ValueError("oversample must be positive")
    normalized = _normalized_source(source, source_pixel_scale_arcsec)
    height, width = target_shape
    fine_scale = target_pixel_scale_arcsec / oversample
    x, y = _grid(height * oversample, width * oversample, fine_scale)
    alpha_x, alpha_y = sie_deflection(x, y, lens)
    shear_x = lens.shear_g1 * x + lens.shear_g2 * y
    shear_y = lens.shear_g2 * x - lens.shear_g1 * y
    beta_x = x - alpha_x - shear_x - float(source_center_x_arcsec)
    beta_y = y - alpha_y - shear_y - float(source_center_y_arcsec)
    source_y = beta_y / source_pixel_scale_arcsec + (normalized.shape[0] - 1) / 2.0
    source_x = beta_x / source_pixel_scale_arcsec + (normalized.shape[1] - 1) / 2.0
    sampled = ndimage.map_coordinates(
        normalized,
        [source_y, source_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.maximum(sampled, 0.0)


def render_expectation(
    source: np.ndarray,
    *,
    source_pixel_scale_arcsec: float,
    intrinsic_flux_nmgy_grz: tuple[float, float, float],
    lens: Lens,
    source_center_x_arcsec: float,
    source_center_y_arcsec: float,
    target_shape: tuple[int, int],
    target_pixel_scale_arcsec: float,
    psf_kernels_grz: tuple[np.ndarray, np.ndarray, np.ndarray],
    psf_convention: Literal["optical_pre_pixel", "effective_post_pixel"],
    oversample: int,
) -> np.ndarray:
    """Return the deterministic expected lensed source flux in HWC nanomaggies."""
    if len(intrinsic_flux_nmgy_grz) != 3 or any(
        (not np.isfinite(value) or value <= 0) for value in intrinsic_flux_nmgy_grz
    ):
        raise ValueError("intrinsic g/r/z fluxes must be finite and positive")
    fine = raytrace_source(
        source,
        source_pixel_scale_arcsec=source_pixel_scale_arcsec,
        lens=lens,
        source_center_x_arcsec=source_center_x_arcsec,
        source_center_y_arcsec=source_center_y_arcsec,
        target_shape=target_shape,
        target_pixel_scale_arcsec=target_pixel_scale_arcsec,
        oversample=oversample,
    )
    height, width = target_shape
    area = target_pixel_scale_arcsec**2
    bands = []
    for flux, kernel in zip(intrinsic_flux_nmgy_grz, psf_kernels_grz, strict=True):
        if psf_convention == "optical_pre_pixel":
            filtered = zero_padded_convolution(fine * float(flux), kernel)
            native_sb = filtered.reshape(height, oversample, width, oversample).mean(
                axis=(1, 3)
            )
            band = native_sb * area
        elif psf_convention == "effective_post_pixel":
            native_sb = fine.reshape(height, oversample, width, oversample).mean(
                axis=(1, 3)
            ) * float(flux)
            band = zero_padded_convolution(native_sb * area, kernel)
        else:
            raise ValueError(f"unsupported PSF convention: {psf_convention}")
        bands.append(np.maximum(band, 0.0))
    return np.stack(bands, axis=-1)


def gaussian_psf_kernels(
    *,
    fwhm_grz_arcsec: tuple[float, float, float],
    target_pixel_scale_arcsec: float,
    oversample: int,
    convention: Literal["optical_pre_pixel", "effective_post_pixel"],
    truncate_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = (
        target_pixel_scale_arcsec / oversample
        if convention == "optical_pre_pixel"
        else target_pixel_scale_arcsec
    )
    return tuple(
        gaussian_kernel(
            float(fwhm),
            scale,
            truncate_sigma=truncate_sigma,
        )
        for fwhm in fwhm_grz_arcsec
    )  # type: ignore[return-value]


def source_only_lensed_r_match(
    structured: np.ndarray, sersic: np.ndarray
) -> tuple[np.ndarray, float]:
    """Named secondary only: preserve SED while matching noiseless lensed r flux."""
    flux_a = float(np.asarray(structured, dtype=np.float64)[..., 1].sum())
    flux_b = float(np.asarray(sersic, dtype=np.float64)[..., 1].sum())
    if flux_a <= 0 or flux_b <= 0:
        raise ValueError("cannot match non-positive lensed r-band flux")
    scale = flux_a / flux_b
    return np.asarray(sersic, dtype=np.float64) * scale, float(scale)


def coupled_photon_draw(
    structured_expectation_nmgy: np.ndarray,
    sersic_expectation_nmgy: np.ndarray,
    *,
    gain_e_per_nmgy: float,
    noise_key_base: str,
    draw_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Draw exact Poisson marginals with a common-component coupling.

    At each pixel, ``Poisson(min(lambda_a, lambda_b))`` is shared. Independent
    Poisson increments supply each arm's excess intensity. Equal means therefore
    produce identical counts, while unequal means retain exact arm marginals.
    """
    a = np.asarray(structured_expectation_nmgy, dtype=np.float64)
    b = np.asarray(sersic_expectation_nmgy, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 3
        or a.shape[-1] != 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(a < 0)
        or np.any(b < 0)
        or gain_e_per_nmgy <= 0
    ):
        raise ValueError("invalid expected source flux or gain")
    lambda_a = a * gain_e_per_nmgy
    lambda_b = b * gain_e_per_nmgy
    common_lambda = np.minimum(lambda_a, lambda_b)
    namespace = "P_src_v2_photon"
    shared_key = f"{noise_key_base}:draw-{draw_index}:shared"
    a_key = f"{noise_key_base}:draw-{draw_index}:structured-extra"
    b_key = f"{noise_key_base}:draw-{draw_index}:sersic-extra"
    shared = np.random.default_rng(stable_seed(namespace, shared_key)).poisson(
        common_lambda
    )
    extra_a = np.random.default_rng(stable_seed(namespace, a_key)).poisson(
        lambda_a - common_lambda
    )
    extra_b = np.random.default_rng(stable_seed(namespace, b_key)).poisson(
        lambda_b - common_lambda
    )
    return (
        (shared + extra_a) / gain_e_per_nmgy,
        (shared + extra_b) / gain_e_per_nmgy,
        {
            "shared_noise_key": shared_key,
            "structured_extra_noise_key": a_key,
            "sersic_extra_noise_key": b_key,
        },
    )
