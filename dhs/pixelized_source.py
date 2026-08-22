"""Ray-trace provenance-bound pixelized sources through the injection engine."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from dhs.injection_engine import (
    InjectionResult,
    LensParams,
    _fft_gaussian_blur,
    _shear_deflection,
    _sie_deflection,
    _torch_meshgrid_arcsec,
)


def inject_pixelized_source_sie(
    *,
    host_nmgy_hwc: torch.Tensor,
    source_surface_brightness: torch.Tensor,
    source_pixel_scale_arcsec: float,
    source_center_x_arcsec: float,
    source_center_y_arcsec: float,
    unlensed_flux_nmgy_grz: tuple[float, float, float],
    lens: LensParams,
    target_pixel_scale_arcsec: float,
    psf_fwhm_r_arcsec: float,
    seed: int,
    subpixel_oversample: int = 4,
    add_poisson_noise: bool = True,
    gain_e_per_nmgy: float = 150.0,
    psf_fwhm_scale_g: float = 1.05,
    psf_fwhm_scale_z: float = 0.94,
) -> InjectionResult:
    """Inject one normalized pixelized source with the standard SIE renderer.

    ``source_surface_brightness`` is a two-dimensional source-plane map. It is
    normalized internally so that its integral is one; both members of a
    morphology pair therefore receive exactly the same unlensed band fluxes.
    """
    if host_nmgy_hwc.ndim != 3 or host_nmgy_hwc.shape[-1] != 3:
        raise ValueError("Host cutout must be HWC with three bands")
    if source_surface_brightness.ndim != 2:
        raise ValueError("Pixelized source must be two-dimensional")
    if source_pixel_scale_arcsec <= 0 or target_pixel_scale_arcsec <= 0:
        raise ValueError("Pixel scales must be positive")
    if not torch.isfinite(source_surface_brightness).all():
        raise ValueError("Pixelized source contains non-finite values")
    device = host_nmgy_hwc.device
    dtype = host_nmgy_hwc.dtype
    source = source_surface_brightness.to(device=device, dtype=dtype).clamp(min=0.0)
    source_integral = (
        source.sum() * source_pixel_scale_arcsec * source_pixel_scale_arcsec
    )
    if not torch.isfinite(source_integral) or source_integral <= 0:
        raise ValueError("Pixelized source has no positive surface brightness")
    source = source / source_integral
    height, width, _ = host_nmgy_hwc.shape
    sub = max(1, int(subpixel_oversample))
    sub_scale = target_pixel_scale_arcsec / sub
    target_x, target_y = _torch_meshgrid_arcsec(
        height * sub,
        width * sub,
        sub_scale,
        device,
        dtype,
    )
    theta_e = torch.tensor(lens.theta_e_arcsec, device=device, dtype=dtype)
    q_lens = torch.tensor(lens.q_lens, device=device, dtype=dtype)
    phi_lens = torch.tensor(lens.phi_lens_rad, device=device, dtype=dtype)
    x0 = torch.tensor(lens.x0_arcsec, device=device, dtype=dtype)
    y0 = torch.tensor(lens.y0_arcsec, device=device, dtype=dtype)
    alpha_x, alpha_y = _sie_deflection(
        target_x,
        target_y,
        theta_e,
        q_lens,
        phi_lens,
        x0,
        y0,
    )
    shear_x, shear_y = _shear_deflection(
        target_x,
        target_y,
        torch.tensor(lens.shear_g1, device=device, dtype=dtype),
        torch.tensor(lens.shear_g2, device=device, dtype=dtype),
    )
    beta_x = target_x - alpha_x - shear_x - source_center_x_arcsec
    beta_y = target_y - alpha_y - shear_y - source_center_y_arcsec

    source_height, source_width = source.shape
    source_x = beta_x / source_pixel_scale_arcsec + (source_width - 1) / 2.0
    source_y = beta_y / source_pixel_scale_arcsec + (source_height - 1) / 2.0
    normalized_x = 2.0 * source_x / (source_width - 1) - 1.0
    normalized_y = 2.0 * source_y / (source_height - 1) - 1.0
    grid = torch.stack([normalized_x, normalized_y], dim=-1)[None, ...]
    sampled = functional.grid_sample(
        source[None, None, ...],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    if sub > 1:
        sampled = sampled.view(height, sub, width, sub).mean(dim=(1, 3))
    target_pixel_area = target_pixel_scale_arcsec * target_pixel_scale_arcsec
    band_maps = torch.stack(
        [
            sampled * float(flux) * target_pixel_area
            for flux in unlensed_flux_nmgy_grz
        ]
    )

    fwhm_values = [
        psf_fwhm_r_arcsec * psf_fwhm_scale_g,
        psf_fwhm_r_arcsec,
        psf_fwhm_r_arcsec * psf_fwhm_scale_z,
    ]
    blurred = []
    for band, fwhm in zip(band_maps, fwhm_values, strict=True):
        sigma_pixels = torch.tensor(
            fwhm / (2.355 * target_pixel_scale_arcsec),
            device=device,
            dtype=dtype,
        )
        blurred.append(
            _fft_gaussian_blur(band[None, ...], sigma_pixels[None])[0]
        )
    injection = torch.stack(blurred)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if add_poisson_noise and gain_e_per_nmgy > 0:
        expected_electrons = injection.clamp(min=0.0) * gain_e_per_nmgy
        observed_electrons = torch.poisson(
            expected_electrons,
            generator=generator,
        )
        injection = injection + (
            observed_electrons - expected_electrons
        ) / gain_e_per_nmgy
    injected = host_nmgy_hwc.permute(2, 0, 1) + injection
    return InjectionResult(
        injected=injected[None, ...],
        injection_only=injection[None, ...],
        meta={
            "theta_e_arcsec": theta_e[None],
            "q_lens": q_lens[None],
            "source_center_x_arcsec": torch.tensor(
                [source_center_x_arcsec],
                device=device,
                dtype=dtype,
            ),
            "source_center_y_arcsec": torch.tensor(
                [source_center_y_arcsec],
                device=device,
                dtype=dtype,
            ),
            "source_plane_integral_before_normalization": source_integral[None],
            "source_plane_pixel_scale_arcsec": torch.tensor(
                [source_pixel_scale_arcsec],
                device=device,
                dtype=dtype,
            ),
        },
    )
