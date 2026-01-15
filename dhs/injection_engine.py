"""Injection engine for strong-lens selection-function calibration.

This module is intentionally self-contained (NumPy + PyTorch only) to avoid the
unit pitfalls we hit previously with external lensing packages.

Key choices:
  - Lens model: SIS + external shear (gamma1, gamma2). Ray-shooting (lens
    equation) is implemented directly in arcsec units.
  - Source model: compact Sérsic profile with optional clumps, normalized to a
    specified *total unlensed* flux in nanomaggies.
  - Flux units: nanomaggies (AB zeropoint 22.5). Surface brightness is handled
    as nmgy / arcsec^2; pixel flux is SB * pixel_area.
  - PSF: Gaussian convolution performed via FFT with per-sample sigma.

This is designed for injection-recovery (selection function), not for
training data generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


AB_ZP = 22.5
AB_ZEROPOINT_MAG = AB_ZP  # alias used by other modules


def nmgy_from_abmag(mag: torch.Tensor) -> torch.Tensor:
    """Convert AB magnitude to nanomaggies."""
    return torch.pow(10.0, (AB_ZP - mag) / 2.5)


@dataclass
class LensParams:
    theta_e_arcsec: float
    shear_g1: float = 0.0
    shear_g2: float = 0.0
    # lens center offset (arcsec) relative to cutout center
    x0_arcsec: float = 0.0
    y0_arcsec: float = 0.0
    # SIE axis ratio (b/a, minor/major; 0 < q_lens <= 1; q_lens=1 is SIS)
    q_lens: float = 1.0
    # SIE position angle (radians, measured E of N)
    phi_lens_rad: float = 0.0


@dataclass
class SourceParams:
    # source center in source plane (arcsec)
    beta_x_arcsec: float
    beta_y_arcsec: float
    # Sérsic
    re_arcsec: float
    n_sersic: float
    q: float
    phi_rad: float
    # total unlensed flux (nanomaggies) per band
    flux_nmgy_g: float
    flux_nmgy_r: float
    flux_nmgy_z: float
    # optional clumps
    n_clumps: int = 0
    clump_frac: float = 0.25


@dataclass
class InjectionResult:
    injected: torch.Tensor  # (B, 3, H, W) nmgy
    injection_only: torch.Tensor  # (B, 3, H, W) nmgy
    meta: Dict[str, torch.Tensor]


def _torch_meshgrid_arcsec(
    h: int, w: int, pixel_scale: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x,y coordinate grids in arcsec centered at (0,0)."""
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy = (torch.arange(h, device=device, dtype=dtype) - cy) * pixel_scale
    xx = (torch.arange(w, device=device, dtype=dtype) - cx) * pixel_scale
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return x, y


def _sis_deflection(
    x: torch.Tensor,
    y: torch.Tensor,
    theta_e: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SIS deflection alpha = theta_E * (theta - theta0) / |theta - theta0|.

    DEPRECATED: Use _sie_deflection with q_lens=1.0 instead.
    Retained for reference and verification tests.
    """
    dx = x - x0
    dy = y - y0
    r = torch.sqrt(dx * dx + dy * dy + eps)
    ax = theta_e * dx / r
    ay = theta_e * dy / r
    return ax, ay


def _sie_deflection(
    x: torch.Tensor,
    y: torch.Tensor,
    theta_e: torch.Tensor,
    q_lens: torch.Tensor,
    phi_lens: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SIE deflection angles (Kormann et al. 1994; Keeton 2001).

    For an ellipsoidal isothermal mass distribution with axis ratio q_lens
    (b/a, 0 < q <= 1) and position angle phi_lens (radians), the deflection in
    the lens principal-axis frame (x', y') is:

        w  = sqrt(q^2 * x'^2 + y'^2 + eps^2)
        f  = sqrt(1 - q^2)
        alpha_x' = theta_E * sqrt(q) / f * arctan(f * x' / w)
        alpha_y' = theta_E * sqrt(q) / f * arctanh(f * y' / w)

    For q -> 1 (SIS limit), L'Hôpital gives alpha = theta_E * (x',y') / r,
    which is the standard SIS deflection. A branch at |1 - q| < 1e-6 handles
    this limit to avoid 0/0.

    Parameters
    ----------
    q_lens : scalar tensor, axis ratio b/a in (0, 1]
    phi_lens : scalar tensor, PA in radians
    eps : softening to avoid division by zero at the origin
    """
    # Shift to lens center
    dx = x - x0
    dy = y - y0

    # Rotate to lens principal axis frame
    cos_p = torch.cos(phi_lens)
    sin_p = torch.sin(phi_lens)
    xp = cos_p * dx + sin_p * dy
    yp = -sin_p * dx + cos_p * dy

    q_val = float(q_lens.item()) if q_lens.dim() == 0 else float(q_lens)

    if abs(q_val - 1.0) < 1e-6:
        # SIS limit: alpha = theta_E * (xp, yp) / r
        # Use same softening convention as _sis_deflection (eps added directly)
        r = torch.sqrt(xp * xp + yp * yp + eps)
        ax_p = theta_e * xp / r
        ay_p = theta_e * yp / r
    else:
        f = torch.sqrt(1.0 - q_lens * q_lens)
        w = torch.sqrt(q_lens * q_lens * (xp * xp) + yp * yp + eps * eps)
        sq = torch.sqrt(q_lens)
        # arctan is safe for all real arguments
        ax_p = theta_e * sq / f * torch.atan(f * xp / w.clamp(min=eps))
        # arctanh argument: |f*y'/w| < 1 by construction (f < 1 and w >= |y|*f)
        # but clamp for numerical safety near boundaries
        atanh_arg = (f * yp / w.clamp(min=eps)).clamp(-0.9999, 0.9999)
        ay_p = theta_e * sq / f * torch.atanh(atanh_arg)

    # Rotate back to sky frame
    ax = cos_p * ax_p - sin_p * ay_p
    ay = sin_p * ax_p + cos_p * ay_p

    return ax, ay


def _shear_deflection(
    x: torch.Tensor,
    y: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """External shear deflection: alpha = Gamma * theta.

    With Gamma = [[g1, g2], [g2, -g1]]
    """
    ax = g1 * x + g2 * y
    ay = g2 * x - g1 * y
    return ax, ay


def _rotate(x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(phi)
    s = torch.sin(phi)
    xr = c * x + s * y
    yr = -s * x + c * y
    return xr, yr


def _sersic_source_integral(
    re: torch.Tensor,
    n: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Analytical total flux integral of the Sersic profile in the SOURCE PLANE.

    For a Sersic profile S(r) = exp(-b_n * ((r_ell/R_e)^(1/n) - 1)) with
    elliptical radius r_ell = sqrt(x^2 + y^2/q^2), the total integral is:

        I = 2 * pi * q * n * R_e^2 * exp(b_n) * Gamma(2n) / b_n^(2n)

    where b_n = 2n - 1/3 (Ciotti & Bertin 1999 approximation, accurate to
    ~1% for n > 0.36).

    This is used for flux normalization: dividing by this integral converts the
    dimensionless shape to surface brightness (nmgy/arcsec^2) such that the
    source-plane total flux equals the specified unlensed flux. Because lensing
    conserves surface brightness, the IMAGE-plane total flux then equals
    mu_eff * flux_unlensed, where mu_eff is the effective magnification.

    References:
        Graham & Driver (2005), PASA, 22, 118, eq. 4
        Ciotti & Bertin (1999), A&A, 352, 447
    """
    b_n = 2.0 * n - 1.0 / 3.0
    # Compute in log-space for numerical stability
    log_integral = (
        math.log(2.0 * math.pi)
        + torch.log(q.clamp(min=1e-8))
        + torch.log(n.clamp(min=1e-8))
        + 2.0 * torch.log(re.clamp(min=1e-12))
        + b_n
        + torch.lgamma(2.0 * n)
        - 2.0 * n * torch.log(b_n.clamp(min=1e-8))
    )
    return torch.exp(log_integral).clamp(min=1e-30)


def _sersic_shape(
    bx: torch.Tensor,
    by: torch.Tensor,
    beta_x0: torch.Tensor,
    beta_y0: torch.Tensor,
    re: torch.Tensor,
    n: torch.Tensor,
    q: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Return an unnormalized Sérsic surface-brightness shape (dimensionless)."""
    # shift
    x = bx - beta_x0
    y = by - beta_y0
    # rotate into source frame
    xr, yr = _rotate(x, y, phi)
    # elliptical radius
    r_ell = torch.sqrt((xr * xr) + (yr * yr) / (q * q + 1e-8) + 1e-12)
    # Sérsic profile (up to scale)
    # b_n ≈ 2n - 1/3 (Ciotti & Bertin 1999); keeps Re as half-light radius.
    # Accurate to <1% for n > 0.5. See _sersic_source_integral docstring.
    b_n = 2.0 * n - 1.0 / 3.0
    t = torch.pow(torch.clamp(r_ell / (re + 1e-8), min=0.0), 1.0 / (n + 1e-8))
    prof = torch.exp(-b_n * (t - 1.0))
    return prof


def _add_clumps(
    base: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
    beta_x0: torch.Tensor,
    beta_y0: torch.Tensor,
    re: torch.Tensor,
    q: torch.Tensor,
    phi: torch.Tensor,
    n_clumps: torch.Tensor,
    clump_frac: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Add small Gaussian clumps to the source profile.

    Implemented in a stable way: for each sample we draw clump centers within
    ~Re and sum Gaussians. The clumps are blended into base via a fractional
    mixture weight (clump_frac).

    LIMITATIONS AND CAVEATS (documented in MNRAS_RAW_NOTES.md Section 7.7.8):

    1. The mixing formula ``(1-f)*base + f*(mean + std*clumps)`` is
       phenomenological, not physically derived from star-forming region models.
    2. Clump brightnesses are scaled by the base profile's mean+std, which is
       ad hoc and creates a coupling between the smooth Sersic and clump
       amplitudes that has no physical basis.
    3. For clumped sources, the total flux is approximate: it can deviate from
       the analytical Sersic prediction by up to ~clump_frac (typically <45%).
       Since the flux normalization uses the analytical source-plane integral
       of the *smooth* Sersic, adding clumps redistributes flux but does not
       conserve it exactly. This is a minor effect compared to the magnification
       correction (which was a 500-3000% error before fixing).
    4. The clump model is adequate for probing the classifier's sensitivity to
       morphological substructure but should NOT be interpreted as a realistic
       model of star-forming regions in lensed sources. For a physical clump
       model, see e.g. Cava et al. (2018), Dessauges-Zavadsky et al. (2017).
    """
    # base: (B,H,W)
    B, H, W = base.shape
    device = base.device
    dtype = base.dtype
    out = base.clone()
    if torch.max(n_clumps).item() <= 0:
        return out

    # coordinate grid relative to source center in the *source plane*
    x = bx[None, :, :].expand(B, H, W)
    y = by[None, :, :].expand(B, H, W)

    # per-sample draws
    for i in range(B):
        k = int(n_clumps[i].item())
        if k <= 0:
            continue
        # draw clump centers in an ellipse of scale ~Re
        # use normal draws and rescale to avoid bias to edge
        cx = torch.randn((k,), generator=rng, device=device, dtype=dtype)
        cy = torch.randn((k,), generator=rng, device=device, dtype=dtype)
        cx = cx * (0.6 * re[i]) + beta_x0[i]
        cy = cy * (0.6 * re[i] * q[i]) + beta_y0[i]
        # rotate centers
        # (apply inverse rotation to express in sky frame)
        c = torch.cos(phi[i])
        s = torch.sin(phi[i])
        dx = cx - beta_x0[i]
        dy = cy - beta_y0[i]
        cx_sky = beta_x0[i] + c * dx - s * dy
        cy_sky = beta_y0[i] + s * dx + c * dy

        # clump widths: compact
        sigma = torch.clamp(0.25 * re[i], min=0.02, max=0.15)
        amp = torch.ones((k,), device=device, dtype=dtype)
        # randomize clump weights
        amp = amp * torch.rand((k,), generator=rng, device=device, dtype=dtype)
        amp = amp / (amp.sum() + 1e-8)

        cl = torch.zeros((H, W), device=device, dtype=dtype)
        for j in range(k):
            dx = x[i] - cx_sky[j]
            dy = y[i] - cy_sky[j]
            cl = cl + amp[j] * torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma + 1e-12))

        # normalize clumps to match base integral scale, then mix
        cl = cl / (cl.max() + 1e-8)
        mix = torch.clamp(clump_frac[i], 0.0, 0.9)
        out[i] = (1.0 - mix) * out[i] + mix * (out[i].mean() + out[i].std() * cl)

    return out


def _fft_gaussian_blur(
    img: torch.Tensor,
    sigma_pix: torch.Tensor,
) -> torch.Tensor:
    """Gaussian blur via FFT for per-sample sigmas.

    img: (B,H,W)
    sigma_pix: (B,) in pixel units
    """
    B, H, W = img.shape
    device = img.device
    dtype = img.dtype
    # frequency grid (cycles/pixel)
    fy = torch.fft.fftfreq(H, device=device, dtype=dtype)  # (H,)
    fx = torch.fft.fftfreq(W, device=device, dtype=dtype)  # (W,)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    k2 = (kx * kx + ky * ky)[None, :, :]  # (1,H,W)
    # sigma in pixels -> transfer function exp(-2*pi^2*sigma^2*k^2)
    sig2 = (sigma_pix * sigma_pix).view(B, 1, 1)
    Hf = torch.exp(-2.0 * (math.pi**2) * sig2 * k2)
    F = torch.fft.fftn(img, dim=(-2, -1))
    out = torch.fft.ifftn(F * Hf, dim=(-2, -1)).real
    return out


def inject_sis_shear(
    host_nmgy_hwc: torch.Tensor,
    lens: LensParams,
    source: SourceParams,
    pixel_scale: float,
    psf_fwhm_r_arcsec: float,
    psf_fwhm_scale_g: float = 1.05,
    psf_fwhm_scale_z: float = 0.94,
    core_suppress_radius_pix: Optional[int] = None,
    seed: int = 1337,
    subpixel_oversample: int = 4,
    add_poisson_noise: bool = False,
    gain_e_per_nmgy: float = 150.0,
    add_sky_noise: bool = False,
) -> InjectionResult:
    """Inject a lensed source into a single host cutout.

    Format boundary (Q3.2):
        Input:  host_nmgy_hwc is (H, W, 3) HWC torch float32 in nanomaggies.
        Output: InjectionResult.injected and .injection_only are (1, 3, H, W) CHW torch.
        Callers must transpose HWC->CHW before feeding to preprocess_stack.

    subpixel_oversample: sub-pixel oversampling factor (e.g. 4 means 4x4 sub-pixels
        per pixel). Higher values give more accurate flux calibration for compact
        sources at the cost of computation. Set to 1 to disable (point-sampling).
        Default 4 reduces pixel-sampling bias from ~46% to <2% for R_e=0.15" sources.

    psf_fwhm_scale_g, psf_fwhm_scale_z: band-dependent PSF scaling factors
        relative to r-band. Atmospheric seeing scales as lambda^{-1/5}
        (Kolmogorov turbulence), giving g ≈ 1.07x and z ≈ 0.94x r-band.
        Previous default was z=1.00 (identical to r-band), corrected to
        0.94 based on Kolmogorov scaling. The g-band default (1.05)
        is close to the physical value (1.07).

    add_poisson_noise: if True, add Poisson noise to the injected arc signal.
        Real arcs have Poisson noise proportional to arc flux; omitting this
        makes bright injections anomalously smooth (at mag-18, Poisson noise
        is 12x sky noise, so absence is detectable by a high-AUC CNN).
        Default False for backward compatibility.

    gain_e_per_nmgy: approximate conversion factor from nanomaggies to
        photo-electrons, used for Poisson noise calculation. For a typical
        DR10 coadd with ~30 exposures at 90s each, gain ~ 150 e-/nmgy is
        a reasonable order-of-magnitude estimate. This is an approximation;
        real gain varies with depth and band.

    add_sky_noise: if True, add per-band Gaussian background-texture noise
        to the injected arc stamp, with sigma matched to the host cutout's
        background noise level (measured from the outer sky annulus via MAD).
        NOTE: This is NOT modelling additional sky photons — the host cutout
        already contains real sky/read noise. The purpose is empirical
        texture matching: without it, bright injected arcs appear anomalously
        smooth against the noisy host background, a visual artefact
        detectable by both humans and CNNs.  The physically correct arc-flux
        noise is the Poisson shot noise (add_poisson_noise); this Gaussian
        layer is an additional texture-matching step.  The paper should
        describe this as "background-texture matching noise".
        BUG FIX (2026-02-16): previously absent, causing injections to appear
        sharper than real arcs in ground-based imaging.
    """
    assert host_nmgy_hwc.ndim == 3 and host_nmgy_hwc.shape[-1] == 3
    device = host_nmgy_hwc.device
    dtype = host_nmgy_hwc.dtype
    H, W, _ = host_nmgy_hwc.shape
    sub = max(1, subpixel_oversample)

    # coordinate grids in arcsec — at sub-pixel resolution if oversampling
    sub_scale = pixel_scale / sub
    Hs, Ws = H * sub, W * sub
    x, y = _torch_meshgrid_arcsec(Hs, Ws, sub_scale, device, dtype)

    # lens equation: beta = theta - alpha_lens - alpha_shear
    # Uses SIE deflection (generalizes SIS; q_lens=1 reduces to SIS exactly)
    theta_e = torch.tensor(lens.theta_e_arcsec, device=device, dtype=dtype)
    x0 = torch.tensor(lens.x0_arcsec, device=device, dtype=dtype)
    y0 = torch.tensor(lens.y0_arcsec, device=device, dtype=dtype)
    q_lens = torch.tensor(lens.q_lens, device=device, dtype=dtype)
    phi_lens = torch.tensor(lens.phi_lens_rad, device=device, dtype=dtype)
    ax_sis, ay_sis = _sie_deflection(x, y, theta_e, q_lens, phi_lens, x0, y0)
    g1 = torch.tensor(lens.shear_g1, device=device, dtype=dtype)
    g2 = torch.tensor(lens.shear_g2, device=device, dtype=dtype)
    ax_sh, ay_sh = _shear_deflection(x, y, g1, g2)
    bx = x - ax_sis - ax_sh
    by = y - ay_sis - ay_sh

    # source shape (dimensionless) — evaluated at sub-pixel resolution
    beta_x0 = torch.tensor(source.beta_x_arcsec, device=device, dtype=dtype)
    beta_y0 = torch.tensor(source.beta_y_arcsec, device=device, dtype=dtype)
    re = torch.tensor(source.re_arcsec, device=device, dtype=dtype)
    n = torch.tensor(source.n_sersic, device=device, dtype=dtype)
    q = torch.tensor(source.q, device=device, dtype=dtype)
    phi = torch.tensor(source.phi_rad, device=device, dtype=dtype)
    shape_hi = _sersic_shape(bx, by, beta_x0, beta_y0, re, n, q, phi)
    shape_hi = shape_hi.clamp(min=0.0)

    # Downsample from (Hs, Ws) to (H, W) by averaging sub×sub blocks
    if sub > 1:
        shape = shape_hi.view(H, sub, W, sub).mean(dim=(1, 3))
    else:
        shape = shape_hi

    # optional clumps — applied at pixel resolution (clumps are morphological
    # perturbations, not high-frequency features, so sub-pixel evaluation is
    # unnecessary; using pixel-scale source-plane coords for consistency)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    bx_pix, by_pix = bx, by
    if sub > 1:
        # Get pixel-resolution source-plane coordinates by subsampling center points
        x_pix, y_pix = _torch_meshgrid_arcsec(H, W, pixel_scale, device, dtype)
        ax_pix, ay_pix = _sie_deflection(x_pix, y_pix, theta_e, q_lens, phi_lens, x0, y0)
        axs_pix, ays_pix = _shear_deflection(x_pix, y_pix, g1, g2)
        bx_pix = x_pix - ax_pix - axs_pix
        by_pix = y_pix - ay_pix - ays_pix
    shape = _add_clumps(
        shape[None, :, :],
        bx_pix,
        by_pix,
        beta_x0[None],
        beta_y0[None],
        re[None],
        q[None],
        phi[None],
        torch.tensor([source.n_clumps], device=device, dtype=torch.int64),
        torch.tensor([source.clump_frac], device=device, dtype=dtype),
        rng,
    )[0]

    # Normalize using the ANALYTICAL source-plane Sersic integral.
    #
    # Why analytical, not image-plane sum?
    # Lensing conserves surface brightness: I_obs(theta) = I_source(beta(theta)).
    # The image-plane integral of the lensed profile equals mu_eff times the
    # source-plane integral (where mu_eff is the effective magnification).
    # If we normalized by the image-plane sum, we would divide out the
    # magnification and get total_image_flux = flux_unlensed (WRONG).
    # By normalizing with the source-plane integral, the magnification is
    # preserved: total_image_flux = mu_eff * flux_unlensed (CORRECT).
    #
    # Note: clumps are added before this point as a morphological perturbation.
    # They may change the total flux by up to ~clump_frac relative to the pure
    # Sersic prediction. This is a documented approximation (see MNRAS_RAW_NOTES
    # Section 7.7.8).
    pix_area = pixel_scale * pixel_scale
    source_integral = _sersic_source_integral(re, n, q)

    # band fluxes (these are UNLENSED source fluxes in nanomaggies)
    flux_r = torch.tensor(source.flux_nmgy_r, device=device, dtype=dtype)
    flux_g = torch.tensor(source.flux_nmgy_g, device=device, dtype=dtype)
    flux_z = torch.tensor(source.flux_nmgy_z, device=device, dtype=dtype)

    sb_r = shape * (flux_r / source_integral)  # nmgy/arcsec^2
    sb_g = shape * (flux_g / source_integral)
    sb_z = shape * (flux_z / source_integral)

    # pixel flux maps (nmgy)
    inj_r = sb_r * pix_area
    inj_g = sb_g * pix_area
    inj_z = sb_z * pix_area

    # PSF blur per band (Gaussian, per-sample sigma)
    def blur(img2d: torch.Tensor, fwhm_arcsec: float) -> torch.Tensor:
        sigma_pix = torch.tensor(fwhm_arcsec / (2.355 * pixel_scale), device=device, dtype=dtype)
        return _fft_gaussian_blur(img2d[None, :, :], sigma_pix[None])[0]

    inj_r = blur(inj_r, psf_fwhm_r_arcsec)
    inj_g = blur(inj_g, psf_fwhm_r_arcsec * psf_fwhm_scale_g)
    inj_z = blur(inj_z, psf_fwhm_r_arcsec * psf_fwhm_scale_z)

    injection = torch.stack([inj_g, inj_r, inj_z], dim=0)  # (3,H,W)

    # core suppression ablation
    if core_suppress_radius_pix is not None and core_suppress_radius_pix > 0:
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        yy = torch.arange(H, device=device, dtype=dtype) - cy
        xx = torch.arange(W, device=device, dtype=dtype) - cx
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        rr = torch.sqrt(X * X + Y * Y)
        mask = (rr >= float(core_suppress_radius_pix)).to(dtype)
        injection = injection * mask[None, :, :]

    # Optionally add Poisson noise to arc signal.
    # Real arcs contribute shot noise proportional to sqrt(flux * gain).
    # Without this, bright injections are anomalously smooth — a statistical
    # signature detectable by a high-AUC CNN.
    #
    # RNG FIX (D06):  Poisson and background-
    # texture noise now use the per-injection `rng` generator (seeded from
    # `seed` above).  Previously these used the GLOBAL torch RNG, which:
    #   (a) made results non-reproducible across re-runs, and
    #   (b) broke the "paired comparison" claim because two conditions sharing
    #       the same seed would draw different noise realisations.
    if add_poisson_noise and gain_e_per_nmgy > 0:
        # Convert arc flux to photo-electrons, draw exact Poisson, convert back.
        # torch.poisson handles all lambda correctly:
        #   lambda=0 -> returns 0 (no noise on zero-flux pixels)
        #   lambda>0 -> exact Poisson draw
        # BUGFIX (D03 post-review): Previous implementation used a Gaussian
        # approximation with clamp(min=1.0), which injected noise of std
        # ~1 electron into ALL zero-flux pixels (~95% of the stamp). This
        # noise (~0.007 nmgy) corrupted the annulus normalization, inflating
        # the MAD by ~2.5x and compressing the normalized arc signal by the
        # same factor. The effect was catastrophic for faint-arc detection.
        arc_electrons = (injection.clamp(min=0.0) * gain_e_per_nmgy)
        noisy_electrons = torch.poisson(arc_electrons, generator=rng)
        noise_electrons = noisy_electrons - arc_electrons
        injection = injection + noise_electrons / gain_e_per_nmgy

    # Optionally add per-band Gaussian background-texture noise to the arc
    # stamp so the injection region has the same noise texture as the host.
    # Sigma is estimated per band from the outer sky annulus of the host
    # cutout (robust MAD estimator).
    #
    # IMPORTANT: This is NOT physically modelling additional sky photons.
    # The host cutout already contains all real sky/read noise.  The purpose
    # is to prevent the CNN from detecting injections via anomalously smooth
    # arc regions -- a visual artefact seen during manual inspection
    # (2026-02-16).  Physically, the correct arc-flux noise is the Poisson
    # term above; this Gaussian layer is an empirical texture-matching step.
    # The paper should describe it as "background-texture matching noise",
    # not literal sky noise addition.
    if add_sky_noise:
        host_chw_np = host_nmgy_hwc.permute(2, 0, 1).cpu().numpy()
        for band_idx in range(3):
            sigma = estimate_sigma_pix_from_cutout(
                host_chw_np, band_idx=band_idx,
                sky_r_inner_pix=35, sky_r_outer_pix=48,
            )
            if np.isfinite(sigma) and sigma > 0:
                noise = torch.randn(
                    injection[band_idx].shape,
                    generator=rng, device=device, dtype=dtype,
                ) * float(sigma)
                injection[band_idx] = injection[band_idx] + noise

    host_chw = host_nmgy_hwc.permute(2, 0, 1)
    injected = host_chw + injection

    meta = {
        "theta_e_arcsec": torch.tensor([lens.theta_e_arcsec], device=device, dtype=dtype),
        "shear_g1": torch.tensor([lens.shear_g1], device=device, dtype=dtype),
        "shear_g2": torch.tensor([lens.shear_g2], device=device, dtype=dtype),
        "q_lens": torch.tensor([lens.q_lens], device=device, dtype=dtype),
        "phi_lens_rad": torch.tensor([lens.phi_lens_rad], device=device, dtype=dtype),
        "beta_x_arcsec": torch.tensor([source.beta_x_arcsec], device=device, dtype=dtype),
        "beta_y_arcsec": torch.tensor([source.beta_y_arcsec], device=device, dtype=dtype),
        "re_arcsec": torch.tensor([source.re_arcsec], device=device, dtype=dtype),
        "n_sersic": torch.tensor([source.n_sersic], device=device, dtype=dtype),
        "flux_nmgy_r": torch.tensor([source.flux_nmgy_r], device=device, dtype=dtype),
        "psf_fwhm_r": torch.tensor([psf_fwhm_r_arcsec], device=device, dtype=dtype),
    }

    return InjectionResult(injected=injected[None, ...], injection_only=injection[None, ...], meta=meta)


def sample_source_params(
    rng: np.random.Generator,
    theta_e_arcsec: float,
    r_mag_range: Tuple[float, float] = (23.0, 26.0),
    beta_frac_range: Tuple[float, float] = (0.10, 0.40),
    re_arcsec_range: Tuple[float, float] = (0.15, 0.50),
    n_range: Tuple[float, float] = (0.5, 2.0),
    q_range: Tuple[float, float] = (0.3, 1.0),
    g_minus_r_mu_sigma: Tuple[float, float] = (1.15, 0.30),
    r_minus_z_mu_sigma: Tuple[float, float] = (0.85, 0.20),
    clumps_prob: float = 0.0,
    # Q1.16 fix: clumps params were previously hardcoded in the function body.
    # Now they are explicit parameters so the AST-based prior validator can
    # check them against configs/injection_priors.yaml.
    clumps_n_range: Tuple[int, int] = (1, 4),
    clumps_frac_range: Tuple[float, float] = (0.15, 0.45),
    # Sensitivity analysis overrides
    re_scale: float = 1.0,
    gmr_shift: float = 0.0,
    rmz_shift: float = 0.0,
) -> SourceParams:
    """Sample physically plausible source parameters.

    We sample beta as a fraction of theta_E. For SIS, beta < theta_E produces
    multiple images and a realistic magnification distribution.

    Prior ranges (updated 2026-02-13 after code review):

    g_minus_r_mu_sigma: (1.15, 0.30) -- changed from (0.2, 0.25).
        BUG FIX (2026-02-16): Previous defaults were rest-frame blue galaxy
        colours and did not account for cosmological K-correction. Sources
        lensed at z~1-3 have OBSERVER-FRAME g-r ~ 0.8-1.5 in Legacy Survey
        bands due to redshift.  Measured from arc-region annulus (8-18 px)
        of 388 real Tier-A lenses: mean=1.23, std=0.39, median=1.28.
        Initial conservative estimate N(1.0, 0.35); refined to N(1.15, 0.30)
        after visual inspection showed slight residual colour offset.

    r_minus_z_mu_sigma: (0.85, 0.20) -- changed from (0.1, 0.25).
        BUG FIX (2026-02-16): Same K-correction issue as g-r.
        Measured from arc-region annulus of Tier-A lenses: mean=0.95,
        std=0.27, median=0.89.  Initial conservative estimate N(0.75, 0.25);
        refined to N(0.85, 0.20) after visual inspection.

    beta_frac_range: (0.10, 0.40) -- narrowed from (0.1, 1.0).
        BUG FIX (2026-02-16): Area-weighted sampling with upper bound
        at 1.0 (or even 0.55 in experiments) heavily biases toward high
        beta_frac values where the source is far from the caustic. This
        produces widely separated double/quad images instead of extended
        tangential arcs. Real lenses with visible arcs have sources near
        the caustic (low beta_frac). Narrowing to (0.10, 0.40) favours
        near-caustic configurations producing arc-like morphologies.

    re_arcsec_range: (0.15, 0.50) -- raised minimum from 0.05.
        BUG FIX (2026-02-16): Sources with Re < 0.15" are unresolvable at
        Legacy Survey ~1" seeing and produce point-like lensed images that
        look like binary stars, not gravitational arcs. Manual visual
        inspection confirmed that Re_min=0.05 creates obviously fake
        double-image artefacts. Minimum 0.15" ensures lensed images
        overlap and merge into smooth arcs at ground-based resolution.
        Literature:
        - Collett (2015): typical 0.1-0.5" at z~1-3
        - Observed: z~1 late-type 0.3-0.8", z~2 0.2-0.5"

    n_range: (0.5, 2.0) -- narrowed from (0.5, 4.0).
        BUG FIX (2026-02-16): Concentrated profiles (n > 2, e.g. de
        Vaucouleurs n=4) produce compact source light distributions that
        get lensed into point-like doubles, not smooth arcs. Manual visual
        inspection showed pie-chart-like segmented artefacts. High-z lensed
        sources are predominantly disk galaxies (n ~ 1).
        - Collett (2015): n = 1 (exponential disk) for high-z sources
        - Herle et al. (2024): while their CNN selects n >= 2.55, that
          selection bias should not be built into the injection prior.

    clumps_prob: 0.0 -- disabled from 0.6.
        BUG FIX (2026-02-16): Clumps (1-4 Gaussian blobs at 60% prob,
        up to 45% flux) create additional bright spots at separate lensed
        image positions, producing obviously fake multi-blob artefacts
        visible in manual inspection. Disabled until base arc morphology
        is validated. Can be re-enabled with reduced parameters if needed.
    """
    # source offset in polar coordinates in source plane
    # Area-weighted sampling: for uniform source density on the sky,
    # P(beta) d_beta proportional to 2*pi*beta*d_beta (annular area element).
    # With beta_frac = beta / theta_E, we need P(beta_frac) proportional to
    # beta_frac, achieved by: beta_frac = sqrt(uniform(lo^2, hi^2)).
    lo2 = beta_frac_range[0] ** 2
    hi2 = beta_frac_range[1] ** 2
    beta_frac = math.sqrt(rng.uniform(lo2, hi2))
    beta = beta_frac * theta_e_arcsec
    ang = rng.uniform(0, 2 * math.pi)
    beta_x = beta * math.cos(ang)
    beta_y = beta * math.sin(ang)

    re = rng.uniform(re_arcsec_range[0], re_arcsec_range[1]) * re_scale
    n = rng.uniform(n_range[0], n_range[1])
    q = rng.uniform(q_range[0], q_range[1])
    phi = rng.uniform(0, 2 * math.pi)

    r_mag = rng.uniform(r_mag_range[0], r_mag_range[1])
    # colors (with optional sensitivity shift)
    gmr = rng.normal(g_minus_r_mu_sigma[0], g_minus_r_mu_sigma[1]) + gmr_shift
    rmz = rng.normal(r_minus_z_mu_sigma[0], r_minus_z_mu_sigma[1]) + rmz_shift
    g_mag = r_mag + gmr
    z_mag = r_mag - rmz

    flux_r = float(10 ** ((AB_ZP - r_mag) / 2.5))
    flux_g = float(10 ** ((AB_ZP - g_mag) / 2.5))
    flux_z = float(10 ** ((AB_ZP - z_mag) / 2.5))

    n_clumps = 0
    clump_frac = 0.0
    if rng.uniform() < clumps_prob:
        n_clumps = int(rng.integers(clumps_n_range[0], clumps_n_range[1]))
        clump_frac = float(rng.uniform(clumps_frac_range[0], clumps_frac_range[1]))

    return SourceParams(
        beta_x_arcsec=beta_x,
        beta_y_arcsec=beta_y,
        re_arcsec=re,
        n_sersic=n,
        q=q,
        phi_rad=phi,
        flux_nmgy_g=flux_g,
        flux_nmgy_r=flux_r,
        flux_nmgy_z=flux_z,
        n_clumps=n_clumps,
        clump_frac=clump_frac,
    )


def sample_lens_params(
    rng: np.random.Generator,
    theta_e_arcsec: float,
    shear_sigma: float = 0.05,
    center_sigma_arcsec: float = 0.05,
    q_lens_range: Tuple[float, float] = (0.5, 1.0),
    # Sensitivity analysis override
    q_lens_range_override: Optional[Tuple[float, float]] = None,
) -> LensParams:
    """Sample lens nuisance parameters (shear, centroid jitter, ellipticity).

    q_lens is drawn from uniform(q_lens_range[0], q_lens_range[1]).
    When q_lens_range = (1.0, 1.0), this reduces to a pure SIS.
    phi_lens is drawn from uniform(0, pi) (PA has pi symmetry).

    If q_lens_range_override is provided, it replaces q_lens_range.
    """
    qlr = q_lens_range_override if q_lens_range_override is not None else q_lens_range
    g1 = float(rng.normal(0.0, shear_sigma))
    g2 = float(rng.normal(0.0, shear_sigma))
    x0 = float(rng.normal(0.0, center_sigma_arcsec))
    y0 = float(rng.normal(0.0, center_sigma_arcsec))
    q_lens = float(rng.uniform(qlr[0], qlr[1]))
    phi_lens = float(rng.uniform(0.0, math.pi))
    return LensParams(
        theta_e_arcsec=theta_e_arcsec,
        shear_g1=g1,
        shear_g2=g2,
        x0_arcsec=x0,
        y0_arcsec=y0,
        q_lens=q_lens,
        phi_lens_rad=phi_lens,
    )


def estimate_sigma_pix_from_psfdepth(
    psfdepth_nmgy_invvar: float,
    psf_fwhm_arcsec: float,
    pixel_scale: float,
) -> float:
    """Approximate per-pixel noise sigma from psfdepth.

    Assumes a Gaussian PSF and independent pixels. This is imperfect for coadds
    (correlated noise), but it provides a consistent proxy for SNR diagnostics.

    psfdepth is inverse variance of PSF-flux (nmgy^-2), as defined in the
    DR10 Tractor catalog schema (https://www.legacysurvey.org/dr10/catalogs/).

    NOTE: The DR10 brick-summary files define psfdepth differently (as a
    5-sigma AB magnitude). This function expects the Tractor/sweep catalog
    value, which is in 1/nanomaggy^2. The pipeline's manifest uses Tractor
    catalog columns, so this is correct. Do not confuse with brick-summary
    psfdepth, which uses a different convention (5-sigma AB magnitude).
    """
    if psfdepth_nmgy_invvar <= 0 or not np.isfinite(psfdepth_nmgy_invvar):
        return float("nan")
    sigma_flux = 1.0 / math.sqrt(psfdepth_nmgy_invvar)  # nmgy
    sigma_psf_pix = psf_fwhm_arcsec / (2.355 * pixel_scale)
    # continuous approximation: sum(PSF^2) ≈ 1/(4*pi*sigma^2)
    sum_psf2 = 1.0 / (4.0 * math.pi * sigma_psf_pix * sigma_psf_pix + 1e-12)
    sigma_pix = sigma_flux * math.sqrt(sum_psf2)
    return sigma_pix


def estimate_sigma_pix_from_cutout(
    host_chw: np.ndarray,
    band_idx: int = 1,
    sky_r_inner_pix: int = 35,
    sky_r_outer_pix: int = 48,
) -> float:
    """Estimate per-pixel noise sigma directly from the host cutout sky ring.

    This is an alternative to psfdepth-based estimation that avoids any
    ambiguity in psfdepth unit interpretation. It measures the actual noise
    in the image by computing robust MAD in a sky annulus.

    Estimates pixel noise directly from the host cutout by computing
    robust MAD in an outer sky annulus, per band.

    Parameters
    ----------
    host_chw : (C, H, W) numpy array in nanomaggies
    band_idx : which band to use (0=g, 1=r, 2=z)
    sky_r_inner_pix, sky_r_outer_pix : sky annulus radii in pixels
        Default (35, 48) avoids galaxy light and stays inside 101x101 stamp.

    Returns
    -------
    sigma_pix : per-pixel noise sigma in nanomaggies
    """
    img = host_chw[band_idx]  # (H, W)
    H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = np.arange(H) - cy
    xx = np.arange(W) - cx
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    rr = np.sqrt(X * X + Y * Y)
    mask = (rr >= sky_r_inner_pix) & (rr <= sky_r_outer_pix)
    vals = img[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return float("nan")
    # Robust sigma via MAD (median absolute deviation)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = mad * 1.4826  # MAD -> sigma for Gaussian
    return sigma if sigma > 0 else float("nan")


def arc_annulus_snr(
    injection_only_chw: torch.Tensor,
    sigma_pix_r: float,
    annulus_r_pix: Tuple[int, int] = (4, 16),
    core_r_pix: int = 3,
) -> float:
    """Compute a simple annulus SNR proxy on the r band for diagnostics."""
    inj_r = injection_only_chw[1]  # (H,W)
    H, W = inj_r.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, device=inj_r.device, dtype=torch.float32) - cy
    xx = torch.arange(W, device=inj_r.device, dtype=torch.float32) - cx
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    rr = torch.sqrt(X * X + Y * Y)
    m = (rr >= annulus_r_pix[0]) & (rr <= annulus_r_pix[1]) & (rr >= core_r_pix)
    vals = inj_r[m]
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0 or not np.isfinite(sigma_pix_r) or sigma_pix_r <= 0:
        return float("nan")
    # matched-filter-ish proxy: sum(signal)/sqrt(N)*sigma
    snr = (vals.sum() / (math.sqrt(float(vals.numel())) * sigma_pix_r)).item()
    return float(snr)
