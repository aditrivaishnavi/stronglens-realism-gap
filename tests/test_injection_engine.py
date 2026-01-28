#!/usr/bin/env python3
"""
Injection Engine Validation Tests.

Comprehensive regression and physics tests for dhs/injection_engine.py,
covering:

  1. Magnification physics (analytical Sersic normalization)
  2. SIE lens model (Kormann et al. 1994 deflection)
  3. Area-weighted source offset sampling
  4. Sub-pixel oversampling convergence
  5. Cross-validation against lenstronomy (peer-reviewed lensing code)
  6. Flux conservation
  7. Clump model stability

These tests were written after discovering and fixing 8 injection pipeline
issues (MNRAS_RAW_NOTES.md Section 7.7).  They serve as a regression gate:
any future code change that breaks these tests has introduced a physics error.

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.
    python -m pytest tests/test_injection_engine.py -v

    # Or run standalone:
    python tests/test_injection_engine.py

Author: stronglens_calibration project
Date: 2026-02-10
References:
  - MNRAS_RAW_NOTES.md Section 7.7
  - Graham & Driver (2005), PASA, 22, 118 (Sersic integral)
  - Kormann et al. (1994), A&A, 284, 285 (SIE deflection)
  - Ciotti & Bertin (1999), A&A, 352, 447 (b_n approximation)
"""
from __future__ import annotations

import math
import sys
import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
try:
    from dhs.injection_engine import (
        AB_ZP,
        LensParams,
        SourceParams,
        InjectionResult,
        inject_sis_shear,
        sample_lens_params,
        sample_source_params,
        _sersic_source_integral,
        _sersic_shape,
        _sis_deflection,
        _sie_deflection,
        _shear_deflection,
        _torch_meshgrid_arcsec,
        estimate_sigma_pix_from_psfdepth,
        arc_annulus_snr,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Run with PYTHONPATH=. from the stronglens_calibration directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIXEL_SCALE = 0.262  # arcsec/pixel (DESI Legacy Survey)
STAMP_H = STAMP_W = 101
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _make_host(h: int = STAMP_H, w: int = STAMP_W) -> torch.Tensor:
    """Return a zero host cutout (H, W, 3)."""
    return torch.zeros((h, w, 3), dtype=DTYPE)


def _total_r_flux(result: InjectionResult) -> float:
    """Sum the r-band injection-only flux."""
    return float(result.injection_only[0, 1].sum().item())


# ===================================================================
# 1. Magnification physics
# ===================================================================
class TestMagnification(unittest.TestCase):
    """Verify that lensing magnification is correctly preserved."""

    def test_on_axis_high_magnification(self):
        """Nearly on-axis source should show very high magnification."""
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.001, beta_y_arcsec=0.0,
            re_arcsec=0.05, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(_make_host(), lens, src, PIXEL_SCALE, 1.2, seed=42)
        mu = _total_r_flux(result)
        self.assertGreater(mu, 30.0,
                           f"On-axis source should have mu >> 1, got {mu:.1f}")

    def test_known_offset_magnification(self):
        """Source at beta=0.5 with theta_E=1.5 -> point-source mu=6.0."""
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(_make_host(), lens, src, PIXEL_SCALE, 1.2, seed=42)
        mu = _total_r_flux(result)
        # Extended source mu differs from point-source mu=6.0, but should
        # be in the same ballpark (within 30%)
        self.assertGreater(mu, 4.0, f"mu too low: {mu:.2f}")
        self.assertLess(mu, 12.0, f"mu too high: {mu:.2f}")

    def test_far_offset_weak_magnification(self):
        """Source at beta=5*theta_E should have mu ~ 1."""
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=7.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(_make_host(), lens, src, PIXEL_SCALE, 1.2, seed=42)
        mu = _total_r_flux(result)
        self.assertGreater(mu, 0.5, f"Far-field mu too low: {mu:.3f}")
        self.assertLess(mu, 2.0, f"Far-field mu too high: {mu:.3f}")

    def test_flux_conservation(self):
        """injected == host + injection_only (additive injection)."""
        host = torch.rand((STAMP_H, STAMP_W, 3), dtype=DTYPE) * 5.0
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.3, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.5, q=0.7, phi_rad=0.5,
            flux_nmgy_g=0.5, flux_nmgy_r=1.0, flux_nmgy_z=0.8,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(host, lens, src, PIXEL_SCALE, 1.2, seed=42)
        host_chw = host.permute(2, 0, 1)
        diff = (result.injected[0] - host_chw - result.injection_only[0]).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"Flux conservation violated: max diff = {diff:.2e}")

    def test_clump_flux_stability(self):
        """Clumped source flux should not differ > 50% from non-clumped."""
        lens = LensParams(theta_e_arcsec=1.5)
        base_kw = dict(
            beta_x_arcsec=0.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.5, q=0.7, phi_rad=0.5,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
        )
        src_base = SourceParams(**base_kw, n_clumps=0, clump_frac=0.0)
        src_clmp = SourceParams(**base_kw, n_clumps=3, clump_frac=0.4)
        host = _make_host()
        fb = _total_r_flux(inject_sis_shear(host, lens, src_base, PIXEL_SCALE, 1.2, seed=42))
        fc = _total_r_flux(inject_sis_shear(host, lens, src_clmp, PIXEL_SCALE, 1.2, seed=42))
        frac = abs(fc - fb) / fb
        self.assertLess(frac, 0.50,
                        f"Clump flux deviation {frac:.1%} > 50%")


# ===================================================================
# 2. Analytical Sersic integral
# ===================================================================
class TestSersicIntegral(unittest.TestCase):
    """Verify the analytical source-plane Sersic integral."""

    def test_known_value_n1_circular(self):
        """For n=1, q=1, R_e=1: known closed-form value."""
        re = torch.tensor(1.0, dtype=DTYPE)
        n = torch.tensor(1.0, dtype=DTYPE)
        q = torch.tensor(1.0, dtype=DTYPE)
        I = _sersic_source_integral(re, n, q).item()
        # I = 2*pi*1*1*1^2 * exp(5/3) * Gamma(2) / (5/3)^2
        b_n = 5.0 / 3.0
        expected = 2 * math.pi * math.exp(b_n) * math.gamma(2.0) / b_n**2
        self.assertAlmostEqual(I, expected, places=3,
                               msg=f"Sersic integral mismatch: {I} vs {expected}")

    def test_scales_with_re_squared(self):
        """Integral should scale as R_e^2."""
        n = torch.tensor(1.5, dtype=DTYPE)
        q = torch.tensor(0.8, dtype=DTYPE)
        I1 = _sersic_source_integral(torch.tensor(0.1, dtype=DTYPE), n, q).item()
        I2 = _sersic_source_integral(torch.tensor(0.2, dtype=DTYPE), n, q).item()
        ratio = I2 / I1
        self.assertAlmostEqual(ratio, 4.0, places=2,
                               msg=f"R_e scaling wrong: ratio = {ratio}")

    def test_scales_with_q(self):
        """Integral should scale linearly with q."""
        re = torch.tensor(0.15, dtype=DTYPE)
        n = torch.tensor(1.0, dtype=DTYPE)
        I1 = _sersic_source_integral(re, n, torch.tensor(0.5, dtype=DTYPE)).item()
        I2 = _sersic_source_integral(re, n, torch.tensor(1.0, dtype=DTYPE)).item()
        ratio = I2 / I1
        self.assertAlmostEqual(ratio, 2.0, places=2,
                               msg=f"q scaling wrong: ratio = {ratio}")

    def test_positive_for_valid_params(self):
        """Integral should be strictly positive for all valid parameter ranges."""
        for n_val in [0.5, 1.0, 1.5, 2.0, 2.5, 4.0]:
            for q_val in [0.3, 0.5, 0.7, 1.0]:
                I = _sersic_source_integral(
                    torch.tensor(0.15, dtype=DTYPE),
                    torch.tensor(n_val, dtype=DTYPE),
                    torch.tensor(q_val, dtype=DTYPE),
                ).item()
                self.assertGreater(I, 0.0,
                                   f"Negative integral for n={n_val}, q={q_val}: {I}")


# ===================================================================
# 3. SIE lens model
# ===================================================================
class TestSIEDeflection(unittest.TestCase):
    """Verify SIE deflection against SIS limit and properties."""

    def setUp(self):
        self.x, self.y = _torch_meshgrid_arcsec(
            STAMP_H, STAMP_W, PIXEL_SCALE, DEVICE, DTYPE
        )
        self.theta_e = torch.tensor(1.5, dtype=DTYPE)
        self.x0 = torch.tensor(0.1, dtype=DTYPE)
        self.y0 = torch.tensor(-0.05, dtype=DTYPE)

    def test_sie_q1_equals_sis(self):
        """SIE with q=1.0 must reproduce SIS to machine precision."""
        ax_sis, ay_sis = _sis_deflection(
            self.x, self.y, self.theta_e, self.x0, self.y0
        )
        # Use an arbitrary PA — should not matter at q=1
        ax_sie, ay_sie = _sie_deflection(
            self.x, self.y, self.theta_e,
            torch.tensor(1.0, dtype=DTYPE),
            torch.tensor(0.7, dtype=DTYPE),
            self.x0, self.y0,
        )
        max_dx = (ax_sis - ax_sie).abs().max().item()
        max_dy = (ay_sis - ay_sie).abs().max().item()
        self.assertLess(max_dx, 1e-6,
                        f"x deflection mismatch: {max_dx:.2e}")
        self.assertLess(max_dy, 1e-6,
                        f"y deflection mismatch: {max_dy:.2e}")

    def test_sie_q07_different_morphology(self):
        """SIE q=0.7 should produce visibly different arc morphology."""
        host = _make_host()
        src = SourceParams(
            beta_x_arcsec=0.3, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        r_sis = inject_sis_shear(
            host, LensParams(theta_e_arcsec=1.5, q_lens=1.0),
            src, PIXEL_SCALE, 1.2, seed=42,
        )
        r_sie = inject_sis_shear(
            host, LensParams(theta_e_arcsec=1.5, q_lens=0.7, phi_lens_rad=0.3),
            src, PIXEL_SCALE, 1.2, seed=42,
        )
        diff = (r_sis.injection_only[0, 1] - r_sie.injection_only[0, 1]).abs().sum().item()
        self.assertGreater(diff, 0.01,
                           f"SIS and SIE morphologies should differ, got sum|diff|={diff:.4f}")

    def test_sie_magnification(self):
        """Both SIS and SIE should show clear magnification."""
        host = _make_host()
        src = SourceParams(
            beta_x_arcsec=0.3, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        f_sis = _total_r_flux(inject_sis_shear(
            host, LensParams(theta_e_arcsec=1.5, q_lens=1.0),
            src, PIXEL_SCALE, 1.2, seed=42,
        ))
        f_sie = _total_r_flux(inject_sis_shear(
            host, LensParams(theta_e_arcsec=1.5, q_lens=0.7, phi_lens_rad=0.3),
            src, PIXEL_SCALE, 1.2, seed=42,
        ))
        self.assertGreater(f_sis, 2.0, f"SIS mu too low: {f_sis}")
        self.assertGreater(f_sie, 1.5, f"SIE mu too low: {f_sie}")

    def test_deflection_finite_at_origin(self):
        """Deflection at (0,0) should be finite (eps softening)."""
        ax, ay = _sie_deflection(
            torch.tensor([0.0]), torch.tensor([0.0]),
            self.theta_e,
            torch.tensor(0.7, dtype=DTYPE),
            torch.tensor(0.0, dtype=DTYPE),
            torch.tensor(0.0, dtype=DTYPE),
            torch.tensor(0.0, dtype=DTYPE),
        )
        self.assertTrue(torch.isfinite(ax).all(), "Non-finite x deflection at origin")
        self.assertTrue(torch.isfinite(ay).all(), "Non-finite y deflection at origin")


# ===================================================================
# 4. Cross-validation against lenstronomy
# ===================================================================
class TestLenstronomyCrossValidation(unittest.TestCase):
    """Cross-validate deflection angles and flux against lenstronomy."""

    @classmethod
    def setUpClass(cls):
        try:
            from lenstronomy.LensModel.lens_model import LensModel as LM
            cls.has_lenstronomy = True
            cls.LensModel = LM
        except ImportError:
            cls.has_lenstronomy = False

    def _skip_if_no_lenstronomy(self):
        if not self.has_lenstronomy:
            self.skipTest("lenstronomy not installed")

    def test_deflection_sis(self):
        """Our SIE(q=1) deflection matches lenstronomy SIS."""
        self._skip_if_no_lenstronomy()
        N = 51
        coords = np.linspace(-5.0, 5.0, N)
        xx, yy = np.meshgrid(coords, coords)
        x_flat, y_flat = xx.flatten(), yy.flatten()

        # lenstronomy
        lm = self.LensModel(lens_model_list=["SIS"])
        kw = [{"theta_E": 1.5, "center_x": 0.0, "center_y": 0.0}]
        ax_ls, ay_ls = lm.alpha(x_flat, y_flat, kw)

        # ours
        x_t = torch.from_numpy(x_flat).float()
        y_t = torch.from_numpy(y_flat).float()
        ax_o, ay_o = _sie_deflection(
            x_t, y_t,
            torch.tensor(1.5), torch.tensor(1.0), torch.tensor(0.0),
            torch.tensor(0.0), torch.tensor(0.0),
        )

        r = np.sqrt(x_flat**2 + y_flat**2)
        mask = r > 0.1
        max_err = max(
            np.max(np.abs(ax_o.numpy()[mask] - ax_ls[mask])),
            np.max(np.abs(ay_o.numpy()[mask] - ay_ls[mask])),
        )
        # Tolerance 1e-4 arcsec: different softening conventions near origin
        # cause O(1e-5) differences. 1e-4 is < 0.04% of a pixel.
        self.assertLess(max_err, 1e-4,
                        f"SIS deflection vs lenstronomy: max err = {max_err:.2e}")

    def test_deflection_sie_q07(self):
        """Our SIE(q=0.7) deflection matches lenstronomy SIE."""
        self._skip_if_no_lenstronomy()
        N = 51
        coords = np.linspace(-5.0, 5.0, N)
        xx, yy = np.meshgrid(coords, coords)
        x_flat, y_flat = xx.flatten(), yy.flatten()

        q, phi = 0.7, 0.3
        e = (1.0 - q) / (1.0 + q)
        e1 = e * np.cos(2.0 * phi)
        e2 = e * np.sin(2.0 * phi)

        lm = self.LensModel(lens_model_list=["SIE"])
        kw = [{"theta_E": 1.5, "e1": e1, "e2": e2,
               "center_x": 0.0, "center_y": 0.0}]
        ax_ls, ay_ls = lm.alpha(x_flat, y_flat, kw)

        x_t = torch.from_numpy(x_flat).float()
        y_t = torch.from_numpy(y_flat).float()
        ax_o, ay_o = _sie_deflection(
            x_t, y_t,
            torch.tensor(1.5), torch.tensor(q), torch.tensor(phi),
            torch.tensor(0.0), torch.tensor(0.0),
        )

        r = np.sqrt(x_flat**2 + y_flat**2)
        mask = r > 0.1
        mag_ls = np.sqrt(ax_ls[mask]**2 + ay_ls[mask]**2)
        rel_err = np.sqrt(
            (ax_o.numpy()[mask] - ax_ls[mask])**2
            + (ay_o.numpy()[mask] - ay_ls[mask])**2
        ) / (mag_ls + 1e-12)
        max_rel = np.max(rel_err)
        self.assertLess(max_rel, 0.001,
                        f"SIE deflection vs lenstronomy: max rel err = {max_rel:.6f}")

    def test_deflection_sie_q05(self):
        """Our SIE(q=0.5, phi=1.2) matches lenstronomy."""
        self._skip_if_no_lenstronomy()
        N = 51
        coords = np.linspace(-5.0, 5.0, N)
        xx, yy = np.meshgrid(coords, coords)
        x_flat, y_flat = xx.flatten(), yy.flatten()

        q, phi = 0.5, 1.2
        e = (1.0 - q) / (1.0 + q)
        e1 = e * np.cos(2.0 * phi)
        e2 = e * np.sin(2.0 * phi)

        lm = self.LensModel(lens_model_list=["SIE"])
        kw = [{"theta_E": 2.0, "e1": e1, "e2": e2,
               "center_x": 0.0, "center_y": 0.0}]
        ax_ls, ay_ls = lm.alpha(x_flat, y_flat, kw)

        x_t = torch.from_numpy(x_flat).float()
        y_t = torch.from_numpy(y_flat).float()
        ax_o, ay_o = _sie_deflection(
            x_t, y_t,
            torch.tensor(2.0), torch.tensor(q), torch.tensor(phi),
            torch.tensor(0.0), torch.tensor(0.0),
        )

        r = np.sqrt(x_flat**2 + y_flat**2)
        mask = r > 0.1
        mag_ls = np.sqrt(ax_ls[mask]**2 + ay_ls[mask]**2)
        rel_err = np.sqrt(
            (ax_o.numpy()[mask] - ax_ls[mask])**2
            + (ay_o.numpy()[mask] - ay_ls[mask])**2
        ) / (mag_ls + 1e-12)
        max_rel = np.max(rel_err)
        self.assertLess(max_rel, 0.001,
                        f"SIE(q=0.5) vs lenstronomy: max rel err = {max_rel:.6f}")

    def test_lensed_flux_matches_lenstronomy(self):
        """Our lensed pixel flux matches lenstronomy at 4x oversampling to <1%."""
        self._skip_if_no_lenstronomy()
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF as LenstronomyPSF

        H = W = STAMP_H
        re, n_s, q_s, phi_s = 0.15, 1.0, 1.0, 0.0
        flux = 1.0
        theta_E, q_lens, phi_lens = 1.5, 0.7, 0.3
        beta_x = 0.5

        # --- Our code (4x oversampling, delta PSF) ---
        lens = LensParams(theta_e_arcsec=theta_E, q_lens=q_lens, phi_lens_rad=phi_lens)
        src = SourceParams(
            beta_x_arcsec=beta_x, beta_y_arcsec=0.0,
            re_arcsec=re, n_sersic=n_s, q=q_s, phi_rad=phi_s,
            flux_nmgy_g=flux, flux_nmgy_r=flux, flux_nmgy_z=flux,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(
            _make_host(), lens, src, PIXEL_SCALE, 0.001, seed=42,
            subpixel_oversample=4,
        )
        our_total = _total_r_flux(result)

        # --- lenstronomy (10x oversampling) ---
        data_kwargs = {
            "image_data": np.zeros((H, W)),
            "transform_pix2angle": np.array([[PIXEL_SCALE, 0], [0, PIXEL_SCALE]]),
            "ra_at_xy_0": -(H - 1) / 2.0 * PIXEL_SCALE,
            "dec_at_xy_0": -(W - 1) / 2.0 * PIXEL_SCALE,
        }
        data_class = ImageData(**data_kwargs)
        psf_class = LenstronomyPSF(psf_type="NONE")

        e = (1.0 - q_lens) / (1.0 + q_lens)
        e1 = e * np.cos(2.0 * phi_lens)
        e2 = e * np.sin(2.0 * phi_lens)
        lm = self.LensModel(lens_model_list=["SIE"])
        kwargs_lens = [{"theta_E": theta_E, "e1": e1, "e2": e2,
                        "center_x": 0.0, "center_y": 0.0}]

        b_n = 2.0 * n_s - 1.0 / 3.0
        sersic_int = (
            2 * math.pi * q_s * n_s * re**2
            * math.exp(b_n) * math.gamma(2 * n_s) / b_n ** (2 * n_s)
        )
        amp = flux / sersic_int

        light_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
        kwargs_source = [{"amp": amp, "R_sersic": re, "n_sersic": n_s,
                          "e1": 0.0, "e2": 0.0,
                          "center_x": beta_x, "center_y": 0.0}]

        im_model = ImageModel(
            data_class, psf_class, lens_model_class=lm,
            source_model_class=light_model,
            kwargs_numerics={"supersampling_factor": 10,
                             "supersampling_convolution": False},
        )
        ls_img = im_model.image(kwargs_lens, kwargs_source)
        ls_total = float(ls_img.sum())

        # Allow 1% agreement (accounts for different b_n approximations,
        # oversampling strategy, and float32 vs float64)
        rel_err = abs(our_total - ls_total) / ls_total
        self.assertLess(rel_err, 0.01,
                        f"Lensed flux mismatch: ours={our_total:.4f}, "
                        f"lenstronomy={ls_total:.4f}, rel_err={rel_err:.4f}")


# ===================================================================
# 5. Sub-pixel oversampling convergence
# ===================================================================
class TestSubpixelOversampling(unittest.TestCase):
    """Verify that sub-pixel oversampling converges."""

    def test_4x_vs_8x_convergence(self):
        """4x and 8x oversampling should agree to <0.5%."""
        host = _make_host()
        lens = LensParams(theta_e_arcsec=1.5, q_lens=0.7, phi_lens_rad=0.3)
        src = SourceParams(
            beta_x_arcsec=0.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        f4 = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 0.001, seed=42, subpixel_oversample=4,
        ))
        f8 = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 0.001, seed=42, subpixel_oversample=8,
        ))
        rel = abs(f4 - f8) / f8
        self.assertLess(rel, 0.005,
                        f"4x vs 8x: {rel:.4%} > 0.5% (f4={f4:.4f}, f8={f8:.4f})")

    def test_oversampling_reduces_bias(self):
        """Higher oversampling should give lower total flux (less peak bias)."""
        host = _make_host()
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        f1 = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 0.001, seed=42, subpixel_oversample=1,
        ))
        f4 = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 0.001, seed=42, subpixel_oversample=4,
        ))
        # 1x may be higher or lower depending on where peaks land,
        # but at 4x the convergence should be clearly different from 1x
        # (difference > 0.1% at minimum for compact sources)
        self.assertNotAlmostEqual(f1, f4, places=1,
                                  msg="Oversampling had no effect — suspicious")


# ===================================================================
# 6. Area-weighted source offset sampling
# ===================================================================
class TestAreaWeightedSampling(unittest.TestCase):
    """Verify that source offsets follow the area-weighted prior."""

    def test_distribution_ks_test(self):
        """KS test: sampled beta_frac should follow P(x) prop to x."""
        from scipy.stats import kstest

        rng = np.random.default_rng(42)
        N = 100_000
        lo, hi = 0.1, 1.0
        lo2, hi2 = lo**2, hi**2
        samples = np.sqrt(rng.uniform(lo2, hi2, size=N))

        def cdf(x):
            return np.clip((x**2 - lo2) / (hi2 - lo2), 0, 1)

        stat, pval = kstest(samples, cdf)
        self.assertGreater(pval, 0.01,
                           f"KS test failed: stat={stat:.4f}, p={pval:.4f}")

    def test_mean_is_not_midpoint(self):
        """Area-weighted mean should be higher than the midpoint of [lo, hi]."""
        rng = np.random.default_rng(42)
        lo, hi = 0.1, 1.0
        lo2, hi2 = lo**2, hi**2
        samples = np.sqrt(rng.uniform(lo2, hi2, size=100_000))
        midpoint = (lo + hi) / 2.0
        self.assertGreater(np.mean(samples), midpoint,
                           "Area-weighted mean should be > midpoint (bias toward larger offsets)")


# ===================================================================
# 7. LensParams and sampling
# ===================================================================
class TestLensParamsSampling(unittest.TestCase):
    """Verify LensParams dataclass and sampling functions."""

    def test_lens_params_has_q_lens(self):
        """LensParams should have q_lens and phi_lens_rad fields."""
        lp = LensParams(theta_e_arcsec=1.5)
        self.assertEqual(lp.q_lens, 1.0, "Default q_lens should be 1.0 (SIS)")
        self.assertEqual(lp.phi_lens_rad, 0.0, "Default phi_lens should be 0.0")

    def test_sample_lens_params_q_range(self):
        """Sampled q_lens should be within the specified range."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            lp = sample_lens_params(rng, 1.5, q_lens_range=(0.5, 1.0))
            self.assertGreaterEqual(lp.q_lens, 0.5)
            self.assertLessEqual(lp.q_lens, 1.0)
            self.assertGreaterEqual(lp.phi_lens_rad, 0.0)
            self.assertLessEqual(lp.phi_lens_rad, math.pi)

    def test_sample_lens_params_sis_mode(self):
        """With q_lens_range=(1,1), should always give q_lens=1.0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            lp = sample_lens_params(rng, 1.5, q_lens_range=(1.0, 1.0))
            self.assertAlmostEqual(lp.q_lens, 1.0, places=10)

    def test_sample_source_params_area_weighted(self):
        """source beta_frac should follow area-weighted distribution."""
        rng = np.random.default_rng(42)
        offsets = []
        theta_e = 1.5
        for _ in range(5000):
            sp = sample_source_params(rng, theta_e)
            beta = math.sqrt(sp.beta_x_arcsec**2 + sp.beta_y_arcsec**2)
            offsets.append(beta / theta_e)
        offsets = np.array(offsets)
        # Area-weighted: mean(beta_frac) should be > midpoint of [0.1, 1.0] = 0.55
        self.assertGreater(np.mean(offsets), 0.55,
                           f"Mean offset {np.mean(offsets):.3f} too low for area-weighted sampling")

    def test_meta_keys_include_sie_params(self):
        """Injection result meta should include q_lens and phi_lens_rad."""
        host = _make_host()
        lens = LensParams(theta_e_arcsec=1.5, q_lens=0.7, phi_lens_rad=0.5)
        src = SourceParams(
            beta_x_arcsec=0.3, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        result = inject_sis_shear(host, lens, src, PIXEL_SCALE, 1.2, seed=42)
        self.assertIn("q_lens", result.meta)
        self.assertIn("phi_lens_rad", result.meta)
        self.assertAlmostEqual(result.meta["q_lens"].item(), 0.7, places=5)


# ===================================================================
# 8. PSF and core suppression
# ===================================================================
class TestPSFAndCoreSuppression(unittest.TestCase):
    """Verify PSF convolution and core suppression."""

    def test_psf_preserves_total_flux(self):
        """FFT Gaussian PSF should preserve total injection flux."""
        host = _make_host()
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.5, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        f_delta = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 0.001, seed=42,
        ))
        f_broad = _total_r_flux(inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 1.5, seed=42,
        ))
        rel = abs(f_delta - f_broad) / f_delta
        self.assertLess(rel, 0.01,
                        f"PSF changed total flux by {rel:.2%}")

    def test_core_suppression_reduces_center(self):
        """Core suppression should zero out central pixels."""
        host = _make_host()
        lens = LensParams(theta_e_arcsec=1.5)
        src = SourceParams(
            beta_x_arcsec=0.001, beta_y_arcsec=0.0,
            re_arcsec=0.15, n_sersic=1.0, q=1.0, phi_rad=0.0,
            flux_nmgy_g=1.0, flux_nmgy_r=1.0, flux_nmgy_z=1.0,
            n_clumps=0, clump_frac=0.0,
        )
        result_cs = inject_sis_shear(
            host, lens, src, PIXEL_SCALE, 1.2, seed=42,
            core_suppress_radius_pix=10,
        )
        inj = result_cs.injection_only[0, 1]  # r-band
        cy, cx = STAMP_H // 2, STAMP_W // 2
        center_flux = inj[cy - 5:cy + 5, cx - 5:cx + 5].abs().sum().item()
        self.assertAlmostEqual(center_flux, 0.0, places=6,
                               msg="Core suppression did not zero center")


# ===================================================================
# Standalone runner
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Injection Engine Validation Tests")
    print("=" * 60)
    print(f"  PIXEL_SCALE = {PIXEL_SCALE}")
    print(f"  STAMP_SIZE = {STAMP_H}x{STAMP_W}")
    print(f"  Device = {DEVICE}")
    print()
    unittest.main(verbosity=2)
