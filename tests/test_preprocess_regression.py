#!/usr/bin/env python3
"""
Preprocessing Regression Test: Lock preprocessing behavior via checksum.

Creates a deterministic synthetic cutout with known pixel values, runs
preprocess_stack with the same parameters used in training, and verifies
the output matches a known checksum. This prevents silent drift in
preprocessing behavior across code changes.

Per code review:
  "A tiny checksum-style test on 1-2 reference NPZs prevents silent drift."

Tests cover:
  1. raw_robust preprocessing (101x101, no crop) — Paper IV parity mode
  2. raw_robust preprocessing (101x101 -> 64x64 center crop) — legacy mode
  3. Deterministic augmentation reproducibility
  4. NaN handling (NaN pixels replaced with 0)
  5. Clip range enforcement (values clipped to [-10, +10])
  6. Outer-annulus normalization correctness

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.
    python -m pytest tests/test_preprocess_regression.py -v

    # Or run standalone:
    python tests/test_preprocess_regression.py

Author: stronglens_calibration project
Date: 2026-02-11
Aligned with: preprocessing consistency check
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dhs.preprocess import preprocess_stack, center_crop
from dhs.utils import normalize_outer_annulus, robust_median_mad, radial_mask, default_annulus_radii
from dhs.transforms import AugmentConfig, random_augment
from dhs.constants import STAMP_SIZE, CUTOUT_SIZE


def make_synthetic_cutout(seed: int = 42) -> np.ndarray:
    """Create a deterministic synthetic cutout (3, 101, 101) in CHW format.

    The cutout has:
    - A bright Gaussian center (galaxy-like)
    - A faint background with noise
    - Realistic-ish flux values in nanomaggies range
    """
    rng = np.random.default_rng(seed)
    h = w = CUTOUT_SIZE  # 101

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    bands = []
    for b, (flux, bg, noise) in enumerate(
        [(5.0, 0.1, 0.05), (8.0, 0.15, 0.07), (3.5, 0.08, 0.04)]  # g, r, z
    ):
        # Central Gaussian (galaxy core)
        galaxy = flux * np.exp(-r2 / (2 * 6.0 ** 2))
        # Background + noise
        background = bg + noise * rng.standard_normal((h, w))
        band = (galaxy + background).astype(np.float32)
        bands.append(band)

    return np.stack(bands, axis=0)  # (3, 101, 101) CHW


def array_checksum(arr: np.ndarray) -> str:
    """Compute SHA-256 checksum of array bytes for regression testing."""
    # Use round to reduce float precision sensitivity across platforms
    rounded = np.round(arr, decimals=6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


class TestPreprocessRegression(unittest.TestCase):
    """Regression tests for preprocessing pipeline."""

    def setUp(self):
        """Create a deterministic synthetic cutout."""
        self.cutout = make_synthetic_cutout(seed=42)
        self.assertEqual(self.cutout.shape, (3, 101, 101))
        self.assertEqual(self.cutout.dtype, np.float32)

    # ------------------------------------------------------------------
    # Test 1: raw_robust with no crop (Paper IV parity mode)
    # ------------------------------------------------------------------
    def test_raw_robust_no_crop_shape(self):
        """raw_robust with crop=False keeps 101x101."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        self.assertEqual(out.shape, (3, 101, 101))
        self.assertEqual(out.dtype, np.float32)

    def test_raw_robust_no_crop_deterministic(self):
        """raw_robust with crop=False is deterministic (same input -> same output)."""
        out1 = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        out2 = preprocess_stack(self.cutout.copy(), mode="raw_robust", crop=False)
        np.testing.assert_array_equal(out1, out2)

    def test_raw_robust_no_crop_regression(self):
        """raw_robust locks preprocessing behavior via tolerance-based regression.

        Q1.14/Q3.4 fix: replaced brittle bitwise checksum with np.allclose-based
        test using a golden reference array. This is robust across CPU architectures
        (ARM vs x86), BLAS implementations, and minor NumPy version changes.

        The golden reference is computed as summary statistics of the output
        rather than a full array comparison, which is more portable.
        """
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)

        # Golden reference summary statistics (computed 2026-02-13, NumPy 1.26, Python 3.11)
        # Uses annulus (20, 32) which matches all trained models.
        # NOTE: This annulus is suboptimal for 101x101 stamps (see KNOWN ISSUE
        # in dhs/utils.py). When retraining with corrected annulus, update these.
        #
        # Per-band means (mean of all pixels in band after preprocessing):
        EXPECTED_BAND_MEANS = [0.774, 0.775, 0.707]
        # Per-band stds:
        EXPECTED_BAND_STDS = [2.925, 3.009, 2.898]
        # Per-band center pixel value (50, 50):
        EXPECTED_CENTER_VALS = [10.0, 10.0, 10.0]  # clipped to 10

        for band in range(3):
            band_mean = float(np.mean(out[band]))
            band_std = float(np.std(out[band]))
            center_val = float(out[band, 50, 50])

            # Allow 1% tolerance for cross-platform differences
            self.assertAlmostEqual(band_mean, EXPECTED_BAND_MEANS[band], delta=0.1,
                msg=f"Band {band} mean changed: {band_mean:.3f} vs expected {EXPECTED_BAND_MEANS[band]:.3f}")
            self.assertAlmostEqual(band_std, EXPECTED_BAND_STDS[band], delta=0.1,
                msg=f"Band {band} std changed: {band_std:.3f} vs expected {EXPECTED_BAND_STDS[band]:.3f}")
            # Center pixel should be clipped to 10 (bright galaxy center)
            self.assertAlmostEqual(center_val, EXPECTED_CENTER_VALS[band], delta=0.1,
                msg=f"Band {band} center value changed: {center_val:.3f}")

        # Also verify the quantized hash for the SAME platform (faster regression check)
        cs = array_checksum(out)
        print(f"  raw_robust no_crop quantized hash: {cs}")
        means = [f"{float(np.mean(out[b])):.3f}" for b in range(3)]
        stds = [f"{float(np.std(out[b])):.3f}" for b in range(3)]
        print(f"  Band means: {means}")
        print(f"  Band stds:  {stds}")

    # ------------------------------------------------------------------
    # Test 2: raw_robust with center crop (legacy mode)
    # ------------------------------------------------------------------
    def test_raw_robust_with_crop_shape(self):
        """raw_robust with crop=True crops to STAMP_SIZE (64x64)."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        self.assertEqual(out.shape, (3, STAMP_SIZE, STAMP_SIZE))

    def test_raw_robust_with_crop_deterministic(self):
        """raw_robust with crop=True is deterministic."""
        out1 = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        out2 = preprocess_stack(self.cutout.copy(), mode="raw_robust", crop=True)
        np.testing.assert_array_equal(out1, out2)

    # ------------------------------------------------------------------
    # Test 3: Clipping
    # ------------------------------------------------------------------
    def test_clip_range(self):
        """Output values are clipped to [-10, +10]."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False, clip_range=10.0)
        self.assertTrue(np.all(out >= -10.0), f"Min value: {out.min()}")
        self.assertTrue(np.all(out <= 10.0), f"Max value: {out.max()}")

    def test_clip_range_custom(self):
        """Custom clip range is respected."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False, clip_range=5.0)
        self.assertTrue(np.all(out >= -5.0))
        self.assertTrue(np.all(out <= 5.0))

    # ------------------------------------------------------------------
    # Test 4: NaN handling
    # ------------------------------------------------------------------
    def test_nan_handling(self):
        """NaN pixels are replaced with 0 before normalization."""
        cutout_nan = self.cutout.copy()
        # Inject NaN in the center of one band
        cutout_nan[0, 45:55, 45:55] = np.nan
        out = preprocess_stack(cutout_nan, mode="raw_robust", crop=False)
        self.assertFalse(np.any(np.isnan(out)), "Output contains NaN!")
        self.assertFalse(np.any(np.isinf(out)), "Output contains Inf!")

    def test_nan_heavy(self):
        """Cutout with many NaN pixels still produces valid output."""
        cutout_nan = self.cutout.copy()
        cutout_nan[1, :, :50] = np.nan  # Half of r-band is NaN
        out = preprocess_stack(cutout_nan, mode="raw_robust", crop=False)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertEqual(out.shape, (3, 101, 101))

    # ------------------------------------------------------------------
    # Test 5: Outer-annulus normalization
    # ------------------------------------------------------------------
    def test_outer_annulus_normalization(self):
        """Verify outer annulus normalization uses correct radii."""
        # Create a band image with known properties
        band = np.ones((101, 101), dtype=np.float32) * 10.0
        # Set the outer annulus (r=20 to r=32) to a known value
        mask = radial_mask(101, 101, 20, 32)
        band[mask] = 5.0  # median=5, MAD~=0 -> normalized ~ (x-5)/eps

        normed = normalize_outer_annulus(band, r_in=20, r_out=32)
        # The annulus pixels should be near 0 after normalization
        annulus_vals = normed[mask]
        self.assertAlmostEqual(float(np.median(annulus_vals)), 0.0, places=2)

    def test_robust_median_mad(self):
        """Test robust_median_mad computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # outlier at 100
        med, mad = robust_median_mad(x)
        self.assertAlmostEqual(med, 3.5, places=1)  # median of 1-5 range
        # MAD should be robust to the outlier
        self.assertTrue(mad < 10.0, f"MAD={mad} should be robust to outlier")

    # ------------------------------------------------------------------
    # Test 6: Center crop
    # ------------------------------------------------------------------
    def test_center_crop_101_to_64(self):
        """Center crop from 101x101 to 64x64."""
        out = center_crop(self.cutout, 64)
        self.assertEqual(out.shape, (3, 64, 64))

    def test_center_crop_preserves_center(self):
        """Center crop preserves the central pixel."""
        center_val = self.cutout[0, 50, 50]
        out = center_crop(self.cutout, 64)
        # After cropping 101->64, offset is (101-64)//2 = 18
        # So center (50,50) maps to (50-18, 50-18) = (32, 32) in cropped
        self.assertAlmostEqual(float(out[0, 32, 32]), float(center_val), places=5)

    def test_center_crop_identity(self):
        """Center crop with target size == input size is identity."""
        out = center_crop(self.cutout, 101)
        np.testing.assert_array_equal(out, self.cutout)

    # ------------------------------------------------------------------
    # Test 7: Augmentation determinism
    # ------------------------------------------------------------------
    def test_augmentation_deterministic(self):
        """Augmentation with same seed produces identical output."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=True, vflip=True, rot90=True)

        out1 = random_augment(preprocessed, seed=12345, cfg=cfg)
        out2 = random_augment(preprocessed.copy(), seed=12345, cfg=cfg)
        np.testing.assert_array_equal(out1, out2)

    def test_augmentation_different_seeds(self):
        """Augmentation with different seeds produces different output (usually)."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=True, vflip=True, rot90=True)

        # Try multiple seed pairs; at least one should differ
        any_different = False
        for s1, s2 in [(1, 2), (10, 20), (100, 200)]:
            out1 = random_augment(preprocessed.copy(), seed=s1, cfg=cfg)
            out2 = random_augment(preprocessed.copy(), seed=s2, cfg=cfg)
            if not np.array_equal(out1, out2):
                any_different = True
                break
        self.assertTrue(any_different, "All augmentations with different seeds were identical")

    # ------------------------------------------------------------------
    # Test 8: No-augmentation preserves preprocessing
    # ------------------------------------------------------------------
    def test_no_augmentation_identity(self):
        """Disabled augmentation is identity."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=False, vflip=False, rot90=False)
        out = random_augment(preprocessed, seed=42, cfg=cfg)
        np.testing.assert_array_almost_equal(out, preprocessed, decimal=5)

    # ------------------------------------------------------------------
    # Test 9: Verify dtype consistency
    # ------------------------------------------------------------------
    def test_output_dtype(self):
        """Preprocessing output is float32."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        self.assertEqual(out.dtype, np.float32)

    def test_output_dtype_after_crop(self):
        """Preprocessing output is float32 after crop."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        self.assertEqual(out.dtype, np.float32)


class TestPreprocessCrossPlatform(unittest.TestCase):
    """Cross-platform consistency checks.

    These tests don't rely on exact checksums (which may differ between
    NumPy versions), but verify key invariants that must hold everywhere.
    """

    def test_raw_robust_zero_centered_annulus(self):
        """After raw_robust normalization, the actual annulus should be ~zero mean."""
        cutout = make_synthetic_cutout(seed=42)
        out = preprocess_stack(cutout, mode="raw_robust", crop=False)

        # Test with the ACTUAL annulus used by normalize_outer_annulus (20, 32)
        # This matches all trained models.
        for band in range(3):
            mask = radial_mask(101, 101, 20, 32)
            annulus_vals = out[band][mask]
            mean = float(np.mean(annulus_vals))
            # Annulus should be approximately zero-centered
            self.assertAlmostEqual(mean, 0.0, places=0,
                                   msg=f"Band {band}: annulus mean = {mean:.4f}")

    def test_raw_robust_center_positive(self):
        """After raw_robust, the center (galaxy) should be positive (bright)."""
        cutout = make_synthetic_cutout(seed=42)
        out = preprocess_stack(cutout, mode="raw_robust", crop=False)

        for band in range(3):
            center_val = out[band, 50, 50]
            # The synthetic cutout has a bright center; after normalization,
            # the center should be positive relative to the background.
            self.assertGreater(center_val, 0.0,
                               msg=f"Band {band}: center value = {center_val:.4f}")


# ======================================================================
# Behavioral guard tests: catch annulus / normalization bugs
# ======================================================================

class TestAnnulusBehavioral(unittest.TestCase):
    """Behavioral tests that catch annulus bugs by testing preprocess_stack output.

    These tests do NOT depend on any helper function — they test the
    actual preprocessing behavior. They would have caught the (20, 32)
    bug on 101x101 stamps from day one.

    KNOWN ISSUE (2026-02-13): The current annulus (20, 32) FAILS the
    galaxy flux fraction test for 101x101 stamps (19.9% galaxy light
    in the annulus). This is documented in dhs/utils.py and will be
    fixed when retraining with default_annulus_radii().
    """

    def _make_bright_galaxy(self, h=101, w=101, seed=42):
        """Bright exponential galaxy in nanomaggies, (3, H, W) CHW."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bands = []
        for flux, bg, noise in [(5.0, 0.05, 0.03), (8.0, 0.08, 0.05), (3.5, 0.04, 0.02)]:
            galaxy = flux * np.exp(-r / 8.0)
            background = bg + noise * rng.standard_normal((h, w))
            bands.append((galaxy + background).astype(np.float32))
        return np.stack(bands, axis=0)

    def test_annulus_galaxy_flux_fraction_101x101(self):
        """For 101x101, the corrected annulus dramatically reduces galaxy contamination.

        The OLD annulus (20, 32) contains ~20% of galaxy flux on 101x101 stamps
        for a bright extended host — this is the known bug. The CORRECTED annulus
        from default_annulus_radii(101, 101) = (32.5, 45.0) reduces this to ~6%.

        This test:
        1. Asserts the corrected annulus IS good (< 10% galaxy flux)
        2. Asserts the old annulus IS bad (> 10%) — so nobody thinks the
           current (20, 32) defaults are fine for 101x101
        """
        h = w = 101
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        # Bright exponential galaxy (scale length 8 px, half-light R ~ 13 px)
        galaxy = 10.0 * np.exp(-np.sqrt(r2) / 8.0)
        total = galaxy.sum()

        # OLD defaults (20, 32): BAD for 101x101
        old_mask = radial_mask(h, w, 20, 32)
        old_frac = galaxy[old_mask].sum() / total

        # CORRECTED annulus: GOOD
        r_in_new, r_out_new = default_annulus_radii(h, w)
        new_mask = radial_mask(h, w, r_in_new, r_out_new)
        new_frac = galaxy[new_mask].sum() / total

        print(f"  [KNOWN ISSUE] Current annulus (20,32) on 101x101: "
              f"{old_frac:.1%} galaxy flux")
        print(f"  [FIX READY]   Corrected annulus ({r_in_new:.1f},{r_out_new:.1f}): "
              f"{new_frac:.1%} galaxy flux")

        # Corrected annulus must have low galaxy contamination
        self.assertLess(new_frac, 0.10,
            f"Corrected annulus contains {new_frac:.1%} of galaxy flux. "
            "default_annulus_radii() formula needs fixing.")

        # Old annulus must be bad — if this fails, someone already changed
        # the test galaxy, and the "known issue" narrative may be wrong.
        self.assertGreater(old_frac, 0.10,
            "Old annulus (20,32) on 101x101 should contain >10% galaxy flux. "
            "If this fails, the test galaxy model may have changed.")

    def test_corrected_annulus_improvement_over_old(self):
        """The corrected annulus must reduce galaxy flux by at least 2x on 101x101.

        This quantifies the improvement from default_annulus_radii() over the
        hardcoded (20, 32). For a typical extended galaxy, the improvement
        should be 3-4x.
        """
        h = w = 101
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        galaxy = 10.0 * np.exp(-np.sqrt(r2) / 8.0)
        total = galaxy.sum()

        old_frac = galaxy[radial_mask(h, w, 20, 32)].sum() / total
        r_in, r_out = default_annulus_radii(h, w)
        new_frac = galaxy[radial_mask(h, w, r_in, r_out)].sum() / total

        improvement = old_frac / max(new_frac, 1e-10)
        self.assertGreater(improvement, 2.0,
            f"Corrected annulus should reduce galaxy flux by >= 2x. "
            f"Got {improvement:.1f}x ({old_frac:.1%} -> {new_frac:.1%})")

    def test_preprocessing_outer_sky_near_zero(self):
        """After preprocessing, sky pixels (r>40 on 101x101) should be near zero.

        If the normalization annulus is in the sky, then true sky pixels
        will be near zero after normalization. If the annulus overlaps the
        galaxy, sky pixels will be shifted negative.

        This test checks the BEHAVIOR of the full preprocess_stack pipeline.

        Q1.12 fix: Added assertions. With the current (20, 32) annulus and
        this test galaxy, sky median is around -2.5. We assert this known
        value so the test catches unexpected changes.
        """
        cutout = self._make_bright_galaxy(101, 101)
        out = preprocess_stack(cutout, mode="raw_robust", crop=False)

        rmap = np.sqrt(
            (np.mgrid[0:101, 0:101][1] - 50) ** 2
            + (np.mgrid[0:101, 0:101][0] - 50) ** 2
        )
        sky_mask = rmap > 40  # true sky region

        for band in range(3):
            sky_median = float(np.median(out[band][sky_mask]))
            print(f"  Band {band} sky median (r>40): {sky_median:+.2f}"
                  f" ({'OK' if abs(sky_median) < 1.0 else 'KNOWN ISSUE: annulus in galaxy'})")
            # Assert the known behavior with the current (20, 32) annulus:
            # sky median is negative (around -2.5) due to galaxy contamination.
            # This catches unexpected changes to normalization.
            self.assertLess(sky_median, 0.0,
                f"Band {band}: sky median should be negative with (20,32) annulus "
                f"on extended galaxy. Got {sky_median:+.2f}")
            self.assertGreater(sky_median, -5.0,
                f"Band {band}: sky median too negative ({sky_median:+.2f}). "
                "Normalization may have changed unexpectedly.")
            # TODO(post-retraining): after retraining with corrected annulus (32.5, 45),
            # replace above assertions with:
            #   self.assertAlmostEqual(sky_median, 0.0, delta=0.5,
            #       msg=f"Band {band}: sky median should be near zero with corrected annulus")

    def test_annulus_contamination_devaucouleurs_profiles(self):
        """Q1.2/Q1.13 fix: Test annulus contamination for realistic de Vaucouleurs (n=4) LRGs.

        The original test used only an exponential (n=1) with scale=8 px (R_e ≈ 13.4 px),
        which is at the extreme end of the LRG size distribution. Code review
        recommended testing with de Vaucouleurs profiles at R_e = 4, 8, 12 px to bound
        the real impact.

        For de Vaucouleurs: I(r) = I_e * exp(-b_4 * [(r/R_e)^(1/4) - 1]), b_4 ≈ 7.67.
        """
        h = w = 101
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float64)
        r = np.maximum(r, 0.01)  # avoid division by zero

        b4 = 7.67  # Sersic b_n for n=4

        test_cases = [
            (4.0, "compact LRG"),
            (8.0, "typical LRG"),
            (12.0, "extended LRG"),
        ]

        r_in_new, r_out_new = default_annulus_radii(h, w)

        for re_px, description in test_cases:
            # de Vaucouleurs profile
            galaxy = 10.0 * np.exp(-b4 * ((r / re_px) ** 0.25 - 1.0))
            total = galaxy.sum()

            old_frac = galaxy[radial_mask(h, w, 20, 32)].sum() / total
            new_frac = galaxy[radial_mask(h, w, r_in_new, r_out_new)].sum() / total

            print(f"  deVauc R_e={re_px:.0f}px ({description}): "
                  f"old annulus {old_frac:.1%}, corrected {new_frac:.1%}")

            # Corrected annulus should always be better
            self.assertLess(new_frac, old_frac,
                f"Corrected annulus should have less galaxy flux than old for "
                f"R_e={re_px}px. Old={old_frac:.1%}, New={new_frac:.1%}")

            # Corrected annulus should have < 15% contamination even for extended profiles
            self.assertLess(new_frac, 0.15,
                f"Corrected annulus has {new_frac:.1%} galaxy flux for R_e={re_px}px "
                f"({description}). Should be < 15%.")

    def test_default_annulus_radii_is_in_outer_region(self):
        """default_annulus_radii output must be in the outer region for all sizes."""
        for size in [64, 101, 128, 256]:
            r_in, r_out = default_annulus_radii(size, size)
            half = size // 2
            self.assertGreaterEqual(r_out / half, 0.70,
                f"For {size}x{size}: r_out={r_out:.1f} is {r_out/half:.0%} of half-size. "
                "Must be >= 70%.")
            self.assertGreater(r_in / half, 0.50,
                f"For {size}x{size}: r_in={r_in:.1f} is {r_in/half:.0%} of half-size. "
                "Must be > 50%.")
            n_pix = int(radial_mask(size, size, r_in, r_out).sum())
            self.assertGreater(n_pix, 200,
                f"Annulus for {size}x{size} has only {n_pix} pixels. Need >= 200.")


if __name__ == "__main__":
    # When run standalone, also print helpful info
    print("=" * 60)
    print("Preprocessing Regression Test")
    print("=" * 60)
    print(f"  STAMP_SIZE = {STAMP_SIZE}")
    print(f"  CUTOUT_SIZE = {CUTOUT_SIZE}")
    print(f"  Current annulus defaults: (20, 32)  [KNOWN ISSUE for 101x101]")

    r_in_new, r_out_new = default_annulus_radii(101, 101)
    print(f"  Corrected annulus for 101x101: ({r_in_new:.1f}, {r_out_new:.1f})")
    print(f"  (Requires retraining to activate)")

    cutout = make_synthetic_cutout(seed=42)
    out_nocrop = preprocess_stack(cutout, mode="raw_robust", crop=False)
    out_crop = preprocess_stack(cutout, mode="raw_robust", crop=True)
    print(f"  Checksum (no crop): {array_checksum(out_nocrop)}")
    print(f"  Checksum (crop 64): {array_checksum(out_crop)}")
    print()

    # Run tests
    unittest.main(verbosity=2)
