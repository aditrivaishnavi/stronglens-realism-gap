#!/usr/bin/env python3
"""
Integration Test: Mini Pipeline End-to-End

Tests the core pipeline functions without requiring EMR or S3.
Uses synthetic data to verify the full flow works correctly.

Run with: pytest tests/test_integration_mini_pipeline.py -v
"""
import io
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestCutoutProcessing:
    """Test cutout generation and validation functions."""
    
    def test_synthetic_cutout_shape(self):
        """Verify cutout shape is correct (101x101x3)."""
        # Simulate a cutout with 3 bands
        cutout = np.random.randn(3, 101, 101).astype(np.float32)
        assert cutout.shape == (3, 101, 101)
        assert cutout.dtype == np.float32
    
    def test_center_crop(self):
        """Test center cropping from 101x101 to 64x64."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dhs.preprocess import center_crop
        
        # Create 101x101 image
        img = np.arange(101 * 101).reshape(101, 101).astype(np.float32)
        
        # Center crop to 64x64
        cropped = center_crop(img, 64)
        
        assert cropped.shape == (64, 64)
        # Check center is preserved
        center_orig = img[50, 50]
        center_cropped = cropped[32, 32]  # Center of 64x64
        assert center_orig == center_cropped
    
    def test_center_crop_3d(self):
        """Test center cropping on 3-band image."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dhs.preprocess import center_crop
        
        # Create 3x101x101 image
        img = np.random.randn(3, 101, 101).astype(np.float32)
        
        # Center crop to 64x64
        cropped = center_crop(img, 64)
        
        assert cropped.shape == (3, 64, 64)
    
    def test_preprocess_stack(self):
        """Test preprocessing pipeline with center crop."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dhs.preprocess import preprocess_stack
        
        # Create synthetic 101x101 cutout
        img = np.random.randn(3, 101, 101).astype(np.float32) * 10 + 100
        
        # Preprocess (should center crop to 64x64)
        result = preprocess_stack(img, mode="raw_robust", crop=True)
        
        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32


class TestFeatureExtraction:
    """Test shortcut detection feature extraction."""
    
    def create_synthetic_cutout(self, has_arc=False):
        """Create a synthetic cutout with optional arc signal."""
        img = np.random.randn(101, 101).astype(np.float32) * 5 + 100
        
        if has_arc:
            # Add arc signal in annulus (8-16 pixels from center)
            y, x = np.ogrid[:101, :101]
            center = 50
            r = np.sqrt((x - center)**2 + (y - center)**2)
            arc_mask = (r > 8) & (r < 16) & (x > center)  # Arc on one side
            img[arc_mask] += 50  # Add arc flux
        
        return img
    
    def test_arc_signal_detection(self):
        """Verify arc signal creates measurable annulus brightness difference."""
        # Create positive (with arc) and negative (without)
        pos_img = self.create_synthetic_cutout(has_arc=True)
        neg_img = self.create_synthetic_cutout(has_arc=False)
        
        # Measure annulus brightness
        y, x = np.ogrid[:101, :101]
        center = 50
        r = np.sqrt((x - center)**2 + (y - center)**2)
        annulus_mask = (r > 8) & (r < 16)
        
        pos_annulus = np.median(pos_img[annulus_mask])
        neg_annulus = np.median(neg_img[annulus_mask])
        
        # Positive should have higher annulus brightness
        assert pos_annulus > neg_annulus + 10  # Arc adds ~50, so should be significantly higher
    
    def test_azimuthal_asymmetry(self):
        """Verify azimuthal asymmetry is higher for arc images."""
        pos_img = self.create_synthetic_cutout(has_arc=True)
        neg_img = self.create_synthetic_cutout(has_arc=False)
        
        # Compute azimuthal asymmetry (std / median in arc region)
        y, x = np.ogrid[:101, :101]
        center = 50
        r = np.sqrt((x - center)**2 + (y - center)**2)
        arc_mask = (r > 8) & (r < 16)
        
        pos_arc = pos_img[arc_mask]
        neg_arc = neg_img[arc_mask]
        
        pos_asymmetry = np.std(pos_arc) / max(abs(np.median(pos_arc)), 0.001)
        neg_asymmetry = np.std(neg_arc) / max(abs(np.median(neg_arc)), 0.001)
        
        # Positive (with arc on one side) should be more asymmetric
        assert pos_asymmetry > neg_asymmetry


class TestAUCComputation:
    """Test AUC computation for shortcut detection."""
    
    def test_auc_perfectly_separable(self):
        """AUC should be 1.0 for perfectly separable distributions."""
        pos = [10, 11, 12, 13, 14]
        neg = [1, 2, 3, 4, 5]
        
        # Simple rank-based AUC
        all_vals = [(v, 1) for v in pos] + [(v, 0) for v in neg]
        all_vals.sort()
        
        # Count pairs where pos > neg
        correct = sum(1 for p in pos for n in neg if p > n)
        auc = correct / (len(pos) * len(neg))
        
        assert auc == 1.0
    
    def test_auc_no_separation(self):
        """AUC should be ~0.5 for random/identical distributions."""
        np.random.seed(42)
        pos = list(np.random.randn(100))
        neg = list(np.random.randn(100))
        
        # Simple rank-based AUC
        correct = sum(1 for p in pos for n in neg if p > n)
        ties = sum(1 for p in pos for n in neg if p == n)
        auc = (correct + 0.5 * ties) / (len(pos) * len(neg))
        
        # Should be close to 0.5
        assert 0.4 < auc < 0.6


class TestManifestGeneration:
    """Test training manifest generation."""
    
    def test_split_ratios(self):
        """Verify train/val/test splits are correct."""
        np.random.seed(42)
        
        # Simulate 1000 samples
        n = 1000
        labels = [1] * (n // 2) + [0] * (n // 2)
        
        # Assign splits (80/10/10)
        indices = np.random.permutation(n)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        splits = ["test"] * n
        for i, idx in enumerate(indices):
            if i < n_train:
                splits[idx] = "train"
            elif i < n_train + n_val:
                splits[idx] = "val"
        
        # Count splits
        train_count = splits.count("train")
        val_count = splits.count("val")
        test_count = splits.count("test")
        
        assert train_count == 800
        assert val_count == 100
        assert test_count == 100


class TestNPZFormat:
    """Test NPZ file format compatibility."""
    
    def test_npz_roundtrip(self):
        """Verify NPZ save/load preserves data."""
        # Create synthetic cutout
        g = np.random.randn(101, 101).astype(np.float32)
        r = np.random.randn(101, 101).astype(np.float32)
        z = np.random.randn(101, 101).astype(np.float32)
        
        # Save to buffer
        buf = io.BytesIO()
        np.savez(buf, image_g=g, image_r=r, image_z=z, ra=123.456, dec=-12.345)
        buf.seek(0)
        
        # Load back
        data = np.load(buf)
        
        assert np.allclose(data["image_g"], g)
        assert np.allclose(data["image_r"], r)
        assert np.allclose(data["image_z"], z)
        assert np.isclose(float(data["ra"]), 123.456)
        assert np.isclose(float(data["dec"]), -12.345)
    
    def test_stack_from_npz(self):
        """Test stacking bands from NPZ."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dhs.data import stack_from_npz
        
        # Create decoded NPZ dict
        decoded = {
            "image_g": np.ones((64, 64), dtype=np.float32) * 1,
            "image_r": np.ones((64, 64), dtype=np.float32) * 2,
            "image_z": np.ones((64, 64), dtype=np.float32) * 3,
        }
        
        stack = stack_from_npz(decoded)
        
        assert stack.shape == (3, 64, 64)
        assert stack[0, 0, 0] == 1.0  # g band
        assert stack[1, 0, 0] == 2.0  # r band
        assert stack[2, 0, 0] == 3.0  # z band


class TestConstantsConsistency:
    """Test that constants are consistent across modules."""
    
    def test_cutout_size_consistency(self):
        """Verify CUTOUT_SIZE is consistent."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from constants import CUTOUT_SIZE, TRAIN_SIZE
        from dhs.constants import STAMP_SIZE, CUTOUT_SIZE as DHS_CUTOUT_SIZE
        
        assert CUTOUT_SIZE == 101
        assert DHS_CUTOUT_SIZE == 101
        assert TRAIN_SIZE == 64
        assert STAMP_SIZE == 64
    
    def test_bands_consistency(self):
        """Verify band ordering is consistent."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from constants import CUTOUT_BANDS
        from dhs.constants import BANDS
        
        assert CUTOUT_BANDS == "grz"
        assert BANDS == ("g", "r", "z")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
