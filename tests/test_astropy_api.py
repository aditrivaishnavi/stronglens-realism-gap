#!/usr/bin/env python3
"""
Unit tests for Astropy and FITS API behavior.

These tests verify correct handling of:
1. Column name case sensitivity (DR10 uses lowercase)
2. Byte string decoding (char[] columns return bytes)
3. NaN/Inf handling in float columns
4. Endianness issues
5. Data type conversions

Run with: python -m pytest tests/test_astropy_api.py -v
"""
import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# DR10 SCHEMA CONSTANTS (verified from legacysurvey.org/dr10/catalogs)
# =============================================================================

DR10_COLUMN_NAMES = {
    # Identification - ALL LOWERCASE
    "release": "int16",
    "brickid": "int32",
    "brickname": "char[8]",
    "objid": "int32",
    "brick_primary": "boolean",
    
    # Position - ALL LOWERCASE
    "ra": "float64",
    "dec": "float64",
    "ra_ivar": "float32",
    "dec_ivar": "float32",
    
    # Morphology - ALL LOWERCASE
    "type": "char[3]",  # PSF, REX, DEV, EXP, SER, DUP
    
    # Quality flags - ALL LOWERCASE
    "maskbits": "int32",
    "fitbits": "int16",
    "ebv": "float32",
    
    # Photometry - ALL LOWERCASE
    "flux_g": "float32",
    "flux_r": "float32",
    "flux_z": "float32",
    "flux_w1": "float32",
    
    # Observational - ALL LOWERCASE
    "nobs_g": "int16",
    "nobs_r": "int16",
    "nobs_z": "int16",
    "psfsize_g": "float32",
    "psfsize_r": "float32",
    "psfsize_z": "float32",
    "psfdepth_g": "float32",
    "psfdepth_r": "float32",
    "psfdepth_z": "float32",
    "galdepth_g": "float32",
    "galdepth_r": "float32",
    "galdepth_z": "float32",
    
    # Morphology - ALL LOWERCASE
    "shape_r": "float32",
    "shape_e1": "float32",
    "shape_e2": "float32",
    "sersic": "float32",
    
    # Galactic extinction - ALL LOWERCASE
    "mw_transmission_g": "float32",
    "mw_transmission_r": "float32",
    "mw_transmission_z": "float32",
}


class TestDR10ColumnNames:
    """Verify DR10 uses lowercase column names."""
    
    def test_all_dr10_columns_are_lowercase(self):
        """DR10 sweep files use lowercase column names."""
        for col in DR10_COLUMN_NAMES.keys():
            assert col == col.lower(), f"DR10 column '{col}' should be lowercase"
    
    def test_common_columns_exist(self):
        """Essential columns must be present in DR10 schema."""
        essential = ["ra", "dec", "type", "brickname", "objid", "maskbits",
                     "flux_g", "flux_r", "flux_z", "nobs_z"]
        for col in essential:
            assert col in DR10_COLUMN_NAMES, f"Essential column '{col}' missing from DR10 schema"


class TestAstropyFITSBehavior:
    """Test astropy.io.fits behavior with realistic data."""
    
    @pytest.fixture
    def mock_dr10_fits(self, tmp_path):
        """Create a mock DR10 sweep FITS file with correct column names."""
        from astropy.io import fits
        from astropy.table import Table
        
        # Create mock data with DR10-style lowercase columns
        n_rows = 5
        data = {
            'ra': np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64),
            'dec': np.array([10.0, 20.0, -10.0, -20.0, 0.0], dtype=np.float64),
            'type': np.array(['SER', 'DEV', 'EXP', 'REX', 'PSF'], dtype='U3'),  # char[3]
            'brickname': np.array(['1000p100', '1500p200', '2000m100', '2500m200', '3000p000'], dtype='U8'),
            'objid': np.array([1, 2, 3, 4, 5], dtype=np.int32),
            'maskbits': np.array([0, 0, 2, 0, 32], dtype=np.int32),
            'fitbits': np.array([0, 0, 0, 1, 0], dtype=np.int16),
            'flux_g': np.array([10.5, 5.0, 2.0, np.nan, 1.0], dtype=np.float32),
            'flux_r': np.array([15.0, 8.0, 3.0, 1.0, np.inf], dtype=np.float32),
            'flux_z': np.array([20.0, 10.0, 4.0, 0.5, -1.0], dtype=np.float32),
            'nobs_z': np.array([5, 3, 8, 1, 10], dtype=np.int16),
            'shape_r': np.array([1.5, 2.0, 3.5, 0.5, np.nan], dtype=np.float32),
            'ebv': np.array([0.02, 0.03, 0.01, 0.05, 0.02], dtype=np.float32),
        }
        
        table = Table(data)
        fits_path = tmp_path / "mock_sweep_dr10.fits"
        table.write(fits_path, format='fits', overwrite=True)
        
        return str(fits_path)
    
    def test_column_names_are_lowercase(self, mock_dr10_fits):
        """Column names from astropy should be lowercase."""
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            col_names = data.dtype.names
            
            # ALL column names should be lowercase
            for col in col_names:
                assert col == col.lower(), f"Column name '{col}' is not lowercase"
            
            # Verify specific columns exist as lowercase
            assert 'ra' in col_names, "Column 'ra' should exist (lowercase)"
            assert 'RA' not in col_names, "Column 'RA' should NOT exist (uppercase)"
            assert 'type' in col_names, "Column 'type' should exist (lowercase)"
            assert 'TYPE' not in col_names, "Column 'TYPE' should NOT exist (uppercase)"
    
    def test_string_column_returns_bytes_or_str(self, mock_dr10_fits):
        """
        Test how astropy handles char[] columns.
        
        CRITICAL: char[] columns may return numpy byte strings that need decoding!
        """
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            type_val = data['type'][0]
            
            # astropy may return either bytes or str depending on version
            # We need to handle both
            print(f"Type column raw value: {type_val!r}")
            print(f"Type of type column: {type(type_val)}")
            
            # Safe conversion
            if isinstance(type_val, bytes):
                decoded = type_val.decode('utf-8').strip()
            else:
                decoded = str(type_val).strip()
            
            assert decoded == 'SER', f"Expected 'SER', got '{decoded}'"
    
    def test_nan_handling_in_floats(self, mock_dr10_fits):
        """Test that NaN values are handled correctly."""
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            
            # flux_g[3] should be NaN
            flux_g_3 = data['flux_g'][3]
            assert np.isnan(flux_g_3), "flux_g[3] should be NaN"
            
            # NaN comparisons
            assert not (flux_g_3 > 0), "NaN > 0 should be False"
            assert not (flux_g_3 < 0), "NaN < 0 should be False"
            assert not (flux_g_3 == 0), "NaN == 0 should be False"
            
            # Safe NaN check before conversion
            if flux_g_3 is not None and not np.isnan(flux_g_3):
                safe_val = float(flux_g_3)
            else:
                safe_val = None
            
            assert safe_val is None, "NaN should convert to None safely"
    
    def test_inf_handling_in_floats(self, mock_dr10_fits):
        """Test that Inf values are handled correctly."""
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            
            # flux_r[4] should be Inf
            flux_r_4 = data['flux_r'][4]
            assert np.isinf(flux_r_4), "flux_r[4] should be Inf"
            
            # Inf comparisons
            assert flux_r_4 > 0, "Inf > 0 should be True"
            assert flux_r_4 > 1e10, "Inf > 1e10 should be True"
    
    def test_negative_flux_handling(self, mock_dr10_fits):
        """Test handling of negative flux values."""
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            
            # flux_z[4] should be -1.0 (negative)
            flux_z_4 = data['flux_z'][4]
            assert flux_z_4 < 0, "flux_z[4] should be negative"
            
            # Magnitude calculation should handle this
            if flux_z_4 is not None and flux_z_4 > 0:
                mag = 22.5 - 2.5 * np.log10(flux_z_4)
            else:
                mag = None
            
            assert mag is None, "Negative flux should result in None magnitude"
    
    def test_safe_column_access_pattern(self, mock_dr10_fits):
        """
        Demonstrate the CORRECT pattern for accessing DR10 columns.
        
        This is the pattern that should be used in spark_negative_sampling.py
        """
        from astropy.io import fits
        
        with fits.open(mock_dr10_fits) as hdul:
            data = hdul[1].data
            col_names = [c.lower() for c in data.dtype.names]  # Normalize to lowercase
            
            row = data[0]
            
            # CORRECT: Use lowercase column names
            ra = float(row['ra'])
            dec = float(row['dec'])
            
            # CORRECT: Decode byte strings for char[] columns
            type_raw = row['type']
            if isinstance(type_raw, bytes):
                galaxy_type = type_raw.decode('utf-8').strip().upper()
            else:
                galaxy_type = str(type_raw).strip().upper()
            
            # CORRECT: Check column existence with lowercase
            nobs_z = int(row['nobs_z']) if 'nobs_z' in col_names else 0
            
            # CORRECT: Handle NaN/Inf in float columns
            flux_g_raw = row['flux_g']
            if flux_g_raw is not None and np.isfinite(flux_g_raw):
                flux_g = float(flux_g_raw)
            else:
                flux_g = None
            
            assert ra == 100.0
            assert dec == 10.0
            assert galaxy_type == 'SER'
            assert nobs_z == 5
            assert flux_g == 10.5


class TestFitsioVsAstropy:
    """Compare fitsio and astropy behavior."""
    
    @pytest.fixture
    def mock_dr10_fits(self, tmp_path):
        """Create a mock DR10 sweep FITS file."""
        from astropy.io import fits
        from astropy.table import Table
        
        n_rows = 3
        data = {
            'ra': np.array([100.0, 150.0, 200.0], dtype=np.float64),
            'dec': np.array([10.0, 20.0, -10.0], dtype=np.float64),
            'type': np.array(['SER', 'DEV', 'EXP'], dtype='U3'),
            'brickname': np.array(['1000p100', '1500p200', '2000m100'], dtype='U8'),
            'nobs_z': np.array([5, 3, 8], dtype=np.int16),
        }
        
        table = Table(data)
        fits_path = tmp_path / "mock_sweep_fitsio.fits"
        table.write(fits_path, format='fits', overwrite=True)
        
        return str(fits_path)
    
    def test_fitsio_column_names(self, mock_dr10_fits):
        """Test fitsio column name behavior."""
        try:
            import fitsio
        except ImportError:
            pytest.skip("fitsio not installed")
        
        data = fitsio.read(mock_dr10_fits)
        col_names = data.dtype.names
        
        # fitsio also uses lowercase for DR10 files
        for col in col_names:
            assert col == col.lower(), f"Column '{col}' should be lowercase in fitsio"
    
    def test_fitsio_string_handling(self, mock_dr10_fits):
        """Test fitsio string column handling."""
        try:
            import fitsio
        except ImportError:
            pytest.skip("fitsio not installed")
        
        data = fitsio.read(mock_dr10_fits)
        type_val = data['type'][0]
        
        print(f"fitsio type raw: {type_val!r}")
        print(f"fitsio type type: {type(type_val)}")
        
        # fitsio may also return bytes
        if isinstance(type_val, bytes):
            decoded = type_val.decode('utf-8').strip()
        else:
            decoded = str(type_val).strip()
        
        assert decoded == 'SER', f"Expected 'SER', got '{decoded}'"


class TestColumnAccessFunctions:
    """Test utility functions for safe column access."""
    
    def test_safe_float_extraction(self):
        """Test safe float extraction handling NaN/Inf."""
        
        def safe_float(value):
            """Safely extract float, returning None for NaN/Inf/invalid."""
            if value is None:
                return None
            try:
                f = float(value)
                if not np.isfinite(f):
                    return None
                return f
            except (ValueError, TypeError):
                return None
        
        assert safe_float(10.5) == 10.5
        assert safe_float(np.float32(10.5)) == 10.5
        assert safe_float(np.nan) is None
        assert safe_float(np.inf) is None
        assert safe_float(-np.inf) is None
        assert safe_float(None) is None
        assert safe_float("invalid") is None
    
    def test_safe_string_extraction(self):
        """Test safe string extraction from FITS char[] columns."""
        
        def safe_string(value):
            """Safely extract string, handling bytes and whitespace."""
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode('utf-8').strip()
            return str(value).strip()
        
        assert safe_string('SER') == 'SER'
        assert safe_string(b'SER') == 'SER'
        assert safe_string('  SER  ') == 'SER'
        assert safe_string(b'  SER  ') == 'SER'
        assert safe_string(None) is None
    
    def test_safe_int_extraction(self):
        """Test safe int extraction."""
        
        def safe_int(value, default=0):
            """Safely extract int, returning default for invalid."""
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        assert safe_int(5) == 5
        assert safe_int(np.int16(5)) == 5
        assert safe_int(5.9) == 5
        assert safe_int(None) == 0
        assert safe_int(None, default=-1) == -1


def test_current_code_uses_wrong_case():
    """
    AUDIT: Verify that current code uses WRONG UPPERCASE column names.
    
    This test documents the bug in spark_negative_sampling.py
    """
    import re
    
    # Path to the spark job
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'emr', 'spark_negative_sampling.py'
    )
    
    if not os.path.exists(script_path):
        pytest.skip(f"Script not found: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find all column accesses like row["COLUMN_NAME"]
    pattern = r'row\["([A-Z_]+)"\]'
    uppercase_cols = set(re.findall(pattern, content))
    
    print(f"Found UPPERCASE column accesses: {uppercase_cols}")
    
    # These should all be lowercase in DR10!
    expected_wrong = {'RA', 'DEC', 'TYPE', 'NOBS_Z', 'MASKBITS', 'FLUX_G', 'FLUX_R', 
                      'FLUX_Z', 'FLUX_W1', 'SHAPE_R', 'BRICKNAME', 'OBJID', 'PSFSIZE_G',
                      'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                      'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'EBV', 'SHAPE_E1', 
                      'SHAPE_E2', 'SERSIC', 'FITBITS', 'MW_TRANSMISSION_G', 
                      'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z'}
    
    # This assertion SHOULD FAIL, documenting the bug
    if uppercase_cols:
        pytest.fail(
            f"BUG FOUND: Code uses UPPERCASE column names {uppercase_cols} "
            f"but DR10 uses lowercase! This will cause KeyError or silent failures."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
