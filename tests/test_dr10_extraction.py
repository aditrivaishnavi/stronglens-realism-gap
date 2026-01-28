#!/usr/bin/env python3
"""
Local test for DR10 column extraction.

Verifies the fixes for:
1. Lowercase column names (DR10 standard)
2. Byte string decoding for char[] columns
3. NaN/Inf handling in float columns
4. Safe extraction functions

Run with: python tests/test_dr10_extraction.py
"""
import os
import sys
import tempfile
import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_dr10_fits(output_path: str):
    """Create a mock DR10 sweep FITS file with realistic data."""
    from astropy.io import fits
    from astropy.table import Table
    
    n_rows = 10
    
    # Create data with DR10-style lowercase columns
    # Note: astropy Table will preserve column name case
    data = {
        # Position columns (float64)
        'ra': np.array([100.0, 150.0, 200.0, 250.0, 300.0, 
                        110.0, 160.0, 210.0, 260.0, 310.0], dtype=np.float64),
        'dec': np.array([10.0, 20.0, -10.0, -20.0, 0.0,
                         15.0, 25.0, -15.0, -25.0, 5.0], dtype=np.float64),
        
        # Type column (char[3]) - will be stored as bytes in FITS
        'type': np.array(['SER', 'DEV', 'EXP', 'REX', 'PSF',
                          'SER', 'DEV', 'EXP', 'REX', 'DUP'], dtype='U3'),
        
        # Brickname column (char[8])
        'brickname': np.array(['1000p100', '1500p200', '2000m100', '2500m200', '3000p000',
                               '1100p150', '1600p250', '2100m150', '2600m250', '3100p050'], dtype='U8'),
        
        # Integer columns
        'objid': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32),
        'maskbits': np.array([0, 0, 2, 0, 32, 0, 0, 0, 0, 0], dtype=np.int32),
        'fitbits': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int16),
        'nobs_z': np.array([5, 3, 8, 1, 10, 4, 6, 9, 2, 7], dtype=np.int16),
        
        # Float columns (with some NaN/Inf for testing)
        'flux_g': np.array([10.5, 5.0, 2.0, np.nan, 1.0, 
                            8.0, 4.0, 3.0, 0.5, 6.0], dtype=np.float32),
        'flux_r': np.array([15.0, 8.0, 3.0, 1.0, np.inf, 
                            12.0, 7.0, 4.0, 1.5, 9.0], dtype=np.float32),
        'flux_z': np.array([20.0, 10.0, 4.0, 0.5, -1.0, 
                            18.0, 9.0, 5.0, 2.0, 11.0], dtype=np.float32),
        'flux_w1': np.array([5.0, 3.0, 1.5, 0.3, 0.5, 
                             4.0, 2.5, 2.0, 0.8, 3.5], dtype=np.float32),
        
        # More float columns
        'shape_r': np.array([1.5, 2.0, 3.5, 0.5, np.nan, 
                             1.2, 1.8, 2.5, 0.8, 1.0], dtype=np.float32),
        'ebv': np.array([0.02, 0.03, 0.01, 0.05, 0.02,
                         0.025, 0.035, 0.015, 0.045, 0.03], dtype=np.float32),
        
        # PSF size columns
        'psfsize_g': np.array([1.2, 1.3, 1.1, 1.5, 1.4,
                               1.25, 1.35, 1.15, 1.45, 1.3], dtype=np.float32),
        'psfsize_r': np.array([1.1, 1.2, 1.0, 1.4, 1.3,
                               1.15, 1.25, 1.05, 1.35, 1.2], dtype=np.float32),
        'psfsize_z': np.array([1.0, 1.1, 0.9, 1.3, 1.2,
                               1.05, 1.15, 0.95, 1.25, 1.1], dtype=np.float32),
    }
    
    table = Table(data)
    table.write(output_path, format='fits', overwrite=True)
    print(f"Created mock DR10 FITS file: {output_path}")
    return output_path


def test_column_names():
    """Test that column names are lowercase in the FITS file."""
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits_path = f.name
    
    try:
        create_mock_dr10_fits(fits_path)
        
        with fits.open(fits_path) as hdul:
            col_names = hdul[1].data.dtype.names
            
            print("\n=== Column Name Test ===")
            print(f"Column names: {col_names}")
            
            # All should be lowercase
            for col in col_names:
                assert col == col.lower(), f"Column '{col}' should be lowercase"
            
            # Verify key columns exist
            assert 'ra' in col_names
            assert 'dec' in col_names
            assert 'type' in col_names
            assert 'brickname' in col_names
            
            print("PASS: All column names are lowercase")
    
    finally:
        os.unlink(fits_path)


def test_byte_string_handling():
    """Test that byte strings from char[] columns are handled correctly."""
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits_path = f.name
    
    try:
        create_mock_dr10_fits(fits_path)
        
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            
            print("\n=== Byte String Test ===")
            
            # Check type column
            type_val = data['type'][0]
            print(f"type[0] raw value: {type_val!r}")
            print(f"type[0] type: {type(type_val)}")
            
            # Safe decoding
            if isinstance(type_val, bytes):
                decoded = type_val.decode('utf-8', errors='replace').strip()
            else:
                decoded = str(type_val).strip()
            
            print(f"Decoded type: '{decoded}'")
            assert decoded == 'SER', f"Expected 'SER', got '{decoded}'"
            
            # Check brickname column
            brickname_val = data['brickname'][0]
            print(f"\nbrickname[0] raw value: {brickname_val!r}")
            
            if isinstance(brickname_val, bytes):
                decoded_brick = brickname_val.decode('utf-8', errors='replace').strip()
            else:
                decoded_brick = str(brickname_val).strip()
            
            print(f"Decoded brickname: '{decoded_brick}'")
            assert decoded_brick == '1000p100', f"Expected '1000p100', got '{decoded_brick}'"
            
            print("PASS: Byte strings decoded correctly")
    
    finally:
        os.unlink(fits_path)


def test_nan_inf_handling():
    """Test that NaN/Inf values are handled correctly."""
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits_path = f.name
    
    try:
        create_mock_dr10_fits(fits_path)
        
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            
            print("\n=== NaN/Inf Handling Test ===")
            
            # flux_g[3] should be NaN
            flux_g_3 = data['flux_g'][3]
            print(f"flux_g[3]: {flux_g_3} (is NaN: {np.isnan(flux_g_3)})")
            assert np.isnan(flux_g_3), "flux_g[3] should be NaN"
            
            # Safe extraction function
            def safe_float(value):
                if value is None:
                    return None
                try:
                    f = float(value)
                    if not np.isfinite(f):
                        return None
                    return f
                except (ValueError, TypeError):
                    return None
            
            # NaN should become None
            result = safe_float(flux_g_3)
            print(f"safe_float(NaN): {result}")
            assert result is None, "NaN should convert to None"
            
            # flux_r[4] should be Inf
            flux_r_4 = data['flux_r'][4]
            print(f"\nflux_r[4]: {flux_r_4} (is Inf: {np.isinf(flux_r_4)})")
            assert np.isinf(flux_r_4), "flux_r[4] should be Inf"
            
            # Inf should become None
            result = safe_float(flux_r_4)
            print(f"safe_float(Inf): {result}")
            assert result is None, "Inf should convert to None"
            
            # Normal value should work
            flux_g_0 = data['flux_g'][0]
            result = safe_float(flux_g_0)
            print(f"\nsafe_float({flux_g_0}): {result}")
            assert result == 10.5, f"Expected 10.5, got {result}"
            
            print("PASS: NaN/Inf handled correctly")
    
    finally:
        os.unlink(fits_path)


def test_safe_extraction_pattern():
    """Test the complete safe extraction pattern used in process_fits_file."""
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits_path = f.name
    
    try:
        create_mock_dr10_fits(fits_path)
        
        # Import our utility functions
        from emr.sampling_utils import flux_to_mag, get_type_bin, get_nobs_z_bin
        
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            col_names_lower = set(c.lower() for c in data.dtype.names)
            
            print("\n=== Full Extraction Pattern Test ===")
            
            # Safe extraction helpers (matching the code in spark_negative_sampling.py)
            def safe_string(value):
                if value is None:
                    return None
                if isinstance(value, bytes):
                    return value.decode('utf-8', errors='replace').strip()
                return str(value).strip()
            
            def safe_float(value):
                if value is None:
                    return None
                try:
                    f = float(value)
                    if not np.isfinite(f):
                        return None
                    return f
                except (ValueError, TypeError):
                    return None
            
            def safe_int(value, default=0):
                if value is None:
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default
            
            # Process first row
            row = data[0]
            
            # Extract fields using the corrected pattern
            ra = safe_float(row['ra'])
            dec = safe_float(row['dec'])
            
            type_raw = row['type'] if 'type' in col_names_lower else None
            galaxy_type = safe_string(type_raw).upper() if type_raw is not None else "OTHER"
            
            nobs_z = safe_int(row['nobs_z']) if 'nobs_z' in col_names_lower else 0
            
            flux_g = safe_float(row['flux_g']) if 'flux_g' in col_names_lower else None
            flux_r = safe_float(row['flux_r']) if 'flux_r' in col_names_lower else None
            flux_z = safe_float(row['flux_z']) if 'flux_z' in col_names_lower else None
            
            brickname_raw = row['brickname'] if 'brickname' in col_names_lower else None
            brickname = safe_string(brickname_raw) if brickname_raw is not None else "unknown"
            objid = safe_int(row['objid']) if 'objid' in col_names_lower else 0
            
            print(f"ra: {ra}")
            print(f"dec: {dec}")
            print(f"type: {galaxy_type}")
            print(f"nobs_z: {nobs_z}")
            print(f"flux_g: {flux_g}")
            print(f"flux_r: {flux_r}")
            print(f"flux_z: {flux_z}")
            print(f"brickname: {brickname}")
            print(f"objid: {objid}")
            
            # Verify values
            assert ra == 100.0
            assert dec == 10.0
            assert galaxy_type == 'SER'
            assert nobs_z == 5
            assert flux_g == 10.5
            assert brickname == '1000p100'
            assert objid == 1
            
            # Test type_bin and nobs_z_bin
            type_bin = get_type_bin(galaxy_type)
            nobs_z_bin = get_nobs_z_bin(nobs_z)
            
            print(f"\ntype_bin: {type_bin}")
            print(f"nobs_z_bin: {nobs_z_bin}")
            
            assert type_bin == 'SER'
            assert nobs_z_bin == '3-5'
            
            # Test magnitude conversion
            mag_g = flux_to_mag(flux_g)
            print(f"\nmag_g: {mag_g}")
            assert mag_g is not None and abs(mag_g - 19.947) < 0.01
            
            print("\nPASS: Full extraction pattern works correctly")
    
    finally:
        os.unlink(fits_path)


def test_row_with_nan():
    """Test extraction of a row with NaN values."""
    from astropy.io import fits
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits_path = f.name
    
    try:
        create_mock_dr10_fits(fits_path)
        
        from emr.sampling_utils import flux_to_mag
        
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            col_names_lower = set(c.lower() for c in data.dtype.names)
            
            print("\n=== NaN Row Extraction Test ===")
            
            def safe_float(value):
                if value is None:
                    return None
                try:
                    f = float(value)
                    if not np.isfinite(f):
                        return None
                    return f
                except (ValueError, TypeError):
                    return None
            
            # Row 3 has NaN flux_g and row 4 has Inf flux_r
            row = data[3]
            
            flux_g = safe_float(row['flux_g'])
            print(f"Row 3 flux_g: {flux_g}")
            assert flux_g is None, "NaN flux should become None"
            
            mag_g = flux_to_mag(flux_g)
            print(f"Row 3 mag_g: {mag_g}")
            assert mag_g is None, "Magnitude from None flux should be None"
            
            row = data[4]
            flux_r = safe_float(row['flux_r'])
            print(f"Row 4 flux_r: {flux_r}")
            assert flux_r is None, "Inf flux should become None"
            
            print("PASS: NaN/Inf rows handled correctly")
    
    finally:
        os.unlink(fits_path)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DR10 EXTRACTION TESTS")
    print("=" * 60)
    
    try:
        test_column_names()
        test_byte_string_handling()
        test_nan_inf_handling()
        test_safe_extraction_pattern()
        test_row_with_nan()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
