# Technical Specifications

**Last Updated:** 2026-02-09

This document provides exact technical specifications for the stronglens-realism-gap pipeline.

---

## 1. NPZ Cutout Format

### File Format
- **Extension:** `.npz` (NumPy compressed archive)
- **Compression:** `np.savez_compressed()`

### Keys and Contents

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `cutout` | `(101, 101, 3)` | float32 | Image data: (height, width, channels) |
| `meta_galaxy_id` | scalar | str | Unique galaxy identifier |
| `meta_ra` | scalar | float64 | Right Ascension (degrees) |
| `meta_dec` | scalar | float64 | Declination (degrees) |
| `meta_type` | scalar | str | Tractor morphology type (SER/DEV/REX/EXP) |
| `meta_nobs_z` | scalar | int | Number of z-band exposures |
| `meta_psfsize_z` | scalar | float32 | PSF FWHM in z-band (arcsec) |
| `meta_flux_r` | scalar | float32 | r-band flux (nanomaggies) |
| `meta_cutout_url` | scalar | str | Source URL for cutout |
| `meta_download_timestamp` | scalar | str | ISO timestamp of download |

### Channel Order
- Channel 0: g-band
- Channel 1: r-band  
- Channel 2: z-band

### Units
- **Pixel values:** nanomaggies (DR10 standard)
- **Pixel scale:** 0.262 arcsec/pixel
- **Field of view:** 101 × 0.262 = 26.5 arcsec

### Generation Code
```python
# From emr/spark_generate_cutouts.py:319
np.savez_compressed(buffer, cutout=cutout, **{f"meta_{k}": v for k, v in metadata.items()})
```

---

## 2. Manifest Schema

### Training Manifest (scripts/generate_training_manifest.py)

| Column | Type | Description |
|--------|------|-------------|
| `s3_key` | str | S3 key path to .npz file |
| `s3_path` | str | Full S3 URI (s3://bucket/key) |
| `galaxy_id` | str | Unique identifier |
| `filename` | str | NPZ filename |
| `label` | int | 1=positive (lens), 0=negative |
| `cutout_type` | str | "positive" or "negative" |
| `split` | str | "train", "val", or "test" |
| `ra` | float64 | (optional) Right Ascension |
| `dec` | float64 | (optional) Declination |

### Negative Pool Manifest (emr/spark_negative_sampling.py)

Full schema from `configs/negative_sampling_v1.yaml`:

| Column | Type | Description |
|--------|------|-------------|
| `galaxy_id` | str | `{brickname}_{objid}` |
| `brickname` | str | DR10 brick identifier |
| `objid` | int | Object ID within brick |
| `ra` | float64 | Right Ascension |
| `dec` | float64 | Declination |
| `type` | str | Tractor morphology type |
| `nobs_z` | int | z-band exposure count |
| `nobs_z_bin` | str | "1-2", "3-5", "6-10", "11+" |
| `type_bin` | str | "SER", "DEV", "REX", "OTHER" |
| `flux_g/r/z/w1` | float32 | Flux in nanomaggies |
| `mag_g/r/z` | float32 | AB magnitude |
| `g_minus_r`, `r_minus_z`, `z_minus_w1` | float32 | Colors |
| `psfsize_g/r/z` | float32 | PSF FWHM (arcsec) |
| `psfdepth_g/r/z` | float32 | 5σ point source depth |
| `galdepth_g/r/z` | float32 | Galaxy depth |
| `ebv` | float32 | E(B-V) extinction |
| `maskbits` | int | DR10 mask flags |
| `healpix_64` | int64 | HEALPix index (nside=64) |
| `healpix_128` | int64 | HEALPix index (nside=128) |
| `split` | str | "train", "val", "test" |
| `pool` | str | "N1", "N2", or "positive" |
| `sweep_file` | str | Source sweep filename |
| `pipeline_version` | str | Version string |
| `git_commit` | str | Git commit hash |

---

## 3. Split Script and HEALPix Implementation

### Location
`emr/sampling_utils.py`

### HEALPix Computation

```python
def compute_healpix(ra: float, dec: float, nside: int) -> int:
    """
    Compute HEALPix index for given coordinates.
    Uses healpy if available, falls back to manual calculation.
    """
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return int(hp.ang2pix(nside, theta, phi, nest=True))
```

**Configuration:** `nside=128` (from `configs/negative_sampling_v1.yaml`)

### Split Assignment

```python
def assign_split(healpix_idx: int, allocations: Dict[str, float], seed: int = 42) -> str:
    """
    Assign train/val/test split based on HEALPix index.
    Uses deterministic hash to ensure reproducibility.
    """
    hash_input = f"{healpix_idx}_{seed}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    hash_value = int.from_bytes(hash_bytes[:4], "big") / (2**32)
    
    cumulative = 0.0
    for split, proportion in sorted(allocations.items()):
        cumulative += proportion
        if hash_value < cumulative:
            return split
    return "train"
```

**Allocations:** train=0.70, val=0.15, test=0.15

### Exclusion Radius Enforcement

```python
def is_near_known_lens(
    ra: float,
    dec: float,
    known_coords: List[Tuple[float, float]],
    radius_arcsec: float,  # 11" = 5" + 2×θE_max
) -> bool:
    """
    Check if position is within exclusion radius of any known lens.
    Uses KD-tree for O(log n) lookup instead of O(n) linear scan.
    """
    # Uses scipy.spatial.cKDTree with spherical coordinates
    # Converts to Cartesian for proper distance calculation
```

**Exclusion radius:** 11 arcsec (5" buffer + 2×3.0" maximum θE)

---

## 4. Training Configuration

### configs/unpaired_matched_raw.yaml

```yaml
dataset:
  parquet_path: s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/
  mode: unpaired_manifest
  manifest_path: ./manifests/unpaired_matched.parquet
  preprocessing: raw_robust
  seed: 1337

augment:
  core_dropout_prob: 0.0
  az_shuffle_prob: 0.0

train:
  epochs: 50
  batch_size: 128
  lr: 1e-4
  weight_decay: 1e-4
  num_workers: 6
  device: cuda
  early_stopping_patience: 10
  out_dir: ./runs/R3_unpaired_matched_raw
  mixed_precision: true
```

### Shortcut Gates (dhs/gates.py)

```python
@dataclass
class GateResults:
    core_lr_auc: float       # Should be < 0.55 (near random)
    radial_profile_auc: float  # Should be < 0.60

def run_shortcut_gates(xs: np.ndarray, ys: np.ndarray) -> GateResults:
    """
    Gate 1.6: Core-only baseline - logistic regression on center 16×16 pixels
    Gate: radial profile - logistic regression on azimuthal median profile
    """
    return GateResults(
        core_lr_auc=core_lr_auc(xs, ys),
        radial_profile_auc=radial_lr_auc(xs, ys)
    )
```

**Usage:**
```bash
python dhs/scripts/run_gates.py --config configs/unpaired_matched_raw.yaml --split test --n 2048
```

**Pass Criteria:**
- `core_lr_auc < 0.55` (core-only classifier near random)
- `radial_profile_auc < 0.60` (radial profile not predictive)

---

## 5. Validation Reports

### S3 Location
```
s3://darkhaloscope/stronglens-realism-gap/validation/20260209_085438/
```

### Output Files

| File | Description |
|------|-------------|
| `summary.json` | Aggregate statistics and gate results |
| `per_cutout_metrics.parquet` | Per-cutout quality and feature metrics |
| `shortcut_auc_by_feature.parquet` | AUC for each shortcut feature |
| `failed_cutouts.parquet` | List of cutouts that failed quality gates |

### Summary Schema (summary.json)

```json
{
  "pipeline_version": "1.0.0",
  "timestamp": "2026-02-09T08:54:38+00:00",
  "total_cutouts": 421189,
  "positives": 5101,
  "negatives": 416088,
  "quality_pass_rate": 0.9987,
  "gates": {
    "core_brightness_ratio": {
      "value": 1.02,
      "threshold": 1.10,
      "passed": true
    },
    "nan_rejection_rate": {
      "value": 0.0013,
      "threshold": 0.05,
      "passed": true
    }
  },
  "shortcut_features": {
    "core_brightness_r": {"auc": 0.52, "ci_lower": 0.49, "flag": "green"},
    "annulus_brightness_r": {"auc": 0.71, "ci_lower": 0.68, "flag": "yellow"},
    "azimuthal_asymmetry": {"auc": 0.68, "ci_lower": 0.65, "flag": "expected_physics"}
  }
}
```

### Per-Cutout Metrics Schema

| Column | Type | Description |
|--------|------|-------------|
| `galaxy_id` | str | Galaxy identifier |
| `type` | str | "positive" or "negative" |
| `quality_ok` | bool | Passed all quality gates |
| `size_ok` | bool | Correct 101×101×3 shape |
| `total_nan_frac` | float | Fraction NaN overall |
| `core_nan_frac` | float | Fraction NaN in core |
| `core_brightness_r` | float | Mean r-band in r<8px |
| `annulus_brightness_r` | float | Mean r-band in 4-16px annulus |
| `outer_brightness_r` | float | Mean r-band in 16-28px annulus |
| `azimuthal_asymmetry` | float | Std of azimuthal sectors |
| `mad_r` | float | Median absolute deviation |
| `radial_gradient_r` | float | Core/outer brightness ratio |

---

## 6. Injection-Recovery Plan

### NOT APPLICABLE FOR CURRENT APPROACH

**Important:** The project uses **real-image training**, NOT injection-based training. This was a deliberate pivot from Plan B (injection-based) which failed due to sim-to-real gap.

### For Future Selection Function Measurement

If injection-recovery is needed for selection function calibration (Phase 5), the plan is:

#### Grid Axes (from IMPLEMENTATION_CHECKLIST.md)
- **θE:** 0.5-3.0" in 0.25" steps (11 bins)
- **PSF:** 0.9-1.8" in 0.15" steps (7 bins)
- **Depth:** 22.5-24.5 in 0.5 mag steps (5 bins)
- **Total cells:** 385

#### Minimum Injections
- **Per cell:** 200 minimum
- **Total:** ~77,000 injections minimum

#### Parameter Priors (Section 3A in checklist)
- **Source magnitude:** r-band 22-26 unlensed
- **Magnification (μ):** 5-30
- **Target arc_snr:** 0-5 with tail to ~10
- **Colors (g-r, r-z):** Match observed lens colors within ±0.2 mag

#### Storage Format
Injections would be stored as parquet with columns:
- `galaxy_id`, `ra`, `dec`
- `theta_e`, `source_mag_r`, `magnification`
- `arc_snr`, `detected` (boolean at threshold)
- `p_lens` (model score)

---

## 7. Paper Figure/Table List

### Must-Have Figures (from IMPLEMENTATION_CHECKLIST.md 7.6)

| # | Figure | Description |
|---|--------|-------------|
| 1 | **Data/Split Schematic** | Flow diagram showing positive catalog → negatives → cutouts → train/val/test |
| 2 | **Score Distributions by Stratum** | p(lens) histograms faceted by nobs_z, PSF, depth bins |
| 3 | **Selection Function Heatmaps** | C(θE, PSF) at fixed depth values (3 panels) |
| 4 | **Failure Mode Gallery** | 4×4 grid of representative false negatives and false positives with counts |
| 5 | **Independent Validation Table** | Recall on spectroscopically confirmed Tier-A anchors by stratum |

### Additional Figures (from project guidelines)

| # | Figure | Description |
|---|--------|-------------|
| 6 | **Reliability Diagram** | Calibration curve showing predicted vs actual probability |
| 7 | **ROC/PR Curves** | By stratum (nobs_z bins) |
| 8 | **Contaminant FPR by Category** | Bar chart: rings, spirals, mergers, edge-on disks |

### Tables

| # | Table | Description |
|---|-------|-------------|
| 1 | **Data Summary** | Positive/negative counts, Tier-A/B breakdown |
| 2 | **Stratification Distribution** | Count by (nobs_z_bin × type_bin) |
| 3 | **Recall by Stratum** | Tier-A recall with bootstrap CIs |
| 4 | **FPR by Contaminant** | False positive rate per morphology category |
| 5 | **Hyperparameters** | Architecture, training, augmentation settings |
| 6 | **Selection Function Lookup** | Completeness C(θE, PSF, depth) as downloadable table |

### Novelty Statement (7.8)

> "We provide a detector-audit framework for DR10 strong-lens searches, including injection-calibrated completeness surfaces and a condition- and confuser-resolved false-positive taxonomy, enabling bias-aware use of ML lens catalogs."

### Claims to Avoid (7.7)

- Overall precision in survey
- Cosmology constraints
- "Complete" lens sample claims
- Outperforming Huang et al. without matched protocol

---

*This document should be updated as specifications change.*
