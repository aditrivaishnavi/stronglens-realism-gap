# Sim-to-Real Validation Suite

Comprehensive validation of the injection-recovery pipeline for the MNRAS selection function paper.

## Directory Structure

```
sim_to_real_validations/
  README.md                        # This file
  real_lens_scoring.py             # Score real confirmed lenses; recall vs FPR
  confuser_morphology_test.py      # Test if model detects galaxy morphology (shortcut)
  bright_arc_injection_test.py     # Injection at varying source magnitude (brightness gap)
  sim_to_real_validation.py        # Comprehensive sim-to-real diagnostic (SNR, brightness, histograms)
```

## Related Scripts (in scripts/)

```
scripts/
  selection_function_grid.py       # Main injection-recovery grid (308 cells, 200 inj/cell)
  validate_injections.py           # Runtime QA: flux conservation, score distributions, visual pages
  sensitivity_analysis.py          # 8-perturbation systematic uncertainty analysis
```

## Related Tests (in tests/)

```
tests/
  test_injection_engine.py         # 28 physics/regression tests for the injection engine
```

## Results Directories

```
results/
  selection_function_v4_finetune/  # Main selection function grid output
  injection_validation_v4/         # Injection QA (CSV, JSON, 20 PNG pages)
  sensitivity_v4_corrected/        # Corrected sensitivity analysis (9 perturbation CSVs)
  sim_to_real_validation/          # Sim-to-real diagnostic (JSON, NPZ, histograms)
```

## Usage

All scripts require `PYTHONPATH=.` from the `stronglens_calibration/code` directory on NFS, and a CUDA GPU.

```bash
cd /lambda/nfs/.../stronglens_calibration/code
export PYTHONPATH=.

# 1. Score real lenses
python sim_to_real_validations/real_lens_scoring.py \
  --checkpoint /path/to/best.pt --manifest /path/to/manifest.parquet \
  --out-dir results/real_lens_scoring

# 2. Confuser morphology test
python sim_to_real_validations/confuser_morphology_test.py \
  --checkpoint /path/to/best.pt --manifest /path/to/manifest.parquet \
  --out-dir results/confuser_morphology

# 3. Bright arc injection test
python sim_to_real_validations/bright_arc_injection_test.py \
  --checkpoint /path/to/best.pt --manifest /path/to/manifest.parquet \
  --out-dir results/bright_arc_test

# 4. Comprehensive sim-to-real validation
python sim_to_real_validations/sim_to_real_validation.py \
  --checkpoint /path/to/best.pt --manifest /path/to/manifest.parquet \
  --selection-function-csv results/selection_function_v4_finetune/selection_function.csv \
  --injection-validation-csv results/injection_validation_v4/injection_validation.csv \
  --out-dir results/sim_to_real_validation
```
