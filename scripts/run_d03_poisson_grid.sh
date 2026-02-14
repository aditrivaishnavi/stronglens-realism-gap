#!/usr/bin/env bash
# run_d03_poisson_grid.sh — D03: Selection function grid with Poisson noise + combined diagnostic
# Expects to be run from the stronglens_calibration/ directory with venv active.
#
# Phase 1 computations from the code review action plan:
#   1. Full selection function grid with --add-poisson-noise (~4 GPU-hours)
#   2. Combined Poisson + clip_range=20 bright-arc diagnostic (~10 min GPU)
#
# Run:
#   cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
#   source .venv-lambda3/bin/activate
#   export PYTHONPATH=.
#   bash scripts/run_d03_poisson_grid.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/results"
RUN_ID="D03_$(date +%Y%m%d)_poisson_grid"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D03 Selection Function Grid with Poisson Noise"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo "==========================================="
mkdir -p "$OUT_DIR"

# ============================================================
# [1/2] Selection Function Grid with Poisson Noise
# ============================================================
# This is the highest-priority computation from the code review
# action plan. Code review confirmed this is essential.
#
# Compared to the previous grid run (without Poisson noise):
#   - Adds shot noise to injected arcs (gain=150 e-/nmgy)
#   - Produces physically more correct completeness map
#   - Expected: marginal completeness rises from ~3.5% to ~5-7%
#
# Grid: 11 theta_E x 7 PSF x 5 depth = 385 cells x 500 injections = 192,500 injections
# Estimated runtime: ~4 GPU-hours on GH200
echo ""
echo "--- [1/2] Selection Function Grid with Poisson Noise (~4 GPU-hours) ---"
echo "  Start: $(date)"
python "$PROJECT_DIR/scripts/selection_function_grid.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/selection_function_poisson" \
    --host-split val \
    --host-max 20000 \
    --thresholds 0.3 0.5 0.7 \
    --fpr-targets 0.001 0.0001 \
    --injections-per-cell 500 \
    --add-poisson-noise \
    --seed 1337
echo "  Done: $(date)"

# ============================================================
# [2/2] Combined Poisson + clip_range=20 Bright-Arc Diagnostic
# ============================================================
# Additional test: test if Poisson noise and wider clip_range
# have additive effects on bright-arc detection.
#
# Results so far (individual effects at mag 20-21, p>0.3):
#   - Baseline (no Poisson, clip=10): 27.5%
#   - Poisson alone (clip=10):        45.0%  (+17.5pp)
#   - clip_range=20 alone:            37.0%  (+9.5pp)
#
# If combined reaches ~50-60%, this demonstrates two simple fixes
# close roughly half the gap to real-lens recall (89.3%).
#
# NOTE: clip_range=20 requires retraining for production use,
# but this is a valid diagnostic result for the paper.
echo ""
echo "--- [2/2] Poisson + clip_range=20 Combined Diagnostic (~10 min GPU) ---"
echo "  Start: $(date)"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/poisson_clip20_combined" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --clip-range 20 \
    --seed 42
echo "  Done: $(date)"

echo ""
echo "==========================================="
echo " D03 Complete"
echo " Results: $OUT_DIR"
echo "==========================================="
echo "  - selection_function_poisson/: Full grid with Poisson noise"
echo "  - poisson_clip20_combined/:    Bright-arc test with Poisson + clip=20"
echo ""
echo " Next: rsync results back to laptop and prepare analysis"
