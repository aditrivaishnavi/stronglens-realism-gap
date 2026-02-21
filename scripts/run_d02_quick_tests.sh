#!/usr/bin/env bash
# run_d02_quick_tests.sh — Phase 1-2 quick-win diagnostics
# Expects to be run from the stronglens-realism-gap/ directory with venv active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap/results"
RUN_ID="D02_$(date +%Y%m%d)_quick_tests"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D02 Quick Tests"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo "==========================================="
mkdir -p "$OUT_DIR"

# ============================================================
# Phase 1: Quick Tests (< 30 min GPU total)
# ============================================================

# --- [1a/7] Clip-range sweep: clip_range=20 (GPU) ---
echo ""
echo "--- [1a/7] Bright Arc Test — clip_range=20 (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/clip_range_20" \
    --beta-frac-range 0.1 0.55 \
    --clip-range 20 \
    2>&1 | tee "$OUT_DIR/clip_range_20.log"

# --- [1b/7] Clip-range sweep: clip_range=50 (GPU) ---
echo ""
echo "--- [1b/7] Bright Arc Test — clip_range=50 (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/clip_range_50" \
    --beta-frac-range 0.1 0.55 \
    --clip-range 50 \
    2>&1 | tee "$OUT_DIR/clip_range_50.log"

# --- [1c/7] Clip-range sweep: clip_range=100 (GPU) ---
echo ""
echo "--- [1c/7] Bright Arc Test — clip_range=100 (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/clip_range_100" \
    --beta-frac-range 0.1 0.55 \
    --clip-range 100 \
    2>&1 | tee "$OUT_DIR/clip_range_100.log"

# --- [2/7] Poisson noise test (GPU) ---
echo ""
echo "--- [2/7] Bright Arc Test — with Poisson noise (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/poisson_noise" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    2>&1 | tee "$OUT_DIR/poisson_noise.log"

# --- [3/7] Poisson + clip_range=50 combined (GPU) ---
echo ""
echo "--- [3/7] Bright Arc Test — Poisson noise + clip_range=50 (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/poisson_clip50" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --clip-range 50 \
    2>&1 | tee "$OUT_DIR/poisson_clip50.log"

# --- [4/7] Tier-A-only real lens scoring (GPU) ---
echo ""
echo "--- [4/7] Tier-A Only Real Lens Scoring (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/real_lens_scoring.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/tier_a_scoring" \
    --tier-a-only \
    2>&1 | tee "$OUT_DIR/tier_a_scoring.log"

# ============================================================
# Phase 2: Baseline Comparisons (< 60 min GPU)
# ============================================================

# --- [5/7] Unrestricted beta-frac baseline (GPU) ---
echo ""
echo "--- [5/7] Unrestricted Beta-Frac Baseline [0.1, 1.0] (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/unrestricted_bf" \
    --beta-frac-range 0.1 1.0 \
    2>&1 | tee "$OUT_DIR/unrestricted_bf.log"

# --- [6/7] Also run Tier-A + Tier-B scoring (for comparison) ---
echo ""
echo "--- [6/7] All-Tier Real Lens Scoring (GPU) ---"
python "$PROJECT_DIR/sim_to_real_validations/real_lens_scoring.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/all_tier_scoring" \
    2>&1 | tee "$OUT_DIR/all_tier_scoring.log"

# ============================================================
# Phase 3: Healpix investigation (CPU)
# ============================================================

# --- [7/7] Healpix investigation ---
echo ""
echo "--- [7/7] HEALPix NaN Investigation (CPU) ---"
python "$PROJECT_DIR/scripts/investigate_healpix_nan.py" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/healpix_investigation" \
    2>&1 | tee "$OUT_DIR/healpix_investigation.log"

echo ""
echo "==========================================="
echo " D02 Quick Tests Complete"
echo " Results in: $OUT_DIR"
echo "==========================================="
ls -la "$OUT_DIR"/
