#!/usr/bin/env bash
# run_d05_full_reeval.sh — D05: Full independent re-evaluation of all injection realism experiments
#
# PURPOSE: Re-run ALL key experiments from D01-D04 in a single script to:
#   1. Independently verify historical results (D01-D04)
#   2. Fill gaps (missing "Poisson only" bright-arc test)
#   3. Run new diagnostics (gain sweep sanity test)
#   4. Produce a single, consistent, verified dataset for analysis
#
# EXPERIMENTS (10 total):
#   [1] Bright-arc baseline (no Poisson, clip=10, bf [0.1, 0.55]) — reproduces D01
#   [2] Bright-arc Poisson FIXED only (clip=10, bf [0.1, 0.55]) — NEW, missing from D04
#   [3] Bright-arc clip=20 (no Poisson, bf [0.1, 0.55]) — reproduces D02
#   [4] Bright-arc Poisson+clip=20 (bf [0.1, 0.55]) — reproduces D04 combined
#   [5] Bright-arc unrestricted (no Poisson, clip=10, bf [0.1, 1.0]) — reproduces D02
#   [6] Gain sweep sanity (gain=1e12, Poisson ON, clip=10, bf [0.1, 0.55]) — Additional diagnostic
#   [7] Selection function grid baseline (no Poisson) — reproduces D04 grid
#   [8] Selection function grid Poisson — reproduces D04 grid
#   [9] Linear probe (real Tier-A vs low-bf injections) — reproduces D01
#   [10] Tier-A scoring — reproduces D02
#
# SEEDS: bright-arc=42, grid=1337, probe=42, tier-a=42 (same as D01-D04)
#
# Run:
#   cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
#   source .venv-lambda3/bin/activate
#   export PYTHONPATH=.
#   nohup bash scripts/run_d05_full_reeval.sh > d05_log.txt 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/results"
RUN_ID="D05_$(date +%Y%m%d)_full_reeval"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D05 Full Re-evaluation"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo " Start:  $(date)"
echo "==========================================="
mkdir -p "$OUT_DIR"

# Common bright-arc args
BRIGHT_ARC="$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py"
BA_COMMON=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --seed 42
)

# Common grid args
GRID_SCRIPT="$PROJECT_DIR/scripts/selection_function_grid.py"
GRID_COMMON=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --host-split val
    --host-max 20000
    --thresholds 0.3 0.5 0.7
    --fpr-targets 0.001 0.0001
    --injections-per-cell 500
    --seed 1337
)

# ============================================================
# [1/10] Bright-arc baseline (reproduces D01 restricted beta_frac)
# ============================================================
echo ""
echo "--- [1/10] Bright-arc baseline (no Poisson, clip=10, bf [0.1,0.55]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 0.55 \
    --out-dir "$OUT_DIR/ba_baseline"
echo "  Done:  $(date)"

# ============================================================
# [2/10] Bright-arc Poisson FIXED only (NEW — missing from D04)
# ============================================================
echo ""
echo "--- [2/10] Bright-arc Poisson FIXED only (clip=10, bf [0.1,0.55]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --out-dir "$OUT_DIR/ba_poisson_fixed"
echo "  Done:  $(date)"

# ============================================================
# [3/10] Bright-arc clip=20 (reproduces D02 clip_range_20)
# ============================================================
echo ""
echo "--- [3/10] Bright-arc clip=20 (no Poisson, bf [0.1,0.55]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 0.55 \
    --clip-range 20 \
    --out-dir "$OUT_DIR/ba_clip20"
echo "  Done:  $(date)"

# ============================================================
# [4/10] Bright-arc Poisson+clip=20 (reproduces D04 combined)
# ============================================================
echo ""
echo "--- [4/10] Bright-arc Poisson+clip=20 (bf [0.1,0.55]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --clip-range 20 \
    --out-dir "$OUT_DIR/ba_poisson_clip20"
echo "  Done:  $(date)"

# ============================================================
# [5/10] Bright-arc unrestricted (reproduces D02 unrestricted_bf)
# ============================================================
echo ""
echo "--- [5/10] Bright-arc unrestricted (no Poisson, clip=10, bf [0.1,1.0]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 1.0 \
    --out-dir "$OUT_DIR/ba_unrestricted"
echo "  Done:  $(date)"

# ============================================================
# [6/10] Gain sweep sanity test (gain=1e12, should converge to baseline)
# ============================================================
echo ""
echo "--- [6/10] Gain sweep: gain=1e12 (Poisson ON but negligible) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --gain-e-per-nmgy 1000000000000 \
    --out-dir "$OUT_DIR/ba_gain_1e12"
echo "  Done:  $(date)"

# ============================================================
# [7/10] Selection function grid baseline (reproduces D04 no-Poisson)
# ============================================================
echo ""
echo "--- [7/10] Selection function grid baseline (no Poisson) ---"
echo "  Start: $(date)"
python -u "$GRID_SCRIPT" \
    "${GRID_COMMON[@]}" \
    --out-dir "$OUT_DIR/grid_no_poisson"
echo "  Done:  $(date)"

# ============================================================
# [8/10] Selection function grid Poisson (reproduces D04 Poisson)
# ============================================================
echo ""
echo "--- [8/10] Selection function grid Poisson fixed ---"
echo "  Start: $(date)"
python -u "$GRID_SCRIPT" \
    "${GRID_COMMON[@]}" \
    --add-poisson-noise \
    --out-dir "$OUT_DIR/grid_poisson_fixed"
echo "  Done:  $(date)"

# ============================================================
# [9/10] Linear probe (reproduces D01 embedding/probe)
# ============================================================
echo ""
echo "--- [9/10] Linear probe (real Tier-A vs low-bf injections) ---"
echo "  Start: $(date)"
python -u "$PROJECT_DIR/scripts/feature_space_analysis.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/linear_probe" \
    --n-samples 500 \
    --theta-e 1.5 \
    --target-mag 19.0 \
    --seed 42
echo "  Done:  $(date)"

# ============================================================
# [10/10] Tier-A scoring (reproduces D02 tier_a_scoring)
# ============================================================
echo ""
echo "--- [10/10] Tier-A scoring ---"
echo "  Start: $(date)"
python -u "$PROJECT_DIR/sim_to_real_validations/real_lens_scoring.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/tier_a_scoring" \
    --tier-a-only \
    --seed 42
echo "  Done:  $(date)"

echo ""
echo "==========================================="
echo " D05 Complete"
echo " Results: $OUT_DIR"
echo " End:     $(date)"
echo "==========================================="
echo ""
echo "Sub-directories:"
echo "  ba_baseline/         — [1] Bright-arc baseline (reproduces D01)"
echo "  ba_poisson_fixed/    — [2] Bright-arc Poisson FIXED only (NEW)"
echo "  ba_clip20/           — [3] Bright-arc clip=20 (reproduces D02)"
echo "  ba_poisson_clip20/   — [4] Bright-arc Poisson+clip=20 (reproduces D04)"
echo "  ba_unrestricted/     — [5] Bright-arc unrestricted (reproduces D02)"
echo "  ba_gain_1e12/        — [6] Gain sweep sanity test (NEW)"
echo "  grid_no_poisson/     — [7] Grid baseline (reproduces D04)"
echo "  grid_poisson_fixed/  — [8] Grid Poisson (reproduces D04)"
echo "  linear_probe/        — [9] Linear probe (reproduces D01)"
echo "  tier_a_scoring/      — [10] Tier-A scoring (reproduces D02)"
echo ""
echo "Next: rsync results back to laptop, run build_d04_comparison.py, prepare analysis."
