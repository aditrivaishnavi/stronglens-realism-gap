#!/usr/bin/env bash
# run_d06_corrected_priors.sh — D06: Full rerun with corrected injection priors
#
# CONTEXT:
#   D05 was the "full independent re-evaluation" but it used the OLD injection
#   priors (the defaults in injection_engine.py before the 2026-02-16 fixes).
#   Meanwhile, the paper (v10) Section 3.2 was updated to describe the CORRECTED
#   priors (K-corrected colours, narrowed beta_frac/n/Re, disabled clumps).
#   This created an internal inconsistency: the paper claims corrected priors
#   but all result numbers came from the old priors.
#
#   D06 fixes this by re-running ALL experiments with the corrected priors.
#   The engine defaults now encode the corrected priors, so scripts that use
#   engine defaults automatically pick them up. The key additions are:
#     - --save-cutouts on grid scripts (user requirement: save all 110k cutouts)
#     - 2-phase bright-arc (generate then score, for checkpointing)
#     - beta_frac_range for "restricted" bright-arc = engine default (0.10, 0.40)
#       instead of old (0.1, 0.55)
#
#   NOTE ON SKY NOISE: --add-sky-noise is NOT used in D06. The paper (v10,
#   lines 180-183) states: "the injected images already contain the survey's
#   background and host-galaxy noise and artefacts. The hypothesised missing
#   texture is primarily the shot noise associated with the added arc flux
#   itself." Adding extra Gaussian noise would double-count background variance
#   and contradict the paper's physical story. Only Poisson shot noise
#   (--add-poisson-noise) is the physically justified additional noise term.
#   Code review confirmed this assessment.
#
# CORRECTED PRIORS (now engine defaults, cross-checked vs configs/injection_priors.yaml):
#   beta_frac_range: (0.10, 0.40)   [was (0.1, 1.0)  — engine default pre-fix]
#   re_arcsec_range: (0.15, 0.50)   [was (0.05, 0.50) — engine default pre-fix]
#   n_range:         (0.5, 2.0)     [was (0.5, 4.0)   — engine default pre-fix]
#   g-r colour:      N(1.15, 0.30)  [was N(0.2, 0.25) — engine default pre-fix]
#   r-z colour:      N(0.85, 0.20)  [was N(0.1, 0.25) — engine default pre-fix]
#   clumps_prob:     0.0            [was 0.6           — engine default pre-fix]
#
# NOTE: D05 bright-arc used explicit --beta-frac-range 0.1 0.55 (not the engine
#   default). D06 bright-arc restricted uses the engine default (0.10, 0.40).
#   D05 grid used engine default (0.1, 1.0); D06 grid uses (0.10, 0.40).
#
# EXPERIMENTS (10 total):
#   [1] Bright-arc baseline (no Poisson, clip=10, bf default=0.10-0.40)
#   [2] Bright-arc Poisson only (clip=10, bf default)
#   [3] Bright-arc clip=20 (no Poisson, bf default)
#   [4] Bright-arc Poisson+clip=20 (bf default)
#   [5] Bright-arc unrestricted (no Poisson, clip=10, bf [0.1,1.0])
#   [6] Gain sweep sanity (gain=1e12, Poisson ON, clip=10, bf default)
#   [7] Selection function grid baseline (no Poisson, save cutouts)
#   [8] Selection function grid Poisson (save cutouts)
#   [9] Linear probe (real Tier-A vs low-bf injections)
#   [10] Tier-A scoring (no injection, unchanged from D05)
#
# SEEDS: bright-arc=42, grid=1337, probe=42, tier-a=42 (same as D01-D05)
#
# DISK ESTIMATE:
#   Grid cutouts: ~110k x 100KB = ~11 GB per condition (~22 GB total)
#   Bright-arc cutouts: ~9,600 cutouts x 100KB = ~1 GB
#   Total: ~23 GB
#
# Run:
#   cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
#   source .venv-lambda3/bin/activate
#   export PYTHONPATH=.
#   nohup bash scripts/run_d06_corrected_priors.sh > d06_log.txt 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/results"
RUN_ID="D06_$(date +%Y%m%d)_corrected_priors"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D06 Full Rerun with Corrected Priors"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo " Start:  $(date)"
echo "==========================================="
mkdir -p "$OUT_DIR"

# --- Script paths ---
BRIGHT_ARC="$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py"
GRID_SCRIPT="$PROJECT_DIR/scripts/selection_function_grid.py"
PROBE_SCRIPT="$PROJECT_DIR/scripts/feature_space_analysis.py"
TIER_A_SCRIPT="$PROJECT_DIR/sim_to_real_validations/real_lens_scoring.py"

# Common bright-arc args (2-phase: generate then score)
# NOTE: No --beta-frac-range => uses engine default (0.10, 0.40) = corrected priors
# NOTE: No --add-sky-noise => paper states host already contains survey noise;
#       only Poisson shot noise is the physically justified addition.
BA_COMMON=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --seed 42
)

# Common grid args
GRID_COMMON=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --host-split val
    --host-max 20000
    --thresholds 0.3 0.5 0.7
    --fpr-targets 0.001 0.0001
    --injections-per-cell 500
    --seed 1337
    --save-cutouts
)

# ============================================================
# [1/10] Bright-arc baseline (corrected priors, no Poisson)
# ============================================================
echo ""
echo "--- [1/10] Bright-arc baseline (no Poisson, clip=10, corrected priors) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --phase both \
    --out-dir "$OUT_DIR/ba_baseline"
echo "  Done:  $(date)"

# ============================================================
# [2/10] Bright-arc Poisson only (corrected priors)
# ============================================================
echo ""
echo "--- [2/10] Bright-arc Poisson only (clip=10, corrected priors) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --add-poisson-noise \
    --phase both \
    --out-dir "$OUT_DIR/ba_poisson"
echo "  Done:  $(date)"

# ============================================================
# [3/10] Bright-arc clip=20 (corrected priors)
# ============================================================
echo ""
echo "--- [3/10] Bright-arc clip=20 (no Poisson, corrected priors) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --clip-range 20 \
    --phase both \
    --out-dir "$OUT_DIR/ba_clip20"
echo "  Done:  $(date)"

# ============================================================
# [4/10] Bright-arc Poisson+clip=20 (corrected priors)
# ============================================================
echo ""
echo "--- [4/10] Bright-arc Poisson+clip=20 (corrected priors) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --add-poisson-noise \
    --clip-range 20 \
    --phase both \
    --out-dir "$OUT_DIR/ba_poisson_clip20"
echo "  Done:  $(date)"

# ============================================================
# [5/10] Bright-arc unrestricted (bf [0.1,1.0], corrected priors)
# ============================================================
echo ""
echo "--- [5/10] Bright-arc unrestricted (no Poisson, clip=10, bf [0.1,1.0]) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --beta-frac-range 0.1 1.0 \
    --phase both \
    --out-dir "$OUT_DIR/ba_unrestricted"
echo "  Done:  $(date)"

# ============================================================
# [6/10] Gain sweep sanity test (gain=1e12, corrected priors)
# ============================================================
echo ""
echo "--- [6/10] Gain sweep: gain=1e12 (Poisson ON but negligible) ---"
echo "  Start: $(date)"
python -u "$BRIGHT_ARC" \
    "${BA_COMMON[@]}" \
    --add-poisson-noise \
    --gain-e-per-nmgy 1000000000000 \
    --phase both \
    --out-dir "$OUT_DIR/ba_gain_1e12"
echo "  Done:  $(date)"

# ============================================================
# [7/10] Selection function grid baseline (no Poisson, save cutouts)
# ============================================================
echo ""
echo "--- [7/10] Grid baseline (no Poisson, save cutouts) ---"
echo "  Start: $(date)"
python -u "$GRID_SCRIPT" \
    "${GRID_COMMON[@]}" \
    --save-cutouts-dir "$OUT_DIR/grid_no_poisson/cutouts" \
    --out-dir "$OUT_DIR/grid_no_poisson"
echo "  Done:  $(date)"

# ============================================================
# [8/10] Selection function grid Poisson (save cutouts)
# ============================================================
echo ""
echo "--- [8/10] Grid Poisson (save cutouts) ---"
echo "  Start: $(date)"
python -u "$GRID_SCRIPT" \
    "${GRID_COMMON[@]}" \
    --add-poisson-noise \
    --save-cutouts-dir "$OUT_DIR/grid_poisson/cutouts" \
    --out-dir "$OUT_DIR/grid_poisson"
echo "  Done:  $(date)"

# ============================================================
# [9/10] Linear probe (real Tier-A vs low-bf injections)
# ============================================================
echo ""
echo "--- [9/10] Linear probe (corrected priors) ---"
echo "  Start: $(date)"
python -u "$PROBE_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/linear_probe" \
    --n-samples 500 \
    --theta-e 1.5 \
    --target-mag 19.0 \
    --seed 42
echo "  Done:  $(date)"

# ============================================================
# [10/10] Tier-A scoring (no injection, unchanged)
# ============================================================
echo ""
echo "--- [10/10] Tier-A scoring (unchanged, no injection involved) ---"
echo "  Start: $(date)"
python -u "$TIER_A_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/tier_a_scoring" \
    --tier-a-only \
    --seed 42
echo "  Done:  $(date)"

echo ""
echo "==========================================="
echo " D06 Complete"
echo " Results: $OUT_DIR"
echo " End:     $(date)"
echo "==========================================="
echo ""
echo "Sub-directories:"
echo "  ba_baseline/         — [1] Bright-arc baseline (corrected priors)"
echo "  ba_poisson/          — [2] Bright-arc Poisson only (corrected priors)"
echo "  ba_clip20/           — [3] Bright-arc clip=20 (corrected priors)"
echo "  ba_poisson_clip20/   — [4] Bright-arc Poisson+clip=20 (corrected priors)"
echo "  ba_unrestricted/     — [5] Bright-arc unrestricted bf [0.1,1.0] (corrected priors)"
echo "  ba_gain_1e12/        — [6] Gain sweep sanity test (corrected priors)"
echo "  grid_no_poisson/     — [7] Grid baseline + cutouts (corrected priors)"
echo "  grid_poisson/        — [8] Grid Poisson + cutouts (corrected priors)"
echo "  linear_probe/        — [9] Linear probe (corrected priors)"
echo "  tier_a_scoring/      — [10] Tier-A scoring (unchanged)"
echo ""
echo "KEY DIFFERENCES vs D05:"
echo "  1. All injection scripts now use CORRECTED engine defaults"
echo "     (K-corrected colours, narrowed beta_frac/n/Re, clumps disabled)"
echo "  2. --save-cutouts on grid experiments (110k cutouts saved)"
echo "  3. Bright-arc uses 2-phase (generate then score) with checkpointing"
echo "  4. Restricted beta_frac now (0.10, 0.40) [was (0.1, 0.55)]"
echo "  5. Poisson/noise RNG is per-injection seeded (deterministic + paired)"
echo "  6. NO --add-sky-noise (host already contains survey noise; only"
echo "     Poisson shot noise is physically justified per paper Section 3.2)"
echo ""
echo "Next: Update paper Section 4 with D06 numbers, prepare analysis."
