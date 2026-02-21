#!/usr/bin/env bash
# store_d06_provenance.sh — Record SHA256 checksums for D06 code provenance.
#
# The D06 results have git_hash="unknown" because lambda3 is not a
# git repo. This script
# records file hashes as a cryptographic audit trail.
#
# Usage:
#   cd /lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap
#   bash scripts/store_d06_provenance.sh

set -euo pipefail

BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap"
D06_DIR="$BASE/results/D06_corrected_priors"
OUT="$D06_DIR/provenance.json"

echo "Computing SHA256 checksums for D06 provenance..."

# Files to hash: all code that executed during D06, plus data inputs
FILES=(
    # Core injection engine
    "$BASE/dhs/injection_engine.py"
    # Experiment scripts
    "$BASE/scripts/selection_function_grid.py"
    "$BASE/scripts/feature_space_analysis.py"
    "$BASE/sim_to_real_validations/bright_arc_injection_test.py"
    "$BASE/scripts/run_d06_corrected_priors.sh"
    # Configuration
    "$BASE/configs/injection_priors.yaml"
    # Analysis scripts
    "$BASE/scripts/analyze_d06_results.py"
    "$BASE/scripts/analyze_poisson_diagnostics.py"
    # Model checkpoint
    "$BASE/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
    # Manifest
    "$BASE/manifests/training_parity_70_30_v1.parquet"
)

# Build JSON
echo "{" > "$OUT"
echo "  \"generated_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$OUT"
echo "  \"hostname\": \"$(hostname)\"," >> "$OUT"
echo "  \"files\": {" >> "$OUT"

FIRST=true
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        HASH=$(sha256sum "$f" | awk '{print $1}')
        SIZE=$(stat --printf="%s" "$f" 2>/dev/null || stat -f "%z" "$f" 2>/dev/null || echo "unknown")
        REL=${f#"$BASE/"}
    else
        HASH="FILE_NOT_FOUND"
        SIZE="0"
        REL=${f#"$BASE/"}
    fi

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$OUT"
    fi
    printf "    \"%s\": {\"sha256\": \"%s\", \"size_bytes\": %s}" "$REL" "$HASH" "$SIZE" >> "$OUT"
done

echo "" >> "$OUT"
echo "  }" >> "$OUT"
echo "}" >> "$OUT"

echo "Provenance saved to: $OUT"
cat "$OUT"
