#!/usr/bin/env bash
# sync_to_lambda3.sh — Rsync code from laptop to Lambda3 NFS
# Usage: ./scripts/sync_to_lambda3.sh [--dry-run]
set -euo pipefail

# Configuration — edit these for your setup
LAMBDA3_HOST="${LAMBDA3_HOST:-lambda3}"
REMOTE_DIR="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN] No files will be transferred."
fi

echo "=== Syncing code to $LAMBDA3_HOST:$REMOTE_DIR ==="
echo "Source: $PROJECT_DIR/"

rsync -avz $DRY_RUN \
    --exclude '.venv*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '*.zip' \
    --exclude 'results/' \
    --exclude '.editor_cache/' \
    "$PROJECT_DIR/" \
    "$LAMBDA3_HOST:$REMOTE_DIR/"

echo ""
echo "=== Sync complete ==="
echo "SSH to Lambda3:  ssh $LAMBDA3_HOST"
echo "Activate venv:   source $REMOTE_DIR/.venv-lambda3/bin/activate"
echo "Run diagnostics: cd $REMOTE_DIR && bash scripts/run_diagnostics.sh"
