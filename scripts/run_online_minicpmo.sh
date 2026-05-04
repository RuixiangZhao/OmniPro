#!/bin/bash
# =============================================================================
# OmniProact-Bench — MiniCPM-o 4.5 Duplex Online Evaluation
# =============================================================================
# One-shot script: install deps + full evaluation on a fresh machine.
# Run on a dedicated machine with 8 GPUs (transformers will be pinned to 4.51).
#
# Usage:
#   bash scripts/run_online_minicpmo.sh                  # Full run (all 2700 samples)
#   LIMIT=8 bash scripts/run_online_minicpmo.sh          # Quick test (8/task)
#   bash scripts/run_online_minicpmo.sh --tasks instant_event_alert --limit 4
#   bash scripts/run_online_minicpmo.sh --gpt-judge      # enable LLM judge for EN/SSI
#
# NOTE: This script pins transformers==4.51.0. Run on a DEDICATED machine
#       (not shared with probe models that need transformers 4.57+).
# =============================================================================

set -e

# ── Install MiniCPM-o deps (idempotent, skips if already satisfied) ──────────
echo "[setup] Checking MiniCPM-o dependencies..."
_needs_install=0
python3 -c "import transformers; assert transformers.__version__.startswith('4.51')" 2>/dev/null || _needs_install=1
python3 -c "import stepaudio2" 2>/dev/null || _needs_install=1
if [ "$_needs_install" -eq 1 ]; then
    echo "[setup] Installing MiniCPM-o deps (transformers==4.51, minicpmo-utils, etc.)..."
    bash "$(dirname "$0")/install_minicpmo_deps.sh"
else
    echo "[setup] Dependencies already satisfied, skipping install."
fi

MODEL="minicpm-o"
MODEL_TAG="MiniCPM-o-4.5-Duplex"
MODEL_PATH="/path/to/pretrained_models/MiniCPM-o-4_5"

source "$(dirname "$0")/_online_common.sh"
