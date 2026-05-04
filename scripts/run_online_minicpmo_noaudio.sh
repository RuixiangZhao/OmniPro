#!/bin/bash
# =============================================================================
# Online-mode evaluation: MiniCPM-o 4.5 Duplex (NoAudio) — Vision-only variant
# =============================================================================
# Evaluator skips audio extraction, model ignores audio.
# Same deps as audio version (transformers==4.51, minicpmo-utils).
#
# Usage:
#   bash scripts/run_online_minicpmo_noaudio.sh                  # Full run
#   LIMIT=8 bash scripts/run_online_minicpmo_noaudio.sh          # Quick test
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

MODEL="minicpm-o-noaudio"
MODEL_TAG="MiniCPM-o-4.5-Duplex-NoAudio"
MODEL_PATH="/path/to/pretrained_models/MiniCPM-o-4_5"

source "$(dirname "$0")/_online_common.sh"
