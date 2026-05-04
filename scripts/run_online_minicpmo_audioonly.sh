#!/bin/bash
# =============================================================================
# OmniProact-Bench — MiniCPM-o 4.5 Duplex Online Evaluation (Audio-Only)
# =============================================================================
# Audio-only ablation: drops video frames, keeps only audio input.
# Same deps as run_online_minicpmo.sh (transformers 4.51, stepaudio2, etc.)
#
# Usage:
#   bash scripts/run_online_minicpmo_audioonly.sh                  # Full run
#   LIMIT=50 bash scripts/run_online_minicpmo_audioonly.sh         # 50/task
#   bash scripts/run_online_minicpmo_audioonly.sh --gpt-judge      # enable LLM judge
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

MODEL="minicpm-o-audioonly"
MODEL_TAG="MiniCPM-o-4.5-Duplex-AudioOnly"
MODEL_PATH="/path/to/pretrained_models/MiniCPM-o-4_5"

source "$(dirname "$0")/_online_common.sh"
