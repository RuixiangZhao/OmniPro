#!/bin/bash
# =============================================================================
# OmniProact-Bench — LiveStar-8B Online Evaluation
# =============================================================================
# One-shot script: install deps + full evaluation on a fresh machine.
# LiveStar is vision-only (no audio) and uses the SVeD response-silence
# decoding; the per-tick cost is ≈ two forward passes over the cumulative
# frames, so long videos will be slow.
#
# Usage:
#   bash scripts/run_online_livestar.sh                   # Full run (all 2700 samples)
#   LIMIT=4 bash scripts/run_online_livestar.sh           # Quick smoke test
#   bash scripts/run_online_livestar.sh --limit 8
#   bash scripts/run_online_livestar.sh --tasks instant_event_alert,event_narration
#   bash scripts/run_online_livestar.sh --decode-factor 1.06   # less talkative
#   bash scripts/run_online_livestar.sh --max-tokens 128
#   bash scripts/run_online_livestar.sh --gpt-judge
# =============================================================================

set -e

# ── Install deps (idempotent) ────────────────────────────────────────────────
LIVESTAR_PKGS="${LIVESTAR_PKGS:-/path/to/livestar_pkgs}"
echo "[setup] Checking LiveStar dependencies..."

_needs_install=0
PYTHONPATH="${LIVESTAR_PKGS}:${PYTHONPATH:-}" python3 -c "import transformers; assert transformers.__version__.startswith('4.37')" 2>/dev/null || _needs_install=1

if [ "$_needs_install" -eq 1 ]; then
    echo "[setup] Installing LiveStar deps (transformers 4.37 shim, etc.)..."
    bash "$(dirname "$0")/install_livestar_deps.sh"
else
    echo "[setup] Dependencies already satisfied, skipping install."
fi

# Verify shim works
PYTHONPATH="${LIVESTAR_PKGS}:${PYTHONPATH:-}" python3 -c "
import transformers
print(f'  transformers={transformers.__version__} (shim)')
assert transformers.__version__.startswith('4.37'), f'Expected 4.37.x, got {transformers.__version__}'
" || { echo "[ERROR] LiveStar deps verification failed"; exit 1; }

# ── Model config ─────────────────────────────────────────────────────────────
MODEL="livestar"
MODEL_TAG="LiveStar-8B"
MODEL_PATH="${MODEL_PATH:-/path/to/huggingface_cache/LiveStar_8B}"

# LiveStar-specific defaults forwarded through EXTRA_ARGS.
DECODE_FACTOR="${DECODE_FACTOR:-1.04}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_FRAMES_SEG="${MAX_FRAMES_SEG:-60}"

# Parse LiveStar-specific flags out of $@ before _online_common.sh sees them.
REMAINING=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --decode-factor) DECODE_FACTOR="$2"; shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --max-frames-seg) MAX_FRAMES_SEG="$2"; shift 2 ;;
        *)               REMAINING+=("$1");  shift ;;
    esac
done
set -- "${REMAINING[@]}"

EXTRA_ARGS="${EXTRA_ARGS:-} --decode_factor ${DECODE_FACTOR} --max_new_speak_tokens ${MAX_TOKENS} --max_frames_per_segment ${MAX_FRAMES_SEG}"

# Use the LiveStar-pinned transformers 4.37 via PYTHONPATH shim.
export PYTHONPATH="${LIVESTAR_PKGS}:${PYTHONPATH:-}"

source "$(dirname "$0")/_online_common.sh"
