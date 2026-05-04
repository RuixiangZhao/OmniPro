#!/bin/bash
# =============================================================================
# OmniProact-Bench — MiniCPM-o 4.5 (Audio+Visual) Probe Evaluation
# =============================================================================
# Usage:
#   bash scripts/run_probe_minicpmo.sh          # Full run (2700 samples)
#   LIMIT=2 bash scripts/run_probe_minicpmo.sh  # Quick smoke test
#
# Environment:
#   - Requires transformers == 4.51.0 (MiniCPM-o pinned requirement)
#   - Uses the SAME deps as online mode (install_minicpmo_deps.sh)
#   - Model: ~9B params, ~19GB VRAM per GPU (bf16)
#   - Audio+Visual by default
#
# NOTE: This script pins transformers==4.51.0. Run on a DEDICATED machine
#       (not shared with models that need transformers 4.52+/4.57+).
# =============================================================================

set -e

MODEL="minicpm-o"
MODEL_TAG="MiniCPM-o-4.5"
MODEL_PATH="/path/to/pretrained_models/MiniCPM-o-4_5"

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

OUTDIR="${BENCH_DIR}/results/probe/${MODEL_TAG}"
TASKS="instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction"
LIMIT="${LIMIT:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CLIP_CACHE="${CLIP_CACHE:-${BENCH_DIR}/clip_cache}"

LOG_DIR="${BENCH_DIR}/omniproact_logs"
mkdir -p "$CLIP_CACHE" "$LOG_DIR" "$OUTDIR"
LOG="$LOG_DIR/${MODEL_TAG}_probe_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "$LOG"
echo " OmniProact-Bench Probe: $MODEL_TAG"        | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Model path : $MODEL_PATH"                  | tee -a "$LOG"
echo "  Output dir : $OUTDIR"                      | tee -a "$LOG"
echo "  Limit/task : $LIMIT (0=all)"               | tee -a "$LOG"
echo "  Num GPUs   : $NUM_GPUS"                    | tee -a "$LOG"
echo "  Started at : $(date)"                      | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# ── Step 1: Environment Setup (reuse online deps) ────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment setup..." | tee -a "$LOG"

command -v nvidia-smi &>/dev/null || { echo "ERROR: nvidia-smi not found" | tee -a "$LOG"; exit 1; }
command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found" | tee -a "$LOG"; exit 1; }

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPUs available: $GPU_COUNT" | tee -a "$LOG"
[ "$GPU_COUNT" -lt "$NUM_GPUS" ] && NUM_GPUS=$GPU_COUNT

# Use the same install script as online mode
_needs_install=0
python3 -c "import transformers; assert transformers.__version__.startswith('4.51')" 2>/dev/null || _needs_install=1
python3 -c "import stepaudio2" 2>/dev/null || _needs_install=1
python3 -c "import minicpmo" 2>/dev/null || _needs_install=1
if [ "$_needs_install" -eq 1 ]; then
    echo "  Installing MiniCPM-o deps (transformers==4.51, minicpmo-utils, etc.)..." | tee -a "$LOG"
    bash "${BENCH_DIR}/scripts/install_minicpmo_deps.sh" 2>&1 | tee -a "$LOG"
else
    echo "  Dependencies already satisfied, skipping install." | tee -a "$LOG"
fi

# Extra deps for probe mode
_install_if_missing() {
    local pkg="$1"; local pip_name="${2:-$1}"
    python3 -c "import $pkg" 2>/dev/null || {
        echo "  Installing $pip_name..." | tee -a "$LOG"
        pip install --quiet "$pip_name" 2>&1 | tail -1
    }
}
_install_if_missing decord decord
_install_if_missing tqdm tqdm

python3 -c "
import torch, transformers
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  transformers={transformers.__version__}')
assert transformers.__version__.startswith('4.51'), \
    f'MiniCPM-o requires transformers 4.51.x, got {transformers.__version__}'
import minicpmo
print('  minicpmo-utils: OK')
" 2>&1 | tee -a "$LOG"

echo "  Environment OK!" | tee -a "$LOG"

# ── Step 2: Check Model Weights ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Checking model weights..." | tee -a "$LOG"

[ -f "$MODEL_PATH/config.json" ] || {
    echo "  Model not found at $MODEL_PATH, downloading..." | tee -a "$LOG"
    # source your proxy/env setup script here if needed
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM-o-4.5', local_dir='$MODEL_PATH')" 2>&1 | tee -a "$LOG"
}
echo "  Model path: $MODEL_PATH" | tee -a "$LOG"
[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: config.json not found" | tee -a "$LOG"; exit 1; }

[ -f "data/benchmark.json" ] || { echo "ERROR: data/benchmark.json not found. See README.md for data setup."; exit 1; }

# ── Step 3: Run Evaluation ───────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 3] Running probe evaluation..." | tee -a "$LOG"

python3 scripts/run_probe.py \
    --model "$MODEL" \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --num_gpus "$NUM_GPUS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --clip_cache_dir "$CLIP_CACHE" \
    --output_dir "$OUTDIR" 2>&1 | tee -a "$LOG"

# ── Step 4: Compute Metrics ──────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 4] Computing metrics..." | tee -a "$LOG"

python3 scripts/compute_metrics.py \
    --pred_dir "$OUTDIR" --tolerance 3,5 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "[DONE] $MODEL_TAG  $(date)" | tee -a "$LOG"
echo "  Results in: $OUTDIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
