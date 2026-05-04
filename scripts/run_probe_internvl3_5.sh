#!/bin/bash
# =============================================================================
# OmniProact-Bench — InternVL3.5-8B (Vision-only) Full Probe
# =============================================================================
# Usage:
#   bash scripts/run_probe_internvl3.sh          # Full run (2700 samples)
#   LIMIT=2 bash scripts/run_probe_internvl3.sh  # Quick smoke test
#
# Environment:
#   - Requires transformers >= 4.52.1 (InternVL3.5 native support)
#   - Requires: flash-attn, decord, torchvision, pillow
#   - Model: ~8B params, ~18GB VRAM per GPU (bf16)
#   - Vision-only (no audio)
# =============================================================================

set -e

MODEL="internvl3-8b"
MODEL_TAG="InternVL3.5-8B"
MODEL_PATH="/path/to/pretrained_models/InternVL3_5-8B"

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

# ── Step 1: Environment Setup ────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment setup..." | tee -a "$LOG"

command -v nvidia-smi &>/dev/null || { echo "ERROR: nvidia-smi not found" | tee -a "$LOG"; exit 1; }
command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found" | tee -a "$LOG"; exit 1; }

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPUs available: $GPU_COUNT" | tee -a "$LOG"
[ "$GPU_COUNT" -lt "$NUM_GPUS" ] && NUM_GPUS=$GPU_COUNT

# Install dependencies
_install_if_missing() {
    local pkg="$1"; local pip_name="${2:-$1}"
    python3 -c "import $pkg" 2>/dev/null || {
        echo "  Installing $pip_name..." | tee -a "$LOG"
        pip install --quiet "$pip_name" 2>&1 | tail -1
    }
}
_install_if_missing torch torch
_install_if_missing decord decord
_install_if_missing tqdm tqdm
_install_if_missing PIL pillow
_install_if_missing torchvision torchvision
_install_if_missing accelerate accelerate
_install_if_missing numpy numpy
python3 -c "import flash_attn" 2>/dev/null || {
    echo "  Installing flash-attn..." | tee -a "$LOG"
    pip install flash-attn --no-build-isolation 2>&1 | tail -3
}

# InternVL3.5 requires transformers >= 4.52.1
CURRENT_TF=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
TF_OK=$(python3 -c "from packaging import version; print(int(version.parse('$CURRENT_TF') >= version.parse('4.52.1')))" 2>/dev/null || echo "0")
if [ "$TF_OK" != "1" ]; then
    echo "  Upgrading transformers to >=4.52.1 (InternVL3.5 requirement, current=$CURRENT_TF)..." | tee -a "$LOG"
    pip install 'transformers>=4.52.1' 2>&1 | tail -2
fi

python3 -c "
import torch, transformers
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  transformers={transformers.__version__}')
from packaging import version
assert version.parse(transformers.__version__) >= version.parse('4.52.1'), \
    f'transformers {transformers.__version__} < 4.52.1 required for InternVL3.5'
print('  InternVL3.5 support: OK')
" 2>&1 | tee -a "$LOG"

echo "  Environment OK!" | tee -a "$LOG"

# ── Step 2: Check Model Weights ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Checking model weights..." | tee -a "$LOG"

[ -f "$MODEL_PATH/config.json" ] || {
    echo "  Model not found at $MODEL_PATH, downloading..." | tee -a "$LOG"
    # source your proxy/env setup script here if needed
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('OpenGVLab/InternVL3_5-8B', local_dir='$MODEL_PATH')" 2>&1 | tee -a "$LOG"
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
