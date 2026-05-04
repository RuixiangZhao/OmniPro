#!/bin/bash
# =============================================================================
# OmniProact-Bench — Qwen2.5-Omni-7B (Audio ON) Full Probe Evaluation
# =============================================================================
# Usage:
#   bash scripts/run_probe_qwen2_5_omni.sh          # Full run
#   LIMIT=2 bash scripts/run_probe_qwen2_5_omni.sh  # Quick test
# =============================================================================

set -e

MODEL="qwen2.5-omni"
MODEL_TAG="Qwen2.5-Omni-7B"
MODEL_PATH="/path/to/pretrained_models/Qwen2.5-Omni-7B"

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

OUTDIR="${BENCH_DIR}/results/probe/${MODEL_TAG}"
TASKS="instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction"
LIMIT="${LIMIT:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CLIP_CACHE="${CLIP_CACHE:-/path/to/OmniProact-Bench/clip_cache}"

LOG_DIR="/path/to/OmniProact-Bench/omniproact_logs"
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

# ── Environment Check ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment check..." | tee -a "$LOG"

[ -d "$MODEL_PATH" ] || { echo "ERROR: Model not found: $MODEL_PATH" | tee -a "$LOG"; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "ERROR: nvidia-smi not found" | tee -a "$LOG"; exit 1; }
command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found" | tee -a "$LOG"; exit 1; }

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPUs: $GPU_COUNT" | tee -a "$LOG"
[ "$GPU_COUNT" -lt "$NUM_GPUS" ] && NUM_GPUS=$GPU_COUNT

# Install dependencies
_install_if_missing() {
    local pkg="$1"; local pip_name="${2:-$1}"
    python3 -c "import $pkg" 2>/dev/null || pip install --quiet "$pip_name"
}
_install_if_missing torch torch
_install_if_missing transformers transformers
_install_if_missing accelerate accelerate
_install_if_missing decord decord
_install_if_missing tqdm tqdm
_install_if_missing PIL pillow
python3 -c "import flash_attn" 2>/dev/null || pip install flash-attn --no-build-isolation
python3 -c "import qwen_omni_utils" 2>/dev/null || pip install --quiet qwen-omni-utils || \
    { echo "ERROR: Failed to install qwen-omni-utils" | tee -a "$LOG"; exit 1; }
python3 -c "import soundfile" 2>/dev/null || pip install --quiet soundfile 2>/dev/null || true

python3 -c "
import torch, transformers, flash_attn, decord, qwen_omni_utils
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  transformers={transformers.__version__}')
print(f'  flash_attn={flash_attn.__version__}')
print(f'  qwen_omni_utils OK')
" 2>&1 | tee -a "$LOG"

[ -f "data/benchmark.json" ] || { echo "ERROR: data/benchmark.json not found. See README.md for data setup."; exit 1; }
echo "  Environment OK!" | tee -a "$LOG"

# ── Run Evaluation ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Running probe evaluation..." | tee -a "$LOG"

python3 scripts/run_probe.py \
    --model "$MODEL" \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --num_gpus "$NUM_GPUS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --clip_cache_dir "$CLIP_CACHE" \
    --output_dir "$OUTDIR" 2>&1 | tee -a "$LOG"

# ── Compute Metrics ───────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 3] Computing metrics..." | tee -a "$LOG"

python3 scripts/compute_metrics.py \
    --pred_dir "$OUTDIR" --tolerance 3,5 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[DONE] $MODEL_TAG  $(date)" | tee -a "$LOG"
