#!/bin/bash
# =============================================================================
# OmniProact-Bench — Qwen3-VL-8B Full Probe Evaluation
# =============================================================================
# One-shot script: environment setup + verification + full evaluation.
#
# Usage:
#   bash scripts/run_probe_qwen3_vl.sh          # Full run (all 2700 samples)
#   LIMIT=2 bash scripts/run_probe_qwen3_vl.sh  # Quick test (2 per task)
#
# Resume: The script supports automatic resume. If interrupted, just re-run.
# =============================================================================

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="qwen3-vl"
MODEL_TAG="Qwen3-VL-8B"
MODEL_PATH="/path/to/pretrained_models/Qwen3-VL-8B-Instruct"

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

OUTDIR="${BENCH_DIR}/results/probe/${MODEL_TAG}"
TASKS="instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction"
LIMIT="${LIMIT:-0}"   # 0 = all samples (300 per task = 2700 total)
NUM_GPUS="${NUM_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CLIP_CACHE="${CLIP_CACHE:-/path/to/OmniProact-Bench/clip_cache}"

LOG_DIR="/path/to/OmniProact-Bench/omniproact_logs"
mkdir -p "$CLIP_CACHE" "$LOG_DIR" "$OUTDIR"
LOG="$LOG_DIR/${MODEL_TAG}_probe_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "$LOG"
echo " OmniProact-Bench Probe: $MODEL_TAG"        | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Bench dir  : $BENCH_DIR"                   | tee -a "$LOG"
echo "  Model path : $MODEL_PATH"                  | tee -a "$LOG"
echo "  Output dir : $OUTDIR"                      | tee -a "$LOG"
echo "  Tasks      : all 9"                        | tee -a "$LOG"
echo "  Limit/task : $LIMIT (0=all)"               | tee -a "$LOG"
echo "  Num GPUs   : $NUM_GPUS"                    | tee -a "$LOG"
echo "  Started at : $(date)"                      | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# ── Step 1: Environment Check & Install ───────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment check..." | tee -a "$LOG"

# Check model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH" | tee -a "$LOG"
    exit 1
fi
echo "  Model path OK" | tee -a "$LOG"

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found" | tee -a "$LOG"
    exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  GPUs: $GPU_COUNT" | tee -a "$LOG"
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "  WARNING: Requested $NUM_GPUS GPUs but only $GPU_COUNT available. Using $GPU_COUNT." | tee -a "$LOG"
    NUM_GPUS=$GPU_COUNT
fi

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found" | tee -a "$LOG"
    exit 1
fi
echo "  ffmpeg OK" | tee -a "$LOG"

# Check/install Python dependencies
echo "  Checking Python packages..." | tee -a "$LOG"

_install_if_missing() {
    local pkg="$1"
    local pip_name="${2:-$1}"
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo "  Installing $pip_name..." | tee -a "$LOG"
        pip install --quiet "$pip_name"
    fi
}

_install_if_missing torch torch
_install_if_missing transformers transformers
_install_if_missing accelerate accelerate
_install_if_missing decord decord
_install_if_missing tqdm tqdm
_install_if_missing PIL pillow

# flash_attn — critical for performance
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "  Installing flash-attn (may take several minutes)..." | tee -a "$LOG"
    pip install flash-attn --no-build-isolation 2>&1 | tail -3
fi

# qwen_vl_utils — critical for Qwen3-VL
if ! python3 -c "import qwen_vl_utils" 2>/dev/null; then
    echo "  Installing qwen-vl-utils..." | tee -a "$LOG"
    pip install --quiet qwen-vl-utils 2>/dev/null || \
        pip install --quiet "qwen-vl-utils[decord]" 2>/dev/null || \
        { echo "ERROR: Failed to install qwen-vl-utils" | tee -a "$LOG"; exit 1; }
fi

# Final verification
python3 -c "
import torch, transformers, flash_attn, decord, qwen_vl_utils
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  transformers={transformers.__version__}')
print(f'  flash_attn={flash_attn.__version__}')
print(f'  decord OK, qwen_vl_utils OK')
" 2>&1 | tee -a "$LOG"

# Check benchmark data
if [ ! -f "data/benchmark.json" ]; then
    echo "ERROR: data/benchmark.json not found. See README.md for data setup." | tee -a "$LOG"
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('data/benchmark.json'))))")
echo "  Benchmark: $SAMPLE_COUNT samples" | tee -a "$LOG"

echo "  Environment OK!" | tee -a "$LOG"

# ── Step 2: Run Probe Evaluation ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Running probe evaluation..." | tee -a "$LOG"

CMD=(python3 scripts/run_probe.py
     --model "$MODEL"
     --model_path "$MODEL_PATH"
     --tasks "$TASKS"
     --limit "$LIMIT"
     --num_gpus "$NUM_GPUS"
     --max_new_tokens "$MAX_NEW_TOKENS"
     --clip_cache_dir "$CLIP_CACHE"
     --output_dir "$OUTDIR")

echo "  CMD: ${CMD[*]}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

"${CMD[@]}" 2>&1 | tee -a "$LOG"

# ── Step 3: Compute Metrics ───────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 3] Computing metrics..." | tee -a "$LOG"

python3 scripts/compute_metrics.py \
    --pred_dir "$OUTDIR" \
    --tolerance 3,5 2>&1 | tee -a "$LOG"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo " DONE: $MODEL_TAG"                           | tee -a "$LOG"
echo " Results: $OUTDIR"                           | tee -a "$LOG"
echo " Metrics: $OUTDIR/metrics.json"              | tee -a "$LOG"
echo " Log:     $LOG"                              | tee -a "$LOG"
echo " Ended:   $(date)"                           | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
