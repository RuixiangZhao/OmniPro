#!/bin/bash
# =============================================================================
# OmniProact-Bench — LiveStar-8B Probe Evaluation
# =============================================================================
# Usage:
#   bash scripts/run_probe_livestar.sh          # Full run
#   LIMIT=2 bash scripts/run_probe_livestar.sh  # Smoke test
#
# Environment:
#   - Requires transformers == 4.37.2 (LiveStar pinned, auto-installed as shim)
#   - Model: InternViT + InternLM2 ~8B, ~18GB VRAM (bf16)
#   - Vision-only (no audio)
# =============================================================================

set -e

MODEL="livestar"
MODEL_TAG="LiveStar-8B"
MODEL_PATH="${MODEL_PATH:-/path/to/huggingface_cache/LiveStar_8B}"

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

# Install transformers 4.37 shim (LiveStar requirement)
LIVESTAR_PKGS="${LIVESTAR_PKGS:-/path/to/livestar_pkgs}"
if [ ! -d "$LIVESTAR_PKGS/transformers" ]; then
    echo "  Installing LiveStar deps..." | tee -a "$LOG"
    bash "${BENCH_DIR}/scripts/install_livestar_deps.sh" 2>&1 | tee -a "$LOG"
fi
export PYTHONPATH="${LIVESTAR_PKGS}:${PYTHONPATH:-}"

# Verify
python3 -c "
import transformers
assert transformers.__version__.startswith('4.37'), f'Need 4.37.x, got {transformers.__version__}'
print(f'  transformers={transformers.__version__} (shim)')
import torch
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}')
print('  LiveStar deps: OK')
" 2>&1 | tee -a "$LOG"

echo "  Environment OK!" | tee -a "$LOG"

# ── Step 2: Check Model Weights ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Checking model weights..." | tee -a "$LOG"
[ -d "$MODEL_PATH" ] || { echo "ERROR: Model not found at $MODEL_PATH" | tee -a "$LOG"; exit 1; }
echo "  Model path: $MODEL_PATH" | tee -a "$LOG"

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
