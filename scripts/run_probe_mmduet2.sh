#!/bin/bash
# =============================================================================
# OmniProact-Bench — MMDuet2 Probe Evaluation
# =============================================================================
# Usage:
#   bash scripts/run_probe_mmduet2.sh          # Full run
#   LIMIT=2 bash scripts/run_probe_mmduet2.sh  # Smoke test
#
# Environment:
#   - Requires transformers == 4.49.x (MMDuet2 pinned, auto-installed as shim)
#   - Requires flash-attn
#   - Model: Qwen2.5-VL-3B + MMDuet2 LoRA, ~8GB VRAM (bf16)
#   - Vision-only (no audio)
# =============================================================================

set -e

MODEL="mmduet2"
MODEL_TAG="MMDuet2"
MODEL_PATH="${MODEL_PATH:-wangyueqian/MMDuet2}"

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

# Install MMDuet2 deps (transformers 4.49 shim + flash-attn)
MMDUET2_PKGS="${MMDUET2_PKGS:-/path/to/mmduet2_pkgs}"
_needs_install=0
PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}" python3 -c "import transformers; assert transformers.__version__.startswith('4.49')" 2>/dev/null || _needs_install=1
python3 -c "import flash_attn" 2>/dev/null || _needs_install=1

if [ "$_needs_install" -eq 1 ]; then
    echo "  Installing MMDuet2 deps..." | tee -a "$LOG"
    bash "${BENCH_DIR}/scripts/install_mmduet2_deps.sh" 2>&1 | tee -a "$LOG"
    # Patch version check
    if grep -q "require_version_core" "${MMDUET2_PKGS}/transformers/dependency_versions_check.py" 2>/dev/null; then
        sed -i 's/require_version_core(deps\[pkg\])/pass/' "${MMDUET2_PKGS}/transformers/dependency_versions_check.py"
    fi
fi
export PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/path/to/huggingface_cache}"

python3 -c "
import transformers, flash_attn, torch
print(f'  transformers={transformers.__version__} (shim)')
print(f'  flash_attn={flash_attn.__version__}')
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}')
" 2>&1 | tee -a "$LOG"

echo "  Environment OK!" | tee -a "$LOG"

# ── Step 2: Check Model Weights ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Checking model weights..." | tee -a "$LOG"
echo "  Model: $MODEL_PATH (resolved via HF_HOME=$HF_HOME)" | tee -a "$LOG"

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
