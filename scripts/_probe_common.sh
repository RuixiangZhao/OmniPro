# Shared probe-mode launcher. Sourced by per-model entry scripts.
#
# Inputs (set by the caller before sourcing this file):
#   MODEL           — value for --model (e.g. qwen3-vl, qwen2.5-omni, gemini-3-flash, ...)
#   MODEL_PATH      — value for --model_path (optional for API models)
#   OUTDIR          — target result directory (results/probe/<tag>)
#   MODEL_TAG       — short name used in log file names
#
# Optional overrides:
#   TASKS           — comma list (default: all 9 tasks)
#   LIMIT           — samples per task (default: 50)
#   NUM_GPUS        — parallel workers (default: 8)
#   MAX_NEW_TOKENS  — model output cap (default: 1024)
#   TOLERANCE       — "3,5" (default: 3,5)
#   EXTRA_ARGS      — appended to run_probe.py

set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"

TASKS="${TASKS:-instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction}"
LIMIT="${LIMIT:-0}"   # 0 = all samples (300 per task = 2700 total)
NUM_GPUS="${NUM_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOLERANCE="${TOLERANCE:-3,5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CLIP_CACHE="${CLIP_CACHE:-/path/to/OmniProact-Bench/clip_cache}"
LOG_DIR="${LOG_DIR:-/path/to/OmniProact-Bench/omniproact_logs}"
mkdir -p "$CLIP_CACHE" "$LOG_DIR" "$OUTDIR"

LOG="$LOG_DIR/${MODEL_TAG}_probe.log"
: > "$LOG"

echo "==========================================" | tee -a "$LOG"
echo "[probe] $MODEL_TAG  (model=$MODEL)"           | tee -a "$LOG"
echo "  tasks       : $TASKS"                       | tee -a "$LOG"
echo "  limit/task  : $LIMIT"                       | tee -a "$LOG"
echo "  num_gpus    : $NUM_GPUS"                    | tee -a "$LOG"
echo "  output_dir  : $OUTDIR"                      | tee -a "$LOG"
echo "  started at  : $(date)"                      | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

CMD=(python3 scripts/run_probe.py
     --model "$MODEL"
     --tasks "$TASKS"
     --limit "$LIMIT"
     --num_gpus "$NUM_GPUS"
     --max_new_tokens "$MAX_NEW_TOKENS"
     --clip_cache_dir "$CLIP_CACHE"
     --output_dir "$OUTDIR")

if [ -n "${MODEL_PATH:-}" ]; then
    CMD+=(--model_path "$MODEL_PATH")
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

echo "[run] ${CMD[*]}" | tee -a "$LOG"
"${CMD[@]}" 2>&1 | tee -a "$LOG"

echo ""                                             | tee -a "$LOG"
echo "[done] $(date)"                               | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "[metrics]"                                    | tee -a "$LOG"
python3 scripts/compute_metrics.py \
    --pred_dir "$OUTDIR" --tolerance "$TOLERANCE" 2>&1 | tee -a "$LOG"

echo ""                                             | tee -a "$LOG"
echo "[all done] $MODEL_TAG  $(date)"               | tee -a "$LOG"
