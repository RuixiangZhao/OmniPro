# Shared online-mode launcher. Sourced by per-model entry scripts.
#
# Inputs (set by the caller before sourcing this file):
#   MODEL         — value for --model (e.g. minicpm-o, dummy-perfect)
#   MODEL_TAG     — short name used in log file & result dir
#   MODEL_PATH    — value for --model_path (optional for dummy models)
#
# Optional overrides (env var or first-wins CLI flags):
#   TASKS         — comma list (default: all 9 tasks)
#   LIMIT         — samples per task (default: 50; 0 = all)
#   NUM_GPUS      — parallel workers (default: 8)
#   FPS           — streaming tick rate (default: 1.0)
#   TOLERANCE     — temporal tolerance seconds (default: 3)
#   GPT_JUDGE     — "1" to run LLM judge on EN/SSI after scoring
#   GPT_PASS      — pass_score threshold (default: 3)
#   EXTRA_ARGS    — appended to run_online.py

set -eu

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"

# --- Credentials for the LLM judge (override with env var if needed) -------
# Used by compute_online_metrics.py --gpt_judge for EN / SSI scoring.
: "${GEMINI_API_KEY:=YOUR_API_KEY_HERE}"
export GEMINI_API_KEY

# --- CLI argument parsing (overrides env vars) -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)      TASKS="$2";      shift 2 ;;
        --limit)      LIMIT="$2";      shift 2 ;;
        --num-gpus)   NUM_GPUS="$2";   shift 2 ;;
        --fps)        FPS="$2";        shift 2 ;;
        --tolerance)  TOLERANCE="$2";  shift 2 ;;
        --gpt-judge)  GPT_JUDGE=1;     shift ;;
        --gpt-pass)   GPT_PASS="$2";   shift 2 ;;
        --output-dir) OUTDIR="$2";     shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        *)            EXTRA_ARGS="${EXTRA_ARGS:-} $1"; shift ;;
    esac
done

TASKS="${TASKS:-instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction}"
LIMIT="${LIMIT:-0}"   # 0 = all samples (300 per task = 2700 total)
NUM_GPUS="${NUM_GPUS:-8}"
FPS="${FPS:-1.0}"
TOLERANCE="${TOLERANCE:-3}"
GPT_JUDGE="${GPT_JUDGE:-0}"
GPT_PASS="${GPT_PASS:-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUTDIR="${OUTDIR:-${BENCH_DIR}/results/online/${MODEL_TAG}}"
LOG_DIR="${LOG_DIR:-/path/to/OmniProact-Bench/omniproact_logs}"

mkdir -p "$OUTDIR" "$LOG_DIR"
LOG="$LOG_DIR/${MODEL_TAG}_online.log"
: > "$LOG"

echo "==========================================" | tee -a "$LOG"
echo "[online] $MODEL_TAG  (model=$MODEL)"         | tee -a "$LOG"
echo "  tasks       : $TASKS"                      | tee -a "$LOG"
echo "  limit/task  : $LIMIT"                      | tee -a "$LOG"
echo "  num_gpus    : $NUM_GPUS"                   | tee -a "$LOG"
echo "  fps         : $FPS"                        | tee -a "$LOG"
echo "  tolerance   : $TOLERANCE"                  | tee -a "$LOG"
echo "  gpt_judge   : $GPT_JUDGE (pass>=$GPT_PASS)"| tee -a "$LOG"
echo "  output_dir  : $OUTDIR"                     | tee -a "$LOG"
echo "  started at  : $(date)"                     | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# --- Optional model-weight sanity check ------------------------------------
if [ -n "${MODEL_PATH:-}" ] && [ -d "$MODEL_PATH" ]; then
    NUM_SHARDS=$(ls "$MODEL_PATH"/model-*.safetensors 2>/dev/null | wc -l)
    echo "[check] model_path=$MODEL_PATH  shards=$NUM_SHARDS" | tee -a "$LOG"
    if [ "$NUM_SHARDS" -eq 0 ]; then
        echo "[warn] no safetensors shards found (may be fine for API models)" \
            | tee -a "$LOG"
    fi
elif [ -n "${MODEL_PATH:-}" ] && echo "$MODEL_PATH" | grep -q "/"; then
    # Looks like a HuggingFace repo ID (contains "/" but not a local dir)
    # e.g. "wangyueqian/MMDuet2" — skip directory check, let from_pretrained resolve it
    echo "[check] model_path=$MODEL_PATH  (HF repo ID, will resolve via HF cache)" | tee -a "$LOG"
elif [ -n "${MODEL_PATH:-}" ]; then
    echo "[ERROR] MODEL_PATH='$MODEL_PATH' not found" | tee -a "$LOG"
    exit 1
fi

# --- Build and run inference command ---------------------------------------
CMD=(python3 scripts/run_online.py
     --model "$MODEL"
     --tasks "$TASKS"
     --num_gpus "$NUM_GPUS"
     --fps "$FPS"
     --output_dir "$OUTDIR")

if [ -n "${MODEL_PATH:-}" ]; then
    CMD+=(--model_path "$MODEL_PATH")
fi
if [ "$LIMIT" -gt 0 ]; then
    CMD+=(--limit "$LIMIT")
fi
if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

echo "[run] ${CMD[*]}" | tee -a "$LOG"
"${CMD[@]}" 2>&1 | tee -a "$LOG"

echo ""                                             | tee -a "$LOG"
echo "[done] inference  $(date)"                    | tee -a "$LOG"

# --- Score + metrics --------------------------------------------------------
echo "==========================================" | tee -a "$LOG"
echo "[metrics] scoring and aggregating"           | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

SCORE_CMD=(python3 scripts/compute_online_metrics.py
           --pred_dir "$OUTDIR"
           --tolerance "$TOLERANCE")

if [ "$GPT_JUDGE" = "1" ]; then
    SCORE_CMD+=(--gpt_judge --gpt_pass_score "$GPT_PASS")
fi

echo "[run] ${SCORE_CMD[*]}"                        | tee -a "$LOG"
"${SCORE_CMD[@]}" 2>&1 | tee -a "$LOG"

echo ""                                             | tee -a "$LOG"
echo "[all done] $MODEL_TAG  $(date)"               | tee -a "$LOG"
echo "Results in: $OUTDIR"                          | tee -a "$LOG"
