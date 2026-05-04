#!/bin/bash
# =============================================================================
# OmniProact-Bench — Gemini-3-Flash-Preview (Audio ON) Full Probe
# =============================================================================
# API model — does not need GPU. NUM_GPUS controls parallel worker count.
#
# Usage:
#   bash scripts/run_probe_gemini.sh          # Full run
#   LIMIT=2 bash scripts/run_probe_gemini.sh  # Quick test
# =============================================================================

set -e

MODEL="gemini-3-flash"
MODEL_TAG="gemini-3-flash-preview"

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

OUTDIR="${BENCH_DIR}/results/probe/${MODEL_TAG}"
TASKS="instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction"
LIMIT="${LIMIT:-0}"
NUM_GPUS="${NUM_GPUS:-8}"   # Controls parallel workers (not actual GPUs for API model)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CLIP_CACHE="${CLIP_CACHE:-/path/to/OmniProact-Bench/clip_cache}"

# Gemini needs its own compressed-clip cache
export OMNIPROACT_GEMINI_CACHE="${OMNIPROACT_GEMINI_CACHE:-/path/to/OmniProact-Bench/gemini_clip_cache}"

LOG_DIR="/path/to/OmniProact-Bench/omniproact_logs"
mkdir -p "$CLIP_CACHE" "$LOG_DIR" "$OUTDIR" "$OMNIPROACT_GEMINI_CACHE"
LOG="$LOG_DIR/${MODEL_TAG}_probe_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "$LOG"
echo " OmniProact-Bench Probe: $MODEL_TAG"        | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Output dir : $OUTDIR"                      | tee -a "$LOG"
echo "  Limit/task : $LIMIT (0=all)"               | tee -a "$LOG"
echo "  Workers    : $NUM_GPUS"                    | tee -a "$LOG"
echo "  Started at : $(date)"                      | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# ── Environment Check ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment check..." | tee -a "$LOG"

command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found" | tee -a "$LOG"; exit 1; }

_install_if_missing() {
    local pkg="$1"; local pip_name="${2:-$1}"
    python3 -c "import $pkg" 2>/dev/null || pip install --quiet "$pip_name"
}
_install_if_missing requests requests
_install_if_missing tqdm tqdm
_install_if_missing decord decord
_install_if_missing PIL pillow

python3 -c "
import requests, tqdm, decord
print(f'  requests OK, tqdm OK, decord OK')
" 2>&1 | tee -a "$LOG"

[ -f "data/benchmark.json" ] || { echo "ERROR: data/benchmark.json not found. See README.md for data setup."; exit 1; }
echo "  Environment OK!" | tee -a "$LOG"

# ── Run Evaluation ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Running probe evaluation..." | tee -a "$LOG"

python3 scripts/run_probe.py \
    --model "$MODEL" \
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
