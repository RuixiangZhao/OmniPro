#!/bin/bash
# =============================================================================
# OmniProact-Bench — Gemini-3-Flash (Audio-Only) Probe Evaluation
# =============================================================================
# Audio-only ablation: sends ONLY the audio track (mp3) to the Gemini API,
# no video frames. No GPU required.
#
# Usage:
#   bash scripts/run_probe_gemini_audioonly.sh                  # Full run
#   LIMIT=50 bash scripts/run_probe_gemini_audioonly.sh         # 50/task
#   LIMIT=2 bash scripts/run_probe_gemini_audioonly.sh          # Quick test
#   NUM_WORKERS=16 bash scripts/run_probe_gemini_audioonly.sh   # More parallelism
# =============================================================================

set -e

MODEL="gemini-3-flash-audioonly"
MODEL_TAG="gemini-3-flash-preview-AudioOnly"

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

OUTDIR="${BENCH_DIR}/results/probe/${MODEL_TAG}"
TASKS="instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction"
LIMIT="${LIMIT:-0}"
# Gemini is API-based, no GPU needed. NUM_WORKERS controls HTTP concurrency.
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CLIP_CACHE="${CLIP_CACHE:-/path/to/OmniProact-Bench/clip_cache}"

LOG_DIR="${BENCH_DIR}/omniproact_logs"
mkdir -p "$CLIP_CACHE" "$LOG_DIR" "$OUTDIR"
LOG="$LOG_DIR/${MODEL_TAG}_probe_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee "$LOG"
echo " OmniProact-Bench Probe: $MODEL_TAG"        | tee -a "$LOG"
echo "  (Audio-Only ablation via Gemini API)"      | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Output dir  : $OUTDIR"                     | tee -a "$LOG"
echo "  Limit/task  : $LIMIT (0=all)"              | tee -a "$LOG"
echo "  Workers     : $NUM_WORKERS"                | tee -a "$LOG"
echo "  Started at  : $(date)"                     | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# ── Environment Check ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 1] Environment check..." | tee -a "$LOG"

command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found (needed for audio extraction)" | tee -a "$LOG"; exit 1; }

python3 -c "import requests; print('  requests OK')" 2>&1 | tee -a "$LOG" || \
    { pip install --quiet requests && echo "  installed requests"; }

[ -f "data/benchmark.json" ] || { echo "ERROR: data/benchmark.json not found. See README.md for data setup."; exit 1; }
echo "  Environment OK!" | tee -a "$LOG"

# ── Run Evaluation ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Running probe evaluation (Audio-Only)..." | tee -a "$LOG"

python3 scripts/run_probe.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --num_gpus "$NUM_WORKERS" \
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
