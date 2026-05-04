#!/bin/bash
# =============================================================================
# OmniProact-Bench Probe Mode — Quick Verification
# =============================================================================
# Runs 2 samples per task on a SINGLE model to verify the environment is correct.
# Should complete in ~5-10 minutes on 1 GPU.
#
# Usage:
#   bash scripts/verify_probe.sh [model]
#
# Supported models: qwen3-vl (default), qwen2.5-omni, qwen3-omni, gemini-3-flash
#
# What it checks:
#   1. Data file exists and loads correctly
#   2. Model loads successfully
#   3. Video clipping works (ffmpeg)
#   4. Inference runs end-to-end for all 9 tasks
#   5. Metrics computation works
# =============================================================================

set -e

BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH_DIR"

MODEL="${1:-qwen3-vl}"
VERIFY_DIR="/tmp/omniproact_verify_$$"
CLIP_CACHE="/tmp/omniproact_verify_clips_$$"

echo "============================================"
echo " OmniProact-Bench — Probe Verification"
echo "  Model: $MODEL"
echo "  Temp output: $VERIFY_DIR"
echo "============================================"
echo ""

# ── 1. Check data ─────────────────────────────────────────────────────────────
echo "[1/5] Checking benchmark data..."
if [ ! -f "data/benchmark.json" ]; then
    echo "ERROR: data/benchmark.json not found. See README.md for data setup."
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('data/benchmark.json'))))")
echo "  OK: $SAMPLE_COUNT samples loaded"

# ── 2. Determine model path ──────────────────────────────────────────────────
echo ""
echo "[2/5] Resolving model path..."

case "$MODEL" in
    qwen3-vl)
        MODEL_PATH="/path/to/pretrained_models/Qwen3-VL-8B-Instruct"
        ;;
    qwen2.5-omni)
        MODEL_PATH="/path/to/pretrained_models/Qwen2.5-Omni-7B"
        ;;
    qwen2.5-omni-noaudio)
        MODEL_PATH="/path/to/pretrained_models/Qwen2.5-Omni-7B"
        ;;
    qwen3-omni)
        MODEL_PATH="/path/to/pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
        ;;
    qwen3-omni-noaudio)
        MODEL_PATH="/path/to/pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
        ;;
    gemini-3-flash|gemini-3-flash-noaudio)
        MODEL_PATH=""
        echo "  API model — no local path needed"
        ;;
    *)
        echo "ERROR: Unknown model '$MODEL'"
        echo "  Supported: qwen3-vl, qwen2.5-omni, qwen2.5-omni-noaudio,"
        echo "             qwen3-omni, qwen3-omni-noaudio,"
        echo "             gemini-3-flash, gemini-3-flash-noaudio"
        exit 1
        ;;
esac

if [ -n "$MODEL_PATH" ]; then
    if [ -d "$MODEL_PATH" ]; then
        echo "  OK: $MODEL_PATH exists"
    else
        echo "  ERROR: Model path not found: $MODEL_PATH"
        exit 1
    fi
fi

# ── 3. Run 2 samples on single GPU ───────────────────────────────────────────
echo ""
echo "[3/5] Running inference (2 samples/task, 1 GPU)..."

mkdir -p "$VERIFY_DIR" "$CLIP_CACHE"

CMD=(python3 scripts/run_probe.py
     --model "$MODEL"
     --tasks "instant_event_alert,cumulative_counting,realtime_state_monitor,event_narration,explicit_target_grounding,semantic_condition_alert,dedup_counting,snapshot_counting,sequential_step_instruction"
     --limit 2
     --num_gpus 1
     --max_new_tokens 512
     --clip_cache_dir "$CLIP_CACHE"
     --output_dir "$VERIFY_DIR")

if [ -n "$MODEL_PATH" ]; then
    CMD+=(--model_path "$MODEL_PATH")
fi

echo "  CMD: ${CMD[*]}"
echo ""

START=$(date +%s)
"${CMD[@]}" 2>&1 | tee /tmp/omniproact_verify.log
END=$(date +%s)
ELAPSED=$((END - START))

# ── 4. Check output ───────────────────────────────────────────────────────────
echo ""
echo "[4/5] Checking outputs..."

TOTAL_PREDS=$(cat "$VERIFY_DIR"/*.jsonl 2>/dev/null | wc -l)
NUM_TASKS=$(ls "$VERIFY_DIR"/*.jsonl 2>/dev/null | wc -l)

if [ "$TOTAL_PREDS" -lt 1 ]; then
    echo "  ERROR: No predictions generated!"
    echo "  Check log: /tmp/omniproact_verify.log"
    exit 1
fi

echo "  Generated $TOTAL_PREDS predictions across $NUM_TASKS task files"
echo "  Time: ${ELAPSED}s"

# Show per-task counts
for f in "$VERIFY_DIR"/*.jsonl; do
    task=$(basename "$f" .jsonl)
    count=$(wc -l < "$f")
    printf "    %-35s %d\n" "$task" "$count"
done

# ── 5. Compute metrics ────────────────────────────────────────────────────────
echo ""
echo "[5/5] Computing metrics..."

python3 scripts/compute_metrics.py \
    --pred_dir "$VERIFY_DIR" \
    --tolerance 3,5

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " VERIFICATION PASSED"
echo ""
echo "  Model: $MODEL"
echo "  Predictions: $TOTAL_PREDS"
echo "  Tasks: $NUM_TASKS / 9"
echo "  Time: ${ELAPSED}s for 2 samples/task"
echo ""
echo " Estimated full-run time (300 samples/task, 8 GPUs):"
echo "   ~$((ELAPSED * 150 / 8 / 60)) minutes"
echo ""
echo " To run full evaluation:"
echo "   LIMIT=0 bash scripts/run_probe_${MODEL//./_}.sh"
echo "============================================"

# Cleanup temp
rm -rf "$VERIFY_DIR" "$CLIP_CACHE"
