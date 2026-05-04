#!/bin/bash
# =============================================================================
# OmniProact-Bench — VideoLLaMA2.1-7B-AV (Audio ON) Full Probe
# =============================================================================
# Usage:
#   bash scripts/run_probe_videollama2_av.sh          # Full run (2700)
#   LIMIT=2 bash scripts/run_probe_videollama2_av.sh  # Quick smoke test
# =============================================================================

set -e

MODEL="videollama2-av"
MODEL_TAG="VideoLLaMA2.1-7B-AV"
MODEL_PATH="/path/to/pretrained_models/VideoLLaMA2.1-7B-AV"
VIDEOLLAMA2_PKGS="${VIDEOLLAMA2_PKGS:-${BENCH_DIR}/videollama2_pkgs}"

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

# Install base dependencies
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
_install_if_missing soundfile soundfile
_install_if_missing librosa librosa
_install_if_missing torchaudio torchaudio
_install_if_missing einops einops
_install_if_missing timm timm
_install_if_missing moviepy 'moviepy==1.0.3'
_install_if_missing pytorchvideo pytorchvideo
_install_if_missing peft peft
python3 -c "import flash_attn" 2>/dev/null || {
    echo "  Installing flash-attn..." | tee -a "$LOG"
    pip install flash-attn --no-build-isolation 2>&1 | tail -3
}

# Install transformers 4.42 shim (VideoLLaMA2.1-AV needs ==4.42.3)
echo "  Setting up transformers 4.42 shim -> ${VIDEOLLAMA2_PKGS}" | tee -a "$LOG"
if ! PYTHONPATH="${VIDEOLLAMA2_PKGS}" python3 -c "import transformers; assert transformers.__version__.startswith('4.42')" 2>/dev/null; then
    pip install --target="${VIDEOLLAMA2_PKGS}" --no-deps 'transformers==4.42.3' 2>&1 | tail -1
    pip install --target="${VIDEOLLAMA2_PKGS}" --no-deps --force-reinstall 'tokenizers==0.19.1' 2>&1 | tail -1
    pip install --target="${VIDEOLLAMA2_PKGS}" --no-deps --force-reinstall 'huggingface_hub==0.23.4' 2>&1 | tail -1
    pip install --target="${VIDEOLLAMA2_PKGS}" --no-deps 'peft==0.11.1' 2>&1 | tail -1
    rm -rf "${VIDEOLLAMA2_PKGS}"/tokenizers-0.2[0-9]*.dist-info \
           "${VIDEOLLAMA2_PKGS}"/huggingface_hub-1.*.dist-info 2>/dev/null
fi
export PYTHONPATH="${VIDEOLLAMA2_PKGS}:${PYTHONPATH:-}"
PYTHONPATH="${VIDEOLLAMA2_PKGS}:${PYTHONPATH:-}" python3 -c "import transformers; print(f'  transformers={transformers.__version__}')" 2>&1 | tee -a "$LOG"

# ── Step 2: Check Model Weights ──────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 2] Checking model weights..." | tee -a "$LOG"

[ -f "$MODEL_PATH/config.json" ] || {
    echo "  Model not found at $MODEL_PATH, downloading..." | tee -a "$LOG"
    # source your proxy/env setup script here if needed
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('DAMO-NLP-SG/VideoLLaMA2.1-7B-AV', local_dir='$MODEL_PATH')" 2>&1 | tee -a "$LOG"
}
echo "  Model path: $MODEL_PATH" | tee -a "$LOG"
[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: config.json not found" | tee -a "$LOG"; exit 1; }

# Clone third_party if needed
if [ ! -d "${BENCH_DIR}/third_party/VideoLLaMA2/videollama2" ]; then
    echo "  Cloning VideoLLaMA2 repo (audio_visual branch)..." | tee -a "$LOG"
    # source your proxy/env setup script here if needed
    git clone -b audio_visual https://github.com/DAMO-NLP-SG/VideoLLaMA2.git "${BENCH_DIR}/third_party/VideoLLaMA2" 2>&1 | tail -3
fi

# Verify
python3 -c "
import torch, flash_attn, decord
print(f'  torch={torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  flash_attn={flash_attn.__version__}')
" 2>&1 | tee -a "$LOG"
[ -f "data/benchmark.json" ] || { echo "ERROR: data/benchmark.json not found. See README.md for data setup."; exit 1; }
echo "  Environment OK!" | tee -a "$LOG"

# ── Step 3: Run Evaluation ───────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[Step 3] Running probe evaluation..." | tee -a "$LOG"

# siglip vision tower config pre-downloaded to shared HF cache.
# Set HF_HOME so model can find it offline.
HF_HOME=/path/to/huggingface_cache \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
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
