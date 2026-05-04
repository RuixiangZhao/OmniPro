#!/bin/bash
# =============================================================================
# OmniProact-Bench Probe Mode — Environment Setup
# =============================================================================
# Run this ONCE on each machine before launching probe evaluations.
# Installs all Python dependencies needed for probe-mode models.
#
# Usage:
#   bash scripts/setup_probe_env.sh
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - CUDA 12.x + NVIDIA driver (for GPU models)
#   - ffmpeg available in PATH
# =============================================================================

set -e

echo "============================================"
echo " OmniProact-Bench Probe Environment Setup"
echo "============================================"

# ── 1. Check basic prerequisites ─────────────────────────────────────────────
echo ""
echo "[1/5] Checking prerequisites..."

# Python version
python3 --version || { echo "ERROR: python3 not found"; exit 1; }

# ffmpeg
ffmpeg -version 2>/dev/null | head -1 || { echo "ERROR: ffmpeg not found"; exit 1; }

# CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU models will not work."
fi

# ── 2. Install core Python packages ──────────────────────────────────────────
echo ""
echo "[2/5] Installing core Python packages..."

pip install --quiet --upgrade pip

# Core ML stack (skip if already satisfied to save time)
pip install --quiet \
    torch torchvision torchaudio \
    transformers>=4.51 \
    accelerate \
    decord \
    tqdm \
    requests \
    pillow \
    numpy \
    scipy

# ── 3. Install flash-attention (critical for Qwen models) ────────────────────
echo ""
echo "[3/5] Installing flash-attention..."

if python3 -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} already installed')" 2>/dev/null; then
    echo "  -> Already installed, skipping."
else
    echo "  -> Building flash-attn (this may take 5-10 minutes)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5
fi

# ── 4. Install model-specific utilities ───────────────────────────────────────
echo ""
echo "[4/5] Installing model-specific packages..."

# qwen_vl_utils — needed by Qwen3-VL
pip install --quiet qwen-vl-utils 2>/dev/null || \
    pip install --quiet git+https://github.com/QwenLM/Qwen2.5-VL.git@main#subdirectory=qwen-vl-utils 2>/dev/null || \
    echo "  [WARN] qwen-vl-utils install failed; Qwen3-VL may not work."

# qwen_omni_utils — needed by Qwen2.5-Omni and Qwen3-Omni
pip install --quiet qwen-omni-utils 2>/dev/null || \
    echo "  [WARN] qwen-omni-utils install failed; Qwen Omni models may not work."

# soundfile — may be needed for audio processing
pip install --quiet soundfile librosa 2>/dev/null || true

# ── 5. Verify imports ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying critical imports..."

python3 -c "
import sys
errors = []

# Core
try:
    import torch
    print(f'  torch {torch.__version__}  CUDA={torch.cuda.is_available()}  devices={torch.cuda.device_count()}')
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import transformers
    print(f'  transformers {transformers.__version__}')
except ImportError as e:
    errors.append(f'transformers: {e}')

try:
    import flash_attn
    print(f'  flash_attn {flash_attn.__version__}')
except ImportError as e:
    errors.append(f'flash_attn: {e}')

try:
    import decord
    print(f'  decord OK')
except ImportError as e:
    errors.append(f'decord: {e}')

# Model utils
try:
    import qwen_vl_utils
    print(f'  qwen_vl_utils OK')
except ImportError as e:
    errors.append(f'qwen_vl_utils: {e}')

try:
    import qwen_omni_utils
    print(f'  qwen_omni_utils OK')
except ImportError as e:
    errors.append(f'qwen_omni_utils: {e}')

if errors:
    print()
    print('  WARNINGS (some models may not work):')
    for e in errors:
        print(f'    - {e}')
    sys.exit(0)  # don't fail hard; user may only need subset of models
else:
    print()
    print('  All imports OK!')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next: run the verification script to test"
echo "   bash scripts/verify_probe.sh"
echo ""
echo " Then: run full evaluation with any of:"
echo "   LIMIT=0 bash scripts/run_probe_qwen2_5_omni.sh"
echo "   LIMIT=0 bash scripts/run_probe_qwen3_omni.sh"
echo "   LIMIT=0 bash scripts/run_probe_qwen3_vl.sh"
echo "   LIMIT=0 bash scripts/run_probe_gemini.sh"
echo "   (and their -noaudio variants)"
echo "============================================"
