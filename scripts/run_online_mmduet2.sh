#!/bin/bash
# =============================================================================
# OmniProact-Bench — MMDuet2 Online Evaluation
# =============================================================================
# One-shot script: install deps + full evaluation on a fresh machine.
# MMDuet2 uses text-to-text "NO REPLY" mechanism for proactive interaction.
# Based on Qwen2.5-VL-3B, vision-only, no audio.
#
# CRITICAL: flash-attn must be installed. Without flash_attention_2, the SDPA
# backend builds an O(N²) dense causal mask that causes a 12× slowdown once
# context exceeds ~40K tokens. See DEVLOG.md for profiling details.
#
# Usage:
#   bash scripts/run_online_mmduet2.sh                  # Full run (all 2700 samples)
#   LIMIT=4 bash scripts/run_online_mmduet2.sh          # Quick test (4/task)
#   bash scripts/run_online_mmduet2.sh --tasks event_narration,instant_event_alert
#   bash scripts/run_online_mmduet2.sh --gpt-judge
# =============================================================================

set -e

# ── Install deps (idempotent) ────────────────────────────────────────────────
MMDUET2_PKGS="${MMDUET2_PKGS:-/path/to/mmduet2_pkgs}"
echo "[setup] Checking MMDuet2 dependencies..."

_needs_install=0
PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}" python3 -c "import transformers; assert transformers.__version__.startswith('4.49')" 2>/dev/null || _needs_install=1
python3 -c "import flash_attn" 2>/dev/null || _needs_install=1

if [ "$_needs_install" -eq 1 ]; then
    echo "[setup] Installing MMDuet2 deps (transformers 4.49 shim, flash-attn, etc.)..."
    bash "$(dirname "$0")/install_mmduet2_deps.sh"
    # Patch transformers 4.49 to skip huggingface-hub version check
    # (system huggingface-hub >= 1.0 is incompatible with 4.49's check)
    if grep -q "require_version_core" "${MMDUET2_PKGS}/transformers/dependency_versions_check.py" 2>/dev/null; then
        echo "[setup] Patching transformers 4.49 version check..."
        sed -i 's/require_version_core(deps\[pkg\])/pass  # patched: skip version check/' \
            "${MMDUET2_PKGS}/transformers/dependency_versions_check.py"
    fi
else
    echo "[setup] Dependencies already satisfied, skipping install."
fi

# Verify shim works
PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}" python3 -c "
import transformers, flash_attn
print(f'  transformers={transformers.__version__} (shim)')
print(f'  flash_attn={flash_attn.__version__}')
" || { echo "[ERROR] MMDuet2 deps verification failed"; exit 1; }

MODEL="mmduet2"
MODEL_TAG="MMDuet2"
MODEL_PATH="${MODEL_PATH:-wangyueqian/MMDuet2}"

# MMDuet2 needs transformers 4.49 shim + HF cache for model weights
export PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/path/to/huggingface_cache}"

source "$(dirname "$0")/_online_common.sh"
