#!/bin/bash
# One-time setup for MMDuet2 online evaluation dependencies.
#
# Usage:
#   bash scripts/install_mmduet2_deps.sh
#
# MMDuet2 requires:
#   1. transformers 4.49 (incompatible with system 4.57)
#      → installed into /path/to/mmduet2_pkgs as an isolated shim
#   2. flash-attn ≥2.3 (CRITICAL for performance)
#      → without flash_attention_2, SDPA builds an O(N²) causal mask that
#        causes a 12× slowdown at ~40K tokens (from ~1s/frame to 15s/frame)
#   3. third_party/MMDuet2/ cloned from GitHub
#
# The transformers shim is injected via PYTHONPATH at runtime (see
# scripts/run_online_mmduet2.sh), so it does NOT affect other models.

set -eu
MMDUET2_PKGS="${MMDUET2_PKGS:-/path/to/mmduet2_pkgs}"

echo "=========================================="
echo "Installing MMDuet2 dependencies"
echo "=========================================="

# ── 1. Transformers 4.49 shim (isolated install) ──────────────────────
echo ""
echo "[1/3] Installing transformers 4.49 shim → ${MMDUET2_PKGS}"
pip install --target="${MMDUET2_PKGS}" --no-deps \
    'transformers==4.49.0' \
    'tokenizers>=0.21,<0.22' \
    'huggingface_hub>=0.26' \
    'safetensors>=0.3' \
    2>&1 | tail -5

# ── 2. flash-attn (CRITICAL for performance) ─────────────────────────
echo ""
echo "[2/3] Checking flash-attn..."
if python3 -c 'import flash_attn; print(f"flash-attn {flash_attn.__version__} OK")' 2>/dev/null; then
    echo "  flash-attn already installed."
else
    echo "  flash-attn not found. Installing (this may take several minutes)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5
fi

# ── 3. Other dependencies ────────────────────────────────────────────
echo ""
echo "[3/3] Installing qwen-vl-utils and other deps..."
pip install --target="${MMDUET2_PKGS}" --no-deps qwen-vl-utils 2>&1 | tail -3
pip install decord Pillow numpy tqdm 2>&1 | tail -3

# ── Patch: skip huggingface-hub version check ────────────────────────
# transformers 4.49 requires huggingface-hub <1.0 but system has >=1.0.
# The check is cosmetic (4.49 works fine with newer hub), so we patch it out.
echo ""
echo "[patch] Fixing huggingface-hub version check in transformers 4.49 shim..."
if grep -q "require_version_core" "${MMDUET2_PKGS}/transformers/dependency_versions_check.py" 2>/dev/null; then
    sed -i 's/require_version_core(deps\[pkg\])/pass  # patched: skip version check/' \
        "${MMDUET2_PKGS}/transformers/dependency_versions_check.py"
    echo "  Patched successfully."
else
    echo "  Already patched or file not found, skipping."
fi

# ── Verification ─────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Verifying installation"
echo "=========================================="

PYTHONPATH="${MMDUET2_PKGS}:${PYTHONPATH:-}" python3 - <<'PY'
import importlib, sys

CHECKS = [
    ("transformers (shim)", "transformers"),
    ("torch",               "torch"),
    ("flash_attn",          "flash_attn"),
    ("decord",              "decord"),
]
errors = []
for label, mod in CHECKS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "-")
        print(f"  {label:22s} : {ver}")
    except Exception as e:
        errors.append((label, repr(e)))
        print(f"  {label:22s} : [FAIL] {type(e).__name__}: {e}")

# Critical check: transformers must be 4.49.x
import transformers
if not transformers.__version__.startswith("4.49"):
    errors.append(("transformers version",
                    f"expected 4.49.x, got {transformers.__version__}"))
    print(f"\n  [ERROR] transformers shim version mismatch: {transformers.__version__}")

# Critical check: flash-attn must be available
try:
    import flash_attn
except ImportError:
    errors.append(("flash_attn", "not installed"))
    print("\n  [CRITICAL] flash-attn is NOT installed!")
    print("  Without flash_attention_2, MMDuet2 will be 12× slower at >40K tokens.")
    print("  The SDPA backend builds a dense O(N²) causal mask that causes")
    print("  inference to jump from ~1s/frame to ~15s/frame at ~40K context tokens.")
    print("  Install with: pip install flash-attn --no-build-isolation")

if errors:
    print(f"\n[WARNING] {len(errors)} check(s) failed. See above.")
    sys.exit(1)
PY

echo ""
echo "All checks passed. Run:"
echo "  bash scripts/run_online_mmduet2.sh"
echo ""
echo "NOTE: flash_attention_2 is CRITICAL for MMDuet2 performance."
echo "  Without it, generate() hits a 12× slowdown at ~40K tokens"
echo "  due to SDPA's O(N²) dense causal mask allocation."
