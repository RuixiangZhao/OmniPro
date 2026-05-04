#!/bin/bash
# One-time setup for LiveStar online evaluation dependencies.
#
# Usage:
#   bash scripts/install_livestar_deps.sh
#
# LiveStar requires:
#   1. transformers 4.37.2 (incompatible with newer system versions)
#      -> installed into a local directory as an isolated shim
#   2. flash-attn (optional but recommended for speed)
#   3. torchvision, Pillow, numpy (for image preprocessing)
#
# The transformers shim is injected via PYTHONPATH at runtime (see
# scripts/run_online_livestar.sh), so it does NOT affect other models.

set -eu
LIVESTAR_PKGS="${LIVESTAR_PKGS:-/path/to/livestar_pkgs}"

echo "=========================================="
echo "Installing LiveStar dependencies"
echo "=========================================="

# ── 1. Transformers 4.37 shim (isolated install) ─────────────────────
echo ""
echo "[1/3] Installing transformers 4.37.2 shim -> ${LIVESTAR_PKGS}"
pip install --target="${LIVESTAR_PKGS}" --no-deps \
    'transformers==4.37.2' \
    'tokenizers>=0.15,<0.20' \
    'huggingface_hub>=0.19' \
    'safetensors>=0.3' \
    2>&1 | tail -5

# ── Patch: skip tokenizers version check ─────────────────────────────
# transformers 4.37 requires tokenizers <0.19 but system may have >=0.19.
# The check is cosmetic (4.37 works fine with newer tokenizers), so patch it out.
echo ""
echo "[patch] Fixing version checks in transformers 4.37 shim..."
if grep -q "require_version_core" "${LIVESTAR_PKGS}/transformers/dependency_versions_check.py" 2>/dev/null; then
    sed -i 's/require_version_core(deps\[pkg\])/pass  # patched: skip version check/' \
        "${LIVESTAR_PKGS}/transformers/dependency_versions_check.py"
    echo "  Patched successfully."
else
    echo "  Already patched or file not found, skipping."
fi

# ── 2. flash-attn (optional, improves speed) ────────────────────────
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
echo "[3/3] Checking other deps (torchvision, Pillow, numpy, decord)..."
pip install --quiet torchvision Pillow numpy decord tqdm 2>&1 | tail -3

# ── Verification ─────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Verifying installation"
echo "=========================================="

PYTHONPATH="${LIVESTAR_PKGS}:${PYTHONPATH:-}" python3 - <<'PY'
import importlib, sys

CHECKS = [
    ("transformers (shim)", "transformers"),
    ("torch",               "torch"),
    ("torchvision",         "torchvision"),
    ("Pillow",              "PIL"),
    ("numpy",              "numpy"),
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

# Critical check: transformers must be 4.37.x
import transformers
if not transformers.__version__.startswith("4.37"):
    errors.append(("transformers version",
                    f"expected 4.37.x, got {transformers.__version__}"))
    print(f"\n  [ERROR] transformers shim version mismatch: {transformers.__version__}")

# Optional check: flash-attn
try:
    import flash_attn
    print(f"  {'flash_attn':22s} : {flash_attn.__version__}")
except ImportError:
    print(f"  {'flash_attn':22s} : [WARN] not installed (optional, model works without it)")

if errors:
    print(f"\n[WARNING] {len(errors)} check(s) failed. See above.")
    sys.exit(1)
PY

echo ""
echo "All checks passed. Run:"
echo "  bash scripts/run_online_livestar.sh"
echo ""
