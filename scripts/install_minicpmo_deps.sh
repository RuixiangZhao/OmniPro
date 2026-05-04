#!/bin/bash
# One-time setup for MiniCPM-o 4.5 online evaluation dependencies.
#
# Usage:
#   bash scripts/install_minicpmo_deps.sh
#
# Installs the pinned versions required by the MiniCPM-o Duplex wrapper:
#   - transformers 4.51.0  (required — newer versions break Whisper-attn patch)
#   - minicpmo-utils 1.0.2+ (bundles stepaudio2, Token2Wav, cosyvoice2)
#   - decord / librosa / moviepy / Pillow / numpy / tqdm
#
# Safe to re-run: pip will skip packages that already satisfy the pin.
# Finishes with a smoke-import check so failures surface here, not at run time.

set -eu
echo "=========================================="
echo "Installing MiniCPM-o 4.5 dependencies"
echo "=========================================="

# ── Core pinned packages ───────────────────────────────────────────────
# transformers is strictly pinned; minicpmo-utils bundles stepaudio2 inside
# the wheel so --upgrade is enough (no separate stepaudio2 install needed).
pip install "transformers==4.51.0"         2>&1 | tail -3
pip install -U "minicpmo-utils[all]>=1.0.2" 2>&1 | tail -3
pip install decord librosa moviepy Pillow numpy tqdm 2>&1 | tail -3

echo ""
echo "=========================================="
echo "Verifying installation"
echo "=========================================="

# ── System ffmpeg (used by our audio extractor and by moviepy) ─────────
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  ffmpeg      : $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    echo "  ffmpeg      : [MISSING]  (apt install -y ffmpeg)"
fi

# ── Python packages + runtime imports ──────────────────────────────────
python3 - <<'PY'
import importlib, sys

CHECKS = [
    ("transformers", "transformers"),
    ("torch",        "torch"),
    ("decord",       "decord"),
    ("librosa",      "librosa"),
    ("moviepy",      "moviepy"),
    # minicpmo-utils ships `stepaudio2` inside the wheel. If this import
    # fails, the Duplex model loader will crash with the same message.
    ("stepaudio2",   "stepaudio2"),
]
errors = []
for label, mod in CHECKS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "-")
        print(f"  {label:11s} : {ver}")
    except Exception as e:
        errors.append((label, repr(e)))
        print(f"  {label:11s} : [FAIL] {type(e).__name__}: {e}")

if errors:
    print("\n[ERROR] some imports failed:")
    for lbl, err in errors:
        print(f"  - {lbl}: {err}")
    sys.exit(1)
PY

echo ""
echo "All checks passed. Run:"
echo "  bash scripts/run_online_minicpmo.sh"
