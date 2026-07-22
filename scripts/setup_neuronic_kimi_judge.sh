#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
PYTHON_VERSION="3.10"
KIMI_MODEL="moonshotai/Kimi-VL-A3B-Instruct"
# Pin weights and trust_remote_code to one reviewed upstream revision.
KIMI_REVISION="cc6452511d00c99f3b3bed213e96ab7802c415c8"
KIMI_VENV="${KIMI_VENV:-$CACHE_ROOT/kimi-judge-venv}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$CACHE_ROOT/bin"
cd "$REPO"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$CACHE_ROOT/bin/uv" ]]; then
  UV_BIN="$CACHE_ROOT/bin/uv"
else
  BOOTSTRAP_VENV="$CACHE_ROOT/uv-bootstrap"
  python3 -m venv "$BOOTSTRAP_VENV"
  "$BOOTSTRAP_VENV/bin/python" -m pip install --upgrade uv
  UV_BIN="$BOOTSTRAP_VENV/bin/uv"
fi
if [[ "$UV_BIN" != "$CACHE_ROOT/bin/uv" ]]; then
  ln -sfn "$UV_BIN" "$CACHE_ROOT/bin/uv"
fi

export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/python"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
unset TRANSFORMERS_CACHE

uv python install "$PYTHON_VERSION"
if [[ ! -x "$KIMI_VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$KIMI_VENV"
fi
uv pip install --python "$KIMI_VENV/bin/python" -r scripts/requirements_kimi_judge.txt

"$KIMI_VENV/bin/python" - <<'PY'
import sys
import torch
import transformers

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert transformers.__version__ == "4.51.3", transformers.__version__
print(f"Kimi judge environment OK: Python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")
PY

KIMI_MODEL="$KIMI_MODEL" KIMI_REVISION="$KIMI_REVISION" "$KIMI_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    os.environ["KIMI_MODEL"],
    revision=os.environ["KIMI_REVISION"],
)
print(f"Kimi-VL snapshot ready: {path}")
PY

echo "Kimi judge ready. No API key is required. Shared cache: $CACHE_ROOT"
