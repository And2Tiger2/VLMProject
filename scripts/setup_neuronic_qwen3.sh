#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
PYTHON_VERSION="3.12"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$CACHE_ROOT/bin" "$REPO/segments/gaze_heads_qwen3_8b/runs/slurm"
cd "$REPO"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
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

uv python install "$PYTHON_VERSION"
uv sync --locked --python "$PYTHON_VERSION" --extra qwen --extra judge --extra dev
uv run --python "$PYTHON_VERSION" python -c \
  "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(f'Python {sys.version.split()[0]} OK')"
uv run --python "$PYTHON_VERSION" python -c \
  "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3-VL import OK')"
uv run --python "$PYTHON_VERSION" python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"

echo "Environment ready with Python $PYTHON_VERSION. Shared cache: $CACHE_ROOT"
