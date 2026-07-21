#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
DOWNLOAD_RAW=0
if [[ "${1:-}" == "--download-raw-comics" ]]; then
  DOWNLOAD_RAW=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/prepare_neuronic_qwen3_data.sh [--download-raw-comics]" >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
uv run python scripts/download_gaze_comics.py \
  --out segments/gaze_heads_qwen3_8b/data/eval_comics
uv run python scripts/make_vlmbias_slice.py \
  --split main --n 400 --seed 0 \
  --out segments/vlm_bias_attention/data/vlmbias_400.jsonl
uv run python scripts/make_naturalbench_slice.py \
  --split train --groups 100 --seed 0 \
  --out segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl

if [[ "$DOWNLOAD_RAW" == "1" ]]; then
  archive="$REPO/.cache/datasets/raw_panel_images.tar.gz"
  data_root="$REPO/segments/gaze_heads_qwen3_8b/data"
  target="$data_root/discovery_comics"
  mkdir -p "$(dirname "$archive")" "$data_root"
  if [[ -e "$target" ]]; then
    echo "Raw COMICS target already exists; refusing to overwrite: $target" >&2
    exit 2
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz -O "$archive"
  else
    curl --fail --location --continue-at - \
      https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz --output "$archive"
  fi
  tar -xzf "$archive" -C "$data_root"
  if [[ -d "$data_root/raw_panel_images" ]]; then
    mv "$data_root/raw_panel_images" "$target"
  elif [[ ! -d "$target" ]]; then
    echo "Archive extracted, but expected raw_panel_images/ was not found under $data_root" >&2
    exit 2
  fi
fi

if [[ ! -d segments/gaze_heads_qwen3_8b/data/discovery_comics ]]; then
  echo "Raw COMICS is still missing. Re-run with --download-raw-comics (about 65 GB)." >&2
else
  echo "All Qwen3 Gaze Heads datasets are ready."
fi
