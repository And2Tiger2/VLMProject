#!/usr/bin/env bash
#SBATCH --job-name=gaze-disc
#SBATCH --output=segments/gaze_heads_qwen25/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen25/runs/slurm/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

export PROJ="${PROJ:-/n/fs/pvl-memory/at7979}"
export REPO="${REPO:-$PROJ/VLMProject}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJ/uv_cache}"
export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJ/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJ/hf_cache/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJ/torch_cache}"
export TMPDIR="${TMPDIR:-$PROJ/tmp}"
export PATH="$PROJ/bin:$PATH"

mkdir -p "$REPO/segments/gaze_heads_qwen25/runs/slurm" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$REPO"

START_COMIC_IDX="${START_COMIC_IDX:?Set START_COMIC_IDX, e.g. 0}"
MAX_COMICS="${MAX_COMICS:-50}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen25}"
COMICS_ROOT="${COMICS_ROOT:-$SEGMENT_ROOT/data/comics}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/gaze_discovery_${START_COMIC_IDX}_${MAX_COMICS}}"

if [[ ! -d "$COMICS_ROOT" ]]; then
  echo "Comics root not found: $COMICS_ROOT" >&2
  exit 1
fi

cmd=(
  uv run python scripts/discover_qwen25_gaze_heads.py
  --comics-root "$COMICS_ROOT"
  --out-dir "$OUT_DIR"
  --model-id "$MODEL_ID"
  --device-map "$DEVICE_MAP"
  --start-comic-idx "$START_COMIC_IDX"
  --max-comics "$MAX_COMICS"
)

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
