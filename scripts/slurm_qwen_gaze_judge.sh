#!/usr/bin/env bash
#SBATCH --job-name=qwen-judge
#SBATCH --output=segments/gaze_heads_qwen25/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen25/runs/slurm/%x-%j.err
#SBATCH --time=12:00:00
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

GENERATIONS="${GENERATIONS:?Set GENERATIONS to a generations.jsonl file.}"
OUT_DIR="${OUT_DIR:-$(dirname "$GENERATIONS")/qwen_judge}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

cmd=(
  uv run python scripts/judge_qwen25_gaze_generations.py
  --generations "$GENERATIONS"
  --out-dir "$OUT_DIR"
  --model-id "$MODEL_ID"
  --device-map "$DEVICE_MAP"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --limit "$LIMIT"
  --resume
)

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
