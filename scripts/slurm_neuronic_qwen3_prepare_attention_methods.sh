#!/usr/bin/env bash
#SBATCH --job-name=q3-attn-prep
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
STAGE="${STAGE:?STAGE is required}"
SEEDS_COLON="${SEEDS_COLON:-0:1:2}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/segments/gaze_heads_qwen3_8b/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

IFS=: read -r -a SEEDS <<< "$SEEDS_COLON"
uv run python scripts/prepare_qwen3_attention_methods.py "$STAGE" \
  --seeds "${SEEDS[@]}"
