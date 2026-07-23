#!/usr/bin/env bash
#SBATCH --job-name=q3-ctrl-agg
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
TOP_K="${TOP_K:-100}"
N_SHARDS="${N_SHARDS:-2}"
SHARD_SIZE="${SHARD_SIZE:-50}"
REFERENCE_SEED="${REFERENCE_SEED:-42}"
REFERENCE_COMICS="${REFERENCE_COMICS:-500}"
LOW_SEED="${LOW_SEED:-42}"
IFS=: read -r -a PAPER_SEEDS <<< "${PAPER_SEEDS_COLON:?PAPER_SEEDS_COLON is required}"
IFS=: read -r -a MATCHED_SEEDS <<< "${MATCHED_SEEDS_COLON:?MATCHED_SEEDS_COLON is required}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run python scripts/aggregate_qwen3_control_distribution.py \
  --segment-root "$SEGMENT_ROOT" \
  --paper-seeds "${PAPER_SEEDS[@]}" \
  --matched-seeds "${MATCHED_SEEDS[@]}" \
  --low-seed "$LOW_SEED" \
  --reference-seed "$REFERENCE_SEED" \
  --reference-comics "$REFERENCE_COMICS" \
  --n-comics "$((N_SHARDS * SHARD_SIZE))" \
  --top-k "$TOP_K"
