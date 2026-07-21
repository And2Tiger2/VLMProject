#!/usr/bin/env bash
#SBATCH --job-name=q3-gaze-merge
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
BASE_SEED="${BASE_SEED:-42}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-10}"
SEED="$((BASE_SEED + ${SLURM_ARRAY_TASK_ID:-0}))"
OUT_DIR="$SEGMENT_ROOT/runs/gaze_discovery_seed${SEED}_merged"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

shard_dirs=()
for ((shard = 0; shard < N_SHARDS; shard++)); do
  start="$((shard * SHARD_SIZE))"
  shard_dirs+=("$SEGMENT_ROOT/runs/gaze_discovery_seed${SEED}_${start}_${SHARD_SIZE}")
done

uv run python scripts/merge_qwen25_gaze_discovery_shards.py \
  --shard-dirs "${shard_dirs[@]}" --out-dir "$OUT_DIR"
uv run python scripts/validate_qwen3_gaze_stage.py discovery \
  --run-dir "$OUT_DIR" --min-samples "$((N_SHARDS * SHARD_SIZE))"
