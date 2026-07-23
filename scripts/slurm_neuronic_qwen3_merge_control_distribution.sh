#!/usr/bin/env bash
#SBATCH --job-name=q3-ctrl-merge
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-2}"
TOP_K="${TOP_K:-100}"
IFS=: read -r -a SPECS <<< "${CONTROL_SPECS_COLON:?CONTROL_SPECS_COLON is required}"
SPEC="${SPECS[${SLURM_ARRAY_TASK_ID:-0}]}"
MODE="${SPEC%@*}"
SEED="${SPEC#*@}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

starts=()
for ((shard = 0; shard < N_SHARDS; shard++)); do
  starts+=("$((shard * SHARD_SIZE))")
done
uv run python scripts/merge_qwen3_control_distribution.py \
  --segment-root "$SEGMENT_ROOT" \
  --mode "$MODE" \
  --seed "$SEED" \
  --top-k "$TOP_K" \
  --starts "${starts[@]}" \
  --shard-size "$SHARD_SIZE"
uv run python scripts/validate_qwen3_gaze_stage.py static \
  --run-dir "$SEGMENT_ROOT/runs/static_control_distribution_${MODE}_seed${SEED}_top${TOP_K}_merged_0_$((N_SHARDS * SHARD_SIZE))"
