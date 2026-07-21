#!/usr/bin/env bash
#SBATCH --job-name=q3-static-merge
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
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-1:10:50:100}"
SEED="$((BASE_SEED + ${SLURM_ARRAY_TASK_ID:-0}))"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

starts=()
for ((shard = 0; shard < N_SHARDS; shard++)); do
  starts+=("$((shard * SHARD_SIZE))")
done
uv run python scripts/merge_gaze_static_topk_batches.py \
  --segment-root "$SEGMENT_ROOT" --run-tag "seed${SEED}" \
  --top-ks "${TOP_KS_ARRAY[@]}" --starts "${starts[@]}" --batch-size "$SHARD_SIZE"

for top_k in "${TOP_KS_ARRAY[@]}"; do
  uv run python scripts/validate_qwen3_gaze_stage.py static \
    --run-dir "$SEGMENT_ROOT/runs/static_narration_seed${SEED}_top${top_k}_merged_0_$((N_SHARDS * SHARD_SIZE))"
done
