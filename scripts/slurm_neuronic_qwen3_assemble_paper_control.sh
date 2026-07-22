#!/usr/bin/env bash
#SBATCH --job-name=q3-paper-merge
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
N_SHARDS="${N_SHARDS:-1}"
SOURCE_GAZE_COMICS="${SOURCE_GAZE_COMICS:-500}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-100}"
SEED="$((BASE_SEED + ${SLURM_ARRAY_TASK_ID:-0}))"
N_COMICS="$((N_SHARDS * SHARD_SIZE))"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

for top_k in "${TOP_KS_ARRAY[@]}"; do
  shards=()
  for ((shard = 0; shard < N_SHARDS; shard++)); do
    start="$((shard * SHARD_SIZE))"
    shards+=("$SEGMENT_ROOT/runs/static_paper_control_seed${SEED}_top${top_k}_${start}_${SHARD_SIZE}")
  done
  gaze_run="$SEGMENT_ROOT/runs/static_narration_seed${SEED}_top${top_k}_merged_0_${SOURCE_GAZE_COMICS}"
  out_dir="$SEGMENT_ROOT/runs/static_paper_replication_seed${SEED}_top${top_k}_merged_0_${N_COMICS}"
  uv run python scripts/assemble_qwen3_static_paper_control.py \
    --existing-gaze-run "$gaze_run" \
    --control-shard-runs "${shards[@]}" \
    --out-dir "$out_dir" \
    --top-k "$top_k" \
    --expected-comics "$N_COMICS"
  uv run python scripts/validate_qwen3_gaze_stage.py static --run-dir "$out_dir"
done
