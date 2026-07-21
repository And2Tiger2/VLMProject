#!/usr/bin/env bash
#SBATCH --job-name=q3-static-judge
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
BASE_SEED="${BASE_SEED:-42}"
N_SEEDS="${N_SEEDS:-1}"
N_SHARDS="${N_SHARDS:-10}"
SHARD_SIZE="${SHARD_SIZE:-50}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-1:10:50:100}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_INDEX="$((TASK_ID / ${#TOP_KS_ARRAY[@]}))"
TOP_K_INDEX="$((TASK_ID % ${#TOP_KS_ARRAY[@]}))"
SEED="$((BASE_SEED + SEED_INDEX))"
TOP_K="${TOP_KS_ARRAY[$TOP_K_INDEX]}"
RUN_DIR="$SEGMENT_ROOT/runs/static_narration_seed${SEED}_top${TOP_K}_merged_0_$((N_SHARDS * SHARD_SIZE))"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is required for the paper-style static judge." >&2
  exit 2
fi
mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run python scripts/score_qwen25_gaze_generations.py \
  --generations "$RUN_DIR/generations.jsonl" \
  --out-dir "$RUN_DIR/anthropic_judge" \
  --judge anthropic --seed "$SEED" --resume
