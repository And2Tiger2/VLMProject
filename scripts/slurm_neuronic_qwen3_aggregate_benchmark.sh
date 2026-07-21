#!/usr/bin/env bash
#SBATCH --job-name=q3-bench-aggregate
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
BASE_SEED="${BASE_SEED:-0}"
N_SEEDS="${N_SEEDS:-1}"
ATTENTION_MODE="${ATTENTION_MODE:-full}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-10:50:100}"
IFS=: read -r -a ALPHAS_ARRAY <<< "${ALPHAS_COLON:-0.25:0.5:1:2:5:10}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/benchmark_attention_${ATTENTION_MODE}}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

seeds=()
for ((index = 0; index < N_SEEDS; index++)); do
  seeds+=("$((BASE_SEED + index))")
done
uv run python scripts/aggregate_vlmbias_gaze_attention_sweep.py \
  --out-dir "$OUT_DIR" --expected-seeds "${seeds[@]}" \
  --expected-top-ks "${TOP_KS_ARRAY[@]}" --expected-alphas "${ALPHAS_ARRAY[@]}" --strict
uv run python scripts/validate_qwen3_gaze_stage.py benchmark --run-dir "$OUT_DIR"
