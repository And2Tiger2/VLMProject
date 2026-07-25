#!/usr/bin/env bash
#SBATCH --job-name=q3-spec-prep
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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-segments/gaze_heads_qwen3_8b/experiments/gaze_specificity_v2}"
REPORT_ROOT="${REPORT_ROOT:-segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2}"
SOURCE_REPORT_ROOT="${SOURCE_REPORT_ROOT:-segments/gaze_heads_qwen3_8b/reports/attention_methods_v1}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/segments/gaze_heads_qwen3_8b/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

IFS=: read -r -a SEEDS <<< "$SEEDS_COLON"
uv run python scripts/prepare_qwen3_gaze_specificity.py "$STAGE" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --report-root "$REPORT_ROOT" \
  --source-report-root "$SOURCE_REPORT_ROOT" \
  --seeds "${SEEDS[@]}"
