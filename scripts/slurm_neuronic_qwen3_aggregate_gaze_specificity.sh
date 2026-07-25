#!/usr/bin/env bash
#SBATCH --job-name=q3-spec-agg
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
STAGE="${STAGE:?STAGE is required}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-segments/gaze_heads_qwen3_8b/experiments/gaze_specificity_v2}"
RUN_ROOT="${RUN_ROOT:-segments/gaze_heads_qwen3_8b/runs/gaze_specificity_v2}"
REPORT_ROOT="${REPORT_ROOT:-segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/segments/gaze_heads_qwen3_8b/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run python scripts/aggregate_qwen3_gaze_specificity.py "$STAGE" \
  --manifest "$EXPERIMENT_ROOT/${STAGE}_manifest.json" \
  --run-root "$RUN_ROOT" \
  --report-root "$REPORT_ROOT"
