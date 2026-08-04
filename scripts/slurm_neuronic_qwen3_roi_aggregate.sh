#!/usr/bin/env bash
#SBATCH --job-name=q3-roi-agg
#SBATCH --output=segments/vlm_bias_attention/runs/slurm/%x-%j.out
#SBATCH --error=segments/vlm_bias_attention/runs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
STAGE="${STAGE:?STAGE is required}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-segments/vlm_bias_attention/experiments/qwen3_roi_attention_v1}"
RUN_ROOT="${RUN_ROOT:-segments/vlm_bias_attention/runs/qwen3_roi_attention_v1}"
REPORT_ROOT="${REPORT_ROOT:-segments/vlm_bias_attention/reports/qwen3_roi_attention_v1}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/segments/vlm_bias_attention/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run python scripts/aggregate_qwen3_roi_attention.py "$STAGE" \
  --manifest "$EXPERIMENT_ROOT/${STAGE}_manifest.json" \
  --run-root "$RUN_ROOT" \
  --report-root "$REPORT_ROOT"
