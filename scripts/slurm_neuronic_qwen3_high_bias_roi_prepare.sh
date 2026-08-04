#!/usr/bin/env bash
#SBATCH --job-name=q3-hbroi-prep
#SBATCH --output=segments/vlm_bias_attention/runs/slurm/%x-%j.out
#SBATCH --error=segments/vlm_bias_attention/runs/slurm/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
STAGE="${STAGE:?STAGE is required}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-segments/vlm_bias_attention/experiments/qwen3_high_bias_roi_attention_v1}"
REPORT_ROOT="${REPORT_ROOT:-segments/vlm_bias_attention/reports/qwen3_high_bias_roi_attention_v1}"
ROI_ROOT="${ROI_ROOT:-segments/vlm_bias_attention/data/vlmbias_high_bias_roi_masks_v1}"
VLMBIAS_DATASET="${VLMBIAS_DATASET:-segments/vlm_bias_attention/data/vlmbias_high_bias_roi_masks_v1/dataset/vlmbias_high_bias_114.jsonl}"

mkdir -p "$CACHE_ROOT/uv" "$REPO/segments/vlm_bias_attention/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run python scripts/prepare_qwen3_high_bias_roi_attention_experiment.py "$STAGE" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --report-root "$REPORT_ROOT" \
  --vlmbias "$VLMBIAS_DATASET" \
  --roi-root "$ROI_ROOT"
