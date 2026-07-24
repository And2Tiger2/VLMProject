#!/usr/bin/env bash
#SBATCH --job-name=q3-attn-method
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
STAGE="${STAGE:?STAGE is required}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1}"
RUN_ROOT="${RUN_ROOT:-segments/gaze_heads_qwen3_8b/runs/attention_methods_v1}"
GAZE_RANKING="${GAZE_RANKING:-segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" \
  "$REPO/segments/gaze_heads_qwen3_8b/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLM_REQUIRE_OVERHEAT_CHECK=1

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} stage=$STAGE task=${SLURM_ARRAY_TASK_ID:-0}"
uv run python scripts/check_neuronic_overheat.py
uv run python scripts/check_neuronic_gpu.py \
  --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
srun uv run python scripts/run_qwen3_attention_method_condition.py \
  --manifest "$EXPERIMENT_ROOT/${STAGE}_manifest.json" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --run-root "$RUN_ROOT" \
  --gaze-ranking "$GAZE_RANKING" \
  --device-map cuda \
  --resume
