#!/usr/bin/env bash
#SBATCH --job-name=q3-gaze-bench-smoke
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
GAZE_RANKING="${GAZE_RANKING:-$SEGMENT_ROOT/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/benchmark_attention_smoke}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} benchmark smoke"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
srun uv run python scripts/run_vlmbias_gaze_attention_sweep.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda \
  --gaze-ranking "$GAZE_RANKING" \
  --out-dir "$OUT_DIR" \
  --limit 20 \
  --naturalbench-limit-groups 5 \
  --seeds 0 \
  --top-ks 10 \
  --alphas 1 5 \
  --no-decode-only \
  --no-do-sample
