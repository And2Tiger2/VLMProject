#!/usr/bin/env bash
#SBATCH --job-name=q3-gaze-disc
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
COMICS_ROOT="${DISCOVERY_COMICS_ROOT:-$SEGMENT_ROOT/data/discovery_comics}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-10}"
BASE_SEED="${BASE_SEED:-42}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_INDEX="$((TASK_ID / N_SHARDS))"
SHARD_INDEX="$((TASK_ID % N_SHARDS))"
SEED="$((BASE_SEED + SEED_INDEX))"
START_COMIC_IDX="${START_COMIC_IDX:-$((SHARD_INDEX * SHARD_SIZE))}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/gaze_discovery_seed${SEED}_${START_COMIC_IDX}_${SHARD_SIZE}}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED shard=$SHARD_INDEX start=$START_COMIC_IDX"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
srun uv run python scripts/discover_qwen25_gaze_heads.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda \
  --comics-root "$COMICS_ROOT" \
  --use-raw \
  --out-dir "$OUT_DIR" \
  --n-samples "$((N_SHARDS * SHARD_SIZE))" \
  --start-comic-idx "$START_COMIC_IDX" \
  --max-comics "$SHARD_SIZE" \
  --seed "$SEED"
