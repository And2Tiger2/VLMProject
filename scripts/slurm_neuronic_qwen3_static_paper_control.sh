#!/usr/bin/env bash
#SBATCH --job-name=q3-paper-ctrl
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
EVAL_COMICS_ROOT="${EVAL_COMICS_ROOT:-$SEGMENT_ROOT/data/eval_comics}"
RANKING_SEED="${RANKING_SEED:-42}"
GAZE_RANKING="${GAZE_RANKING:-$SEGMENT_ROOT/runs/gaze_discovery_seed${RANKING_SEED}_merged/gaze_head_ranking.json}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-1}"
BASE_SEED="${BASE_SEED:-42}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-100}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASKS_PER_SEED="$((N_SHARDS * ${#TOP_KS_ARRAY[@]}))"
SEED_INDEX="$((TASK_ID / TASKS_PER_SEED))"
WITHIN_SEED="$((TASK_ID % TASKS_PER_SEED))"
TOP_K_INDEX="$((WITHIN_SEED / N_SHARDS))"
SHARD_INDEX="$((WITHIN_SEED % N_SHARDS))"
SEED="$((BASE_SEED + SEED_INDEX))"
TOP_K="${TOP_KS_ARRAY[$TOP_K_INDEX]}"
START_COMIC_IDX="$((SHARD_INDEX * SHARD_SIZE))"
OUT_DIR="$SEGMENT_ROOT/runs/static_paper_control_seed${SEED}_top${TOP_K}_${START_COMIC_IDX}_${SHARD_SIZE}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED top_k=$TOP_K start=$START_COMIC_IDX paper_control=layers20-35"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
srun uv run python scripts/run_qwen25_gaze_static_narration.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda \
  --comics-root "$EVAL_COMICS_ROOT" \
  --gaze-ranking "$GAZE_RANKING" \
  --out-dir "$OUT_DIR" \
  --start-comic-idx "$START_COMIC_IDX" \
  --max-comics "$SHARD_SIZE" \
  --top-k-gaze "$TOP_K" \
  --top-k-random "$TOP_K" \
  --control-mode paper \
  --condition-set control \
  --targets-per-strip 6 \
  --max-new-tokens 100 \
  --no-decode-only \
  --seed "$SEED" \
  --resume
uv run python scripts/validate_qwen3_gaze_stage.py static --run-dir "$OUT_DIR"
