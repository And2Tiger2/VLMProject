#!/usr/bin/env bash
#SBATCH --job-name=q3-ctrl-dist
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
EVAL_COMICS_ROOT="${EVAL_COMICS_ROOT:-$SEGMENT_ROOT/data/eval_comics}"
GAZE_RANKING="${GAZE_RANKING:-$SEGMENT_ROOT/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-2}"
TOP_K="${TOP_K:-100}"
IFS=: read -r -a SPECS <<< "${CONTROL_SPECS_COLON:?CONTROL_SPECS_COLON is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SPEC_INDEX="$((TASK_ID / N_SHARDS))"
SHARD_INDEX="$((TASK_ID % N_SHARDS))"
SPEC="${SPECS[$SPEC_INDEX]}"
MODE="${SPEC%@*}"
SEED="${SPEC#*@}"
START_COMIC_IDX="$((SHARD_INDEX * SHARD_SIZE))"
OUT_DIR="$SEGMENT_ROOT/runs/static_control_distribution_${MODE}_seed${SEED}_top${TOP_K}_${START_COMIC_IDX}_${SHARD_SIZE}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} mode=$MODE head_seed=$SEED start=$START_COMIC_IDX"
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
  --control-mode "$MODE" \
  --condition-set control \
  --targets-per-strip 6 \
  --max-new-tokens 100 \
  --no-decode-only \
  --seed "$SEED" \
  --resume
uv run python scripts/validate_qwen3_gaze_stage.py static --run-dir "$OUT_DIR"
