#!/usr/bin/env bash
#SBATCH --job-name=q3-gaze-static-smoke
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
DISCOVERY_COMICS_ROOT="${DISCOVERY_COMICS_ROOT:-$SEGMENT_ROOT/data/discovery_comics}"
EVAL_COMICS_ROOT="${EVAL_COMICS_ROOT:-$SEGMENT_ROOT/data/eval_comics}"
GAZE_RANKING="${GAZE_RANKING:-$SEGMENT_ROOT/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/static_smoke_top10}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

uv run python scripts/validate_qwen3_gaze_stage.py datasets \
  --discovery-root "$DISCOVERY_COMICS_ROOT" --eval-root "$EVAL_COMICS_ROOT"
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} static smoke"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
srun uv run python scripts/run_qwen25_gaze_static_narration.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda \
  --comics-root "$EVAL_COMICS_ROOT" \
  --gaze-ranking "$GAZE_RANKING" \
  --out-dir "$OUT_DIR" \
  --max-comics 5 \
  --top-k-gaze 10 \
  --top-k-random 10 \
  --targets-per-strip 6 \
  --max-new-tokens 80 \
  --no-decode-only
uv run python scripts/validate_qwen3_gaze_stage.py static --run-dir "$OUT_DIR"
