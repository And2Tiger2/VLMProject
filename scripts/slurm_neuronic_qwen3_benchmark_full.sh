#!/usr/bin/env bash
#SBATCH --job-name=q3-gaze-bench
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
RANKING_SEED="${RANKING_SEED:-42}"
GAZE_RANKING="${GAZE_RANKING:-$SEGMENT_ROOT/runs/gaze_discovery_seed${RANKING_SEED}_merged/gaze_head_ranking.json}"
BASE_SEED="${BASE_SEED:-0}"
SEED="$((BASE_SEED + ${SLURM_ARRAY_TASK_ID:-0}))"
ATTENTION_MODE="${ATTENTION_MODE:-full}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/benchmark_attention_${ATTENTION_MODE}}"
DATASET="${DATASET:-segments/vlm_bias_attention/data/vlmbias_400.jsonl}"
NATURALBENCH_DATASET="${NATURALBENCH_DATASET:-segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl}"
LIMIT="${LIMIT:-0}"
NATURALBENCH_LIMIT_GROUPS="${NATURALBENCH_LIMIT_GROUPS:-0}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-10:50:100}"
IFS=: read -r -a ALPHAS_ARRAY <<< "${ALPHAS_COLON:-0.25:0.5:1:2:5:10}"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED attention_mode=$ATTENTION_MODE"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"
attention_flag="--no-decode-only"
if [[ "$ATTENTION_MODE" == "decode" ]]; then
  attention_flag="--decode-only"
elif [[ "$ATTENTION_MODE" != "full" ]]; then
  echo "ATTENTION_MODE must be full or decode, got: $ATTENTION_MODE" >&2
  exit 2
fi
srun uv run python scripts/run_vlmbias_gaze_attention_sweep.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --device-map cuda \
  --gaze-ranking "$GAZE_RANKING" \
  --out-dir "$OUT_DIR" \
  --dataset "$DATASET" \
  --naturalbench-dataset "$NATURALBENCH_DATASET" \
  --limit "$LIMIT" \
  --naturalbench-limit-groups "$NATURALBENCH_LIMIT_GROUPS" \
  --seeds "$SEED" \
  --top-ks "${TOP_KS_ARRAY[@]}" \
  --alphas "${ALPHAS_ARRAY[@]}" \
  "$attention_flag" \
  --skip-run-summary-files \
  --resume
