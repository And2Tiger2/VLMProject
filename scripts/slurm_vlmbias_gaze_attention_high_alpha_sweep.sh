#!/usr/bin/env bash
#SBATCH --job-name=vlmbias-gaze-highalpha
#SBATCH --output=segments/vlm_bias_attention/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/vlm_bias_attention/runs/slurm/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

set -euo pipefail

export PROJ="${PROJ:-/n/fs/pvl-memory/at7979}"
export REPO="${REPO:-$PROJ/VLMProject}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJ/uv_cache}"
export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJ/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PROJ/hf_cache/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJ/torch_cache}"
export TMPDIR="${TMPDIR:-$PROJ/tmp}"
export PATH="$PROJ/bin:$PATH"

mkdir -p "$REPO/segments/vlm_bias_attention/runs/slurm" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$REPO"

DATASET="${DATASET:-segments/vlm_bias_attention/data/vlmbias_400.jsonl}"
NATURALBENCH_DATASET="${NATURALBENCH_DATASET:-segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl}"
OUT_DIR="${OUT_DIR:-segments/vlm_bias_attention/runs/vlmbias_gaze_attention_high_alpha_sweep}"
GAZE_RANKING="${GAZE_RANKING:-segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json}"
LIMIT="${LIMIT:-}"
NATURALBENCH_LIMIT_GROUPS="${NATURALBENCH_LIMIT_GROUPS:-}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
TOP_KS="${TOP_KS:-10 20}"
ALPHAS="${ALPHAS:-10 20 30 50 100}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
DECODE_ONLY="${DECODE_ONLY:-0}"
SKIP_NATURALBENCH="${SKIP_NATURALBENCH:-0}"
RESUME="${RESUME:-1}"

if [[ ! -f "$DATASET" ]]; then
  echo "VLMBias dataset not found: $DATASET" >&2
  exit 1
fi

if [[ "$SKIP_NATURALBENCH" != "1" && ! -f "$NATURALBENCH_DATASET" ]]; then
  echo "NaturalBench dataset not found: $NATURALBENCH_DATASET" >&2
  echo "Set SKIP_NATURALBENCH=1 to run only VLMBias." >&2
  exit 1
fi

if [[ ! -f "$GAZE_RANKING" ]]; then
  echo "Gaze ranking not found: $GAZE_RANKING" >&2
  exit 1
fi

read -r -a top_k_array <<< "$TOP_KS"
read -r -a alpha_array <<< "$ALPHAS"

cmd=(
  uv run python scripts/run_vlmbias_gaze_attention_sweep.py
  --dataset "$DATASET"
  --naturalbench-dataset "$NATURALBENCH_DATASET"
  --out-dir "$OUT_DIR"
  --gaze-ranking "$GAZE_RANKING"
  --device-map "$DEVICE_MAP"
  --seeds "$SEED"
  --top-ks "${top_k_array[@]}"
  --alphas "${alpha_array[@]}"
  --skip-run-summary-files
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ -n "$NATURALBENCH_LIMIT_GROUPS" ]]; then
  cmd+=(--naturalbench-limit-groups "$NATURALBENCH_LIMIT_GROUPS")
fi

if [[ "$DECODE_ONLY" == "1" ]]; then
  cmd+=(--decode-only)
fi

if [[ "$SKIP_NATURALBENCH" == "1" ]]; then
  cmd+=(--skip-naturalbench)
fi

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Seed: $SEED"
echo "Top-k gaze heads: $TOP_KS"
echo "Attention alphas: $ALPHAS"
echo "Output directory: $OUT_DIR"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
