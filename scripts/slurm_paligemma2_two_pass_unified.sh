#!/usr/bin/env bash
#SBATCH --job-name=pg2-2pass-unified
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

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

mkdir -p "$REPO/runs/slurm" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$REPO"

DATASET="${DATASET:-data/vlmbias_400.jsonl}"
OUT_DIR="${OUT_DIR:-runs/paligemma2_two_pass_unified}"
MODEL_ID="${MODEL_ID:-google/paligemma2-3b-mix-448}"
LIMIT="${LIMIT:-}"
SEEDS="${SEEDS:-0}"
DESCRIPTION_PROMPT_VARIANTS="${DESCRIPTION_PROMPT_VARIANTS:-direct original counterfactual structured verification}"
ANSWER_PROMPT_VARIANTS="${ANSWER_PROMPT_VARIANTS:-original}"
ANSWER_MODES="${ANSWER_MODES:-image_again description_only}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET" >&2
  echo "Create it first with scripts/make_vlmbias_slice.py or copy data/ to the cluster." >&2
  exit 1
fi

read -r -a seed_array <<< "$SEEDS"
read -r -a description_prompt_variant_array <<< "$DESCRIPTION_PROMPT_VARIANTS"
read -r -a answer_prompt_variant_array <<< "$ANSWER_PROMPT_VARIANTS"
read -r -a answer_mode_array <<< "$ANSWER_MODES"

cmd=(
  uv run --extra paligemma python scripts/run_paligemma2_two_pass_unified.py
  --dataset "$DATASET"
  --out-dir "$OUT_DIR"
  --model-id "$MODEL_ID"
  --device-map "$DEVICE_MAP"
  --seeds "${seed_array[@]}"
  --description-prompt-variants "${description_prompt_variant_array[@]}"
  --answer-prompt-variants "${answer_prompt_variant_array[@]}"
  --answer-modes "${answer_mode_array[@]}"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
