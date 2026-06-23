#!/usr/bin/env bash
#SBATCH --job-name=pg2-attn-alpha
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err
#SBATCH --time=72:00:00
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

VLMBIAS_DATASET="${VLMBIAS_DATASET:-data/vlmbias_400.jsonl}"
NATURALBENCH_DATASET="${NATURALBENCH_DATASET:-data/naturalbench_100_groups.jsonl}"
OUT_DIR="${OUT_DIR:-runs/paligemma2_attention_alpha_sweep}"
MODEL_ID="${MODEL_ID:-google/paligemma2-3b-mix-448}"
VLMBIAS_LIMIT="${VLMBIAS_LIMIT:-}"
NATURALBENCH_LIMIT_GROUPS="${NATURALBENCH_LIMIT_GROUPS:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ALPHAS="${ALPHAS:-0.05 0.1 0.25 0.5 1.0 2.0 3.0 4.0 6.0 8.0 10.0}"
LAYER_SELECTIONS="${LAYER_SELECTIONS:-last25 last50 all}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
SKIP_NATURALBENCH="${SKIP_NATURALBENCH:-0}"

if [[ ! -f "$VLMBIAS_DATASET" ]]; then
  echo "VLMBias dataset not found: $VLMBIAS_DATASET" >&2
  echo "Create it first with scripts/make_vlmbias_slice.py or copy data/ to the cluster." >&2
  exit 1
fi

if [[ "$SKIP_NATURALBENCH" != "1" && ! -f "$NATURALBENCH_DATASET" ]]; then
  echo "NaturalBench dataset not found: $NATURALBENCH_DATASET" >&2
  echo "Create it first with scripts/make_naturalbench_slice.py or copy data/ to the cluster." >&2
  exit 1
fi

read -r -a seed_array <<< "$SEEDS"
read -r -a alpha_array <<< "$ALPHAS"
read -r -a layer_selection_array <<< "$LAYER_SELECTIONS"

cmd=(
  uv run --extra paligemma python scripts/run_paligemma2_attention_alpha_sweep.py
  --vlmbias-dataset "$VLMBIAS_DATASET"
  --naturalbench-dataset "$NATURALBENCH_DATASET"
  --out-dir "$OUT_DIR"
  --model-id "$MODEL_ID"
  --device-map "$DEVICE_MAP"
  --seeds "${seed_array[@]}"
  --alphas "${alpha_array[@]}"
  --layer-selections "${layer_selection_array[@]}"
)

if [[ -n "$VLMBIAS_LIMIT" ]]; then
  cmd+=(--vlmbias-limit "$VLMBIAS_LIMIT")
fi

if [[ -n "$NATURALBENCH_LIMIT_GROUPS" ]]; then
  cmd+=(--naturalbench-limit-groups "$NATURALBENCH_LIMIT_GROUPS")
fi

if [[ "$SKIP_NATURALBENCH" == "1" ]]; then
  cmd+=(--skip-naturalbench)
fi

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
