#!/usr/bin/env bash
#SBATCH --job-name=vlmbias-2pass-fast
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err
#SBATCH --time=12:00:00
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
OUT_DIR="${OUT_DIR:-runs/vlmbias_two_pass_prompt_variants_fast}"
LIMIT="${LIMIT:-}"
CONDITIONS="${CONDITIONS:-}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET" >&2
  echo "Create it first with scripts/make_vlmbias_slice.py or copy data/ to the cluster." >&2
  exit 1
fi

cmd=(
  uv run python scripts/run_vlmbias_two_pass_prompt_variants.py
  --dataset "$DATASET"
  --out-dir "$OUT_DIR"
  --device-map "$DEVICE_MAP"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ -n "$CONDITIONS" ]]; then
  read -r -a condition_array <<< "$CONDITIONS"
  cmd+=(--conditions "${condition_array[@]}")
fi

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
