#!/usr/bin/env bash
#SBATCH --job-name=q3-mech-prep
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
COUNT_SEED="${COUNT_SEED:-${SEED:-260318523}}"
POINT_SEED="${POINT_SEED:-260525427}"
VLMBIAS_SEED="${VLMBIAS_SEED:-260519250}"
MMMC_SEED="${MMMC_SEED:-260519250}"
MODE="${MODE:-full}"
JOB_STARTED_AT="$(date +%s)"

prepare_mode_args=()
if [[ "$MODE" == "smoke" ]]; then
  prepare_mode_args+=(--smoke)
elif [[ "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" \
  "$REPO/segments/mechanistic_heads_qwen3_8b/runs/slurm"
cd "$REPO"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  echo "refusing mechanistic preparation: tracked worktree changes do not match HEAD" >&2
  exit 2
fi
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

uv run python scripts/generate_counting_data.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/counting_data.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/counting \
  --seed "$COUNT_SEED" --overwrite "${prepare_mode_args[@]}"

uv run python scripts/generate_point_search_data.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/point_search_data.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/point_search \
  --seed "$POINT_SEED" --overwrite "${prepare_mode_args[@]}"

uv run python scripts/prepare_vlmbias_signed_contrasts.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/vlmbias_contrasts.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts \
  --seed "$VLMBIAS_SEED" --overwrite "${prepare_mode_args[@]}"

# This is the exact required Hub loader. The cache and all images remain
# ignored by Git. Tokenization is included in the overnight audit.
HF_HUB_OFFLINE=0 uv run python scripts/prepare_mmmc.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/mmmc.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared \
  --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
  --seed "$MMMC_SEED" --overwrite "${prepare_mode_args[@]}"

for RUN_DIR in \
  segments/mechanistic_heads_qwen3_8b/data/generated/counting \
  segments/mechanistic_heads_qwen3_8b/data/generated/point_search \
  segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts \
  segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared; do
  uv run python scripts/validate_mechanistic_run.py \
    --repo "$REPO" --run-dir "$RUN_DIR" --newer-than-epoch "$JOB_STARTED_AT"
done
