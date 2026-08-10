#!/usr/bin/env bash
#SBATCH --job-name=q3-base-search
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
TASK="${TASK:?TASK is required}"
MODE="${MODE:-full}"
SEED="${SEED:-260809001}"
cd "$REPO"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  echo "refusing frozen-base search run: tracked worktree changes do not match HEAD" >&2
  exit 2
fi
export UV_NO_SYNC=1
export UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"

CONFIG="segments/mechanistic_heads_qwen3_8b/configs/base_search.json"
SMOKE_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="segments/mechanistic_heads_qwen3_8b/configs/base_search_smoke.json"
  SMOKE_ARGS+=(--smoke)
elif [[ "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

case "$TASK" in
  instrumentation)
    OUT="segments/mechanistic_heads_qwen3_8b/reports/base_search_heads/instrumentation"
    uv run python scripts/validate_mechanistic_instrumentation.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/instrumentation_smoke.json \
      --output-dir "$OUT" --seed "$SEED" --device-map cuda --resume --smoke
    ;;
  behavior)
    OUT="segments/mechanistic_heads_qwen3_8b/runs/base_search_behavior/$MODE"
    uv run python scripts/run_base_search_behavior.py \
      --config "$CONFIG" --output-dir "$OUT" --seed "$SEED" --device-map cuda --resume "${SMOKE_ARGS[@]}"
    ;;
  discovery)
    OUT="segments/mechanistic_heads_qwen3_8b/runs/base_search_head_scan/$MODE"
    LAYER_ARGS=()
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      OUT="$OUT/layer-$SLURM_ARRAY_TASK_ID"
      LAYER_ARGS+=(--layers "$SLURM_ARRAY_TASK_ID")
    fi
    uv run python scripts/run_base_search_head_scan.py \
      --config "$CONFIG" --output-dir "$OUT" --seed "$SEED" --device-map cuda --resume \
      "${LAYER_ARGS[@]}" "${SMOKE_ARGS[@]}"
    ;;
  validation)
    OUT="segments/mechanistic_heads_qwen3_8b/runs/base_search_validation/$MODE"
    uv run python scripts/run_base_search_head_validation.py \
      --config "$CONFIG" --output-dir "$OUT" --seed "$SEED" --device-map cuda --resume "${SMOKE_ARGS[@]}"
    ;;
  *)
    echo "unknown base-search task: $TASK" >&2
    exit 2
    ;;
esac

uv run python scripts/validate_mechanistic_run.py --repo "$REPO" --run-dir "$OUT"
