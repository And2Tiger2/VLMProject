#!/usr/bin/env bash
#SBATCH --job-name=q3-base-search-post
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
MODE="${MODE:-full}"
SEED="${SEED:-260809001}"
TASK="${TASK:-ranking}"
RUN_NAMESPACE="${RUN_NAMESPACE:-}"
cd "$REPO"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  echo "refusing frozen-base search postprocess: tracked worktree changes do not match HEAD" >&2
  exit 2
fi
export UV_NO_SYNC=1
export UV_FROZEN=1
CONFIG="segments/mechanistic_heads_qwen3_8b/configs/base_search.json"
SMOKE_ARGS=()
RUN_POLICY=(--resume)
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="segments/mechanistic_heads_qwen3_8b/configs/base_search_smoke.json"
  SMOKE_ARGS+=(--smoke)
  RUN_POLICY=(--overwrite)
fi
if [[ "$TASK" == "ranking" ]]; then
  SOURCE="segments/mechanistic_heads_qwen3_8b/runs/base_search_head_scan/$MODE/base_search_head_scores.tsv"
  OUT="segments/mechanistic_heads_qwen3_8b/reports/base_search_heads/$MODE"
  if [[ -n "$RUN_NAMESPACE" ]]; then
    SOURCE="segments/mechanistic_heads_qwen3_8b/runs/base_search_head_scan/$MODE/$RUN_NAMESPACE/base_search_head_scores.tsv"
    OUT="$OUT/$RUN_NAMESPACE"
  fi
  uv run python scripts/analyze_base_search_heads.py \
    --config "$CONFIG" --source "$SOURCE" --output-dir "$OUT" --seed "$SEED" \
    "${RUN_POLICY[@]}" "${SMOKE_ARGS[@]}"
elif [[ "$TASK" == "seed-summary" ]]; then
  [[ -n "${SEEDS:-}" ]] || { echo "SEEDS is required for seed-summary" >&2; exit 2; }
  IFS=':' read -r -a SEED_VALUES <<< "$SEEDS"
  OUT="segments/mechanistic_heads_qwen3_8b/reports/base_search_heads/$MODE/multiseed"
  uv run python scripts/analyze_base_search_seed_stability.py \
    --config "$CONFIG" --output-dir "$OUT" --seed 0 --resume \
    --mode "$MODE" --seeds "${SEED_VALUES[@]}"
else
  echo "unknown base-search postprocess task: $TASK" >&2
  exit 2
fi
uv run python scripts/validate_mechanistic_run.py --repo "$REPO" --run-dir "$OUT"
