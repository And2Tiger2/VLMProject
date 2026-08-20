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
OUT="segments/mechanistic_heads_qwen3_8b/reports/base_search_heads/$MODE"
uv run python scripts/analyze_base_search_heads.py \
  --config "$CONFIG" --output-dir "$OUT" --seed "$SEED" "${RUN_POLICY[@]}" "${SMOKE_ARGS[@]}"
uv run python scripts/validate_mechanistic_run.py --repo "$REPO" --run-dir "$OUT"
