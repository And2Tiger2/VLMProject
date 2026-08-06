#!/usr/bin/env bash
#SBATCH --job-name=q3-mech-post
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
TASK="${TASK:?TASK is required}"
MODE="${MODE:-full}"
SEED="${SEED:-260318523}"
JOB_STARTED_AT="$(date +%s)"
cd "$REPO"
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export TOKENIZERS_PARALLELISM=false

smoke_args=()
if [[ "$MODE" == "smoke" ]]; then
  smoke_args+=(--smoke)
elif [[ "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

case "$TASK" in
  counting-controls)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/counting_controls"
    CONFIG="segments/mechanistic_heads_qwen3_8b/configs/counting_controls.json"
    if [[ "$MODE" == "smoke" ]]; then
      VERIFY_DIR="$VERIFY_DIR/smoke"
      CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_counting_controls.json"
    fi
    uv run python scripts/analyze_count_head_controls.py \
      --config "$CONFIG" --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --resume "${smoke_args[@]}"
    ;;
  general-importance)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/general_importance"
    CONFIG="segments/mechanistic_heads_qwen3_8b/configs/general_head_importance.json"
    if [[ "$MODE" == "smoke" ]]; then
      VERIFY_DIR="$VERIFY_DIR/smoke"
      CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_general_head_importance.json"
    fi
    uv run python scripts/build_general_head_importance.py \
      --config "$CONFIG" --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --resume "${smoke_args[@]}"
    ;;
  maci-stability)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/maci_stability"
    CONFIG="segments/mechanistic_heads_qwen3_8b/configs/maci_stability.json"
    if [[ "$MODE" == "smoke" ]]; then
      VERIFY_DIR="$VERIFY_DIR/smoke"
      CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_maci_stability.json"
    fi
    uv run python scripts/analyze_maci_head_stability.py \
      --config "$CONFIG" \
      --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --resume "${smoke_args[@]}"
    ;;
  point-reports)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/point_search"
    CONFIG="segments/mechanistic_heads_qwen3_8b/configs/point_search_reports.json"
    if [[ "$MODE" == "smoke" ]]; then
      VERIFY_DIR="$VERIFY_DIR/smoke"
      CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_point_search_reports.json"
    fi
    uv run python scripts/render_point_search_reports.py \
      --config "$CONFIG" --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --resume "${smoke_args[@]}"
    ;;
  atlas)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/atlas"
    CONFIG="segments/mechanistic_heads_qwen3_8b/configs/head_atlas.json"
    if [[ "$MODE" == "smoke" ]]; then
      VERIFY_DIR="$VERIFY_DIR/smoke"
      CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_head_atlas.json"
    fi
    uv run python scripts/render_mechanistic_head_reports.py \
      --config "$CONFIG" --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --resume "${smoke_args[@]}"
    ;;
  *)
    echo "unknown postprocess TASK=$TASK" >&2
    exit 2
    ;;
esac

uv run python scripts/validate_mechanistic_run.py \
  --repo "$REPO" --run-dir "$VERIFY_DIR" --newer-than-epoch "$JOB_STARTED_AT"
