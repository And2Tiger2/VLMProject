#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"
ACTION="${1:-help}"
MODE="${2:-smoke}"
cd "$REPO"

if [[ "$ACTION" == "overnight-smoke" || "$ACTION" == "overnight-all" || "$ACTION" == "overnight-all-resume" ]]; then
  # Resolve the shared environment once before submitting concurrent jobs.
  # Point-search training requires PEFT from the mechanistic extra.
  uv sync --extra qwen --extra mechanistic --extra dev
fi

case "$ACTION" in
  overnight-smoke)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile smoke
    ;;
  overnight-all)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile all --confirm-full
    ;;
  overnight-all-resume)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile all --confirm-full --reuse-prepared
    ;;
  prepare-synthetic)
    uv run python scripts/generate_counting_data.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/counting_data.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/counting \
      --seed 260318523
    uv run python scripts/generate_point_search_data.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/point_search_data.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/point_search \
      --seed 260525427
    uv run python scripts/prepare_vlmbias_signed_contrasts.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/vlmbias_contrasts.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts \
      --seed 260519250
    ;;
  download-mmmc)
    HF_HUB_DISABLE_XET=1 uv run python scripts/prepare_mmmc.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/mmmc.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed 260519250
    ;;
  counting-controls)
    uv run python scripts/analyze_count_head_controls.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/counting_controls.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/counting_controls \
      --seed 260318523 --resume
    ;;
  general-importance)
    uv run python scripts/build_general_head_importance.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/general_head_importance.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/general_importance \
      --seed 0 --resume
    ;;
  atlas)
    uv run python scripts/render_mechanistic_head_reports.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/head_atlas.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/atlas \
      --seed 0 --resume
    ;;
  instrumentation|counting-behavior|counting-vap|counting-heads|counting-validation|waldo-behavior|point-centroids|search-heads|verification-heads|distractor-heads|point-ablation|maci-heads|maci-heads-aligned|maci-ablation|maci-confirm|maci-detector|maci-gated|vlmbias-heads|vlmbias-validation)
    if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
      echo "mode must be smoke or full" >&2
      exit 2
    fi
    if [[ "$MODE" == "full" && "$ACTION" =~ ^(counting-vap|counting-heads|point-centroids|search-heads|verification-heads|distractor-heads|maci-heads|maci-heads-aligned|vlmbias-heads)$ ]]; then
      run_job=$(sbatch --parsable --array=0-35%4 \
        --export "ALL,REPO=$REPO,TASK=$ACTION,MODE=$MODE" \
        scripts/slurm_neuronic_mechanistic_heads.sh)
      aggregate_job=$(sbatch --parsable --dependency "afterok:$run_job" \
        --export "ALL,REPO=$REPO,SOURCE_TASK=$ACTION" \
        scripts/slurm_neuronic_mechanistic_aggregate.sh)
      echo "submitted layer array $run_job and aggregate $aggregate_job"
    else
      sbatch --parsable \
        --export "ALL,REPO=$REPO,TASK=$ACTION,MODE=$MODE" \
        scripts/slurm_neuronic_mechanistic_heads.sh
    fi
    ;;
  help|*)
    echo "usage: $0 {overnight-smoke|overnight-all|overnight-all-resume|prepare-synthetic|download-mmmc|instrumentation|counting-behavior|counting-vap|counting-heads|general-importance|counting-controls|counting-validation|waldo-behavior|point-centroids|search-heads|verification-heads|distractor-heads|point-ablation|maci-heads|maci-heads-aligned|maci-ablation|maci-confirm|maci-detector|maci-gated|vlmbias-heads|vlmbias-validation|atlas} [smoke|full]"
    ;;
esac
