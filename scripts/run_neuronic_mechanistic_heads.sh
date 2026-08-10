#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"
ACTION="${1:-help}"
MODE="${2:-smoke}"
cd "$REPO"

if [[ "$ACTION" == "base-search" || "$ACTION" == "base-search-smoke" ]]; then
  # The frozen-base search path deliberately has no PEFT/LoRA dependency.
  uv sync --frozen --extra qwen --extra dev
  export UV_NO_SYNC=1
  export UV_FROZEN=1
elif [[ "$ACTION" == "overnight-smoke" || "$ACTION" == "overnight-smoke-resume" || "$ACTION" == "overnight-all" || "$ACTION" == "overnight-all-resume" || "$ACTION" == "refresh-generated-data" || "$ACTION" == "point-recovery" ]]; then
  # Resolve the shared environment once before submitting concurrent jobs.
  # Point-search training requires PEFT from the mechanistic extra.
  uv sync --frozen --extra qwen --extra mechanistic --extra dev
  # Every submitted process uses this already-resolved shared environment.
  # Prevent concurrent ``uv run`` calls from trying to mutate it again.
  export UV_NO_SYNC=1
  export UV_FROZEN=1
fi

if [[ "$ACTION" == "overnight-smoke" || "$ACTION" == "overnight-smoke-resume" || "$ACTION" == "overnight-all" || "$ACTION" == "overnight-all-resume" || "$ACTION" == "refresh-generated-data" ]]; then
  # Broad overnight runs archive every revision-stale output. The targeted
  # point recovery action performs its own narrow archive in its submitter.
  archive_args=(--repo "$REPO" --execute)
  if [[ "$ACTION" == "overnight-all" ]]; then
    # Full preparation replaces the smoke-sized JSONLs. Current-SHA smoke
    # checkpoints are therefore incompatible even though the code is current.
    # Move them recoverably before constructing the mandatory full-data smoke
    # barrier; full-resume intentionally retains compatible full checkpoints.
    archive_args+=(--include-current-smoke)
  fi
  uv run python scripts/archive_stale_mechanistic_runs.py "${archive_args[@]}"
fi

case "$ACTION" in
  overnight-smoke)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile smoke
    ;;
  overnight-smoke-resume)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile smoke --reuse-prepared
    ;;
  overnight-all)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile all --confirm-full
    ;;
  overnight-all-resume)
    uv run python scripts/submit_neuronic_mechanistic_overnight.py \
      --repo "$REPO" --profile all --confirm-full --reuse-prepared
    ;;
  point-recovery)
    uv run python scripts/submit_neuronic_point_recovery.py --repo "$REPO"
    ;;
  base-search)
    uv run python scripts/submit_neuronic_base_search.py --repo "$REPO" --profile full
    ;;
  base-search-smoke)
    uv run python scripts/submit_neuronic_base_search.py --repo "$REPO" --profile smoke
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
  maci-stability)
    uv run python scripts/analyze_maci_head_stability.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/maci_stability.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/maci_stability \
      --seed 260318523 --resume
    ;;
  archive-stale)
    uv run python scripts/archive_stale_mechanistic_runs.py --repo "$REPO" --execute
    ;;
  refresh-generated-data)
    # A shared renderer source change invalidates both synthetic manifests.
    # Counting images are unchanged and can be rehashed in place; point/Waldo
    # images are regenerated because their spatial exclusion rule changed.
    uv run python scripts/generate_counting_data.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/counting_data.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/counting \
      --seed 260318523 --resume
    uv run python scripts/generate_point_search_data.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/point_search_data.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/point_search \
      --seed 260525427 --overwrite
    # Semantic-prior coverage and its group split are source-bound just like
    # the synthetic renderers. Rebuild this compact derived dataset whenever
    # the contrast preparer changes; MMMC itself remains reusable.
    uv run python scripts/prepare_vlmbias_signed_contrasts.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/vlmbias_contrasts.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts \
      --seed 260519250 --overwrite
    ;;
  atlas)
    uv run python scripts/render_mechanistic_head_reports.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/head_atlas.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/atlas \
      --seed 0 --resume
    ;;
  instrumentation|counting-behavior|counting-vap|counting-heads|counting-heads-repeat1|counting-heads-repeat2|counting-validation|waldo-behavior|point-centroids|search-heads|verification-heads|distractor-heads|point-ablation|maci-heads|maci-heads-aligned|maci-ablation|maci-confirm|maci-detector|maci-gated|vlmbias-heads|vlmbias-validation)
    if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
      echo "mode must be smoke or full" >&2
      exit 2
    fi
    if [[ "$MODE" == "full" && "$ACTION" =~ ^(counting-vap|counting-heads|counting-heads-repeat1|counting-heads-repeat2|point-centroids|search-heads|verification-heads|distractor-heads|maci-heads|maci-heads-aligned|vlmbias-heads)$ ]]; then
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
    echo "usage: $0 {base-search|base-search-smoke|point-recovery|archive-stale|refresh-generated-data|overnight-smoke|overnight-smoke-resume|overnight-all|overnight-all-resume|prepare-synthetic|download-mmmc|instrumentation|counting-behavior|counting-vap|counting-heads|counting-heads-repeat1|counting-heads-repeat2|general-importance|counting-controls|counting-validation|waldo-behavior|point-centroids|search-heads|verification-heads|distractor-heads|point-ablation|maci-heads|maci-heads-aligned|maci-stability|maci-ablation|maci-confirm|maci-detector|maci-gated|vlmbias-heads|vlmbias-validation|atlas} [smoke|full]"
    ;;
esac
