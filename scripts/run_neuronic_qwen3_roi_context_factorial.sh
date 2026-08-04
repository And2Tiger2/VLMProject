#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"
ACTION="${1:-dry-run}"
cd "$REPO"

bash scripts/run_neuronic_qwen3_high_bias_roi_attention.sh prepare-inputs

case "$ACTION" in
  dry-run)
    uv run python scripts/submit_neuronic_qwen3_roi_context_factorial.py \
      --repo "$REPO" --dry-run
    ;;
  submit)
    uv run python scripts/submit_neuronic_qwen3_roi_context_factorial.py \
      --repo "$REPO"
    ;;
  verify)
    jq '{valid, n_conditions_expected, n_conditions_found, errors, warnings, comparisons}' \
      segments/vlm_bias_attention/reports/qwen3_roi_context_factorial_v1/aggregate_results.json
    cat segments/vlm_bias_attention/reports/qwen3_roi_context_factorial_v1/report.md
    ;;
  *)
    echo "usage: $0 {dry-run|submit|verify}" >&2
    exit 2
    ;;
esac
