#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"

case "${1:-}" in
  submit)
    python3 scripts/submit_neuronic_qwen3.py control-distribution
    ;;
  dry-run)
    python3 scripts/submit_neuronic_qwen3.py \
      --dry-run --skip-preflight control-distribution
    ;;
  verify)
    cat segments/gaze_heads_qwen3_8b/reports/control_distribution_100/aggregate_results.json
    ;;
  *)
    echo "Usage: bash scripts/run_neuronic_qwen3_control_distribution.sh {submit|dry-run|verify}" >&2
    exit 2
    ;;
esac
