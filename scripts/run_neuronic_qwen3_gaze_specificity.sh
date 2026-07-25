#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"

case "${1:-}" in
  full|overnight|repair|controls|tune|final|robustness)
    python3 scripts/submit_neuronic_qwen3_gaze_specificity.py "${1}"
    ;;
  dry-run)
    python3 scripts/submit_neuronic_qwen3_gaze_specificity.py \
      --dry-run --skip-preflight overnight
    ;;
  verify)
    for stage in repair controls tune final robustness; do
      report="segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2/$stage/aggregate_results.json"
      if [[ -f "$report" ]]; then
        echo "===== $stage ====="
        cat "segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2/$stage/report.md"
      fi
    done
    ;;
  *)
    echo "Usage: bash scripts/run_neuronic_qwen3_gaze_specificity.sh {full|overnight|repair|controls|tune|final|robustness|dry-run|verify}" >&2
    exit 2
    ;;
esac
