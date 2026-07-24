#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"

case "${1:-}" in
  full|overnight|smoke|controller|heads|confirm|robustness)
    python3 scripts/submit_neuronic_qwen3_attention_methods.py "${1}"
    ;;
  dry-run)
    python3 scripts/submit_neuronic_qwen3_attention_methods.py \
      --dry-run --skip-preflight overnight
    ;;
  verify)
    for stage in smoke controller heads confirm robustness; do
      report="segments/gaze_heads_qwen3_8b/reports/attention_methods_v1/$stage/aggregate_results.json"
      if [[ -f "$report" ]]; then
        echo "===== $stage ====="
        uv run python -m json.tool "$report"
      fi
    done
    ;;
  recover)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: bash scripts/run_neuronic_qwen3_attention_methods.sh recover FINAL_JOB_ID" >&2
      exit 2
    fi
    python3 scripts/recover_neuronic_qwen3_attention_methods.py "$2"
    ;;
  *)
    echo "Usage: bash scripts/run_neuronic_qwen3_attention_methods.sh {full|overnight|smoke|controller|heads|confirm|robustness|dry-run|verify|recover FINAL_JOB_ID}" >&2
    exit 2
    ;;
esac
