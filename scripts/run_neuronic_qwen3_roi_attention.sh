#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"
ACTION="${1:-dry-run}"
cd "$REPO"

prepare_roi_inputs() {
  local bundle="segments/vlm_bias_attention/assets/vlmbias_roi_masks_v1_runtime.tar.gz"
  local roi_root="segments/vlm_bias_attention/data/vlmbias_roi_masks_v1"
  local manifest="$roi_root/accepted.jsonl"
  local manifest_rows=0
  local mask_files=0

  if [[ -f "$manifest" ]]; then
    manifest_rows="$(wc -l < "$manifest" | tr -d ' ')"
  fi
  if [[ -d "$roi_root/masks" ]]; then
    mask_files="$(find "$roi_root/masks" -type f -name '*.png' | wc -l | tr -d ' ')"
  fi
  if [[ "$manifest_rows" == "141" && "$mask_files" == "141" ]]; then
    echo "ROI runtime inputs already complete: 141 masks"
    return
  fi
  if [[ ! -f "$bundle" ]]; then
    echo "missing tracked ROI runtime bundle: $bundle" >&2
    exit 1
  fi
  mkdir -p "$roi_root"
  tar -xzf "$bundle" -C "$roi_root"
  manifest_rows="$(wc -l < "$manifest" | tr -d ' ')"
  mask_files="$(find "$roi_root/masks" -type f -name '*.png' | wc -l | tr -d ' ')"
  if [[ "$manifest_rows" != "141" || "$mask_files" != "141" ]]; then
    echo "ROI runtime bundle extraction was incomplete" >&2
    exit 1
  fi
  echo "Prepared ROI runtime inputs: 141 masks"
}

case "$ACTION" in
  dry-run)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_roi_attention.py full --dry-run
    ;;
  submit)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_roi_attention.py full
    ;;
  smoke)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_roi_attention.py smoke
    ;;
  prepare-inputs)
    prepare_roi_inputs
    ;;
  verify)
    for stage in smoke tune heads confirm; do
      report="segments/vlm_bias_attention/reports/qwen3_roi_attention_v1/$stage/aggregate_results.json"
      jq '{stage, valid, n_conditions_expected, n_conditions_found, errors, warnings, selection, comparison, mechanics_gate}' "$report"
    done
    ;;
  *)
    echo "usage: $0 {prepare-inputs|dry-run|submit|smoke|verify}" >&2
    exit 2
    ;;
esac
