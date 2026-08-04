#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/n/fs/pvl-memory/at7979/VLMProject}"
ACTION="${1:-dry-run}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
cd "$REPO"

prepare_roi_inputs() {
  local bundle="segments/vlm_bias_attention/assets/vlmbias_high_bias_roi_masks_v1_runtime.tar.gz"
  local roi_root="segments/vlm_bias_attention/data/vlmbias_high_bias_roi_masks_v1"
  local manifest="$roi_root/accepted.jsonl"
  local dataset="$roi_root/dataset/vlmbias_high_bias_114.jsonl"
  local manifest_rows=0
  local dataset_rows=0
  local image_files=0
  local tight_masks=0
  local broad_masks=0

  if [[ -f "$manifest" ]]; then
    manifest_rows="$(wc -l < "$manifest" | tr -d ' ')"
  fi
  if [[ -f "$dataset" ]]; then
    dataset_rows="$(wc -l < "$dataset" | tr -d ' ')"
  fi
  if [[ -d "$roi_root/dataset/images" ]]; then
    image_files="$(find "$roi_root/dataset/images" -type f -name '*.png' | wc -l | tr -d ' ')"
  fi
  if [[ -d "$roi_root/masks_tight" ]]; then
    tight_masks="$(find "$roi_root/masks_tight" -type f -name '*.png' | wc -l | tr -d ' ')"
  fi
  if [[ -d "$roi_root/masks_broad" ]]; then
    broad_masks="$(find "$roi_root/masks_broad" -type f -name '*.png' | wc -l | tr -d ' ')"
  fi
  if [[ "$manifest_rows" == "114" && "$dataset_rows" == "114" && "$image_files" == "114" && "$tight_masks" == "114" && "$broad_masks" == "114" ]]; then
    echo "High-bias ROI runtime inputs already complete: 114 rows/images, 228 masks"
    return
  fi
  if [[ ! -f "$bundle" ]]; then
    echo "missing tracked high-bias ROI runtime bundle: $bundle" >&2
    exit 1
  fi
  mkdir -p "$roi_root"
  tar -xzf "$bundle" -C "$roi_root"
  manifest_rows="$(wc -l < "$manifest" | tr -d ' ')"
  dataset_rows="$(wc -l < "$dataset" | tr -d ' ')"
  image_files="$(find "$roi_root/dataset/images" -type f -name '*.png' | wc -l | tr -d ' ')"
  tight_masks="$(find "$roi_root/masks_tight" -type f -name '*.png' | wc -l | tr -d ' ')"
  broad_masks="$(find "$roi_root/masks_broad" -type f -name '*.png' | wc -l | tr -d ' ')"
  if [[ "$manifest_rows" != "114" || "$dataset_rows" != "114" || "$image_files" != "114" || "$tight_masks" != "114" || "$broad_masks" != "114" ]]; then
    echo "High-bias ROI runtime bundle extraction was incomplete" >&2
    exit 1
  fi
  echo "Prepared high-bias ROI runtime inputs: 114 rows/images, 228 masks"
}

case "$ACTION" in
  dry-run)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_high_bias_roi_attention.py full \
      --repo "$REPO" --max-parallel "$MAX_PARALLEL" --dry-run
    ;;
  submit)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_high_bias_roi_attention.py full \
      --repo "$REPO" --max-parallel "$MAX_PARALLEL"
    ;;
  smoke)
    prepare_roi_inputs
    uv run python scripts/submit_neuronic_qwen3_high_bias_roi_attention.py smoke \
      --repo "$REPO" --max-parallel "$MAX_PARALLEL"
    ;;
  prepare-inputs)
    prepare_roi_inputs
    ;;
  verify)
    for stage in smoke tune heads confirm; do
      report="segments/vlm_bias_attention/reports/qwen3_high_bias_roi_attention_v1/$stage/aggregate_results.json"
      jq '{stage, valid, n_conditions_expected, n_conditions_found, errors, warnings, selection, comparison, mechanics_gate}' "$report"
    done
    ;;
  *)
    echo "usage: $0 {prepare-inputs|dry-run|submit|smoke|verify}" >&2
    exit 2
    ;;
esac
