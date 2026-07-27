#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

ROOT="segments/gaze_heads_qwen3_8b"
MODE="${1:-publish}"
VERIFY_LOG="${TMPDIR:-/tmp}/qwen3_results_verify_$(date -u +%Y%m%dT%H%M%SZ).log"
INDEX="$ROOT/reports/qwen3_results_index.json"
VERIFIED_FILES=()
OPTIONAL_FAMILIES=()

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

ok_json() {
  local report="$1"
  require_file "$report"
  jq -e \
    '(.valid == true) and (((.errors // []) | length) == 0)' \
    "$report" >/dev/null
  VERIFIED_FILES+=("$report")
  jq \
    '{stage, valid, n_conditions_expected, n_conditions_found, errors, warnings}' \
    "$report"
}

run_verification() {
  require_command git
  require_command jq
  require_command uv

  mkdir -p "$ROOT/reports"
  exec > >(tee "$VERIFY_LOG") 2>&1

  echo "Repository: $REPO"
  echo "Commit: $(git rev-parse HEAD)"
  echo "Verification started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "===== DATASETS ====="
  uv run python scripts/validate_qwen3_gaze_stage.py datasets
  ok_json "$ROOT/reports/dataset_validation.json"

  echo "===== THREE-SEED GAZE DISCOVERY ====="
  for seed in 42 43 44; do
    local run="$ROOT/runs/gaze_discovery_seed${seed}_merged"
    uv run python scripts/validate_qwen3_gaze_stage.py discovery \
      --run-dir "$run" \
      --min-samples 500
    ok_json "$run/validation.json"
  done

  echo "===== KIMI JUDGE CALIBRATION ====="
  local calibration="$ROOT/runs/kimi_judge_calibration_60_numbered/calibration_results.json"
  ok_json "$calibration"
  jq \
    '{valid, n_rows, overall_accuracy, per_panel_accuracy, parse_failure_rate, errors}' \
    "$calibration"

  echo "===== STATIC PAPER-CONTROL GENERATIONS ====="
  bash scripts/run_neuronic_qwen3_paper_control.sh verify-full
  for seed in 42 43 44; do
    local run="$ROOT/runs/static_paper_replication_seed${seed}_top100_merged_0_500"
    ok_json "$run/validation.json"
  done

  echo "===== STATIC PAPER-CONTROL KIMI AGGREGATE ====="
  uv run python scripts/aggregate_qwen3_static_paper_judgments.py \
    > "${TMPDIR:-/tmp}/qwen3_static_paper_aggregate.json"
  local static_report="$ROOT/reports/static_paper_kimi_full/aggregate_results.json"
  ok_json "$static_report"
  jq \
    '{valid, aggregate, repeated_gaze_judge_diagnostics, errors, warnings}' \
    "$static_report"

  echo "===== STATIC CONTROL DISTRIBUTION ====="
  uv run python scripts/aggregate_qwen3_control_distribution.py \
    > "${TMPDIR:-/tmp}/qwen3_control_distribution.json"
  local control_report="$ROOT/reports/control_distribution_100/aggregate_results.json"
  ok_json "$control_report"
  jq '{valid, gaze, groups, errors, warnings}' "$control_report"

  echo "===== ORIGINAL FULL ALPHA SWEEP ====="
  local benchmark="$ROOT/runs/benchmark_attention_full"
  if [[ -f "$benchmark/experiment_config.json" ]]; then
    local -a benchmark_seeds=()
    local -a benchmark_topks=()
    local -a benchmark_alphas=()
    mapfile -t benchmark_seeds < <(
      jq -r '.seeds[]' "$benchmark/experiment_config.json"
    )
    mapfile -t benchmark_topks < <(
      jq -r '.top_ks[]' "$benchmark/experiment_config.json"
    )
    mapfile -t benchmark_alphas < <(
      jq -r '.alphas[]' "$benchmark/experiment_config.json"
    )
    ((${#benchmark_seeds[@]} > 0)) || die "benchmark config contains no seeds"
    ((${#benchmark_topks[@]} > 0)) || die "benchmark config contains no top-k values"
    ((${#benchmark_alphas[@]} > 0)) || die "benchmark config contains no alphas"

    uv run python scripts/aggregate_vlmbias_gaze_attention_sweep.py \
      --out-dir "$benchmark" \
      --expected-seeds "${benchmark_seeds[@]}" \
      --expected-top-ks "${benchmark_topks[@]}" \
      --expected-alphas "${benchmark_alphas[@]}" \
      --strict
    uv run python scripts/validate_qwen3_gaze_stage.py benchmark \
      --run-dir "$benchmark"

    local expected_summaries
    expected_summaries=$((
      2 * ${#benchmark_seeds[@]} *
      (1 + ${#benchmark_topks[@]} * ${#benchmark_alphas[@]})
    ))
    jq -e --argjson expected "$expected_summaries" \
      '(.valid == true)
       and (.n_condition_summaries == $expected)
       and (((.errors // []) | length) == 0)' \
      "$benchmark/validation.json" >/dev/null
    VERIFIED_FILES+=("$benchmark/validation.json")
    OPTIONAL_FAMILIES+=("benchmark_attention_full")
    jq '{valid, n_condition_summaries, errors}' "$benchmark/validation.json"
  else
    echo "NOT PRESENT: benchmark_attention_full; skipping it"
  fi

  echo "===== ATTENTION METHODS V1: INFORMATIVE STAGES ONLY ====="
  local v1_experiment="$ROOT/experiments/attention_methods_v1"
  local v1_run="$ROOT/runs/attention_methods_v1"
  local v1_report="$ROOT/reports/attention_methods_v1"
  require_file "$v1_experiment/controller_manifest.json"
  require_file "$v1_experiment/heads_manifest.json"
  require_file "$v1_experiment/robustness_manifest.json"
  for stage in controller heads robustness; do
    uv run python scripts/aggregate_qwen3_attention_methods.py "$stage" \
      --manifest "$v1_experiment/${stage}_manifest.json" \
      --run-root "$v1_run" \
      --report-root "$v1_report" \
      > "${TMPDIR:-/tmp}/qwen3_v1_${stage}.json"
    ok_json "$v1_report/$stage/aggregate_results.json"
  done
  echo "SKIPPED BY DESIGN: v1 smoke"
  echo "SKIPPED AS INVALID: v1 confirm used unequal development/held-out datasets"

  echo "===== GAZE SPECIFICITY V2 ====="
  local v2_experiment="$ROOT/experiments/gaze_specificity_v2"
  local v2_run="$ROOT/runs/gaze_specificity_v2"
  local v2_report="$ROOT/reports/gaze_specificity_v2"
  if [[ -d "$v2_report" || -f "$v2_experiment/repair_manifest.json" ]]; then
    for stage in repair controls tune final robustness; do
      require_file "$v2_experiment/${stage}_manifest.json"
      uv run python scripts/aggregate_qwen3_gaze_specificity.py "$stage" \
        --manifest "$v2_experiment/${stage}_manifest.json" \
        --run-root "$v2_run" \
        --report-root "$v2_report" \
        > "${TMPDIR:-/tmp}/qwen3_v2_${stage}.json"
      ok_json "$v2_report/$stage/aggregate_results.json"
    done
    OPTIONAL_FAMILIES+=("gaze_specificity_v2")
  else
    echo "NOT PRESENT: gaze_specificity_v2; skipping it"
  fi

  write_results_index
  echo "===== VERIFICATION COMPLETE ====="
  echo "Verification log: $VERIFY_LOG"
  echo "Results index: $INDEX"
}

write_results_index() {
  local verified_json
  local optional_json
  verified_json=$(
    printf '%s\n' "${VERIFIED_FILES[@]}" |
      jq -R . |
      jq -s .
  )
  optional_json=$(
    if ((${#OPTIONAL_FAMILIES[@]})); then
      printf '%s\n' "${OPTIONAL_FAMILIES[@]}" | jq -R . | jq -s .
    else
      printf '[]\n'
    fi
  )
  jq -n \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg source_commit "$(git rev-parse HEAD)" \
    --argjson verified_reports "$verified_json" \
    --argjson optional_families_present "$optional_json" \
    '{
      schema_version: 1,
      experiment_family: "Qwen/Qwen3-VL-8B-Instruct gaze heads",
      generated_at_utc: $generated_at_utc,
      source_commit: $source_commit,
      verified_reports: $verified_reports,
      optional_families_present: $optional_families_present,
      included_v1_stages: ["controller", "heads", "robustness"],
      exclusions: {
        smoke_runs: "Mechanics checks are not scientific results.",
        failed_jobs: "Partial and failed Slurm outputs are excluded.",
        v1_confirm: "Invalid comparison: baseline used held-out data while treatments inherited development paths.",
        raw_jsonl: "Raw generations and judgments remain off GitHub; compact aggregates contain graph-ready metrics.",
        slurm_logs: "Scheduler stdout/stderr is transient operational metadata."
      }
    }' > "$INDEX"
}

stage_file() {
  if [[ -f "$1" ]]; then
    git add -- "$1"
  fi
}

stage_force_file() {
  if [[ -f "$1" ]]; then
    git add -f -- "$1"
  fi
}

stage_report_dir() {
  local directory="$1"
  stage_file "$directory/aggregate_results.json"
  stage_file "$directory/report.md"
  stage_file "$directory/selection.json"
}

stage_static_run() {
  local run="$1"
  stage_force_file "$run/experiment_config.json"
  stage_force_file "$run/summary.json"
  stage_force_file "$run/validation.json"
  stage_force_file "$run/kimi_judge/aggregate_results.json"
  stage_force_file "$run/kimi_judge/judgment_config.json"
}

stage_results() {
  git diff --cached --quiet ||
    die "Git index is not empty; commit or unstage existing changes first"

  require_file "$INDEX"
  stage_file "$INDEX"
  stage_file "$ROOT/reports/dataset_validation.json"
  stage_report_dir "$ROOT/reports/static_paper_kimi_full"
  stage_report_dir "$ROOT/reports/control_distribution_100"

  for stage in controller heads robustness; do
    stage_report_dir "$ROOT/reports/attention_methods_v1/$stage"
  done
  for file in \
    methodology.json \
    split_manifest.json \
    controller_manifest.json \
    heads_manifest.json \
    robustness_manifest.json \
    last_submission.json
  do
    stage_file "$ROOT/experiments/attention_methods_v1/$file"
  done

  if [[ -d "$ROOT/reports/gaze_specificity_v2" ]]; then
    for stage in repair controls tune final robustness; do
      stage_report_dir "$ROOT/reports/gaze_specificity_v2/$stage"
    done
    for file in \
      methodology.json \
      split_manifest.json \
      repair_manifest.json \
      controls_manifest.json \
      tune_manifest.json \
      final_manifest.json \
      robustness_manifest.json \
      last_submission.json
    do
      stage_file "$ROOT/experiments/gaze_specificity_v2/$file"
    done
  fi

  for seed in 42 43 44; do
    local run="$ROOT/runs/gaze_discovery_seed${seed}_merged"
    stage_force_file "$run/gaze_head_ranking.json"
    stage_force_file "$run/summary.json"
    stage_force_file "$run/validation.json"
    for file in "$run"/*.npy "$run"/top*_stability.json "$run"/top*_stability.tsv; do
      stage_force_file "$file"
    done
  done

  local calibration="$ROOT/runs/kimi_judge_calibration_60_numbered"
  stage_force_file "$calibration/calibration_results.json"
  stage_force_file "$calibration/kimi_judge/aggregate_results.json"
  stage_force_file "$calibration/kimi_judge/judgment_config.json"

  for seed in 42 43 44; do
    stage_static_run \
      "$ROOT/runs/static_paper_replication_seed${seed}_top100_merged_0_500"
  done

  for seed in {45..54}; do
    stage_static_run \
      "$ROOT/runs/static_control_distribution_paper_seed${seed}_top100_merged_0_100"
    stage_static_run \
      "$ROOT/runs/static_control_distribution_layer_matched_random_seed${seed}_top100_merged_0_100"
  done
  stage_static_run \
    "$ROOT/runs/static_control_distribution_layer_matched_low_seed42_top100_merged_0_100"

  local benchmark="$ROOT/runs/benchmark_attention_full"
  if [[ -d "$benchmark" ]]; then
    stage_force_file "$benchmark/experiment_config.json"
    stage_force_file "$benchmark/validation.json"
    for file in \
      "$benchmark"/*summary_by_seed.tsv \
      "$benchmark"/*summary_aggregate.tsv
    do
      stage_force_file "$file"
    done
  fi

  if git diff --cached --quiet; then
    echo "No new graph-ready result changes were found."
    return
  fi
  audit_staged_results
}

audit_staged_results() {
  git diff --cached --check

  local bad
  bad=$(
    git diff --cached --name-only |
      grep -E \
        '(^|/)(smoke|slurm)(/|$)|generations\.jsonl$|judgments\.jsonl$|\.out$|\.err$' \
      || true
  )
  if [[ -n "$bad" ]]; then
    echo "Excluded artifacts reached the index:" >&2
    echo "$bad" >&2
    die "refusing to publish excluded artifacts"
  fi

  local file
  local bytes
  while IFS= read -r file; do
    bytes=$(git cat-file -s ":$file")
    if ((bytes >= 90000000)); then
      die "staged file is too large for a normal GitHub commit: $bytes bytes $file"
    fi
  done < <(git diff --cached --name-only)

  echo "===== STAGED GRAPH-READY RESULTS ====="
  git diff --cached --stat
  git diff --cached --name-only | sort
}

publish_results() {
  stage_results
  if git diff --cached --quiet; then
    echo "No result changes to commit."
  else
    git commit -m "Add verified Qwen3 experiment results"
  fi
  git push origin HEAD
  echo "Published commit: $(git rev-parse --short HEAD)"
}

case "$MODE" in
  verify)
    run_verification
    ;;
  stage)
    run_verification
    stage_results
    ;;
  publish)
    run_verification
    publish_results
    ;;
  *)
    echo "Usage: bash scripts/publish_neuronic_qwen3_results.sh [verify|stage|publish]" >&2
    exit 2
    ;;
esac
