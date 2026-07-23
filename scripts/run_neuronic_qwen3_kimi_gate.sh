#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mode="${1:-calibrate}"

case "$mode" in
  prepare-captions)
    uv run python scripts/download_gaze_comics.py \
      --out segments/gaze_heads_qwen3_8b/data/eval_comics
    ;;
  calibrate)
    manifest="segments/gaze_heads_qwen3_8b/data/openai_comic_strips_manifest.json"
    if ! uv run python -c \
      'import json,sys; p=json.load(open(sys.argv[1])); assert len(p["rows"]) == 500; assert all(len(r.get("captions", [])) == 6 and all(r["captions"]) for r in p["rows"])' \
      "$manifest"; then
      echo "Caption metadata is missing. Run: bash scripts/run_neuronic_qwen3_kimi_gate.sh prepare-captions" >&2
      exit 2
    fi
    sbatch --parsable \
      --export="ALL,REPO=$PWD,SEGMENT_ROOT=segments/gaze_heads_qwen3_8b" \
      scripts/slurm_neuronic_qwen3_calibrate_kimi.sh
    ;;
  verify-calibration)
    cat segments/gaze_heads_qwen3_8b/runs/kimi_judge_calibration_60_numbered/calibration_results.json
    ;;
  diagnose-calibration)
    uv run python scripts/diagnose_kimi_calibration.py \
      --judgments segments/gaze_heads_qwen3_8b/runs/kimi_judge_calibration_60_numbered/kimi_judge/judgments.jsonl
    ;;
  smoke)
    python3 scripts/submit_neuronic_qwen3.py static-paper-control \
      --seeds 1 --shards 10 --shard-size 50 --top-ks 100 \
      --judge kimi --judge-only --judge-limit 120
    ;;
  verify-smoke)
    cat segments/gaze_heads_qwen3_8b/runs/static_paper_replication_seed42_top100_merged_0_500/kimi_judge_smoke_120_fast/aggregate_results.json
    ;;
  judge-full)
    python3 scripts/submit_neuronic_qwen3.py static-paper-control \
      --seeds 3 --shards 10 --shard-size 50 --top-ks 100 \
      --judge kimi --judge-only
    ;;
  aggregate-full)
    uv run python scripts/aggregate_qwen3_static_paper_judgments.py
    ;;
  *)
    echo "Usage: $0 {prepare-captions|calibrate|verify-calibration|diagnose-calibration|smoke|verify-smoke|judge-full|aggregate-full}" >&2
    exit 2
    ;;
esac
