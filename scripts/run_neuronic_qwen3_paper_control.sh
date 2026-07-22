#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mode="${1:-pilot}"
case "$mode" in
  pilot)
    python3 scripts/submit_neuronic_qwen3.py static-paper-control \
      --seeds 1 \
      --shards 1 \
      --shard-size 50 \
      --top-ks 100 \
      --judge none
    ;;
  verify-pilot)
    uv run python scripts/validate_qwen3_gaze_stage.py static \
      --run-dir segments/gaze_heads_qwen3_8b/runs/static_paper_replication_seed42_top100_merged_0_50
    ;;
  full)
    python3 scripts/submit_neuronic_qwen3.py static-paper-control \
      --seeds 3 \
      --shards 10 \
      --shard-size 50 \
      --top-ks 100 \
      --judge none
    ;;
  verify-full)
    for seed in 42 43 44; do
      uv run python scripts/validate_qwen3_gaze_stage.py static \
        --run-dir "segments/gaze_heads_qwen3_8b/runs/static_paper_replication_seed${seed}_top100_merged_0_500"
    done
    ;;
  dry-run-pilot)
    python3 scripts/submit_neuronic_qwen3.py --dry-run --skip-preflight \
      static-paper-control --seeds 1 --shards 1 --shard-size 50 --top-ks 100 --judge none
    ;;
  dry-run-full)
    python3 scripts/submit_neuronic_qwen3.py --dry-run --skip-preflight \
      static-paper-control --seeds 3 --shards 10 --shard-size 50 --top-ks 100 --judge none
    ;;
  *)
    echo "Usage: $0 {pilot|verify-pilot|full|verify-full|dry-run-pilot|dry-run-full}" >&2
    exit 2
    ;;
esac
