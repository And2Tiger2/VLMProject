#!/usr/bin/env bash
#SBATCH --job-name=q3-mech-agg
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SOURCE_TASK="${SOURCE_TASK:?SOURCE_TASK is required}"
JOB_STARTED_AT="$(date +%s)"
cd "$REPO"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  echo "refusing mechanistic aggregation: tracked worktree changes do not match HEAD" >&2
  exit 2
fi
export UV_NO_SYNC=1
export UV_FROZEN=1
case "$SOURCE_TASK" in
  counting-vap) ROOT=counting_vap ;;
  counting-heads) ROOT=counting_head_scan ;;
  counting-heads-repeat1) ROOT=counting_head_scan_repeat1 ;;
  counting-heads-repeat2) ROOT=counting_head_scan_repeat2 ;;
  point-centroids) ROOT=point_attention_centroids ;;
  search-heads) ROOT=search_head_scan ;;
  verification-heads) ROOT=verification_head_scan ;;
  distractor-heads) ROOT=distractor_head_scan ;;
  maci-heads) ROOT=maci_head_scan ;;
  maci-heads-aligned) ROOT=maci_head_scan_aligned ;;
  vlmbias-heads) ROOT=vlmbias_signed_head_scan ;;
  *) echo "unsupported aggregate task: $SOURCE_TASK" >&2; exit 2 ;;
esac
uv run python scripts/aggregate_mechanistic_shards.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/shard_aggregation.json \
  --input-root "segments/mechanistic_heads_qwen3_8b/runs/$ROOT/full" \
  --output-dir "segments/mechanistic_heads_qwen3_8b/runs/$ROOT/full" \
  --task "$SOURCE_TASK" --seed 0 --resume
uv run python scripts/validate_mechanistic_run.py \
  --repo "$REPO" \
  --run-dir "segments/mechanistic_heads_qwen3_8b/runs/$ROOT/full" \
  --newer-than-epoch "$JOB_STARTED_AT"
