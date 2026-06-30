#!/usr/bin/env bash
#SBATCH --job-name=aggregate-gaze-highalpha
#SBATCH --output=segments/vlm_bias_attention/runs/slurm/%x-%j.out
#SBATCH --error=segments/vlm_bias_attention/runs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

export PROJ="${PROJ:-/n/fs/pvl-memory/at7979}"
export REPO="${REPO:-$PROJ/VLMProject}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJ/uv_cache}"
export TMPDIR="${TMPDIR:-$PROJ/tmp}"
export PATH="$PROJ/bin:$PATH"

mkdir -p "$REPO/segments/vlm_bias_attention/runs/slurm" "$UV_CACHE_DIR" "$TMPDIR"
cd "$REPO"

OUT_DIR="${OUT_DIR:-segments/vlm_bias_attention/runs/vlmbias_gaze_attention_high_alpha_sweep}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
TOP_KS="${TOP_KS:-10 20}"
ALPHAS="${ALPHAS:-10 20 30 50 100}"
STRICT="${STRICT:-1}"

read -r -a seed_array <<< "$SEEDS"
read -r -a top_k_array <<< "$TOP_KS"
read -r -a alpha_array <<< "$ALPHAS"

cmd=(
  uv run python scripts/aggregate_vlmbias_gaze_attention_sweep.py
  --out-dir "$OUT_DIR"
  --expected-seeds "${seed_array[@]}"
  --expected-top-ks "${top_k_array[@]}"
  --expected-alphas "${alpha_array[@]}"
)

if [[ "$STRICT" == "1" ]]; then
  cmd+=(--strict)
fi

echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Output directory: $OUT_DIR"
echo "Command: ${cmd[*]}"

srun "${cmd[@]}"
