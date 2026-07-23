#!/usr/bin/env bash
#SBATCH --job-name=q3-kimi-cal
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
KIMI_VENV="${KIMI_VENV:-$CACHE_ROOT/kimi-judge-venv}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
KIMI_MODEL="${KIMI_MODEL:-moonshotai/Kimi-VL-A3B-Instruct}"
KIMI_REVISION="${KIMI_REVISION:-cc6452511d00c99f3b3bed213e96ab7802c415c8}"
MANIFEST="${MANIFEST:-$SEGMENT_ROOT/data/openai_comic_strips_manifest.json}"
OUT_DIR="${OUT_DIR:-$SEGMENT_ROOT/runs/kimi_judge_calibration_60_numbered}"

if [[ ! -x "$KIMI_VENV/bin/python" ]]; then
  echo "Missing Kimi judge environment: $KIMI_VENV" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$REPO/$SEGMENT_ROOT/runs/slurm"
cd "$REPO"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} kimi_caption_calibration=60"
"$KIMI_VENV/bin/python" scripts/check_neuronic_gpu.py \
  --min-memory-gb "${KIMI_MIN_GPU_MEMORY_GB:-40}"
srun "$KIMI_VENV/bin/python" scripts/calibrate_kimi_gaze_judge.py \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --limit 60 \
  --batch-size 4 \
  --model-id "$KIMI_MODEL" \
  --revision "$KIMI_REVISION" \
  --resume
