#!/usr/bin/env bash
#SBATCH --job-name=q3-ctrl-judge
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
KIMI_VENV="${KIMI_VENV:-$CACHE_ROOT/kimi-judge-venv}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
SHARD_SIZE="${SHARD_SIZE:-50}"
N_SHARDS="${N_SHARDS:-2}"
TOP_K="${TOP_K:-100}"
KIMI_MODEL="${KIMI_MODEL:-moonshotai/Kimi-VL-A3B-Instruct}"
KIMI_REVISION="${KIMI_REVISION:-cc6452511d00c99f3b3bed213e96ab7802c415c8}"
IFS=: read -r -a SPECS <<< "${CONTROL_SPECS_COLON:?CONTROL_SPECS_COLON is required}"
SPEC="${SPECS[${SLURM_ARRAY_TASK_ID:-0}]}"
MODE="${SPEC%@*}"
SEED="${SPEC#*@}"
RUN_DIR="$SEGMENT_ROOT/runs/static_control_distribution_${MODE}_seed${SEED}_top${TOP_K}_merged_0_$((N_SHARDS * SHARD_SIZE))"

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

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} mode=$MODE head_seed=$SEED judge=kimi"
"$KIMI_VENV/bin/python" scripts/check_neuronic_gpu.py \
  --min-memory-gb "${KIMI_MIN_GPU_MEMORY_GB:-40}"
srun "$KIMI_VENV/bin/python" scripts/judge_kimi_gaze_generations.py \
  --generations "$RUN_DIR/generations.jsonl" \
  --out-dir "$RUN_DIR/kimi_judge" \
  --model-id "$KIMI_MODEL" \
  --revision "$KIMI_REVISION" \
  --seed "$SEED" \
  --resume
