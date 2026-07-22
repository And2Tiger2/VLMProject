#!/usr/bin/env bash
#SBATCH --job-name=q3-kimi-judge
#SBATCH --output=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.out
#SBATCH --error=segments/gaze_heads_qwen3_8b/runs/slurm/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
KIMI_VENV="${KIMI_VENV:-$CACHE_ROOT/kimi-judge-venv}"
SEGMENT_ROOT="${SEGMENT_ROOT:-segments/gaze_heads_qwen3_8b}"
BASE_SEED="${BASE_SEED:-42}"
N_SHARDS="${N_SHARDS:-10}"
SHARD_SIZE="${SHARD_SIZE:-50}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"
KIMI_MODEL="${KIMI_MODEL:-moonshotai/Kimi-VL-A3B-Instruct}"
KIMI_REVISION="${KIMI_REVISION:-cc6452511d00c99f3b3bed213e96ab7802c415c8}"
IFS=: read -r -a TOP_KS_ARRAY <<< "${TOP_KS_COLON:-1:10:50:100}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED_INDEX="$((TASK_ID / ${#TOP_KS_ARRAY[@]}))"
TOP_K_INDEX="$((TASK_ID % ${#TOP_KS_ARRAY[@]}))"
SEED="$((BASE_SEED + SEED_INDEX))"
TOP_K="${TOP_KS_ARRAY[$TOP_K_INDEX]}"
RUN_DIR="$SEGMENT_ROOT/runs/static_narration_seed${SEED}_top${TOP_K}_merged_0_$((N_SHARDS * SHARD_SIZE))"
JUDGE_DIR="$RUN_DIR/kimi_judge"
if ((JUDGE_LIMIT > 0)); then
  JUDGE_DIR="${JUDGE_DIR}_smoke_${JUDGE_LIMIT}"
fi

if [[ ! -x "$KIMI_VENV/bin/python" ]]; then
  echo "Missing Kimi judge environment: $KIMI_VENV" >&2
  echo "Run: bash scripts/setup_neuronic_kimi_judge.sh" >&2
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

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=$SEED top_k=$TOP_K judge=kimi limit=$JUDGE_LIMIT"
"$KIMI_VENV/bin/python" scripts/check_neuronic_gpu.py \
  --min-memory-gb "${KIMI_MIN_GPU_MEMORY_GB:-40}"

args=(
  --generations "$RUN_DIR/generations.jsonl"
  --out-dir "$JUDGE_DIR"
  --model-id "$KIMI_MODEL"
  --revision "$KIMI_REVISION"
  --seed "$SEED"
  --resume
)
if ((JUDGE_LIMIT > 0)); then
  args+=(--limit "$JUDGE_LIMIT")
fi
srun "$KIMI_VENV/bin/python" scripts/judge_kimi_gaze_generations.py "${args[@]}"
