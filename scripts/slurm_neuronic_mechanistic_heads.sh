#!/usr/bin/env bash
#SBATCH --job-name=q3-mech
#SBATCH --output=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.out
#SBATCH --error=segments/mechanistic_heads_qwen3_8b/runs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache/vlmproject}"
TASK="${TASK:?TASK is required}"
MODE="${MODE:-smoke}"
SEED="${SEED:-260318523}"
JOB_STARTED_AT="$(date +%s)"

mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" \
  "$REPO/segments/mechanistic_heads_qwen3_8b/runs/slurm"
cd "$REPO"
if ! git diff --quiet -- . || ! git diff --cached --quiet -- .; then
  echo "refusing mechanistic run: tracked worktree changes do not match HEAD" >&2
  exit 2
fi
export PATH="$CACHE_ROOT/bin:$PATH"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
# The submission wrapper performs the sole locked dependency sync before any
# job is enqueued. Never let concurrent GPU jobs alter the shared environment.
export UV_NO_SYNC=1
export UV_FROZEN=1
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Long multimodal prefills have variable allocation sizes. Expandable CUDA
# segments reduce allocator fragmentation after the dense projected-head
# capture was replaced by lazy per-head projection.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

uv run python scripts/check_neuronic_gpu.py --min-memory-gb "${MIN_GPU_MEMORY_GB:-20}"

smoke_args=()
if [[ "$MODE" == "smoke" ]]; then
  smoke_args+=(--smoke)
elif [[ "$MODE" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi
OUT_SUFFIX="$MODE"
POINT_CHECKPOINT_ROOT="segments/mechanistic_heads_qwen3_8b/checkpoints"
if [[ "$MODE" == "smoke" ]]; then
  POINT_CHECKPOINT_ROOT="$POINT_CHECKPOINT_ROOT/smoke"
fi
POINT_ANSWER_CHECKPOINT="$POINT_CHECKPOINT_ROOT/point-answer-lora"
layer_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  layer_args+=(--layers "$SLURM_ARRAY_TASK_ID")
  OUT_SUFFIX="$MODE/layer-$SLURM_ARRAY_TASK_ID"
fi

case "$TASK" in
  instrumentation)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/reports/instrumentation"
    uv run python scripts/validate_mechanistic_instrumentation.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/instrumentation_smoke.json \
      --output-dir segments/mechanistic_heads_qwen3_8b/reports/instrumentation \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  counting-behavior)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/counting_behavior/$OUT_SUFFIX"
    uv run python scripts/run_counting_behavior.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/counting_behavior.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/counting_behavior/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  counting-vap)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/counting_vap/$OUT_SUFFIX"
    uv run python scripts/run_counting_vap.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/counting_vap.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/counting_vap/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  counting-heads|counting-heads-repeat1|counting-heads-repeat2)
    COUNT_SCAN_ROOT="counting_head_scan"
    COUNT_SCAN_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/counting_head_scan.json"
    if [[ "$TASK" == "counting-heads-repeat1" ]]; then
      COUNT_SCAN_ROOT="counting_head_scan_repeat1"
      COUNT_SCAN_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/counting_head_scan_repeat1.json"
    elif [[ "$TASK" == "counting-heads-repeat2" ]]; then
      COUNT_SCAN_ROOT="counting_head_scan_repeat2"
      COUNT_SCAN_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/counting_head_scan_repeat2.json"
    fi
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/$COUNT_SCAN_ROOT/$OUT_SUFFIX"
    uv run python scripts/run_counting_head_scan.py \
      --config "$COUNT_SCAN_CONFIG" \
      --output-dir "$VERIFY_DIR" \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  counting-validation)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/counting_validation/$OUT_SUFFIX"
    COUNT_VALIDATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/counting_validation.json"
    if [[ "$MODE" == "smoke" ]]; then
      COUNT_VALIDATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_counting_validation.json"
    fi
    uv run python scripts/run_counting_head_validation.py \
      --config "$COUNT_VALIDATION_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/counting_validation/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  point-train-all)
    point_conditions=(direct_answer direct_length_matched point_answer shuffled_point_answer)
    point_slugs=(direct-answer direct-length-matched point-answer shuffled-point-answer)
    point_index="${SLURM_ARRAY_TASK_ID:?point-train-all requires array index 0-3}"
    if (( point_index < 0 || point_index >= ${#point_conditions[@]} )); then
      echo "point training array index must be 0-3" >&2
      exit 2
    fi
    VERIFY_DIR="$POINT_CHECKPOINT_ROOT/${point_slugs[$point_index]}-lora"
    uv run python scripts/train_point_search.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/point_search_lora.json \
      --output-dir "$VERIFY_DIR" \
      --condition "${point_conditions[$point_index]}" \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  point-behavior-all)
    behavior_conditions=(base direct_answer direct_length_matched point_answer shuffled_point_answer)
    behavior_slugs=(base direct-answer direct-length-matched point-answer shuffled-point-answer)
    behavior_index="${SLURM_ARRAY_TASK_ID:?point-behavior-all requires array index 0-4}"
    if (( behavior_index < 0 || behavior_index >= ${#behavior_conditions[@]} )); then
      echo "point behavior array index must be 0-4" >&2
      exit 2
    fi
    checkpoint_args=()
    if [[ "${behavior_conditions[$behavior_index]}" != "base" ]]; then
      checkpoint_args+=(--checkpoint "$POINT_CHECKPOINT_ROOT/${behavior_slugs[$behavior_index]}-lora")
    fi
    behavior_root="segments/mechanistic_heads_qwen3_8b/runs/point_behavior"
    if [[ "$MODE" == "smoke" ]]; then
      behavior_root="segments/mechanistic_heads_qwen3_8b/runs/point_behavior_smoke"
    fi
    VERIFY_DIR="$behavior_root/${behavior_conditions[$behavior_index]}"
    uv run python scripts/evaluate_point_search.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/point_search_behavior.json \
      --output-dir "$behavior_root/${behavior_conditions[$behavior_index]}" \
      --condition "${behavior_conditions[$behavior_index]}" \
      --seed "$SEED" --device-map cuda --resume "${checkpoint_args[@]}" "${smoke_args[@]}"
    ;;
  waldo-behavior)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/waldo_behavior/$OUT_SUFFIX"
    uv run python scripts/run_waldo_behavior.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/waldo_behavior.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/waldo_behavior/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${smoke_args[@]}"
    ;;
  search-heads)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/search_head_scan/$OUT_SUFFIX"
    uv run python scripts/run_search_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/search_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/search_head_scan/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  point-centroids)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/point_attention_centroids/$OUT_SUFFIX"
    uv run python scripts/run_point_attention_centroids.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/point_attention_centroids.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/point_attention_centroids/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  verification-heads)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/verification_head_scan/$OUT_SUFFIX"
    uv run python scripts/run_verification_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/verification_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/verification_head_scan/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  distractor-heads)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/distractor_head_scan/$OUT_SUFFIX"
    uv run python scripts/run_distractor_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/distractor_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/distractor_head_scan/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  point-ablation)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/point_head_ablation/$OUT_SUFFIX"
    POINT_ABLATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/point_head_ablation.json"
    if [[ "$MODE" == "smoke" ]]; then
      POINT_ABLATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_point_head_ablation.json"
    fi
    uv run python scripts/run_point_head_ablation.py \
      --config "$POINT_ABLATION_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/point_head_ablation/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --checkpoint "$POINT_ANSWER_CHECKPOINT" \
      --resume "${smoke_args[@]}"
    ;;
  maci-heads)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_head_scan/$OUT_SUFFIX"
    uv run python scripts/run_maci_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/maci_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_head_scan/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  maci-heads-aligned)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_head_scan_aligned/$OUT_SUFFIX"
    uv run python scripts/run_maci_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/maci_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_head_scan_aligned/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --scope all_aligned_prefill \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  maci-ablation)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_ablation/$OUT_SUFFIX"
    MACI_ABLATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/maci_ablation.json"
    if [[ "$MODE" == "smoke" ]]; then
      MACI_ABLATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_maci_ablation.json"
    fi
    uv run python scripts/run_maci_ablation.py \
      --config "$MACI_ABLATION_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_ablation/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  maci-confirm)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_ablation_locked/$OUT_SUFFIX"
    MACI_CONFIRM_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/maci_ablation_locked.json"
    if [[ "$MODE" == "smoke" ]]; then
      MACI_CONFIRM_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_maci_confirmation.json"
    fi
    uv run python scripts/run_maci_ablation.py \
      --config "$MACI_CONFIRM_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_ablation_locked/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  maci-detector)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_detector/$OUT_SUFFIX"
    MACI_DETECTOR_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/maci_detector.json"
    if [[ "$MODE" == "smoke" ]]; then
      MACI_DETECTOR_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_maci_detector.json"
    fi
    uv run python scripts/train_maci_conflict_detector.py \
      --config "$MACI_DETECTOR_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_detector/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  maci-gated)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/maci_gated_intervention/$OUT_SUFFIX"
    MACI_GATED_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/maci_gated_intervention.json"
    if [[ "$MODE" == "smoke" ]]; then
      MACI_GATED_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_maci_gated_intervention.json"
    fi
    uv run python scripts/run_maci_gated_intervention.py \
      --config "$MACI_GATED_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/maci_gated_intervention/$OUT_SUFFIX" \
      --cache-dir segments/mechanistic_heads_qwen3_8b/data/mmmc_cache \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  vlmbias-heads)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/vlmbias_signed_head_scan/$OUT_SUFFIX"
    uv run python scripts/run_vlmbias_signed_head_scan.py \
      --config segments/mechanistic_heads_qwen3_8b/configs/vlmbias_signed_head_scan.json \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/vlmbias_signed_head_scan/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --resume "${layer_args[@]}" "${smoke_args[@]}"
    ;;
  vlmbias-validation)
    VERIFY_DIR="segments/mechanistic_heads_qwen3_8b/runs/vlmbias_head_validation/$OUT_SUFFIX"
    VLMBIAS_VALIDATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/vlmbias_head_validation.json"
    if [[ "$MODE" == "smoke" ]]; then
      VLMBIAS_VALIDATION_CONFIG="segments/mechanistic_heads_qwen3_8b/configs/smoke_vlmbias_head_validation.json"
    fi
    uv run python scripts/run_vlmbias_head_validation.py \
      --config "$VLMBIAS_VALIDATION_CONFIG" \
      --output-dir "segments/mechanistic_heads_qwen3_8b/runs/vlmbias_head_validation/$OUT_SUFFIX" \
      --seed "$SEED" --device-map cuda --resume "${smoke_args[@]}"
    ;;
  *)
    echo "unknown TASK=$TASK" >&2
    exit 2
    ;;
esac

uv run python scripts/validate_mechanistic_run.py \
  --repo "$REPO" --run-dir "$VERIFY_DIR" --newer-than-epoch "$JOB_STARTED_AT"
