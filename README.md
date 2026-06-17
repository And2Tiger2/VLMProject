# VLMProject

Interventions to reduce hallucination and prior-driven bias in VLMs.

This repo currently contains a basic evaluation harness for the
[VLMs are Biased / VLMBias](https://github.com/anvo25/vlms-are-biased)
benchmark. The goal is to make it easy to swap in a local checkpoint or API
adapter and get comparable metrics before testing interventions.

## Setup

```bash
uv sync --extra dev
```

For Qwen2.5-VL local inference:

```bash
uv sync --extra dev --extra qwen
```

## Fixed Eval Settings

Use `adapters.qwen25_vl_eval:make_adapter` for baseline Qwen runs. This keeps
model and decoding settings fixed across VLMBias, NaturalBench, and future
interventions:

- model: `Qwen/Qwen2.5-VL-3B-Instruct`
- decoding: sampled, `do_sample=True`, `temperature=0.7`, `seed=0`
- response length: `max_new_tokens=16`
- image budget: `max_pixels=1048576`, `min_pixels=0`
- device placement: `device_map=auto`

The fixed image budget is important: uncapped NaturalBench images can create
too many visual tokens and cause large attention allocations. The fixed seed is
important because sampled decoding is otherwise noisy across intervention
comparisons.

## Smoke Test

Run the deterministic dummy adapter on the tiny local JSONL:

```bash
uv run vlm-eval \
  --dataset examples/tiny_vlmbias.jsonl \
  --adapter adapters.dummy:make_adapter,mode=bias \
  --out runs/tiny_dummy.jsonl
```

This should produce `accuracy = 0.0` and `bias_aligned_error_rate = 1.0`
because the dummy adapter intentionally returns the expected biased answer.

## VLMBias Eval

Create a fixed local 400-example eval slice:

```bash
uv run python scripts/make_vlmbias_slice.py \
  --split main \
  --n 400 \
  --seed 0 \
  --out data/vlmbias_400.jsonl
```

This writes `data/vlmbias_400.jsonl`, saves images under
`data/vlmbias_400_images/`, and records the topic balance in
`data/vlmbias_400.manifest.json`.

Run a small random slice from Hugging Face:

```bash
uv run vlm-eval \
  --dataset anvo25/vlms-are-biased \
  --split main \
  --limit 200 \
  --shuffle \
  --adapter adapters.qwen25_vl:make_adapter,model_id=Qwen/Qwen2.5-VL-3B-Instruct \
  --out runs/qwen25vl_3b_vlmbias_200.jsonl
```

Or run Qwen2.5-VL-3B on the fixed local slice:

```bash
uv run vlm-eval \
  --dataset data/vlmbias_400.jsonl \
  --adapter adapters.qwen25_vl_eval:make_adapter \
  --out runs/qwen25vl_3b_vlmbias_400.jsonl
```

The CLI writes one JSONL row per prediction and a sibling
`.summary.json` file. Current metrics are intentionally simple:

- `accuracy`: exact normalized match against `ground_truth`
- `bias_aligned_error_rate`: among incorrect examples, fraction where the
  parsed answer exactly matches `expected_bias`
- `bias_aligned_fraction`: fraction of all examples that are bias-aligned
  errors

## Adding Models

Adapters are loaded with `module:factory,arg=value` specs. A factory should
return an object with:

```python
name: str
generate(example: EvalExample) -> str
```

Use `adapters/dummy.py` as the minimal template. The optional
`adapters/qwen25_vl.py` adapter shows the shape for local Transformers
checkpoints.

## NaturalBench Eval

NaturalBench groups each contain two images and two questions. Each group
expands to four model calls, so 100 groups mirrors the 400-call VLMBias run.

Create a fixed local NaturalBench slice:

```bash
uv run python scripts/make_naturalbench_slice.py \
  --split train \
  --groups 100 \
  --seed 0 \
  --out data/naturalbench_100_groups.jsonl
```

Run Qwen2.5-VL-3B on that slice:

```bash
uv run vlm-eval-naturalbench \
  --dataset data/naturalbench_100_groups.jsonl \
  --adapter adapters.qwen25_vl_eval:make_adapter \
  --out runs/qwen25vl_3b_naturalbench_100_groups.jsonl
```

If the run is interrupted, resume it without discarding completed calls:

```bash
uv run vlm-eval-naturalbench \
  --dataset data/naturalbench_100_groups.jsonl \
  --resume \
  --adapter adapters.qwen25_vl_eval:make_adapter \
  --out runs/qwen25vl_3b_naturalbench_100_groups.jsonl
```

The NaturalBench summary reports:

- `Acc`: per-call answer accuracy across all 400 image-question calls.
- `Q_Acc`: consistency for each question across both images.
- `I_Acc`: consistency for each image across both questions.
- `G_Acc`: strict group accuracy; all four calls in a group must be correct.
