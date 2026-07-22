# Qwen3-VL 8B Gaze Heads on Neuronic

This segment targets `Qwen/Qwen3-VL-8B-Instruct`, the default model in the
official Gaze Heads repository. It contains the three experiments needed now:

1. discover gaze heads on raw COMICS;
2. test causal static narration steering on disjoint OpenAI comics;
3. sweep image-attention alpha on VLMBias and NaturalBench.

All real runs use Slurm batch arrays. The commands below are run from the
repository root on a Neuronic login node.

Put the checkout in CS project space rather than a small home directory. The
repository-local cache holds the environment, roughly 16 GB of model weights,
and the raw COMICS archive; downloading and extracting COMICS can temporarily
require about 130 GB. After verifying extraction, the recoverable archive at
`.cache/datasets/raw_panel_images.tar.gz` can be removed to reclaim space.

## One-time setup

```bash
bash scripts/setup_neuronic_qwen3.sh
```

This installs the `qwen` and test extras and downloads the Qwen3-VL 8B weights
to the shared repository cache *before* a GPU is allocated. GPU jobs set
`HF_HUB_OFFLINE=1`, so a missing model fails immediately instead of holding an
idle GPU while downloading. The script provisions a repository-cached Python
3.12 interpreter with `uv`; it does not use Neuronic's incompatible system
Python 3.14.

Prepare the OpenAI comics, VLMBias, and NaturalBench slices in one command:

```bash
bash scripts/prepare_neuronic_qwen3_data.sh
```

To also download and extract the approximately 65 GB raw COMICS corpus used
for paper-protocol discovery:

```bash
bash scripts/prepare_neuronic_qwen3_data.sh --download-raw-comics
```

Expected data layouts:

```text
segments/gaze_heads_qwen3_8b/data/discovery_comics/<comic_id>/<page>_<panel>.jpg
segments/gaze_heads_qwen3_8b/data/eval_comics/<strip_id>/p1.png ... p6.png
segments/vlm_bias_attention/data/vlmbias_400.jsonl
segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl
```

Validate all existing comics, images, prompts, ground-truth answers, counts,
and discovery/evaluation disjointness without downloading or submitting jobs:

```bash
uv run python scripts/validate_qwen3_gaze_stage.py datasets
```

The benchmark inputs are intentionally fixed subsets, not the entire upstream
benchmarks: 400 topic-balanced VLMBias rows and 100 question-type-balanced
NaturalBench groups (400 NaturalBench model calls), both sampled with seed 0.

The discovery and evaluation roots must be disjoint. The launcher rejects an
identical root or overlapping directory IDs.

## Experiment 1: discover gaze heads

One seed, 500 raw COMICS samples, ten 50-sample GPU shards:

```bash
python3 scripts/submit_neuronic_qwen3.py discovery
```

Run three independent discovery seeds in parallel by adding one argument:

```bash
python3 scripts/submit_neuronic_qwen3.py discovery --seeds 3
```

For each seed, the launcher queues GPU shards followed by one dependent merge
and validation job. Seed 42 writes the canonical ranking to:

```text
segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json
```

Raw sampling matches the reference implementation: random comic, random page
with at least six panels, random consecutive six-panel window on that page;
for comics longer than ten pages, the first and last five pages are avoided
when possible.

## Experiment 2: static narration steering

Export the judge key once, then submit the entire generation, merge,
validation, and paper-style Claude judging pipeline:

```bash
python3 scripts/submit_neuronic_qwen3.py static
```

The full defaults produce 24,000 judged generations per seed (500 strips ×
6 targets × 2 conditions × 4 top-k runs), so check the current Anthropic rate
limits and budget first. `--judge none` avoids all API calls without weakening
the generation/attention validation gates.

Multiple random-control seeds require only `--seeds`:

```bash
python3 scripts/submit_neuronic_qwen3.py static --seeds 3
```

Defaults:

- 500 disjoint evaluation strips;
- all six target panels;
- top-k 1, 10, 50, and 100;
- paper-exact non-gaze controls sampled strictly from the bottom 5% of gaze
  scores (so the control can contain fewer than `K` heads when `K` is large);
- greedy generation, 100 tokens;
- `boost_suppress` intervention with bias 10,000;
- **full-sequence steering** during prefill and decode, matching the official
  static/VQA scripts;
- Claude forced 1-of-6 judging; empty, junk, and baseline-identical outputs are
  misses.

If the API key is intentionally unavailable, generation and validation can be
submitted without judging:

```bash
python3 scripts/submit_neuronic_qwen3.py static --judge none
```

For the scientifically useful equal-cardinality control ablation, backfill
with the lowest-scoring remaining heads:

```bash
python3 scripts/submit_neuronic_qwen3.py static --control-mode matched
```

If an API/rate-limit failure interrupts judging, completed judgments are
flushed row by row. Retry only the CPU judge jobs, without rerunning Qwen:

```bash
python3 scripts/submit_neuronic_qwen3.py static --judge-only
```

If GPU workers completed generation but an older shard-level quality gate
returned exit code 2, recover all existing outputs with CPU-only merge and
validation jobs; this never reloads Qwen or regenerates rows:

```bash
python3 scripts/submit_neuronic_qwen3.py static --seeds 3 --merge-only
```

Immediate-EOS empty outputs legitimately have no decode-step attention
telemetry. They are retained and scored as misses. Empty rates above 5% are
reported as quality warnings; rates above 50% remain hard validation failures.

A cheap five-comic mechanics smoke test is one command:

```bash
python3 scripts/submit_neuronic_qwen3.py static --shards 1 --shard-size 5 --top-ks 10 --judge none
```

The dependent validator rejects empty generations, missing attention
telemetry, ineffective target attention, incomplete row counts, and duplicate
experiment keys before results are interpreted.

## Experiment 3: VLMBias + NaturalBench alpha sweep

One seed:

```bash
python3 scripts/submit_neuronic_qwen3.py benchmark
```

Seven seeds, submitted as seven parallel GPU jobs with one dependent strict
aggregation job:

```bash
python3 scripts/submit_neuronic_qwen3.py benchmark --seeds 7
```

Defaults:

- top-k 10, 50, and 100 gaze heads;
- alpha 0.25, 0.5, 1, 2, 5, and 10;
- an independent baseline for every seed;
- all 400 prepared VLMBias rows and all 100 prepared NaturalBench groups (not
  the complete upstream benchmarks);
- full-sequence alpha boosting as the primary intervention;
- strict aggregation that fails if any expected seed/condition is absent.

Decode-only is an explicit ablation and gets a separate output directory:

```bash
python3 scripts/submit_neuronic_qwen3.py benchmark --seeds 7 --attention-mode decode
```

A non-colliding benchmark smoke run is:

```bash
python3 scripts/submit_neuronic_qwen3.py benchmark --limit 20 --naturalbench-limit-groups 5
```

## Scheduler controls

Optional Neuronic scheduling fields are passed directly to every job in the
submission chain:

```bash
python3 scripts/submit_neuronic_qwen3.py --account MY_ACCOUNT --partition MY_PARTITION benchmark --seeds 7
```

Every seed and shard is eligible to run in parallel by default; Slurm starts
them as resources permit. Use `--max-parallel N` when you intentionally want
to cap simultaneously running GPU array tasks. Static judging separately
defaults to two concurrent API clients (`--judge-parallel`).

GPU workers immediately exercise their allocated CUDA device and default to
requiring at least 20 GiB of VRAM. They use BF16 on supported GPUs and FP16 on
older CUDA GPUs. If the cluster constraint name is known, select it with
`--constraint`; raise `--min-gpu-memory-gb` when you specifically want a
larger-memory node. Run the static smoke test before the full eager-attention
jobs on an unfamiliar GPU type.

Before spending GPU time, inspect the exact commands with:

```bash
python3 scripts/submit_neuronic_qwen3.py --dry-run --skip-preflight benchmark --seeds 7
```

`--dependency JOB_ID` can attach a complete experiment to an existing Slurm
job. Every launcher invocation prints `final_job=<id>` for chaining or
monitoring with `squeue -j <id>`.

## Interactive troubleshooting

Interactive allocations are only for short CUDA/import checks:

```bash
salloc --gres=gpu:1 -c 8 --mem=96G --time=00:30:00 srun --pty bash -l
```

Real experiments should use the one-line batch launchers above. The workers
respect `CUDA_VISIBLE_DEVICES`, resume per-condition output where safe, and
write Slurm logs under `segments/gaze_heads_qwen3_8b/runs/slurm/`.

## Protocol references

- Paper: <https://arxiv.org/abs/2606.14703>
- Official implementation: <https://github.com/rohitgandikota/gaze-heads>
  (pipeline audited against commit `668fb489489db7f428f895412c753a2bc1390821`)
- Qwen model: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Princeton computing guide: <https://csguide.cs.princeton.edu/resources>
