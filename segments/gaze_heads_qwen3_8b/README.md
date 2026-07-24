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

Kimi-VL judging uses a separate pinned environment so its upstream
Transformers dependency cannot alter the working Qwen3 environment. Prepare it
once on the login node (no GPU allocation and no API key required):

```bash
bash scripts/setup_neuronic_kimi_judge.sh
```

The checkpoint is public. A Hugging Face token is optional and only helps with
Hub rate limits.

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

Submit the entire generation, merge, validation, and local Kimi-VL judging
pipeline:

```bash
python3 scripts/submit_neuronic_qwen3.py static
```

The full defaults produce 24,000 judged generations per seed (500 strips ×
6 targets × 2 conditions × 4 top-k runs). Judging uses the public
`moonshotai/Kimi-VL-A3B-Instruct` checkpoint locally on one GPU per array task;
there are no API calls or per-call charges. `--judge none` skips judging without
weakening the generation/attention validation gates.

Multiple random-control seeds require only `--seeds`:

```bash
python3 scripts/submit_neuronic_qwen3.py static --seeds 3
```

Defaults:

- 500 disjoint evaluation strips;
- all six target panels;
- top-k 1, 10, 50, and 100;
- paper-exact non-gaze controls: exactly `K` uniformly sampled heads from
  inclusive layers 20--35, excluding the selected gaze heads;
- greedy generation, 100 tokens;
- `boost_suppress` intervention with bias 10,000;
- **full-sequence steering** during prefill and decode, matching the official
  static/VQA scripts;
- Kimi-VL forced 1-of-6 judging with batched, single-class-token decoding;
  empty, junk, and baseline-identical outputs are misses.

Generation and validation can still be submitted without judging:

```bash
python3 scripts/submit_neuronic_qwen3.py static --judge none
```

The public repository's bottom-five-percent behavior remains available only
as an explicitly named ablation. It is not the paper's headline control:

```bash
python3 scripts/submit_neuronic_qwen3.py static --control-mode bottom5
```

The older global low-score exact-cardinality fallback is also retained as a
separate `matched` ablation:

```bash
python3 scripts/submit_neuronic_qwen3.py static --control-mode matched
```

### Correct only the control without rerunning saved gaze outputs

Runs produced before the paper-control correction contain the bottom-5% (58
head) repository control. Preserve those as an ablation and generate a new
control-only artifact. First run a 50-comic, top-100, seed-42 pilot:

```bash
bash scripts/run_neuronic_qwen3_paper_control.sh pilot
```

After it finishes, validate the newly assembled paired artifact:

```bash
bash scripts/run_neuronic_qwen3_paper_control.sh verify-pilot
```

This pairs the new `non_gaze_paper_100` rows with the existing
`gaze_top100` rows; it does not regenerate the gaze condition. If the pilot's
empty rate and outputs are healthy, submit all 500 comics for seeds 42--44:

```bash
bash scripts/run_neuronic_qwen3_paper_control.sh full
```

Then validate all three completed paired runs:

```bash
bash scripts/run_neuronic_qwen3_paper_control.sh verify-full
```

Each control shard records its exact selected head IDs, inclusive layer band,
Git commit, and Slurm provenance. The assembly job rejects the result unless
there are exactly 100 unique control heads, all in layers 20--35, with no
overlap with the top-100 gaze set. Judging is intentionally not submitted by
these wrapper commands; inspect the pilot and calibrate the judge first.

### Calibrate and smoke-test Kimi before full judging

Refresh the evaluation manifest once so it records the six known source
captions for each strip (existing images are not overwritten):

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh prepare-captions
```

Then submit a 60-caption calibration balanced across all six panels:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh calibrate
```

After the job finishes:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh verify-calibration
```

The gate requires at least 70% overall caption-to-panel accuracy, at least 40%
for every panel position, and a valid parse rate. Only after that passes,
submit a 120-row actual-generation smoke test:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh smoke
```

This smoke is deterministically balanced across both conditions and all six
target panels instead of taking the first rows in file order. The judge sees
the strip and answer only; the baseline is used solely for the paper's
token-Jaccard prefilter at threshold 0.90 and is never included in the judge
prompt. Inspect the smoke with:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh verify-smoke
```

Full three-seed judging is deliberately a separate final command:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh judge-full
```

After all three judge tasks complete, strictly validate and aggregate them:

```bash
bash scripts/run_neuronic_qwen3_kimi_gate.sh aggregate-full
```

The aggregator requires 6,000 unique judgments per seed, numbered-panel judge
schema 6, zero parse failures, and exact agreement among the deterministic
gaze judgments repeated across files. It counts gaze rows once, reports each
random-control seed separately, and computes paired 95% intervals by
bootstrapping the 500 comic strips rather than treating panel rows or repeated
gaze copies as independent.

After CPU merging finishes, verify the complete three-seed discovery and
static-generation state before spending more GPU time:

```bash
uv run python scripts/verify_neuronic_qwen3_prejudge.py --seeds 3
```

The command exits 0 only when the datasets, all three discovery rankings, and
all twelve merged static runs are complete and valid.

Before the full judge array, run a 24-row Kimi mechanics smoke test in a
separate output directory:

```bash
python3 scripts/submit_neuronic_qwen3.py static --top-ks 1 --judge-only --judge-limit 24
```

Then judge all merged runs for three seeds without rerunning Qwen:

```bash
python3 scripts/submit_neuronic_qwen3.py static --seeds 3 --judge-only
```

Judgments are flushed row by row and can be resumed by submitting the same
command again. Full outputs are written beneath each merged run in
`kimi_judge/`.

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

### Control-distribution follow-up

The three-control result is too variable to support a specificity claim. The
follow-up estimates that variance on 100 comics while reusing the already
judged seed-42 gaze generations:

- 10 paper controls, each uniformly sampling 100 non-gaze heads from layers
  20--35;
- 10 random controls with exactly the same per-layer histogram as the top-100
  gaze set;
- one deterministic layer-matched control using the lowest-scoring eligible
  heads.

Generation, strict merging, numbered-panel Kimi judging, and CPU aggregation
form one dependency chain:

```bash
bash scripts/run_neuronic_qwen3_control_distribution.sh submit
```

The defaults request one GPU per generation/judge task, cap generation at
eight concurrent tasks, and cap Kimi judging at two. The aggregate counts the
existing gaze condition once and treats each selected control set as a draw;
it reports the empirical gaze percentile, control distributions, paired
comic-bootstrap intervals, and score/accuracy correlations.

After `final_job` completes:

```bash
bash scripts/run_neuronic_qwen3_control_distribution.sh verify
```

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

## Experiment 4: staged attention-controller and head-selection study

The original alpha sweep mixes intervention strength, head count, and
selection on the same benchmark rows. The staged follow-up tests four
controllers (baseline, fixed alpha, target attention mass, and a label-free
confidence gate), then tests head count/layer placement, and finally evaluates
the locked settings on held-out examples.

Inspect the complete Slurm chain:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh dry-run
```

Submit the small mechanics smoke, development controller sweep, development
head sweep, and held-out confirmation as one failure-gated dependency chain:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh full
```

When its final job completes:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh verify
```

If the held-out result is worth tightening, run the separately authorized
three-seed, all-400/all-100 robustness stage:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh robustness
```

To queue the mechanics gate, both development sweeps, held-out confirmation,
and three-seed robustness as one unattended `afterok` graph:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh overnight
```

The launcher records every job ID and the final dependency ID in
`segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1/last_submission.json`.
If any stage fails validation, all later work stays blocked rather than
silently running with missing or unqualified inputs.

The Neuronic temperature checker is mandatory for this workflow. It is
validated before submission and again before model loading in every GPU job.
The default is `/n/fs/vl/scripts_group/check_overheat`; set
`VLM_CHECK_OVERHEAT_DIR` to an accessible directory containing
`check_overheat.py` if your cluster group provides it elsewhere. The workflow
refuses to run without callable `pause_needed()` and `pause()` safeguards.

To replace a failed chain without mixing its partial artifacts into a corrected
run:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh recover FINAL_JOB_ID
```

The exact sweep matrix, controller equations, deterministic 100/300 and 25/75
splits, guardrails, head controls, selection rule, result hierarchy, resource
rationale, and artifact paths are documented in
[ATTENTION_METHODS.md](ATTENTION_METHODS.md).

## Scheduler controls

Optional Neuronic scheduling fields are passed directly to every job in the
submission chain:

```bash
python3 scripts/submit_neuronic_qwen3.py --account MY_ACCOUNT --partition MY_PARTITION benchmark --seeds 7
```

Every seed and shard is eligible to run in parallel by default; Slurm starts
them as resources permit. Use `--max-parallel N` when you intentionally want
to cap simultaneously running GPU array tasks. Static judging separately
defaults to two concurrent one-GPU Kimi workers (`--judge-parallel`). Each
judge worker fails early below 40 GiB of VRAM, selecting the smallest Neuronic
GPU class that safely holds this BF16 checkpoint.

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
