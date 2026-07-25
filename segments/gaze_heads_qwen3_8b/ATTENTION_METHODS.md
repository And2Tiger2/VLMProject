# Qwen3-VL gaze-attention controller experiment

## Research question

Can image-token attention steering on discovered Qwen3-VL gaze heads reduce
VLMBias's bias-aligned answers without degrading correctness on VLMBias or the
less bias-targeted NaturalBench benchmark?

This experiment separates three questions that the older alpha sweep
confounded:

1. Which rule should determine the intervention strength?
2. Which gaze heads or layer band should receive the intervention?
3. Does a setting selected on development data replicate on held-out examples?

The implementation version is `qwen3_attention_methods_v1`. Its
machine-readable methodology, deterministic split manifest, condition
manifests, raw predictions, telemetry, provenance, selections, and reports are
kept separately.

```text
segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1/
segments/gaze_heads_qwen3_8b/runs/attention_methods_v1/
segments/gaze_heads_qwen3_8b/reports/attention_methods_v1/
```

## Fixed inputs and leakage boundary

- Model: `Qwen/Qwen3-VL-8B-Instruct`.
- Head ranking: the merged seed-42 ranking discovered on 500 raw COMICS
  samples.
- Benchmark pool: the already prepared 400-row VLMBias subset and 100-group
  NaturalBench subset.
- Split seed: 2026.
- Development data: 100 VLMBias rows, stratified by topic, and 25 NaturalBench
  groups, stratified by question type and image source.
- Confirmation data: the remaining 300 VLMBias rows and 75 NaturalBench
  groups.

The development and confirmation IDs are disjoint. Head discovery never uses
VLMBias or NaturalBench labels. Controller and head choices use only the
development split. The held-out confirmation split is the primary inferential
result.

> **Post-run correction.** The original v1 confirmation artifacts are not a
> valid held-out comparison. Treatment specs inherited their development split
> and dataset paths while the baseline used the held-out split. The code now
> makes stage routing authoritative and aggregation rejects any stage/path
> mismatch, but existing v1 confirmation outputs remain historical artifacts.
> Use the repaired v2 run documented in
> [GAZE_SPECIFICITY_V2.md](GAZE_SPECIFICITY_V2.md) for held-out inference.

## Attention controllers

All interventions operate inside language-model self-attention. They add a
logit bias to image-token keys for selected heads during both prompt prefill
and autoregressive decoding.

### Baseline

No selected heads and no logit change (`alpha = 0`). It establishes all metric
and invalid-output guardrails.

### Fixed alpha

For every selected head and query, add the same constant to each causally
visible image-token logit:

```text
z'_image = z_image + alpha
```

Development values are `0.5`, `1`, `2`, and `5`.

### Target image-attention mass

Let `m` be a selected head's pre-intervention total attention probability on
image tokens and `t` the requested mass. The exact uncapped positive shift is:

```text
alpha = logit(t) - logit(m)
```

The implementation applies:

```text
alpha_effective = clamp(alpha, minimum=0, maximum=5)
```

Adding this value to every image-token logit reaches `t` exactly when image
tokens are causally visible and the cap is not active. If the head already
places at least `t` mass on the image, it is left unchanged; this controller
never suppresses image attention. Development targets are `0.5`, `0.7`, and
`0.9`. Per-example telemetry records pre-boost mass, effective alpha, achieved
mass, and cap frequency.

### Confidence-gated fixed alpha

This simple, label-free controller first runs the unboosted model greedily. It
computes confidence as the geometric mean probability assigned to the tokens
the model generated. If confidence is below the threshold, it reruns the same
example with fixed `alpha = 2`; otherwise it retains the baseline response.

Development thresholds are `0.3` and `0.6`. The raw baseline response,
confidence, intervention decision, and final attention telemetry are retained.
No answer key, expected bias, or correctness signal enters the gate.

## Stage 1: controller sweep

Before Stage 1, a four-condition mechanics gate runs baseline, fixed alpha 2,
target mass 0.7, and confidence gate 0.6 on 8 development VLMBias rows and 2
development NaturalBench groups. It permits only two simultaneous GPU tasks
and must produce complete unique rows plus controller-specific telemetry. A
failure blocks every larger job through `afterok`.

Every condition uses the global top-100 gaze heads on the development split:

| Family | Conditions |
|---|---|
| Baseline | alpha 0 |
| Fixed | alpha 0.5, 1, 2, 5 |
| Target mass | 0.5, 0.7, 0.9 |
| Confidence gate | thresholds 0.3, 0.6 with fixed alpha 2 |

This is 10 one-GPU conditions. A normal condition makes 200 model calls:
100 VLMBias rows plus four calls for each of 25 NaturalBench groups. A gated
condition can make up to 400 calls because low-confidence examples are rerun.

## Stage 2: head and layer sweep

The best qualified controller is fixed and the following 10 head
configurations are evaluated on the same development rows:

| Configuration | Role |
|---|---|
| Global gaze top 10 | eligible |
| Global gaze top 50 | eligible |
| Global gaze top 100 | eligible |
| Early-layer gaze top 50, layers 0–11 | eligible |
| Middle-layer gaze top 50, layers 12–23 | eligible |
| Late-layer gaze top 50, layers 24–35 | eligible |
| Layer-matched random top 50, seed 55 | control |
| Layer-matched random top 50, seed 56 | control |
| Layer-matched lowest-score top 50 | control |
| Paper-style random top 50 from layers 20–35 | control |

Layer-matched controls reproduce the per-layer histogram of global gaze top
50. All head sets contain exactly the requested number of unique heads and
exclude the reference gaze heads where appropriate. Only gaze-ranked global
or layer-band configurations may be locked for confirmation; a random or
low-score control can diagnose specificity but cannot become the claimed gaze
intervention.

## Development qualification and locking

A non-baseline condition is qualified only if all four checks pass relative
to the development baseline:

- VLMBias accuracy drops by no more than 0.02.
- NaturalBench strict group accuracy (`G_Acc`) drops by no more than 0.05.
- VLMBias invalid-output rate increases by no more than 0.02.
- NaturalBench invalid-output rate increases by no more than 0.02.

Among qualified conditions, selection uses this preregistered lexicographic
objective:

1. minimize VLMBias `bias_aligned_fraction`;
2. maximize VLMBias accuracy;
3. maximize NaturalBench `G_Acc`;
4. maximize NaturalBench call accuracy (`Acc`).

The best condition within each controller family is also retained, even if it
misses a guardrail, so confirmation can report how fixed, target-mass, and
confidence-gated methods behave under one locked head set. If no non-baseline
controller qualifies, or no gaze-ranked head configuration qualifies, the
aggregator exits nonzero and Slurm's `afterok` dependency blocks later stages.

## Stage 3: held-out confirmation

The confirmation array contains four deterministic conditions on 300 held-out
VLMBias rows and 75 held-out NaturalBench groups:

- baseline;
- locked best fixed-alpha controller;
- locked best target-mass controller;
- locked best confidence-gated controller.

All three intervention families use the one gaze-head configuration selected
in Stage 2. Each ordinary condition makes 600 calls; the confidence-gated
condition makes at most 1,200. Selection is not revisited after seeing these
results.

## Optional Stage 4: broader multi-seed robustness

Only after the controller and head selection files exist, an explicit
`robustness` command expands the same four locked conditions over seeds 0, 1,
and 2. It uses temperature 0.7 on all 400 VLMBias rows and all 100 NaturalBench
groups. More seeds may be supplied to the Python launcher.

This analysis measures sampling stability and improves precision, but it is
secondary: it includes development examples, so it cannot replace the held-out
deterministic confirmation.

## Validation and reproducibility

Every worker:

- records the full condition, exact selected head IDs, dataset paths, Git
  commit, Slurm job/task/node provenance, and all generation telemetry;
- writes each prediction immediately and supports exact-config resume;
- requires the expected VLMBias rows, NaturalBench groups, and four calls per
  NaturalBench group;
- rejects duplicate or wrong-cardinality head sets;
- rejects missing attention telemetry for an active intervention;
- records confidence-gate intervention rates and target-mass alpha/cap data.

The split and condition files are deterministic. Re-running preparation
recreates the same split from the same source file hashes.

## Neuronic execution

Inspect the exact dependency chain without submitting:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh dry-run
```

Submit the mechanics gate, controller development, head development, and
held-out confirmation:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh full
```

For an unattended run, queue those four stages plus the three-seed robustness
stage in one command:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh overnight
```

`overnight` is a strict 15-job `afterok` graph: each of the five stages has a
CPU preparation job, a one-GPU condition array, and a CPU validation/aggregate
job. Robustness cannot start unless the mechanics gate passes, a non-baseline
controller passes the development guardrails, a gaze-ranked head setting
passes the development guardrails, and held-out confirmation is complete and
valid. Any failed task blocks every dependent stage.

The submission receipt, exact commands, all job IDs, final job ID, timestamp,
and Git commit are saved to:

```text
segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1/last_submission.json
```

The Princeton-VL `check_overheat` Python module is an Ionic policy and lives in
a Princeton-VL group-owned directory. It is not required for this Neuronic
workflow. Calls through `vlm_eval.overheat.maybe_pause()` therefore treat a
missing or inaccessible checker as unavailable. For a future Ionic run, set
`VLM_REQUIRE_OVERHEAT_CHECK=1` and, if necessary,
`VLM_CHECK_OVERHEAT_DIR` to the accessible directory containing
`check_overheat.py`; strict mode refuses to continue unless the module provides
callable `pause_needed()` and `pause()` functions.

Each GPU task requests one GPU, four CPU cores, and 64 GB host memory. The
model and examples are serial within one worker, so requesting multiple GPUs
would not speed a condition up. The array permits at most eight concurrent
single-GPU workers; Slurm may run fewer as resources permit. This follows the
cluster guidance to demonstrate/utilize the one-GPU case before considering
multi-GPU parallelism. Use `jobstats <jobid>` after the pilot tasks to check
GPU/CPU utilization before increasing resources or concurrency.

If an earlier chain failed, recover it only after pulling the fix:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh recover FINAL_JOB_ID
```

Recovery validates that ID against the saved receipt, cancels exactly the jobs
recorded there, moves the old experiment/run/report directories into
timestamped `*_failed_FINAL_JOB_ID_*` archives, and submits a clean replacement.
It uses gentler concurrency limits of four development workers and two
robustness workers.

After the final job completes:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh verify
```

If `full` was used instead of `overnight` and held-out confirmation is
satisfactory, launch the optional broader run:

```bash
bash scripts/run_neuronic_qwen3_attention_methods.sh robustness
```

Individual stages can be submitted with `controller`, `heads`, or `confirm`.
The mechanics gate alone can be submitted with `smoke`.
Downstream standalone stages require the upstream `selection.json` files.

## Interpretation limits

- Lower bias-aligned fraction is useful only under the stated correctness and
  validity guardrails.
- Attention mass is a mechanistic manipulation check, not proof of causal
  visual reasoning by itself.
- Confidence is sequence-probability confidence, not a calibrated probability
  of answer correctness.
- The prepared datasets are controlled subsets, not the full upstream
  benchmarks.
- The optional all-data multi-seed result is not a fresh test set.
