# Qwen3-VL gaze-head specificity and adaptive steering

## Purpose

This follow-up answers two separate questions:

1. Does steering the COMICS-discovered gaze-head set outperform steering
   equally sized control head sets?
2. After locking the gaze top-50 set, can a fixed, target-mass, or
   confidence-gated controller improve VLMBias without degrading NaturalBench?

The implementation version is `qwen3_gaze_specificity_v2`. It never overwrites
the completed v1 artifacts:

```text
segments/gaze_heads_qwen3_8b/experiments/gaze_specificity_v2/
segments/gaze_heads_qwen3_8b/runs/gaze_specificity_v2/
segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2/
```

The model and seed-42 COMICS gaze ranking remain fixed. Head discovery never
uses VLMBias or NaturalBench labels.

## Data boundary and routing invariant

The deterministic seed-2026 split is unchanged:

- development: 100 stratified VLMBias rows and 25 stratified NaturalBench
  groups;
- held-out: the remaining disjoint 300 VLMBias rows and 75 NaturalBench
  groups;
- robustness: all 400 rows and 100 groups, explicitly secondary because it
  includes development data.

Dataset routing belongs to a stage, not to a condition copied from an earlier
selection report. Preparation removes inherited runtime fields and writes the
stage's split and dataset paths. Aggregation independently checks that every
condition uses the authoritative split-manifest paths, that all paired
conditions have identical IDs, and that output cardinalities are exact.

This invariant repairs the v1 confirmation issue in which treatment specs
inherited `split=dev` while the baseline used `split=confirm`.

## Stage 1: held-out repair

The four previously locked conditions are reconstructed from the v1
development selections and rerun in the new namespace:

- baseline;
- selected fixed-alpha controller;
- selected target-mass controller;
- selected confidence-gated controller.

Every condition uses exactly 300 held-out VLMBias rows and 75 held-out
NaturalBench groups. Results include paired 10,000-replicate cluster-bootstrap
intervals relative to baseline. VLMBias examples and NaturalBench groups are
the resampling units.

## Stage 2: head-set control distribution

All conditions use deterministic decoding, fixed `alpha=0.5`, and the held-out
split:

| Head set | Draws | Role |
|---|---:|---|
| no steering | 1 | baseline |
| global gaze top 50 | 1 | pre-specified treatment |
| layer-matched random top 50 | 20 | primary null distribution |
| paper-style random top 50 from layers 20--35 | 10 | placement diagnostic |
| layer-matched lowest gaze-score top 50 | 1 | score diagnostic |

Each layer-matched set must reproduce the gaze set's per-layer histogram
exactly. The report records exact head IDs, overlaps, per-condition paired
bootstrap intervals, and separate empirical tests for VLMBias accuracy, bias
alignment, invalid rate, NaturalBench call accuracy, strict group accuracy,
and invalid rate.

For the 20 primary random draws, the one-sided add-one empirical p-value is:

```text
p = (1 + number of random draws at least as good as gaze) / 21
```

The minimum attainable value is therefore `1/21 = 0.0476`. The pre-specified
primary specificity endpoint is lower VLMBias bias-aligned fraction, subject
to the existing accuracy, NaturalBench, and validity guardrails. Other
endpoints are reported separately rather than hidden in a composite score.

## Stage 3: top-50 controller interaction sweep

The gaze top-50 head set is fixed before this development-only sweep:

| Family | Development conditions |
|---|---|
| baseline | alpha 0 |
| fixed | alpha 0.25, 0.5, 1, 2, 5 |
| target mass | 0.3, 0.4, 0.5 |
| confidence gate | thresholds 0.7, 0.8, 0.85, 0.9 crossed with alpha 0.5, 1, 2 |

The higher gate thresholds are motivated by the v1 mean baseline confidence of
about 0.83; thresholds 0.3 and 0.6 rarely intervened. Within each family, a
candidate must pass the existing guardrails:

- VLMBias accuracy drop no greater than 0.02;
- NaturalBench `G_Acc` drop no greater than 0.05;
- invalid-rate increase no greater than 0.02 on either dataset.

Qualified settings are ordered by lower VLMBias bias-aligned fraction, higher
VLMBias accuracy, lower VLMBias invalid rate, higher NaturalBench `G_Acc`, and
higher NaturalBench call accuracy. One candidate per family is locked without
reading held-out results. If an entire family misses a guardrail, its best
candidate is retained explicitly as an unqualified diagnostic rather than
blocking the remaining pre-registered stages.

## Stage 4: final held-out evaluation

Five deterministic held-out conditions are run:

- baseline;
- the v1 fixed-alpha-0.5 top-50 anchor;
- the development-selected fixed candidate;
- the development-selected target-mass candidate;
- the development-selected confidence-gated candidate.

The alpha-0.5 anchor is retained even if it duplicates the selected fixed
configuration so the pre-existing hypothesis remains explicit. Each condition
must contain 300/75 held-out units and use identical IDs. Paired cluster
bootstrap intervals are the primary uncertainty analysis.

## Stage 5: sampling robustness

The same five conditions are run on all 400/100 units at temperature 0.7 for
seeds 0, 1, and 2. Reports contain per-condition mean, standard deviation,
range, and same-seed deltas from baseline. This stage evaluates sampling
stability but cannot replace the held-out deterministic result.

## Neuronic execution

Inspect all fifteen submissions without changing cluster state:

```bash
bash scripts/run_neuronic_qwen3_gaze_specificity.sh dry-run
```

Submit every stage as one strict `afterok` chain:

```bash
bash scripts/run_neuronic_qwen3_gaze_specificity.sh overnight
```

The default maximum concurrency is six ordinary conditions, eight control
draws, and four robustness tasks. Each worker requests one GPU, four CPU cores,
and 64 GB host memory through the existing Neuronic condition script. Outputs
are resumable only when the exact experiment configuration matches.

After the printed `final_job` finishes:

```bash
bash scripts/run_neuronic_qwen3_gaze_specificity.sh verify
```

The submission receipt records the Git commit, timestamp, exact commands, all
job IDs, and final job ID:

```text
segments/gaze_heads_qwen3_8b/experiments/gaze_specificity_v2/last_submission.json
```

Individual stages may also be submitted with `repair`, `controls`, `tune`,
`final`, or `robustness`. Standalone `final` and `robustness` require the tune
selection report.

## Interpretation limits

- A change in attention telemetry proves the intervention occurred, not that
  the model used the image causally or correctly.
- Head-draw p-values quantify specificity relative to the implemented null
  family; paper-style and low-score controls remain diagnostics.
- The four primary scientific metrics and two invalid-rate guardrails should
  be read together. A lower bias fraction caused by invalid output is not a
  successful mitigation.
- NaturalBench `G_Acc` is strict: one group is correct only when all four
  calls are correct.
- The prepared 400/100 datasets are controlled subsets rather than the full
  upstream benchmarks.
