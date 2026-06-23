# Experiment Reports

This folder is the curated report layer for the project. Raw model outputs live
under `runs/`; this folder keeps the summary tables, figures, and notes needed
to interpret the experiments without digging through every JSONL.

Unless noted otherwise, VLMBias experiments use:

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Dataset: `data/vlmbias_400.jsonl`
- Calls: 400 examples
- Image cap: `max_pixels=1048576`
- Output length: `max_new_tokens=16`

Metrics:

- `accuracy`: exact normalized match against `ground_truth`
- `bias_aligned_fraction`: fraction of all examples answered with `expected_bias`
- `bias_aligned_error_rate`: among wrong examples, fraction answered with `expected_bias`
- `error_rate`: fraction incorrect

## Baseline Image/No-Image

Files:

- `baselines/vlmbias_baseline_temp07_image_vs_no_image.txt`
- `baselines/vlmbias_baseline_temp10_image_vs_no_image.txt`

Question tested:

Does removing the image reduce bias-aligned errors in the basic one-pass VLMBias
setting?

Setup:

- One-pass answering, no intermediate visual-evidence step
- Prompt mode: baseline, answer-only
- Temperature sweeps:
  - `temperature=0.7`, `seed=0`
  - `temperature=1.0`, `seed=0`
- Conditions:
  - image input
  - no-image input

Main observation:

Removing the image lowers bias-aligned fraction but also lowers accuracy. For
the temp-0.7 run, image baseline has `accuracy=0.1425`,
`bias_aligned_fraction=0.395`; no-image has `accuracy=0.1200`,
`bias_aligned_fraction=0.295`.

## Resolution Ablation

Files:

- `resolution_ablation/summary.tsv`
- `resolution_ablation/summary.png`
- `resolution_ablation/summary.svg`

Question tested:

Does giving Qwen more visual resolution reduce VLMBias bias-aligned errors?

Setup:

- One-pass image answering
- `temperature=0.7`, `seed=0`
- `max_pixels` values:
  - `262144`
  - `524288`
  - `1048576`
  - `1572864`
  - `2097152`

Main observation:

Increasing image budget did not materially reduce bias. Accuracy was flat near
`0.1425` for larger image budgets and bias-aligned fraction stayed around
`0.39-0.40`.

## Original Two-Pass Ablation

Files:

- `two_pass_original/summary.tsv`
- `two_pass_original/summary.png`

Question tested:

Does separating visual-evidence generation from final answering help, and does
the answer pass need to see the image again?

Setup:

- Two passes
- `temperature=0.7`, `seed=0`
- Pass 1:

```text
Describe only the visual evidence needed to answer this question. Do not answer the question.
Question: {question}
```

- Pass 2:

```text
{question}
Visual evidence: {visual_evidence}
Use the visual evidence, not what is normally true. Write 'Final answer:' followed by the answer.
```

- Conditions:
  - `two_pass_image_again`: pass 2 receives visual evidence and image
  - `two_pass_description_only`: pass 2 receives visual evidence only

Main observation:

Description-only pass 2 sharply reduced bias-aligned fraction to `0.1500`;
image-again stayed high at `0.3800`.

## Prompt-Variant Two-Pass Run

Files:

- `prompt_variants_confounded/notes.md`
- `prompt_variants_confounded/fast_summary.tsv`
- `prompt_variants_confounded/legacy_summary.tsv`
- `prompt_variants_confounded/comparison.tsv`
- `prompt_variants_confounded/comparison.png`

Question tested:

Initial attempt to test whether different first-pass description prompts reduce
VLMBias bias.

Setup:

- Two passes
- `temperature=0.7`, `seed=0`
- First-pass variants:
  - neutral
  - counterfactual
  - structured
  - verification
- Answer modes:
  - image again
  - description only

Important caveat:

This run changed the second-pass answer prompt at the same time as the
first-pass description prompt. It is therefore confounded and should be treated
as a diagnostic run, not the main prompt-variant result.

Main observation:

The four variants were nearly identical, which led to the corrected unified
experiment below.

## Unified Two-Pass Prompt Experiment

Files:

- `two_pass_unified/results.png`
- `two_pass_unified/summary_aggregate.tsv`
- `two_pass_unified/summary_by_seed.tsv`
- `two_pass_unified/summary_with_vanilla.tsv`

Question tested:

When the second-pass answer prompt is held fixed, which first-pass evidence
policy best reduces bias-aligned errors? Also, is the anti-prior instruction
itself sufficient even without first-pass evidence?

Setup:

- Two-pass experiment, except `direct` skips pass 1
- Seeds: `0,1,2,3,4`
- Reported bars: seed means with 95% CI over seeds
- Answer prompt variant: original only
- Conditions:
  - `direct`
  - `original`
  - `counterfactual`
  - `structured`
  - `verification`
- Answer modes:
  - `image_again`
  - `description_only`

Pass 1 prompts:

`direct`:

```text
No first pass.
```

`original`:

```text
Describe only the visual evidence needed to answer this question. Do not answer the question.
Question: {question}
```

`counterfactual`:

```text
The image may contradict common assumptions. Describe the visible evidence needed to answer this question. Do not infer what is normally true. Do not answer the question.
Question: {question}
```

`structured`:

```text
Return only this JSON object: {"visible_evidence": "...", "relevant_count_or_attribute": "...", "uncertainty": "low|medium|high"}. Do not answer the question.
Question: {question}
```

`verification`:

```text
Describe the visual evidence needed to answer the question. Then state what common assumption might be misleading. Do not give the final answer.
Question: {question}
```

Pass 2 prompt for non-direct conditions:

```text
{question}
Visual evidence: {visual_evidence}
Use the visual evidence, not what is normally true. Write 'Final answer:' followed by the answer.
```

Pass 2 prompt for direct:

```text
{question}
Use the visual evidence, not what is normally true. Write 'Final answer:' followed by the answer.
```

Vanilla references:

- `vanilla_image*`: earlier temp-0.7 seed-0 image baseline
- `vanilla_no_image*`: earlier temp-0.7 seed-0 no-image baseline
- Asterisk means these are not 5-seed unified runs.

Main observation:

Description-only pass 2 is the main useful intervention. Image-again variants
remain close to vanilla image bias. The direct description-only control also
strongly reduces bias, showing that the anti-prior instruction itself is a
large part of the effect. Original and structured description-only variants
preserve slightly more accuracy while keeping bias low.

## Original Attention Alpha 0.25 Ablation

Files:

- `attention_alpha025/summary.tsv`
- `attention_alpha025/summary.png`

Question tested:

Does adding a small positive constant to image-token attention logits reduce
VLMBias bias?

Setup:

- One-pass VLMBias answering
- `temperature=0.7`, `seed=0`
- Alpha: `0.25`
- Layer selections:
  - baseline/no boost
  - last 25%
  - last 50%
  - all layers

Mechanism:

Before softmax:

```text
attention_logits[..., image_token_positions] += alpha
```

Main observation:

The intervention increased image attention mass but did not reduce bias-aligned
fraction in the alpha=0.25 run.

## Attention Alpha Sweep

Files:

- `attention_alpha_sweep/vlmbias.png`
- `attention_alpha_sweep/vlmbias_summary_aggregate.tsv`
- `attention_alpha_sweep/vlmbias_summary_by_seed.tsv`
- `attention_alpha_sweep/naturalbench.png`
- `attention_alpha_sweep/naturalbench_summary_aggregate.tsv`
- `attention_alpha_sweep/naturalbench_summary_by_seed.tsv`

Question tested:

Does stronger image-token attention boosting reduce VLMBias bias-aligned errors,
and does it hurt NaturalBench?

Setup for the current checked-in results:

- Seed: `0`
- Benchmarks:
  - VLMBias: `data/vlmbias_400.jsonl`
  - NaturalBench: `data/naturalbench_100_groups.jsonl`
- Alpha values:
  - `0.05`
  - `0.1`
  - `0.25`
  - `0.5`
  - `1.0`
  - `2.0`
  - `3.0`
- Layer selections:
  - last 25%
  - last 50%
  - all layers
- Baseline: no layers boosted

The script defaults have since been expanded to seeds `0-9` and alphas
`0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0`, but those expanded
results are not yet reflected in these report artifacts unless new run outputs
are copied into `reports/`.

Current graph error bars:

- VLMBias attention graph: within-run 95% intervals over examples
- NaturalBench graph: within-run 95% intervals over calls/groups
- These are not seed-level confidence intervals because the checked-in sweep
  currently has only seed `0`.

Main observation:

Image attention mass increases strongly with alpha, confirming the mechanism is
active. However, VLMBias bias-aligned fraction does not improve; high alpha can
make it worse. NaturalBench does not show obvious degradation in seed 0, but
more seeds are needed for a robust claim.

## Rebuilding Charts

The chart script is:

```text
scripts/make_completed_experiment_charts.py
```

Run:

```bash
uv run python scripts/make_completed_experiment_charts.py
```

It regenerates:

- `two_pass_unified/results.png`
- `attention_alpha_sweep/vlmbias.png`
- `attention_alpha_sweep/naturalbench.png`
