# Qwen3-VL-8B Mechanistic Head Suite

This additive workstream distinguishes count-carrying, visual-search,
fine-detail verification, distractor-suppression, hallucination-driving, and
hallucination-resisting heads. It reuses the repository's exact Qwen3 loader,
native chat template, dtype/device rules, custom eager-attention interface,
VLMBias metrics, NaturalBench evaluator, gaze rankings, and manifest style.

No existing gaze-head artifact is modified. External data, generated images,
model weights, checkpoints, and activation stores are ignored by Git.

## Current status

- Phase 0 audit: implemented and documented in `IMPLEMENTATION_PLAN.md`.
- Common projected-head/map patching library: implemented and CPU-tested.
- Deterministic generators, exact masks, all behavioral tasks, locked
  validators, matched controls, and report renderers: implemented and
  smoke-tested where no model is required.
- GPU calibration: not run in this checkout.
- MMMC: downloaded with the exact official
  `datasets.load_dataset("ustc-zhangzm/MMMC")` call; the local audit found
  13,446 unambiguous object-conflict pairs and a 256/512/12,678 grouped
  prototype/validation/locked-test split.
- Full scientific scans and training: never launched automatically.

The suite must not be described as a successful replication until its GPU
calibrations, matched controls, and locked confirmations pass.

## Environment

```bash
uv sync --extra dev --extra qwen --extra mechanistic
```

On Neuronic, continue using the repository-cached environment prepared by the
existing Qwen3 setup. The Slurm jobs default to offline model loading so a GPU
is not held while downloading weights.

## Prepare data

Generate full SynDot, constant-complexity, color/shape search, synthetic
Waldo-like, and VLMBias contrast data locally on the login node:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh prepare-synthetic
```

Download and audit official MMMC separately:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh download-mmmc
```

MMMC's official images remain in
`segments/mechanistic_heads_qwen3_8b/data/mmmc_cache/`; only compact audit and
pair metadata are produced. Object-conflict pairing uses this explicit rule:
the conflict row's provided answer is factual, while the paired clean row's
provided answer is the hallucinated/prior candidate. Empty, identical, or
ambiguous pairs are excluded rather than repaired.

For small generator checks without a GPU:

```bash
uv run python scripts/generate_counting_data.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/counting_data.json \
  --output-dir /tmp/qwen3-counting-smoke --seed 17 --smoke

uv run python scripts/generate_point_search_data.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/point_search_data.json \
  --output-dir /tmp/qwen3-search-smoke --seed 17 --smoke
```

## Mandatory instrumentation gate

Run this first on one GPU:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh instrumentation smoke
```

It checks the loaded architecture, identity patch, exact post-`W_O`
reconstruction including bias, exact self no-op, serial/batched tensor patch
agreement, full teacher forcing, token spans, normalized maps, deterministic
generators, grouped splits, normal/custom backend equivalence, cached/uncached
scoring, and complete manifests. Non-smoke scientific runners refuse to start
without a valid instrumentation report.

The expected architecture is 36 language layers by 32 language heads, but this
is read and asserted from `model.config.text_config` and instantiated modules.

## Smoke commands

Each command submits one bounded instrumentation smoke: at most eight examples
and two layers. Point-specific commands require the Point-Answer adapter from
the training step below and refuse to silently scan the base model.

```bash
bash scripts/run_neuronic_mechanistic_heads.sh counting-vap smoke
bash scripts/run_neuronic_mechanistic_heads.sh counting-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh search-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh verification-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh distractor-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh vlmbias-heads smoke
```

Every command supports `--config`, `--output-dir`, `--seed`, `--smoke`,
`--resume`, `--limit`, and opt-in `--overwrite` at the underlying Python CLI.

### One-command overnight submission

Submit data preparation, mandatory instrumentation, four matched Point-Answer
training smokes, and every study smoke as one dependency-safe DAG:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh overnight-smoke
```

The explicit full-suite command first runs all of those gates and then submits
the matched LoRA conditions, behavioral calibrations, all 36-layer discovery
arrays, aggregations, matched controls, locked validations, detector/gated
interventions, reports, and the unified atlas:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh overnight-all
```

`overnight-all` is deliberately explicit because it is a large run. Discovery
arrays are serialized and use `0-35%4`, so no more than four layer-scan GPUs are
requested simultaneously; independent behavioral/training jobs may overlap.
Queue time and full Point-Answer training mean the complete DAG is not
guaranteed to finish in one night. Any failed smoke causes its dependent full
jobs to remain unrun through Slurm's `afterok` dependencies. The complete job
receipt is written to
`segments/mechanistic_heads_qwen3_8b/runs/overnight_submission.json`.

If preparation completed but instrumentation failed, use the validated cached
data without repeating the multi-hour download/generation stage:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh overnight-all-resume
```

This refuses to submit unless every required generated pair file and a valid
MMMC audit already exist.

## Full-run commands

Full runs are opt-in and intentionally separate. Do not submit them until the
corresponding behavioral calibration and instrumentation smoke are valid.

```bash
bash scripts/run_neuronic_mechanistic_heads.sh counting-vap full
bash scripts/run_neuronic_mechanistic_heads.sh counting-heads full
bash scripts/run_neuronic_mechanistic_heads.sh search-heads full
bash scripts/run_neuronic_mechanistic_heads.sh verification-heads full
bash scripts/run_neuronic_mechanistic_heads.sh distractor-heads full
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads full
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads-aligned full
bash scripts/run_neuronic_mechanistic_heads.sh vlmbias-heads full
```

Full scan commands automatically submit 36 one-layer array tasks with at most
four GPUs active, followed by a CPU aggregation job that refuses missing,
duplicated, misnumbered, or wrong-architecture shards. Head candidates within
a layer are also model-microbatched; serial equivalence is tested.

After discovery scans, build cross-task importance, matched controls, and
locked confirmations:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh general-importance
bash scripts/run_neuronic_mechanistic_heads.sh counting-controls
bash scripts/run_neuronic_mechanistic_heads.sh counting-validation full
bash scripts/run_neuronic_mechanistic_heads.sh point-ablation full
bash scripts/run_neuronic_mechanistic_heads.sh maci-ablation full
bash scripts/run_neuronic_mechanistic_heads.sh maci-detector full
bash scripts/run_neuronic_mechanistic_heads.sh maci-gated full
bash scripts/run_neuronic_mechanistic_heads.sh maci-confirm full
bash scripts/run_neuronic_mechanistic_heads.sh vlmbias-validation full
```

## Point-search training

Matched conditions are `base`, `direct_answer`, `direct_length_matched`,
`point_answer`, and `shuffled_point_answer`. A LoRA pilot is always labeled a
modified replication:

```bash
uv run python scripts/train_point_search.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/point_search_lora.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/checkpoints/point-answer-lora \
  --condition point_answer --seed 260525427 --smoke
```

The optional full-weight config uses AdamW-compatible Trainer defaults,
learning rate `1e-5`, cosine scheduling, and 200 warmup steps. It is not called
paper-faithful merely because that config was used.

Run the same behavioral evaluation for every trained condition before any head
scan (substitute each checkpoint and condition in turn):

```bash
uv run python scripts/evaluate_point_search.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/point_search_behavior.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/runs/point_behavior/point-answer \
  --checkpoint segments/mechanistic_heads_qwen3_8b/checkpoints/point-answer-lora \
  --condition point_answer --seed 260525427 --smoke

bash scripts/run_neuronic_mechanistic_heads.sh waldo-behavior smoke
bash scripts/run_neuronic_mechanistic_heads.sh point-centroids smoke
```

## MACI validation and detector

After a prototype signed scan:

```bash
uv run python scripts/run_maci_ablation.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/maci_ablation.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/runs/maci_ablation \
  --seed 260519250 --device-map cuda --smoke

uv run python scripts/train_maci_conflict_detector.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/maci_detector.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/runs/maci_detector \
  --seed 260519250 --device-map cuda --smoke
```

The detector averages last-prefill raw head activations over validated
resisting heads, fits L1 logistic regression on prototype examples, selects its
threshold by validation F1, and reports AUROC/AUPRC/F1/intervention rate on the
locked split.

The MACI validation sweep and locked confirmation are deliberately separate.
Use `maci_ablation.json` for validation-selected k and
`maci_ablation_locked.json` for the paper-style locked result and full matched
control distributions.

## Atlas and reports

Once score TSVs exist:

```bash
uv run python scripts/render_mechanistic_head_reports.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/head_atlas.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/reports/atlas \
  --seed 0
```

The renderer writes a 1,152-row TSV, top-k overlaps, PNG heatmaps, gaze-score
scatterplots, per-family PNG heatmaps, rank correlations, layer distributions,
a clustered atlas, a double-dissociation table, a run manifest, and `STATUS.md`.
Missing studies remain blank;
the renderer never fabricates scores.

## Real Waldo is locked

No real Waldo data is downloaded in this phase. The optional later command and
mandatory license/page-split checks are documented in
`point_search/REAL_WALDO_TRANSFER.md`.

## Tests

```bash
uv run pytest -q
```

Current CPU tests (174 passing in this implementation checkout) cover all common tensor identities, sequence-scoring math,
span invariants, normalization, deterministic generation, grouped leakage,
control sampling, and reproducibility schemas. GPU checks are written by the
mandatory instrumentation job and are not claimed as passed locally.
