# Mechanistic Heads for Qwen3-VL-8B: Implementation Plan

Status: repository audit and implementation completed; GPU instrumentation,
calibration, full scans, training, and locked confirmations remain pending.

This workstream is additive. It must not modify or overwrite any artifact under
`segments/gaze_heads_qwen3_8b/runs/gaze_discovery_*` or any prior VLMBias,
NaturalBench, static-steering, or ROI-attention result.

## Phase 0 audit

### Existing code to reuse

- Model loading, dtype, and device placement:
  `adapters/qwen3_vl.py` and `adapters/qwen25_vl.py`. The mechanistic loader
  will use `AutoProcessor`, `Qwen3VLForConditionalGeneration`, the native
  Qwen3 chat template, `_resolve_torch_dtype`, and `_resolve_device_map`.
- Qwen3 attention implementation and layer discovery:
  `adapters/qwen3_vl_gaze_attention.py` and
  `adapters/qwen25_vl_gaze_attention.py`. The existing custom attention
  backend already exposes the raw `A_h V_h` tensor immediately before heads
  are concatenated and passed through `o_proj`.
- Spatial-token mapping and ROI masks:
  `adapters/qwen3_vl_roi_attention.py` and
  `vlm_eval/qwen3_roi_attention.py`.
- Existing gaze rankings and controls:
  `segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed{42,43,44}_merged/`.
  The seed-42 ranking remains the canonical fixed gaze ranking; seeds 43 and
  44 are stability references.
- VLMBias loading and exact current metrics:
  `vlm_eval/datasets.py`, `vlm_eval/answer.py`, and `vlm_eval/metrics.py`.
- NaturalBench loading and scoring: `vlm_eval/naturalbench.py`.
- Resume/config safety: `vlm_eval/gaze_resume.py`.
- Split and hash conventions: `vlm_eval/qwen3_attention_methods.py`.
- Slurm array, dependency, and preflight conventions:
  `scripts/submit_neuronic_qwen3_attention_methods.py` and the associated
  `scripts/slurm_neuronic_qwen3_*` launchers.
- Existing kernel, dataset, resume, submission, and stage-validation tests in
  `tests/`.

### Verified architecture facts

The installed Transformers Qwen3-VL implementation defines each language
attention block as follows:

1. `q_proj`, `k_proj`, and `v_proj` produce per-head tensors;
2. the attention backend returns `A_h V_h` with shape
   `[batch, query, heads, head_dim]`;
3. heads are reshaped into one concatenated width;
4. `o_proj: Linear(num_heads * head_dim, hidden_size)` is applied once.

Therefore, for head `h`, the exact bias-free projected contribution is:

```text
raw_h = attention_backend_output[:, :, h, :]
W_h = o_proj.weight[:, h*head_dim:(h+1)*head_dim]
projected_h = raw_h @ W_h.T
```

The sum of all projected heads equals the attention output minus the single
`o_proj.bias`; the bias must be added once, never once per head. The runtime
loader will assert the actual number of language layers and heads from
`model.config.text_config` and the instantiated modules. The expected values
are 36 layers and 32 heads, but neither value will be used as an unchecked
hard-coded architecture assumption.

### New implementation files

Reusable library code will live in `vlm_eval/mechanistic_heads/` rather than
being duplicated across study scripts:

- `schema.py`: `PairedExample`, serialization, and pair/split validation.
- `config.py`: uniform CLI options, JSON configuration, and safe output rules.
- `reproducibility.py`: Git/environment/seeds/input-output hashes and resume
  markers.
- `qwen3_runtime.py`: exact existing loader/chat-template wrapper and runtime
  architecture validation.
- `token_spans.py`: system, visual, image-end, prompt, last-prefill, and answer
  span tracing with assertions.
- `likelihood.py`: full answer-sequence teacher-forced log likelihood.
- `capture.py`: layer, attention, MLP, raw-head, attention-map, and projected
  head capture.
- `patching.py`: residual/module/head/map patching and alignment checks.
- `ablations.py`: zero, mean, resample, image-attention knockout, and scaling.
- `scan.py`: layer-batched head scans with configurable microbatching and
  resume checkpoints.
- `controls.py`: repeated layer/attention/norm/entropy/gaze/importance-matched
  control draws.
- `splits.py`: group-aware deterministic split construction and leakage gates.
- `atlas.py`: unified per-head table and overlap/correlation outputs.

Study-specific code, configuration, manifests, reports, and small generated
fixtures will use the requested layout under
`segments/mechanistic_heads_qwen3_8b/`. User-facing CLIs will use the requested
names under `scripts/` and delegate to the shared library.

### Uncertain details that require instrumentation smoke tests

- Exact token IDs and boundaries for system/role/image-end tokens under the
  pinned processor revision. These will be inferred from native chat-template
  encodings and asserted rather than guessed.
- Whether the installed checkpoint has `o_proj.bias`; the code handles either
  case and the reconstruction test decides at runtime.
- Cache behavior and key lengths under generation. Prefill path patching is
  primary; cached decode equivalence will be checked separately.
- Attention backend equivalence. Mechanistic capture requires eager/custom
  attention; logits must be compared with the repository's normal backend
  before scientific runs are allowed.
- MMMC Hub schema and pairing multiplicity beyond fields documented in its
  dataset card. `prepare_mmmc.py` will audit actual rows and refuse ambiguous
  candidate construction.
- Source-paper prompt details unavailable or not rendered reliably (notably
  point-output HTML boxes). Every deviation is listed below and in reports.

### Expected GPU and memory bottlenecks

- Qwen3-VL-8B model weights require approximately 16 GB in FP16/BF16 before
  activations. Full prefill attention maps scale quadratically with sequence
  length and are the largest transient allocation.
- Capturing every layer's raw heads, maps, projected heads, residuals, and MLP
  outputs simultaneously is unsafe. Capture will stream one layer or one
  module family at a time and immediately move compact summaries to CPU.
- A 32-head layer scan can batch counterfactual patches, but sequence length
  and answer length determine whether 32 fits. `--head-microbatch` will fall
  back to 16/8/4/1 without changing semantics.
- Full answer-sequence likelihood costs one teacher-forced pass per candidate
  sequence (or a padded candidate batch). It is more expensive but required;
  first-token logits are diagnostic only.
- Attention-map transplantation requires eager probabilities. It will be
  limited to aligned prefill positions and selected layers/heads.
- Full LoRA or full-weight search training is never part of smoke mode. The
  optional paper-style full-weight configuration is expected to dominate the
  compute budget and is not launched automatically.

## Study implementation sequence

1. Build and unit-test the common schema, likelihood, spans, exact `o_proj`
   decomposition, patching, ablations, batching, and manifests.
2. Generate deterministic SynDot and constant-complexity datasets. Run only
   CPU generator tests and schema smokes locally; publish Neuronic smoke/full
   commands without launching full scans.
3. Generate deterministic color/shape and non-copyright Waldo-like datasets,
   then add matched Direct/Point/Shuffled-Point training configs and causal
   search/verification scans.
4. Download and audit MMMC with `datasets.load_dataset`; materialized data and
   caches remain ignored. Add signed MACI scans and guarded ablations.
5. Build group-safe VLMBias semantic/context/detail contrasts using the current
   loader and definitions; preserve a locked subject/domain-held-out split.
6. Join all 1,152 runtime-verified language heads into the atlas, render TSV,
   PNG, and Markdown reports, and update `reports/STATUS.md` with executed versus
   pending work.

## Exact deviations from the source papers

### Counting Circuits (arXiv:2603.18523)

- Qwen3-VL-8B-Instruct is used throughout this repository; results are a
  methods-based reproduction unless the paper used the identical checkpoint,
  processing revision, prompt, and intervention scope.
- General scores use full candidate-sequence likelihood in addition to the
  paper-style answer-token logit difference.
- Constant-complexity scenes, sham edits, target relocation, renderer seeds,
  and randomized answer codes are explicit modified-replication controls.
- Batched post-`W_O` head patching is an engineering optimization; serial
  equivalence is a mandatory test.

### Binding Visual Features Point by Point (arXiv:2605.25427)

- The paper's HTML-style point boxes are replaced with deterministic plain
  text, e.g. `point=(037,064); object=green-T; answer=present`.
- LoRA is a modified-replication pilot. Only the optional full-weight AdamW
  configuration can approach the paper training setup, and it will be labeled
  paper-faithful only after all remaining differences are verified.
- Visual-key-only transplantation is used when full key sequences differ; the
  transplanted visual slice is renormalized together with unchanged recipient
  nonvisual mass and this rule is recorded per patch.
- The Waldo-like target is synthetic and non-copyright. Real Waldo is a later,
  locked transfer phase and is never downloaded automatically.

### MACI (arXiv:2605.19250)

- Primary scope is the last prefill token; all exactly aligned prefill
  positions are a separately reported secondary analysis because the
  accessible method description does not fully resolve token scope.
- General candidate scoring uses complete answer sequences. A separate
  faithful subset reports single-token candidate coverage.
- MMMC candidates are retained only when factual and hallucinated answers are
  explicit and unambiguous; exclusions are reported rather than repaired by
  invention.
- Qwen3-specific `k` sweeps are validation-selected modified replications and
  remain separate from the paper-style top-30 driving/top-40 resisting result.

### VLMBias adaptation

- Semantic-prior, context, and detail contrasts are new exploratory transfers,
  not claims from the three source papers.
- Existing all-400 analyses are not reused as confirmation. Final effects need
  a subject/domain-held-out split with selection and reporting separated.

## Run and reporting gates

- Every scientific CLI uses `--config`, `--output-dir`, `--seed`, `--smoke`,
  `--resume`, `--limit`, and opt-in `--overwrite`.
- Smoke mode uses at most eight examples and at most two layers unless checking
  global indexing; it writes the real schema and never performs substantive
  training.
- A scientific run must pass the required identity, reconstruction, no-op,
  batched/serial, likelihood, span, attention normalization, determinism,
  leakage, backend, cache, and manifest tests.
- No head family is declared successful from attention mass alone or from the
  same examples used to select it.
- Reports use only the labels: instrumentation smoke test, methods-based
  reproduction, modified replication, paper-faithful replication, failed
  calibration, exploratory transfer, and locked confirmation.
