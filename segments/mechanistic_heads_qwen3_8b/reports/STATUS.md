# Mechanistic Heads Status

## Implemented

- Phase 0 audit and additive data/cache policy; prior gaze artifacts are untouched.
- Runtime-verified Qwen3 architecture and exact post-`W_O` head decomposition.
- Full-sequence candidate likelihood, token spans, capture hooks, exact head and
  module patching, attention-map transplantation, five ablation modes,
  layer-sharded scans, resume checkpoints, hashes, and manifests.
- Deterministic SynDot, constant-eight, color/shape search, and original
  four-feature Waldo-like generators with exact masks and grouped splits.
- Counting VAP/head scan/controls/locked validation; point training, behavior,
  centroid/search/verification/distractor scans and ablations; MMMC preparation,
  MACI scan/ablation/detector/gating; signed VLMBias contrasts/validation and
  NaturalBench retention; unified 1,152-head atlas and PNG reporting.
- Dependency-safe `overnight-smoke` and explicit `overnight-all` Slurm DAGs.
  Full discovery arrays are serialized at four concurrent layer GPUs and all
  downstream validations use `afterok` gates.

## Run

- Repository audit.
- Local CPU unit/integration suite.
- Local smoke generation for counting and point/Waldo-like data.
- Official MMMC was downloaded with
  `datasets.load_dataset("ustc-zhangzm/MMMC")` and audited locally: 36,000
  train rows, 4,000 test rows, and 13,446 unambiguous object-conflict pairs.
  The grouped split contains 256 prototype, 512 validation, and 12,678 locked
  test image IDs. One identical-candidate pair was excluded; pairing success
  was 99.993%. External images/cache remain ignored by Git.
- No GPU scan, model evaluation, or training was launched by implementation.

## Passed

- 174 CPU tests.
- Both synthetic generator smokes write real schemas, images, masks, pairs,
  grouped splits, manifests, input/output hashes, and resume markers.
- Python compilation, JSON config parsing, CLI help, and shell syntax checks.
- Installed Transformers source confirms raw per-head outputs are concatenated
  before one `o_proj`.

## Failed

- No scientific calibration has failed yet because no GPU scientific run has
  been performed. No replication success is claimed.

## Computationally pending

- Mandatory one-GPU instrumentation smoke, including real-model backend/cache
  equivalence and runtime 36×32 assertion.
- Full synthetic dataset generation on Neuronic and Qwen behavioral calibration.
- MMMC Qwen-tokenization audit after the model processor is available (the
  completed metadata/pairing audit intentionally used `--skip-tokenization`).
- Point-model LoRA pilots; optional full-weight training is opt-in.
- All head scans, cross-seed repeats, matched-control distributions, locked
  behavioral confirmations, conflict detector, NaturalBench retention, atlas,
  and result PNGs.

## Every paper deviation

See `../IMPLEMENTATION_PLAN.md`. In brief: Qwen3-VL replaces any source-paper
model mismatch; deterministic text points replace HTML boxes; LoRA is a
modified replication only; synthetic non-copyright Waldo-like data replaces
real Waldo in this phase; MACI reports both last-token and aligned-prefill
scopes; general scoring uses full answer sequences; the three VLMBias
contrasts are exploratory transfers. No replication-success claim is made.
