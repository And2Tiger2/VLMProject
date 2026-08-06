# Mechanistic Heads Status

## Implemented

- Phase 0 audit and additive data/cache policy; prior gaze artifacts are untouched.
- Runtime-verified Qwen3 architecture and exact post-`W_O` head decomposition.
- Full-sequence candidate likelihood with Qwen multimodal metadata, token spans,
  capture hooks, exact lazy post-`W_O` projection, module patching,
  attention-map transplantation, five ablation modes, layer-sharded scans,
  context-bound resume checkpoints, hashes, and manifests.
- Deterministic SynDot, constant-eight, color/shape search, and original
  four-feature Waldo-like generators with exact masks and grouped splits.
- Counting VAP/head scan/controls/locked validation; point training, behavior,
  centroid/search/verification/distractor scans and ablations; MMMC preparation,
  MACI scan/ablation/detector/gating; signed VLMBias contrasts/validation and
  NaturalBench retention; unified 1,152-head atlas and PNG reporting.
- Dependency-safe `overnight-smoke` and explicit `overnight-all` Slurm DAGs.
  Full discovery arrays are serialized at four concurrent layer GPUs and all
  downstream validations use `afterok` gates. Cross-task matched validations
  now wait for the general-importance artifact they consume. Failed branches
  are killed as invalid dependencies, while an `afterany` final status job
  records failures and computationally pending work.

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
- The real Qwen instrumentation gate passed all 12 checks on an L40 at commit
  `9dbf9d0`. The repaired final revision remains pending a new SHA-bound gate.
- Smoke counting behavior and counting-head scan completed on the earlier DAG.
  Other smokes either failed safely or never ran; no full scientific stage ran.

## Passed

- The complete CPU unit/integration suite passes in the audited checkout.
- Both synthetic generator smokes write real schemas, images, masks, pairs,
  grouped splits, manifests, input/output hashes, and resume markers.
- Python compilation, every JSON config, all 26 task CLI imports/help routes,
  shell syntax, dry-run DAG generation, checkpoint context rejection, and
  output-hash validation.
- Resume contexts bind exact datasets, source images, MMMC split fingerprints,
  and adapter identities; smoke/full adapters are isolated; prototype point
  head-discovery families do not overlap validation or locked-test families.
- Installed Transformers source confirms raw per-head outputs are concatenated
  before one `o_proj`.

## Failed

- Earlier smokes found and the implementation now fixes: multimodal
  teacher-forcing length mismatch, custom causal-mask registration, eager
  backend comparison, dense projected-head OOMs, missing PEFT installation,
  mixed-row TSV output, and validation jobs scheduled before their inputs.
- No scientific calibration result exists yet; no replication success is
  claimed.

## Computationally pending

- Mandatory rerun of the one-GPU instrumentation smoke for the final Git SHA.
- Full synthetic dataset generation on Neuronic and Qwen behavioral calibration.
- VLMBias detail contrast unless external original factual images are supplied;
  semantic-prior and context contrasts are prepared.
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
