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
- Waldo-like relocation pairs now keep every distractor fixed, causal pairs
  carry exact target/decoy masks, and the four-candidate target slot is
  randomized deterministically instead of leaking through candidate order.
- Verification pairs hold every distractor fixed while changing one target
  feature; distractor-suppression pairs change exactly one decoy. Counting
  answer-code controls use per-group randomized codebooks and record distinct
  renderer seeds. Two additional deterministic 100-pair counting sets are
  scheduled for genuine cross-seed rank stability.
- Waldo-like scenes include same-canvas full-scene zoom transforms with masks,
  centers, boxes, and cells transformed together. Locked distractor validation
  measures generated target/decoy selections rather than naming a likelihood
  proxy a selection rate.
- Counting VAP/head scan/controls/locked validation; point training, behavior,
  centroid/search/verification/distractor scans and ablations; MMMC preparation,
  MACI scan/ablation/detector/gating; signed VLMBias contrasts/validation and
  NaturalBench retention; unified 1,152-head atlas and PNG reporting.
- Dependency-safe `overnight-smoke`, prepared-data-preserving
  `overnight-smoke-resume`, and explicit `overnight-all` Slurm DAGs. The smoke
  graph now exercises discovery, aggregation, controls, validations, detector,
  gating, reporting, and atlas consumers before any full scientific stage.
  Full discovery arrays are serialized at four concurrent layer GPUs and all
  downstream validations use `afterok` gates. Cross-task matched validations
  now wait for the general-importance artifact they consume. Failed branches
  are killed as invalid dependencies, while an `afterany` final status job
  records failures and computationally pending work.
- Cross-study scan serialization uses `afterany`, so failure in counting,
  point search, or MACI does not suppress scientifically independent later
  scans. Within-study datasets, calibrations, aggregation, and validation
  dependencies remain strict `afterok` gates.
- Every derived ranking, detector, validation, control table, and atlas input
  is bound to a completed manifest, exact file hash, and the current Git SHA;
  an older run can no longer be silently mixed into a new scientific result.
- MACI clean/conflict prompt pairs hold the exact conflict image fixed. The
  paired clean-row image is retained only as audited metadata, so the signed
  head scan changes wording rather than image content.
- Point-search conditions now use distinct matched instructions: direct and
  base conditions request only a count, the length control explicitly forbids
  coordinates, and Point-Answer conditions request the deterministic point
  syntax. This avoids teaching the direct controls a point-output task.
- Coordinate-centroid tracing now uses that same Point-Answer instruction
  while teacher-forcing coordinate answers; it no longer combines a direct
  count-only prompt with a point-form target sequence.
- Point LoRA training disables the generation cache and uses non-reentrant
  gradient checkpointing, bounding decoder activation memory on 48 GB GPUs
  without changing the configured effective batch size.
- Every Slurm entrypoint refuses tracked code/config changes, so a run cannot
  execute bytes that differ from the Git SHA written into its manifest.
- VLMBias semantic-prior contrasts cover all 400 source rows because they do
  not require a mask. Context and detail contrasts remain restricted to the
  114 manually reviewed mask rows, with one subject-grouped split shared by
  every emitted contrast.

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

- The complete CPU unit/integration suite passes in the audited checkout (263
  tests in the final local audit).
- Both synthetic generator smokes write real schemas, images, masks, pairs,
  grouped splits, manifests, input/output hashes, and resume markers.
- Smoke layer and example caps are enforced after config/CLI overrides; MMMC
  smoke selects complete pairs rather than an arbitrary raw-row prefix.
- Python compilation, every JSON config, all 30 standard task CLI imports/help routes
  plus the orchestration/report helpers,
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
- Long-prompt signed scans now offload captures to CPU and lazily project only
  selected heads/positions back onto the active device.
- Distractor-head discovery now uses the matched difference in ablation harm
  between high-decoy and low-decoy scenes, rather than scoring the high-decoy
  intervention alone.
- Smoke-only submissions now prepare bounded smoke datasets; full/resume
  submission still refuses undersized counting, point-search, or MMMC data.
- The old Slurm DAG is not resumable: its VAP, MACI, VLMBias, and point-training
  smoke failures correctly prevented all dependent scientific jobs.
- No scientific calibration result exists yet; no replication success is
  claimed.
- The requirement-by-requirement distinction between implemented code and
  pending empirical evidence is recorded in `../COMPLETION_AUDIT.md`.

## Computationally pending

- Mandatory rerun of the one-GPU instrumentation smoke for the final Git SHA.
- Full synthetic dataset generation on Neuronic and Qwen behavioral calibration.
- VLMBias detail contrast unless external original factual images are supplied;
  semantic-prior and context contrasts are prepared.
- Point-model LoRA pilots; optional full-weight training is opt-in.
- All head scans, scheduled cross-seed repeats, matched-control distributions, locked
  behavioral confirmations, conflict detector, NaturalBench retention, atlas,
  and result PNGs.

## Every paper deviation

See `../IMPLEMENTATION_PLAN.md`. In brief: Qwen3-VL replaces any source-paper
model mismatch; deterministic text points replace HTML boxes; LoRA is a
modified replication only; synthetic non-copyright Waldo-like data replaces
real Waldo in this phase; MACI reports both last-token and aligned-prefill
scopes; general scoring uses full answer sequences; the three VLMBias
contrasts are exploratory transfers. No replication-success claim is made.
