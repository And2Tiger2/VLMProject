# Mechanistic-head suite completion audit

This document distinguishes implementation coverage from empirical completion.
An item marked **implemented** has code and local contract tests. An item marked
**GPU pending** must still produce a current-SHA manifest on Neuronic before it
can support a scientific claim. Nothing below is a claim that a functional head
family has been found.

## Common infrastructure

| Requirement | Status | Evidence / gate |
|---|---|---|
| Runtime architecture discovery | Implemented; L40 verified | `qwen3_runtime.py`; instrumentation asserts 36 language layers, 32 query heads, 8 KV heads, and 128 head dimensions from the loaded model. |
| Full answer-sequence likelihood | Implemented; L40 verified | `likelihood.py`; manual teacher-forcing and cached/uncached checks. |
| System, visual, image-end, user, final-prompt, and answer spans | Implemented; L40 verified | `token_spans.py`; instrumentation span assertions. |
| Layer, attention, MLP, raw `A_hV_h`, maps, and exact post-`W_O` capture | Implemented; L40 verified | `capture.py`, `patching.py`; reconstruction gate includes projection bias once. Projected contributions are lazy to avoid 10–40 GiB derived tensors. |
| Exact projected-head patching | Implemented; L40 verified | Identity, self-subtraction, and batched/serial gates. |
| Attention-map transplantation | Implemented; GPU pending scientifically | Equal-length replacement or explicit equal-cardinality visual slices; recipient visual mass is preserved and maps are renormalized. Normalization checks use a dtype-aware tolerance so valid bfloat16 maps pass while malformed maps are rejected. |
| Zero, mean, resample, image knockout, and scaling ablations | Implemented; GPU pending scientifically | `ablations.py` and locked validation runners. |
| Layer arrays, head microbatching, resume | Implemented | 36 layer shards with at most four concurrent scan GPUs; sequence-aware microbatch bound; checkpoints bind config, inputs, seed, layers, adapter, and Git SHA. |
| Twelve mandatory instrumentation checks | Implemented; passed on L40 at `9dbf9d0` | A fresh pass is required after every code revision before scientific jobs can start. |
| Reproducibility records | Implemented | Every completed run records config, Git SHA, environment, seeds, input/output hashes, and resume metadata. Prepared-data manifests also hash generator source code. |
| Clean-code execution | Implemented | Every Slurm entrypoint refuses tracked worktree changes, preventing manifests from labeling uncommitted code as the recorded Git SHA. |
| Standard CLI and smoke policy | Implemented | All scientific CLIs expose the standard arguments; smoke limits to at most eight examples and one or two layers, refuses config/CLI expansion beyond the layer cap, selects complete MMMC pairs, and writes production schemas. |

## Study A — counting circuits

| Requirement | Status | Evidence / limitation |
|---|---|---|
| SynDot 336×336, 28×28 grid, radius-4 dots, counts 1–10, 4,000/2,000 | Implemented; data generated once | `generate_counting_data.py`; manifest/count gates. |
| Baseline accuracy, MAE, RMSE, off-by-one | Implemented; full GPU pending | `run_counting_behavior.py`; full VAP/head scans require its current-SHA calibration report to meet the configured accuracy and output-validity thresholds. |
| 100 controlled bidirectional mechanistic pairs | Implemented | Paired manifest with grouped split and exact images. |
| Layerwise VAP scopes and attention/MLP/residual modules | Implemented; smoke rerun pending | `run_counting_vap.py`; missing-system rows use a union TSV schema instead of crashing. |
| All-head exact post-`W_O` causal scan and diagnostics | Implemented; GPU pending | Symmetric bidirectional full-sequence margin score plus attention, norm, entropy, and gaze diagnostics. |
| Fixed-eight color and shape controls | Implemented | Exact target/distractor/change masks; sham edits, relocation, multiple deterministic renderer seeds, and truly randomized answer codebooks. |
| Top 10/25/50, gaze, repeated matched controls, low-score controls | Implemented; GPU pending | At least 20 draws for each requested matching family. |
| Necessity/sufficiency/reverse/sham/position/code/renderer validation | Implemented; GPU pending | Zero, mean, resample, donor and reverse patches on locked examples. |
| Split-half and cross-seed stability | Implemented; cross-seed GPU runs pending | Two independently rendered 100-pair repeat datasets and two 36-layer repeat scans are scheduled. The report requires all three pairwise rank correlations plus split-half stability before declaring count heads. |
| Required TSVs and plots | Implemented; rendering pending | VAP, scores, controls, overlap, heatmap, gaze scatter, stability, and double-dissociation outputs. |

## Study B — point search and Waldo-like verification

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Six colors × six shapes; 50 objects; disjoint conjunctions; 2,000 train; six OOD counts × 50 | Implemented | Uses the paper's exact color names, glyph set, six train conjunctions, and ten test conjunctions; generator stores centers, boxes, classes, masks, template IDs, and grouped splits. |
| Base/direct/length-matched/point/shuffled-point training | Implemented; PEFT smoke pending | Matched scenes use condition-specific instructions: direct asks only for a count, the length control forbids coordinates, and Point-Answer requests coordinates. LoRA is labeled modified replication; optional full-weight AdamW/cosine/200-warmup config is documented and not auto-launched. LoRA disables cache and uses non-reentrant gradient checkpointing to bound activation memory. |
| Coordinate-token centroid tracing | Implemented; GPU pending | Per-layer denoised centroids and point RMSE; the trace uses the same Point-Answer prompt as training/evaluation. |
| Per-head attention transplantation and top/bottom/random validation | Implemented; GPU pending | Uses complete emitted localization-sequence likelihood and explicit visual-slice normalization. |
| Non-copyright four-feature target and strong distractors | Implemented | Position, scale, full-scene zoom, clutter, similarity, occlusion, background, presence, and prompt wording vary; images, masks, centers, boxes, and cells remain aligned. |
| Localization, visible-grid, normalized point, verification, four-candidate, presence tasks | Implemented; behavioral calibration pending | Candidate order is deterministically randomized to prevent target-slot leakage. |
| Search relocation pairs | Implemented | Only the identical target moves; all distractors remain pixel/metadata matched. |
| True versus impostor verification pairs | Implemented | Location and every distractor are fixed; only one target feature changes. |
| Distractor-suppression pairs | Implemented | Target and background are fixed; exactly one distractor becomes a strong incorrect-binding decoy. Discovery uses the matched high-decoy minus low-decoy difference in correct-versus-decoy ablation harm; locked generation is required before interpreting selection rates. |
| Gaze/count/OCR/random/matched controls and double dissociation | Implemented; GPU pending | Cross-task head sets are applied to all three locked tasks with at least 20 matched draws. Locked claims fail unless own-task effects beat bottom, cross-task, and jointly matched controls; distractor claims additionally require increased generated decoy selections. |
| Real Waldo transfer | Optional loader only; not executed | Documented Kaggle command, license/class/page audit, page-level split refusal, boxes, and six zoom/crop conditions. No real images are downloaded automatically. |

## Study C — MACI, VLMBias, and the atlas

| Requirement | Status | Evidence / limitation |
|---|---|---|
| MMMC download and audit | Executed once; reusable manifest must match current code | 40,000 rows audited; 13,446 unambiguous object-conflict pairs; 256/512/12,678 grouped prototype/validation/locked examples; candidates and exclusions recorded. |
| Same-image clean/conflict prompt contrast | Implemented | Both runs use the conflict image; the paired clean image remains provenance only. |
| Full-sequence hallucination advantage | Implemented | Single-token coverage is reported rather than assumed. |
| Signed last-prefill and aligned-prefill scans | Implemented; GPU pending | Unequal aligned-prefill lengths are explicitly excluded, never truncated. |
| Top-30 driving, top-40 resisting, joint-30, five random seeds, k sweep | Implemented; GPU pending | Full validation refuses to run unless the requested number of correctly signed heads exists. |
| Sign/rank stability and locked causal directions | Implemented; GPU pending | Failed gates produce `failed calibration`, not a head-family claim. |
| L1 conflict detector and gating comparisons | Implemented; GPU pending | Threshold selected on validation F1; AUROC/AUPRC/F1 and never/always/confidence/random-budget policies are reported. The locked gate requires detector benefit over never and an exactly budget-matched random policy. |
| VLMBias semantic-prior/context/detail contrasts | Semantic/context implemented; detail data pending | Semantic-prior covers all 400 source rows without masks; context/detail use the 114 reviewed-mask rows and share a subject-grouped split. Detail is emitted only when external factual originals exist and processed visual spans align exactly. Rankings stay contrast-specific. |
| Locked role-aware VLMBias validation and NaturalBench retention | Implemented; GPU pending | Reports requested rates, likelihood margin, four transition types, intervention rate, matched controls, and NaturalBench metrics. A result is labeled failed calibration unless all role-aware directions, matched controls, bias-transition checks, and NaturalBench retention gates pass. |
| 1,152-head atlas and plots | Implemented; rendering pending | Joins gaze, count, search, verification, distractor, MACI, three VLMBias scores, norm, and complete-sequence pre-final-norm direct attribution; missing families remain explicitly pending. |

## Operational audit findings

- The first Neuronic DAG failed at teacher-forcing metadata alignment.
- The second exposed invalid eager/custom comparisons and numerical tolerances.
- The current instrumentation revision passed all twelve checks on an L40.
- Subsequent smoke jobs exposed three independent issues now fixed in code:
  union-field TSV output, dense post-`W_O` capture OOM, and missing PEFT setup.
- The smoke DAG now includes every downstream artifact consumer, so a fresh
  smoke validates controls, locked-validation mechanics, detector/gating,
  reports, and atlas assembly—not only the discovery scans that feed them.
- The full DAG now separates scientific prerequisites from resource ordering:
  independent scan families are serialized with `afterany`, while true inputs
  and calibration gates use `afterok`. One study can fail without erasing the
  diagnostic value of later independent studies.
- A new current-SHA instrumentation plus downstream smoke DAG is mandatory.
  Full scans must remain dependency-blocked until every smoke succeeds.

## Remaining empirical gates

1. Run a fresh current-SHA instrumentation and all downstream smokes.
2. Calibrate counting and point behavior before full causal scans.
3. Run full discovery scans and required matched controls.
4. Run cross-seed count repeats before naming count heads.
5. Run locked validations; label failed calibrations honestly.
6. Render the atlas only from current-SHA, hash-verified artifacts.
7. Supply external factual VLMBias originals if the detail contrast is desired.
