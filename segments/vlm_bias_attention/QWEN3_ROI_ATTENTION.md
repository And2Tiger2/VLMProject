# Qwen3-VL localized ROI attention experiment

## Question

Does increasing attention to the small image region altered by VLMBias improve
Qwen3-VL-8B-Instruct more reliably than increasing attention to the entire
image, to an equally sized wrong region, or in non-gaze heads?

The experiment uses the manually reviewed subtraction masks in
`data/vlmbias_roi_masks_v1`. There are 141 unique accepted image/mask groups
covering 160 rows from the fixed 400-row VLMBias slice. Shared prompt variants
for one edited image remain in the same split.

## Intervention

For a selected language-attention head, query position `q`, and key token `k`,
the modified logit is

```text
z'[q,k] = z[q,k] + alpha * 1[k is a selected ROI image token]
attention[q,:] = softmax(z'[q,:])
```

The positive bias is applied before softmax, to every prefill and generated-token
query. Text keys and non-ROI image keys are unchanged. The baseline has zero
selected heads and `alpha = 0`.

Qwen dynamically resizes each image. At generation time, the adapter reads that
example's exact `image_grid_thw` and the processor's `spatial_merge_size`, BOX
resizes the binary pixel mask to the merged visual grid, and selects every token
with at least 5% ROI occupancy. A nonempty pixel mask is guaranteed to select at
least one token. The resulting token count must exactly equal Qwen's image
placeholder count or the run fails.

## Fixed data split

- Smoke: 8 reviewed mask groups (9 prompt rows), sampled within development.
- Development: 50 mask groups (58 prompt rows), stratified by VLMBias topic.
- Confirmation: the other 91 mask groups (102 prompt rows).
- No edited image or mask group is shared between development and confirmation.
- Decoding is deterministic and greedy; the seed affects only reproducible
  control-head or shifted-mask construction.

## Stages

### 1. Smoke mechanics

Four conditions on the smoke split:

- baseline;
- ROI, gaze top 50, `alpha = 2`;
- full image, gaze top 50, `alpha = 2`;
- ROI, layer-matched random 50 heads, `alpha = 2`.

This checks the ROI, full-image, gaze-head, and random-head paths plus telemetry.

### 2. Alpha tuning

Seven conditions on the 50-group development split:

- baseline;
- ROI, gaze top 50, `alpha in {0.5, 1, 2, 5}`;
- full image, gaze top 50, `alpha = 2`;
- spatially shifted ROI, gaze top 50, `alpha = 2`.

The full-image control tests whether localization matters. The shifted-mask
control has the same binary area but targets a different image location. Only
the four true-ROI gaze conditions are eligible to lock the alpha.

### 3. Head-family sweep

At the development-selected alpha, run baseline plus these 11 ROI conditions:

- globally ranked gaze top 10, 50, and 100;
- early-, middle-, and late-layer gaze top 50;
- layer-matched random top 50 with seeds 55 and 56;
- layer-matched random top 100 with seed 55;
- layer-matched lowest-gaze-score top 50;
- paper-style uniformly random top 50 with seed 55.

Only the globally ranked or layer-band gaze sets can be locked. Random and
low-score sets are mechanistic controls.

### 4. Held-out confirmation

On all 91 held-out mask groups, compare:

- baseline;
- the locked ROI alpha/head condition;
- the same heads and alpha applied to the full image;
- the same heads and alpha applied to the shifted mask;
- the true ROI with a fresh layer-matched random head set (seed 57).

The primary result is the locked ROI condition's change in VLMBias accuracy and
bias-aligned fraction on this held-out split. The three controls indicate whether
any change is specifically due to the correct region and discovered gaze heads.

## Selection and metrics

Development candidates must lose no more than 2 percentage points of accuracy
and increase invalid outputs by no more than 3 points versus baseline. Among
qualified eligible conditions, selection maximizes accuracy, then minimizes the
bias-aligned fraction and invalid rate. If none qualify, the pipeline continues
with the best diagnostic candidate and emits a warning; that fallback is not a
positive result.

Each condition records:

- accuracy, bias-aligned fraction/error rate, and invalid rate;
- exact selected `(layer, head)` pairs;
- the exact sparse target-token indices, plus mean/min/max ROI token count and
  target-token fraction;
- total and selected-head attention mass assigned to the target tokens;
- git commit, Slurm job/task IDs, node, config, and raw predictions.

## Neuronic execution

The portable 141-mask runtime bundle is tracked at
`assets/vlmbias_roi_masks_v1_runtime.tar.gz`. The wrapper automatically extracts
it into the ignored data directory before preflight. The fixed VLMBias slice and
images plus the gaze ranking and `gaze_scores.npy` must already be present on
Neuronic; the submitter validates every required input before submitting.

```bash
bash scripts/run_neuronic_qwen3_roi_attention.sh dry-run
bash scripts/run_neuronic_qwen3_roi_attention.sh submit
bash scripts/run_neuronic_qwen3_roi_attention.sh verify
```

The full submission is a dependency chain, so a failed stage prevents later
model-selection stages from running. GPU arrays default to four concurrent jobs
(two for smoke), each requesting one GPU, 64 GB RAM, and six hours. Every example
calls the cluster overheat hook immediately before and after generation.

Outputs are written under:

- manifests: `segments/vlm_bias_attention/experiments/qwen3_roi_attention_v1/`
- predictions: `segments/vlm_bias_attention/runs/qwen3_roi_attention_v1/`
- aggregate reports: `segments/vlm_bias_attention/reports/qwen3_roi_attention_v1/`
