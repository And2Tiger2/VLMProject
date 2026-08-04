# Qwen3 high-bias-topic ROI attention experiment

This experiment repeats the staged Qwen3 localized-attention study on the two VLMBias topics that
produced nearly all bias-aligned baseline errors: **Game Boards** and **Optical Illusion**.

## Data and masks

- 114 evaluation rows: 57 Game Boards and 57 Optical Illusions.
- 79 canonical visual/manipulation groups after collapsing prompt and resolution variants.
- Development: 40 groups / 55 rows (27 Game Boards, 28 Optical Illusions).
- Held-out confirmation: 39 groups / 59 rows (30 Game Boards, 29 Optical Illusions).
- Tight Game Board mask: inferred added/removed row or column band.
- Broad Game Board mask: visible grid and board structure.
- Tight Optical Illusion mask: drawn illusion geometry.
- Broad Optical Illusion mask: dilated illusion geometry and nearby context.

The tracked runtime archive contains the exact 114-row dataset, all 114 source images, and both
binary masks for every row. Development and confirmation never share a canonical visual group.

## Stages

1. `smoke`: baseline; tight/broad gaze-50 alpha 2; full-image gaze-50; tight/broad random-50.
2. `tune`: tight and broad masks with gaze-50 alpha in `{0.5, 1, 2, 5}`, plus full-image and
   shifted-mask controls.
3. `heads`: at the selected mask/alpha, compare gaze top 10/50/100, early/middle/late gaze heads,
   layer-matched random heads, low-score heads, and paper-style random heads.
4. `confirm`: on held-out groups, compare baseline, locked ROI, alternate mask, full-image,
   shifted-mask, and a fresh layer-matched random-head control.

Conditions run as one-GPU Slurm array tasks. Smoke allows two concurrent tasks; later stages allow
four by default. Stage aggregation uses `afterok` dependencies, so tuning and head selections are
locked before downstream manifests are created.

## Metrics

Reports include accuracy, bias-aligned count and fraction, conditional bias-aligned error rate,
invalid-response rate, target-token fraction, and attention telemetry. Accuracy, bias, and invalid
metrics are also reported separately for Game Boards and Optical Illusions.

## Neuronic commands

After pulling the commit:

```bash
bash scripts/run_neuronic_qwen3_high_bias_roi_attention.sh smoke
```

After validating the smoke report, submit the complete dependency chain:

```bash
bash scripts/run_neuronic_qwen3_high_bias_roi_attention.sh submit
```

Inspect all reports with:

```bash
bash scripts/run_neuronic_qwen3_high_bias_roi_attention.sh verify
```
