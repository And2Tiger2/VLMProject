# Gaze attention sweep analysis

## Sources

- Experiment config: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/experiment_config.json`
- Per-run summaries: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/{vlmbias,naturalbench}/*.summary.json`
- Original aggregate TSVs: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/*_summary_*.tsv`
- Analysis outputs: `segments/vlm_bias_attention/reports/gaze_attention_sweep_analysis/`

## Validation

- Found 250 VLMBias summary files and 250 NaturalBench summary files.
- Recomputed aggregates from per-run summary JSON files because the checked-in `*_summary_by_seed.tsv` files only contain seed 0.
- Seed coverage in recomputed aggregates:

| benchmark | min | max |
| --- | --- | --- |
| naturalbench | 10 | 10 |
| vlmbias | 10 | 10 |

## Baselines

- VLMBias baseline accuracy: 0.1215; bias-aligned fraction: 0.2317; bias-aligned error rate: 0.2638.
- NaturalBench baseline Acc: 0.5307; Q_Acc: 0.2330; I_Acc: 0.2150; G_Acc: 0.0250.

## NaturalBench metric definitions

- NaturalBench `Acc` is per-call accuracy: correct model calls divided by all model calls. In this slice that is 400 calls per run.
- NaturalBench `G_Acc` is strict group accuracy: each group has four required calls (`q0_i0`, `q0_i1`, `q1_i0`, `q1_i1`), and the group is correct only if all four are correct. This is why `G_Acc` is much lower than `Acc`.
- VLMBias `accuracy` is a separate benchmark metric: correct answer fraction on the VLMBias questions. It should be compared to NaturalBench `Acc` as another accuracy-style metric, but it is not computed from NaturalBench groups.
- VLMBias `bias_aligned_fraction` is not accuracy. It is the fraction of outputs that align with the dataset's known visual/textual bias direction; lower can be better only if task accuracy is not also harmed.

## Main result

Best VLMBias accuracy is `gaze_top20_alpha2`: accuracy 0.1310, delta 0.0095, with NaturalBench Acc delta -0.0010. The gain is smaller than the seed-to-seed standard deviation for both baseline (0.0125) and this condition (0.0127), so treat it as a small observed shift rather than a robust win.

Best VLMBias accuracy while keeping NaturalBench Acc within 1 point of baseline is `gaze_top20_alpha2`: VLMBias accuracy delta 0.0095, NaturalBench Acc delta -0.0010.

Lowest bias-aligned fraction is `gaze_top20_alpha10`: 0.2140, delta -0.0177. Lower bias-aligned fraction is not enough on its own, because it can coincide with lower accuracy or general degradation.

Worst NaturalBench Acc is `gaze_top5_alpha0p25`: Acc 0.5275, delta -0.0032. Across the recomputed all-seed aggregates, NaturalBench Acc remains within about one point of baseline for every tested condition.

## Top tables

### Best VLMBias accuracy

| condition | accuracy_mean | accuracy_mean_delta_vs_baseline | bias_aligned_fraction_mean | Acc_mean | Acc_mean_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| gaze_top20_alpha2 | 0.1310 | 0.0095 | 0.2373 | 0.5298 | -0.0010 |
| gaze_top10_alpha10 | 0.1293 | 0.0078 | 0.2357 | 0.5343 | 0.0035 |
| gaze_top10_alpha1 | 0.1285 | 0.0070 | 0.2355 | 0.5333 | 0.0025 |
| gaze_top10_alpha5 | 0.1283 | 0.0068 | 0.2417 | 0.5350 | 0.0042 |
| gaze_top20_alpha1 | 0.1277 | 0.0062 | 0.2363 | 0.5340 | 0.0032 |
| gaze_top10_alpha0p25 | 0.1270 | 0.0055 | 0.2330 | 0.5297 | -0.0010 |
| gaze_top10_alpha2 | 0.1268 | 0.0053 | 0.2442 | 0.5382 | 0.0075 |
| gaze_top10_alpha0p5 | 0.1255 | 0.0040 | 0.2370 | 0.5345 | 0.0038 |
| gaze_top1_alpha1 | 0.1253 | 0.0038 | 0.2373 | 0.5317 | 0.0010 |
| gaze_top20_alpha0p25 | 0.1250 | 0.0035 | 0.2328 | 0.5360 | 0.0053 |

### Best VLMBias accuracy with NaturalBench within 1 point

| condition | accuracy_mean | accuracy_mean_delta_vs_baseline | bias_aligned_fraction_mean | Acc_mean | Acc_mean_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| gaze_top20_alpha2 | 0.1310 | 0.0095 | 0.2373 | 0.5298 | -0.0010 |
| gaze_top10_alpha10 | 0.1293 | 0.0078 | 0.2357 | 0.5343 | 0.0035 |
| gaze_top10_alpha1 | 0.1285 | 0.0070 | 0.2355 | 0.5333 | 0.0025 |
| gaze_top10_alpha5 | 0.1283 | 0.0068 | 0.2417 | 0.5350 | 0.0042 |
| gaze_top20_alpha1 | 0.1277 | 0.0062 | 0.2363 | 0.5340 | 0.0032 |
| gaze_top10_alpha0p25 | 0.1270 | 0.0055 | 0.2330 | 0.5297 | -0.0010 |
| gaze_top10_alpha2 | 0.1268 | 0.0053 | 0.2442 | 0.5382 | 0.0075 |
| gaze_top10_alpha0p5 | 0.1255 | 0.0040 | 0.2370 | 0.5345 | 0.0038 |
| gaze_top1_alpha1 | 0.1253 | 0.0038 | 0.2373 | 0.5317 | 0.0010 |
| gaze_top20_alpha0p25 | 0.1250 | 0.0035 | 0.2328 | 0.5360 | 0.0053 |

## Interpretation

The attention intervention changes image attention mass monotonically with alpha, but task outcomes are noisy and not consistently improved. NaturalBench is fairly stable in this completed sweep, so the main concern is not broad VQA collapse; it is that the VLMBias improvements are small relative to seed variation and do not consistently move the bias metrics in the desired direction.

For the current data, gaze attention boosting is not a clean debiasing solution. Top-20 alpha 2 gives the best observed VLMBias accuracy, while top-20 alpha 10 gives the lowest observed bias-aligned fraction; these are different operating points, and neither effect is large enough by itself to claim reliable debiasing.

## Artifacts

- `plots/vlmbias_accuracy_by_alpha.png`
- `plots/vlmbias_bias_aligned_fraction_by_alpha.png`
- `plots/naturalbench_acc_by_alpha.png`
- `plots/naturalbench_g_acc_by_alpha.png`
- `plots/combined_accuracy_bias_comparison.png`
- `plots/combined_accuracy_bias_comparison_faceted_ci.png`
- `plots/boosted_head_attention_mass_by_alpha.png`
- `plots/tradeoff_vlmbias_vs_naturalbench.png`
- `plots/scatter_naturalbench_acc_vs_vlmbias_bias_fraction.png`
- `tables/recomputed_summary_by_seed.tsv`
- `tables/recomputed_summary_aggregate.tsv`
- `tables/joined_vlmbias_naturalbench_tradeoff.tsv`
- `tables/best_vlmbias_accuracy.tsv`
- `tables/best_accuracy_with_naturalbench_within_1pt.tsv`
- `tables/lowest_bias_aligned_fraction.tsv`
- `tables/best_naturalbench_acc.tsv`
