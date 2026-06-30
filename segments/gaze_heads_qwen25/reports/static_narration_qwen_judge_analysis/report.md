# Static narration Qwen-judge analysis

## Sources

- Runs: `segments/gaze_heads_qwen25/runs/static_narration_top{1,5,10,20}_merged_0_500/`
- Judge aggregates: `qwen_judge/aggregate_results.json`
- Judge rows: `qwen_judge/judgments.jsonl`
- Analysis outputs: `segments/gaze_heads_qwen25/reports/static_narration_qwen_judge_analysis/`

## Validation

- Each top-k run has 6,000 judged rows: 3,000 gaze-steered generations and 3,000 non-gaze controls.
- Each condition covers 500 comics x 6 target panels.
- Accuracy is Qwen2.5-VL forced-choice matching of generated narration back to the intended target panel.
- Error bars in the plots are the aggregate judge confidence intervals from `aggregate_results.json`.

## Metric definitions

- Overall accuracy is computed over all 3,000 judged generations for a condition: 500 comics x 6 target panels. It answers: "how often did the judge match the generated text back to the intended panel overall?"
- Panel accuracy slices the same judgment task by target panel. For example, panel-2 accuracy only uses rows where the steered/queried target was panel 2. It answers: "does the method work equally well for each target panel, or is success concentrated in one panel position?"
- These are not separate tasks. Panel accuracy is a diagnostic decomposition of the same overall accuracy.

## Main result

| top_k | gaze_accuracy | non_gaze_accuracy | delta_gaze_minus_non_gaze | gaze_junk_count | non_gaze_junk_count |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1663 | 0.1627 | 0.0037 | 0 | 3 |
| 5 | 0.1593 | 0.1857 | -0.0263 | 13 | 90 |
| 10 | 0.1600 | 0.1463 | 0.0137 | 10 | 79 |
| 20 | 0.1630 | 0.1590 | 0.0040 | 46 | 34 |

The largest positive gaze-minus-control effect is top-10: 0.0137 absolute accuracy. The largest negative effect is top-5: -0.0263.

## Interpretation

The static gaze-description effect is weak. Overall forced-choice accuracies stay close to the 1/6 random baseline, and the best top-k setting improves over its non-gaze control by only 0.0137. This is not evidence that static narration steering reliably makes the model describe the target panel.

The judge is also highly panel-biased, especially toward panel 2. That conclusion comes from `qwen_judge_matched_panel_distribution.tsv`; the most frequent matched panel per condition is:

| top_k | condition | matched_panel | fraction |
| --- | --- | --- | --- |
| 1 | gaze_top1 | 2 | 0.8193 |
| 1 | non_gaze_1 | 2 | 0.8238 |
| 5 | gaze_top5 | 2 | 0.6839 |
| 5 | non_gaze_5 | 2 | 0.6859 |
| 10 | gaze_top10 | 2 | 0.6802 |
| 10 | non_gaze_10 | 2 | 0.5844 |
| 20 | gaze_top20 | 2 | 0.4854 |
| 20 | non_gaze_20 | 2 | 0.7640 |

This panel bias explains why per-panel accuracy can be high for panel 2 while very low for panel 6. See the per-top-k panel plots in `plots/qwen_judge_panel_accuracy_top*.png` and the source table `tables/qwen_judge_panel_summary.tsv`.

## Artifacts

- `plots/qwen_judge_accuracy_by_topk.png`
- `plots/qwen_judge_panel_accuracy_best_topk.png`
- `plots/qwen_judge_panel_accuracy_top1.png`
- `plots/qwen_judge_panel_accuracy_top5.png`
- `plots/qwen_judge_panel_accuracy_top10.png`
- `plots/qwen_judge_panel_accuracy_top20.png`
- `tables/qwen_judge_condition_summary.tsv`
- `tables/qwen_judge_panel_summary.tsv`
- `tables/qwen_judge_matched_panel_distribution.tsv`
- `tables/qwen_judge_gaze_vs_non_gaze.tsv`
