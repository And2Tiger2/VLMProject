# Qwen3 tight-ROI suppression × complementary-context boosting

This is a fast exploratory follow-up to the high-bias Game Boards and Optical
Illusions experiment. It uses all 114 rows, the reviewed tight masks, and the
seed-42 gaze-ranked global top 50 heads.

The four conditions are an exact 2×2 factorial:

| Condition | Tight ROI key-logit bias | Other image-token key-logit bias |
|---|---:|---:|
| baseline | 0 | 0 |
| ROI suppression | -5 | 0 |
| context boosting | 0 | +5 |
| suppression + context boosting | -5 | +5 |

Biases are added before softmax for every prefill and decoding query in the
selected heads. Text tokens and all non-selected heads are unchanged. The
context is exactly the complement of the tight ROI among image tokens.

The four conditions run as four concurrent one-GPU Slurm array tasks, followed
by one CPU aggregation job. Each GPU task has a 40-minute scheduler limit and
uses the cluster overheat check around every generation.

Because this follow-up was designed after examining the earlier held-out
results, it is exploratory. A favorable result should later be confirmed on a
fresh dataset or untouched split.

Run on Neuronic:

```bash
bash scripts/run_neuronic_qwen3_roi_context_factorial.sh submit
```

After the jobs finish:

```bash
bash scripts/run_neuronic_qwen3_roi_context_factorial.sh verify
```
