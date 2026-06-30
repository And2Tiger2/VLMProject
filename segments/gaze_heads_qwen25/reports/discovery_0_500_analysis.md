# Gaze Discovery 0-500 Analysis

Source artifacts:

- `segments/gaze_heads_qwen25/runs/gaze_discovery_{0,50,...,450}_50/`
- `segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/`

## Integrity checks

- Ten shards were merged.
- Each shard processed 50 valid comics.
- The merged result contains 500 valid comics, 36 layers, 16 heads, and 6 panels.
- `gaze_discovery_merged_0_500/gaze_sum.npy` exactly equals the elementwise sum of the ten shard `gaze_sum.npy` files.

## Main ranking

The top 20 gaze heads from the merged 500-comic discovery run are:

| Rank | Head | Score |
| ---: | --- | ---: |
| 1 | L5H10 | 0.158262710 |
| 2 | L0H14 | 0.096555028 |
| 3 | L33H14 | 0.096427558 |
| 4 | L10H12 | 0.093178930 |
| 5 | L12H10 | 0.089326921 |
| 6 | L32H9 | 0.088601526 |
| 7 | L32H6 | 0.081998344 |
| 8 | L33H2 | 0.081357035 |
| 9 | L33H0 | 0.078197409 |
| 10 | L11H1 | 0.077832237 |
| 11 | L33H1 | 0.076148157 |
| 12 | L2H1 | 0.073540942 |
| 13 | L32H0 | 0.073249723 |
| 14 | L32H10 | 0.071924058 |
| 15 | L33H4 | 0.069185614 |
| 16 | L33H11 | 0.068635023 |
| 17 | L32H14 | 0.068232414 |
| 18 | L11H3 | 0.067121661 |
| 19 | L9H13 | 0.066592285 |
| 20 | L33H9 | 0.063195566 |

## Stability across 50-comic shards

- L5H10 is the top-ranked head in all 10 shards.
- L5H10 has merged score 0.158263, which is 1.64x the second-ranked head and 2.03x the tenth-ranked head.
- 16 of the merged top 20 heads appear in the top 20 of all 10 shards.
- 8 of the merged top 20 heads appear in the top 10 of all 10 shards.
- 4 of the merged top 20 heads appear in the top 5 of all 10 shards.
- The merged top 20 heads that do not appear in every shard top 20 are L2H1, L11H3, L9H13, and L33H9.

Shard overlap with the merged ranking:

| Shard start | Top 10 overlap | Top 20 overlap |
| ---: | ---: | ---: |
| 0 | 9/10 | 20/20 |
| 50 | 9/10 | 20/20 |
| 100 | 10/10 | 19/20 |
| 150 | 9/10 | 20/20 |
| 200 | 10/10 | 20/20 |
| 250 | 10/10 | 20/20 |
| 300 | 10/10 | 20/20 |
| 350 | 10/10 | 20/20 |
| 400 | 10/10 | 20/20 |
| 450 | 9/10 | 17/20 |

## Layer pattern

The merged top 20 are concentrated in a few regions:

- Layer 33: 7 heads
- Layer 32: 5 heads
- Layers 0, 2, 5, 9, 10, 12: 1 head each
- Layer 11: 2 heads

So this run identifies one very strong early/mid-layer head, L5H10, plus a broader late-layer cluster around layers 32-33.

## Plots

The intuition plots are stored under `segments/gaze_heads_qwen25/reports/discovery_0_500_plots/`:

- `top20_gaze_scores.svg`: labeled bar plot of the top 20 heads. This is the clearest view of the L5H10 margin.
- `top100_gaze_scores.svg`: wider bar plot of the top 100 heads. This shows how quickly the score decays outside the most important heads.
- `layer_head_score_heatmap.svg`: layer-by-head score heatmap. This shows the late-layer concentration around layers 32-33 plus the isolated strong L5H10 signal.
- `top20_shard_rank_stability.svg`: stability heatmap for the merged top 20 across the 10 shards.

## Conclusion

Experiment 1 gives a robust ordered gaze-head list for Qwen2.5-VL-3B on the 500-comic OpenAI comic-strip setup. The strongest conclusion is that L5H10 is the dominant gaze head under this scoring definition: it ranks first in every 50-comic shard and has a large score margin over the rest of the distribution.

The next most defensible intervention set is the merged top 20, not an arbitrary top 100. The top 20 is highly stable across shards, with 16/20 heads recurring in every shard top 20. For smaller ablations in experiment 2, reasonable cut points are top 1, top 5, top 10, and top 20.

The older local canonical `runs/gaze_discovery/` ranking differs from this pushed sharded result: its top 10 was L28H10, L27H1, L5H10, L33H14, L33H0, L33H2, L32H10, L32H6, L25H14, L33H4, while the merged sharded top 10 is L5H10, L0H14, L33H14, L10H12, L12H10, L32H9, L32H6, L33H2, L33H0, L11H1. Treat `gaze_discovery_merged_0_500` as the authoritative experiment-1 result because it has per-shard reproducibility checks and an exact weighted merge.
