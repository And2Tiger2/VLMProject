# PaliGemma2 Attention Alpha Sweep

Model: `google/paligemma2-3b-mix-448`

Benchmarks:

- VLMBias: `data/vlmbias_400.jsonl`
- NaturalBench: `data/naturalbench_100_groups.jsonl`

Seeds: `0,1,2,3,4,5,6,7,8,9`

Alpha values: `0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0`

Layer selections: `last25`, `last50`, `all`, plus baseline.

## Main Readout

- VLMBias baseline: accuracy `0.141`, bias-aligned fraction `0.267`.
- Lowest VLMBias bias-aligned fraction: `alpha10_all` at `0.028` with accuracy `0.007`.
- Best NaturalBench call accuracy: `alpha8_last25` at `0.736`.
- Worst NaturalBench call accuracy: `alpha10_all` at `0.152`.

## Interpretation

The attention hook is active: image attention mass rises monotonically with alpha, and extreme `all`-layer boosts push image attention near saturation. Moderate boosts do not reliably reduce VLMBias bias and mostly leave NaturalBench stable. Extreme all-layer boosts, especially `alpha10_all`, severely damage NaturalBench and VLMBias accuracy, so the intervention can overpower useful language-model behavior rather than selectively correcting bias.
