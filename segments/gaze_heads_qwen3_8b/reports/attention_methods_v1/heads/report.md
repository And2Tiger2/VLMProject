# Qwen3 attention methods: heads

- Valid: `true`
- Conditions: 10/10
- Errors: 0
- Warnings: 0

## Condition metrics

| Condition | VLMBias acc | Bias-aligned fraction | NaturalBench Acc | NaturalBench G_Acc |
|---|---:|---:|---:|---:|
| global_top10 | 0.0400 | 0.1100 | 0.4900 | 0.0400 |
| global_top50 | 0.0600 | 0.1100 | 0.4900 | 0.0400 |
| global_top100 | 0.0500 | 0.1100 | 0.4800 | 0.0400 |
| early_top50 | 0.0400 | 0.1300 | 0.4500 | 0.0400 |
| middle_top50 | 0.0400 | 0.1000 | 0.4500 | 0.0000 |
| late_top50 | 0.0600 | 0.1100 | 0.4800 | 0.0400 |
| layer_matched_random50_seed55 | 0.0500 | 0.1100 | 0.4800 | 0.0000 |
| layer_matched_random50_seed56 | 0.0500 | 0.1100 | 0.4500 | 0.0000 |
| layer_matched_low50 | 0.0500 | 0.1100 | 0.4600 | 0.0400 |
| paper_random50_seed55 | 0.0500 | 0.1100 | 0.4700 | 0.0400 |

## Locked selection

`global_top50`
