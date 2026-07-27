# Qwen3 attention methods: controller

- Valid: `true`
- Conditions: 10/10
- Errors: 0
- Warnings: 0

## Condition metrics

| Condition | VLMBias acc | Bias-aligned fraction | NaturalBench Acc | NaturalBench G_Acc |
|---|---:|---:|---:|---:|
| baseline | 0.0500 | 0.1100 | 0.4600 | 0.0000 |
| fixed_alpha0p5 | 0.0500 | 0.1100 | 0.4800 | 0.0400 |
| fixed_alpha1 | 0.0500 | 0.1200 | 0.5000 | 0.0400 |
| fixed_alpha2 | 0.0500 | 0.1200 | 0.5200 | 0.0400 |
| fixed_alpha5 | 0.0600 | 0.1200 | 0.5300 | 0.0000 |
| target_mass0p5 | 0.0500 | 0.1200 | 0.5300 | 0.0800 |
| target_mass0p7 | 0.0500 | 0.1200 | 0.5500 | 0.0400 |
| target_mass0p9 | 0.0600 | 0.1400 | 0.5500 | 0.0800 |
| confidence_gate0p3 | 0.0500 | 0.1100 | 0.4600 | 0.0000 |
| confidence_gate0p6 | 0.0600 | 0.1200 | 0.5000 | 0.0400 |

## Locked selection

`fixed_alpha0p5`
