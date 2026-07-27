# Qwen3 attention methods: robustness

- Valid: `true`
- Conditions: 12/12
- Errors: 0
- Warnings: 0

## Condition metrics

| Condition | VLMBias acc | Bias-aligned fraction | NaturalBench Acc | NaturalBench G_Acc |
|---|---:|---:|---:|---:|
| baseline_seed0 | 0.0725 | 0.0975 | 0.4600 | 0.0200 |
| confirm_fixed_fixed_alpha0p5_global_top50_seed0 | 0.0750 | 0.1050 | 0.4825 | 0.0200 |
| confirm_target_mass_target_mass0p5_global_top50_seed0 | 0.0650 | 0.1075 | 0.4650 | 0.0100 |
| confirm_confidence_gate_confidence_gate0p3_global_top50_seed0 | 0.0725 | 0.1025 | 0.4625 | 0.0200 |
| baseline_seed1 | 0.0725 | 0.1125 | 0.4400 | 0.0100 |
| confirm_fixed_fixed_alpha0p5_global_top50_seed1 | 0.0775 | 0.1200 | 0.4375 | 0.0200 |
| confirm_target_mass_target_mass0p5_global_top50_seed1 | 0.0625 | 0.1250 | 0.4500 | 0.0400 |
| confirm_confidence_gate_confidence_gate0p3_global_top50_seed1 | 0.0750 | 0.1125 | 0.4450 | 0.0100 |
| baseline_seed2 | 0.0825 | 0.0975 | 0.4600 | 0.0100 |
| confirm_fixed_fixed_alpha0p5_global_top50_seed2 | 0.0950 | 0.0975 | 0.4600 | 0.0100 |
| confirm_target_mass_target_mass0p5_global_top50_seed2 | 0.0800 | 0.1075 | 0.4625 | 0.0400 |
| confirm_confidence_gate_confidence_gate0p3_global_top50_seed2 | 0.0850 | 0.1000 | 0.4700 | 0.0100 |
