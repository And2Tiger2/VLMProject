# VLMBias Two-Pass Prompt Variants Fast Run

Model: Qwen/Qwen2.5-VL-3B-Instruct
Dataset: data/vlmbias_400.jsonl
Conditions: 4 description prompt variants x image_again/description_only
Decoding: do_sample=True, temperature=0.7, seed=0
Image cap: max_pixels=1048576
Runner: scripts/run_vlmbias_two_pass_prompt_variants.py
Slurm script: scripts/slurm_two_pass_prompt_variants_fast.sh

Summary copied from runs/vlmbias_two_pass_prompt_variants_fast/summary.tsv
