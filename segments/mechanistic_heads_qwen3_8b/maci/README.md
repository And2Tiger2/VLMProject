# MACI and signed VLMBias heads

MMMC is loaded only through `datasets.load_dataset("ustc-zhangzm/MMMC")`, audited,
and split by `image_id`. The object-conflict row's explicit answer is factual;
the paired clean row's explicit answer is the hallucinated/prior candidate.
Empty, identical, cross-split, or ambiguously paired candidates are excluded.

```bash
bash scripts/run_neuronic_mechanistic_heads.sh download-mmmc
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads full
bash scripts/run_neuronic_mechanistic_heads.sh maci-heads-aligned full
uv run python scripts/analyze_maci_head_stability.py --config segments/mechanistic_heads_qwen3_8b/configs/maci_stability.json --output-dir segments/mechanistic_heads_qwen3_8b/reports/maci_stability --seed 0
bash scripts/run_neuronic_mechanistic_heads.sh maci-ablation full
bash scripts/run_neuronic_mechanistic_heads.sh maci-detector full
bash scripts/run_neuronic_mechanistic_heads.sh maci-gated full
bash scripts/run_neuronic_mechanistic_heads.sh maci-confirm full
```

Primary patch scope is the last prefill token. Run the signed scan separately
with `--scope all_aligned_prefill`; unequal sequences are excluded, never
truncated. Validation sweeps k=5/10/20/30/40/50; locked confirmation keeps the
paper-style 30 driving, 40 resisting, joint 30, five random seeds, and 20-draw
matched-control distributions separate.

VLMBias uses three group-safe rankings—semantic prior, context, and detail—then
tests driving suppression, resisting amplification, a joint role-aware policy,
MMMC-detector gating, matched controls, transitions, likelihood margins, exact
VLMBias metrics, and NaturalBench retention. These are exploratory transfers
until the subject/domain-held-out confirmation completes.
