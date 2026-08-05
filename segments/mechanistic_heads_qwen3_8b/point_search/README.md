# Point-by-Point Search

The paper-style generator uses 50 objects, six colors, six shapes, disjoint
target conjunctions, 2,000 training scenes, and OOD target counts
1/2/10/30/40/50. Point output is deterministic text; the length-matched direct
condition is token-matched at runtime, and shuffled supervision uses wrong
distractor points even for singleton targets.

The original non-copyright Waldo-like character has four independently
controlled features. Exact masks cover targets and strong distractors; tasks
include invisible/visible grids, normalized points, same-location verification,
four candidates, and presence. Real Waldo is not downloaded.

```bash
# Train each condition explicitly; LoRA is always a modified replication.
uv run python scripts/train_point_search.py --config segments/mechanistic_heads_qwen3_8b/configs/point_search_lora.json --output-dir segments/mechanistic_heads_qwen3_8b/checkpoints/point-answer-lora --condition point_answer --seed 260525427 --smoke

# The mechanistic scans deliberately refuse to fall back to the base model if
# this Point-Answer adapter is absent.
bash scripts/run_neuronic_mechanistic_heads.sh waldo-behavior smoke
bash scripts/run_neuronic_mechanistic_heads.sh point-centroids smoke
bash scripts/run_neuronic_mechanistic_heads.sh search-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh verification-heads smoke
bash scripts/run_neuronic_mechanistic_heads.sh distractor-heads smoke

# After full scans and general-importance controls:
bash scripts/run_neuronic_mechanistic_heads.sh point-ablation full
uv run python scripts/render_point_search_reports.py --config segments/mechanistic_heads_qwen3_8b/configs/point_search_reports.json --output-dir segments/mechanistic_heads_qwen3_8b/reports/point_search --seed 0
```

Repeat the training and behavioral command for `base`, `direct_answer`,
`direct_length_matched`, `point_answer`, and `shuffled_point_answer`, using a
separate output directory for every condition. After the two-step smoke,
replace the provisional Point-Answer adapter with the full LoRA pilot only by
an explicit `--overwrite`; all head-scan configs point to
`checkpoints/point-answer-lora`. The optional full-weight run instead uses
`configs/point_search_full_weight.json` and must be reported separately from
the LoRA modified replication.

```bash
uv run python scripts/train_point_search.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/point_search_lora.json \
  --output-dir segments/mechanistic_heads_qwen3_8b/checkpoints/point-answer-lora \
  --condition point_answer --seed 260525427 --overwrite
```

Attention transplantation patches aligned visual-key slices only, carries the
recipient visual mass, leaves nonvisual keys unchanged, and refuses unequal
visual-token counts. See `REAL_WALDO_TRANSFER.md` for the locked optional phase.
