# GazeHeads Qwen2.5-VL Segment

This segment adapts the GazeHeads replication workflow to
`Qwen/Qwen2.5-VL-3B-Instruct`.

Contents:

- `papers/`: local GazeHeads paper PDF.
- `data/`: exported six-panel comic folders from `baulab/openai-comic-strips`
  once downloaded.
- `runs/`: gaze-head rankings and steering generations.
- `reports/`: curated summaries and analysis notes.

`data/` and `runs/` are local artifacts ignored by git. Keep environment files,
shared adapters, reusable evaluation code, and tests at the repository root so
other segments can reuse them.

## Pipeline

Run the full ordered pipeline:

```bash
uv run python scripts/run_gaze_pipeline.py \
  --segment-root segments/gaze_heads_qwen25 \
  --device-map auto \
  --judge baseline-only
```

Use `--stages` to run a subset, for example:

```bash
uv run python scripts/run_gaze_pipeline.py \
  --stages discover trajectory static vqa dynamic validate
```

Use `--dry-run` to print the exact commands without executing them.
Use `--resume` on generation/scoring stages or the top-level runner to append to
an existing JSONL and skip rows already present.

For long full runs, shard by comic index and give each shard a suffix so outputs
do not overwrite each other:

```bash
uv run python scripts/preflight_gaze_full_run.py \
  --segment-root segments/gaze_heads_qwen25 \
  --total-comics 500 \
  --shard-size 25 \
  --judge anthropic
```

```bash
uv run python scripts/plan_gaze_full_run.py \
  --segment-root segments/gaze_heads_qwen25 \
  --total-comics 500 \
  --shard-size 25 \
  --judge anthropic \
  --out segments/gaze_heads_qwen25/reports/full_run_plan.sh
```

This writes a shell script with canonical discovery, all shard runs, shard
merges, final scoring, report generation, and validation. Review it, then run it
from the repository root with `bash segments/gaze_heads_qwen25/reports/full_run_plan.sh`.

To run one shard manually:

```bash
uv run python scripts/run_gaze_pipeline.py \
  --segment-root segments/gaze_heads_qwen25 \
  --stages trajectory static vqa dynamic score_static score_vqa score_dynamic validate \
  --device-map auto \
  --start-comic-idx 100 \
  --max-comics 25 \
  --trajectory-comics 25 \
  --run-suffix _100_25 \
  --top-k-gaze 100 \
  --top-k-random 100 \
  --targets-per-strip 1 \
  --include-all-heads \
  --judge baseline-only \
  --resume
```

This writes shard outputs such as `static_narration_100_25/` and validation to
`reports/pipeline_validation_100_25.json`. Omit `--run-suffix` only when writing
the canonical full-run directories. By default, shards read the canonical
`runs/gaze_discovery/gaze_head_ranking.json`; use `--discovery-suffix` only when
you intentionally want a suffixed discovery ranking.

After all shards for a stage finish, merge them into the canonical directory and
score the merged JSONL:

```bash
uv run python scripts/merge_gaze_run_shards.py \
  --segment-root segments/gaze_heads_qwen25 \
  --stage static \
  --suffixes _0_25 _25_25 _50_25

uv run python scripts/score_qwen25_gaze_generations.py \
  --generations segments/gaze_heads_qwen25/runs/static_narration/generations.jsonl \
  --judge anthropic
```

The merge utility supports `trajectory`, `static`, `vqa`, and `dynamic`.

Export the OpenAI comic strips:

```bash
uv run python scripts/download_gaze_comics.py \
  --out segments/gaze_heads_qwen25/data/comics
```

Discover Qwen2.5-VL gaze heads:

```bash
uv run python scripts/discover_qwen25_gaze_heads.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --out-dir segments/gaze_heads_qwen25/runs/gaze_discovery \
  --device-map auto \
  --n-samples 500
```

## Verified Smoke Workflow

For a quick local smoke test before launching the full runs:

```bash
uv run python scripts/run_gaze_pipeline.py \
  --segment-root segments/gaze_heads_qwen25 \
  --device-map auto \
  --judge baseline-only \
  --smoke
```

This runs a one-comic discovery, trajectory tracking, static steering, VQA
steering, dynamic steering, baseline-only scoring, report generation, and smoke
validation.

The same smoke workflow can be run manually. Start with one-comic gaze
discovery:

```bash
uv run python scripts/discover_qwen25_gaze_heads.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --out-dir segments/gaze_heads_qwen25/runs/gaze_discovery_smoke \
  --device-map auto \
  --n-samples 1
```

The smoke run should write `summary.json`, `gaze_head_ranking.json`,
`gaze_scores.npy`, and `mean_panel_attention.npy`.

Then exercise the generation stages against that smoke ranking:

```bash
uv run python scripts/run_qwen25_gaze_narration_trajectory.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery_smoke/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/narration_trajectory_smoke \
  --device-map auto \
  --max-comics 1 \
  --top-k-gaze 5 \
  --control-heads 5 \
  --max-new-tokens 20

uv run python scripts/run_qwen25_gaze_static_narration.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery_smoke/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/static_narration_smoke \
  --device-map auto \
  --max-comics 1 \
  --top-k-gaze 5 \
  --top-k-random 5 \
  --targets-per-strip 1 \
  --max-new-tokens 20

uv run python scripts/run_qwen25_gaze_vqa_steering.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery_smoke/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/vqa_steering_smoke \
  --device-map auto \
  --max-comics 1 \
  --top-k-gaze 5 \
  --top-k-random 5 \
  --max-new-tokens 10

uv run python scripts/run_qwen25_gaze_dynamic_narration.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery_smoke/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/dynamic_narration_smoke \
  --device-map auto \
  --max-comics 1 \
  --top-k-gaze 5 \
  --segment-tokens 5 \
  --max-new-tokens 30
```

Score the smoke generations without an external judge key:

```bash
uv run python scripts/score_qwen25_gaze_generations.py \
  --generations segments/gaze_heads_qwen25/runs/static_narration_smoke/generations.jsonl \
  --judge baseline-only

uv run python scripts/score_qwen25_gaze_generations.py \
  --generations segments/gaze_heads_qwen25/runs/vqa_steering_smoke/generations.jsonl \
  --judge baseline-only

uv run python scripts/score_qwen25_gaze_generations.py \
  --generations segments/gaze_heads_qwen25/runs/dynamic_narration_smoke/generations.jsonl \
  --judge baseline-only
```

Build the smoke report and validate the smoke artifact set:

```bash
uv run python scripts/report_qwen25_gaze_results.py \
  --segment-root segments/gaze_heads_qwen25 \
  --run-suffix _smoke

uv run python scripts/validate_gaze_pipeline.py \
  --segment-root segments/gaze_heads_qwen25 \
  --run-suffix _smoke \
  --out segments/gaze_heads_qwen25/reports/pipeline_validation_smoke.json
```

Track unsteered narration trajectories for gaze heads versus controls:

```bash
uv run python scripts/run_qwen25_gaze_narration_trajectory.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/narration_trajectory \
  --device-map auto \
  --max-comics 5
```

Run static narration steering generations:

```bash
uv run python scripts/run_qwen25_gaze_static_narration.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/static_narration \
  --device-map auto
```

The static steering script writes `generations.jsonl` so judging/scoring can be
applied without rerunning Qwen.

Run VQA steering generations:

```bash
uv run python scripts/run_qwen25_gaze_vqa_steering.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/vqa_steering \
  --device-map auto
```

Run dynamic narration steering generations:

```bash
uv run python scripts/run_qwen25_gaze_dynamic_narration.py \
  --comics-root segments/gaze_heads_qwen25/data/comics \
  --gaze-ranking segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json \
  --out-dir segments/gaze_heads_qwen25/runs/dynamic_narration \
  --device-map auto
```

Score any generation JSONL with the paper-style forced panel judge:

```bash
ANTHROPIC_API_KEY=... uv run --extra judge python scripts/score_qwen25_gaze_generations.py \
  --generations segments/gaze_heads_qwen25/runs/static_narration/generations.jsonl \
  --judge anthropic \
  --resume
```

Build paper-style summary tables after scoring:

```bash
uv run python scripts/report_qwen25_gaze_results.py \
  --segment-root segments/gaze_heads_qwen25
```

This writes `reports/gaze_results_report.json` and
`reports/gaze_results_summary.tsv`. Add `--run-suffix _100_25` to report on a
single shard or `--run-suffix _smoke` for the smoke artifacts.

For a no-key sanity pass, `--judge baseline-only` marks generations that are
effectively unchanged from the unsteered baseline as misses, but it does not
assign panel matches. Dynamic narration JSONL is scored segment-by-segment and
reports per-segment accuracy plus Spearman alignment between the target schedule
and matched panels.

Validate the artifacts from completed or partially completed runs:

```bash
uv run python scripts/validate_gaze_pipeline.py \
  --segment-root segments/gaze_heads_qwen25
```

This writes `segments/gaze_heads_qwen25/reports/pipeline_validation.json`.
Use `--run-suffix _smoke` to validate the smoke directory names instead of the
full-run directory names.
