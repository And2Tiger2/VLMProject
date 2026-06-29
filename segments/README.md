# Research Segments

Each research direction gets its own folder here so papers, datasets, raw runs,
and final reports do not live at the repository root.

Use this structure for each segment:

- `papers/`: source papers and notes that motivate the segment.
- `data/`: local datasets, fixed slices, downloaded assets, and manifests.
- `runs/`: raw model outputs, intermediate JSONL files, and run summaries.
- `reports/`: curated figures, tables, validation summaries, and writeups.

Keep reusable project pieces at the root:

- `vlm_eval/`: shared evaluation logic.
- `adapters/`: shared model adapters.
- `scripts/`: runnable scripts that can target one or more segments.
- `tests/`: shared test coverage.
- `pyproject.toml`, `uv.lock`, `.venv`, `.uv-cache`: environment and dependency
  state.

Current segments:

- `vlm_bias_attention/`: VLMBias/NaturalBench prior-driven bias and attention
  intervention work.
- `gaze_heads_qwen25/`: GazeHeads-style replication and steering experiments on
  Qwen2.5-VL.
