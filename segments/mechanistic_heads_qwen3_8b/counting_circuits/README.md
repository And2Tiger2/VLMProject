# Counting Circuits

Implemented datasets: paper-geometry SynDot (336×336, 28×28 placement grid,
radius-four black circles, counts 1–10), 4,000/2,000 benchmark split, controlled
bidirectional pairs, and fixed-eight color/shape scenes with changed-pixel,
target, distractor, relocation, sham, and randomized-answer-code controls.

Run in this order:

```bash
bash scripts/run_neuronic_mechanistic_heads.sh counting-behavior smoke
bash scripts/run_neuronic_mechanistic_heads.sh counting-vap smoke
bash scripts/run_neuronic_mechanistic_heads.sh counting-heads smoke

bash scripts/run_neuronic_mechanistic_heads.sh counting-behavior full
bash scripts/run_neuronic_mechanistic_heads.sh counting-vap full
bash scripts/run_neuronic_mechanistic_heads.sh counting-heads full
bash scripts/run_neuronic_mechanistic_heads.sh general-importance
bash scripts/run_neuronic_mechanistic_heads.sh counting-controls
bash scripts/run_neuronic_mechanistic_heads.sh counting-validation full
```

Full scans are arrays of 36 one-layer jobs, four concurrent, followed by a
manifest-verifying aggregation job. The locked validator compares count top
10/25/50, gaze top 10/25/50, low-score heads, and at least 20 controls per
matched family under zero, mean, resample, donor, and reverse donor patching.
It stratifies constant complexity, relocation, randomized codes, and matched
shams. Renderer-seed and scan-seed repeats must be added to the control config
before any count-head declaration.
