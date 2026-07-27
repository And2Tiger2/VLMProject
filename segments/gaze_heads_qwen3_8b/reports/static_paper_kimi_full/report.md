# Qwen3-VL static Gaze Heads: corrected paper control

- Comics: 500
- Gaze heads: top-100
- Judge: moonshotai/Kimi-VL-A3B-Instruct with visible panel labels
- Bootstrap unit: comic strip (10000 replicates)
- Gaze accuracy (counted once): 0.706
- Mean control accuracy across seeds: 0.447
- Mean gaze-minus-control effect: +0.261

| Control seed | Gaze accuracy | Control accuracy | Delta | 95% paired comic-bootstrap CI |
|---:|---:|---:|---:|:---|
| 42 | 0.706 | 0.203 | +0.504 | [+0.483, +0.524] |
| 43 | 0.708 | 0.436 | +0.272 | [+0.250, +0.294] |
| 44 | 0.711 | 0.702 | +0.008 | [-0.012, +0.029] |
