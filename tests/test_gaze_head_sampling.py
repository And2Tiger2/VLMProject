from __future__ import annotations

import numpy as np

from scripts.run_qwen25_gaze_static_narration import sample_non_gaze_heads


def test_sample_non_gaze_heads_backfills_to_requested_count() -> None:
    scores = np.arange(20, dtype=np.float64).reshape(4, 5)

    heads = sample_non_gaze_heads(
        n_layers=4,
        n_heads=5,
        exclude={(0, 0), (0, 1)},
        n_select=10,
        seed=0,
        scores=scores,
        max_score=1.0,
    )

    assert len(heads) == 10
    assert len(set(heads)) == 10
    assert (0, 0) not in heads
    assert (0, 1) not in heads
