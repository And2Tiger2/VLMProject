from pathlib import Path

import pytest

from scripts.run_vlmbias_gaze_attention_sweep import _validate_resume_summary


def test_benchmark_resume_rejects_changed_ranking() -> None:
    run_config = {"model_id": "qwen3", "gaze_ranking": "new.json", "limit": None}
    condition = {"condition": "gaze_top10_alpha1", "attention_alpha": 1.0, "top_k_gaze": 10}
    existing_config = {
        **run_config,
        "gaze_ranking": "old.json",
        "benchmark": "vlmbias",
        "condition": condition["condition"],
        "seed": 0,
        "attention_alpha": 1.0,
        "top_k_gaze": 10,
    }
    with pytest.raises(RuntimeError, match="gaze_ranking"):
        _validate_resume_summary(
            {"run_config": existing_config},
            run_config,
            condition,
            0,
            benchmark="vlmbias",
            path=Path("summary.json"),
        )
