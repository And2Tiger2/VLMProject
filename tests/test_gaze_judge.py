from __future__ import annotations

from vlm_eval.gaze_judge import (
    aggregate_dynamic_judgments,
    aggregate_judgments,
    bootstrap_ci,
    normalize_for_match,
    spearman_rho,
    split_dynamic_segments,
    steered_matches_baseline,
    judge_match_target_panel_anthropic,
)


def test_steered_matches_baseline_normalizes_panel_prefix() -> None:
    assert normalize_for_match("Panel 2: A person runs!") == "a person runs"
    assert steered_matches_baseline("Panel 2: A person runs!", "A person runs.")
    assert not steered_matches_baseline("A dog sleeps", "A person runs")


def test_anthropic_judge_marks_empty_without_api_key() -> None:
    judgment = judge_match_target_panel_anthropic(
        strip_image=object(),
        segment_text="\n\t",
        baseline_text="A hero jumps.",
        target_panel=2,
    )

    assert judgment["matched_panel"] is None
    assert judgment["is_junk"] is True
    assert judgment["correct"] is False


def test_bootstrap_and_aggregate_judgments() -> None:
    ci = bootstrap_ci([True, False, True], n_bootstrap=100, seed=0)
    assert ci["accuracy"] == 2 / 3
    assert ci["n"] == 3

    rows = [
        {"condition": "gaze", "target_panel": 1, "judgment": {"correct": True}},
        {"condition": "gaze", "target_panel": 2, "judgment": {"correct": False, "is_junk": True}},
        {"condition": "control", "target_panel": 1, "judgment": {"correct": False, "matches_baseline": True}},
    ]
    aggregate = aggregate_judgments(rows, seed=0)
    assert aggregate["gaze"]["overall"]["accuracy"] == 0.5
    assert aggregate["gaze"]["junk_count"] == 1
    assert aggregate["control"]["baseline_match_count"] == 1


def test_split_dynamic_segments_prefers_panel_markers() -> None:
    text = "Panel 1: A cat runs. Panel 2: A dog jumps. Panel 3: A bird flies."
    assert split_dynamic_segments(text, 3) == [
        "Panel 1: A cat runs.",
        "Panel 2: A dog jumps.",
        "Panel 3: A bird flies.",
    ]
    assert len(split_dynamic_segments("One. Two. Three.", 3)) == 3


def test_spearman_and_dynamic_aggregate() -> None:
    assert spearman_rho([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == -1.0

    rows = [
        {
            "condition": "gaze",
            "segment_judgments": [
                {"target_panel": 2, "matched_panel": 2, "correct": True},
                {"target_panel": 3, "matched_panel": 3, "correct": True},
                {"target_panel": 1, "matched_panel": 2, "correct": False, "is_junk": True},
            ],
        }
    ]
    aggregate = aggregate_dynamic_judgments(rows, seed=0)
    assert aggregate["gaze"]["per_segment_accuracy"]["accuracy"] == 2 / 3
    assert aggregate["gaze"]["junk_segments"] == 1
    assert aggregate["gaze"]["n_strips_for_rho"] == 1
