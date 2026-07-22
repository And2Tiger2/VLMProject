from __future__ import annotations

import json
import tarfile
from pathlib import Path

from scripts.audit_neuronic_qwen3_artifacts import (
    EXPECTED_MODEL,
    audit_artifacts,
    create_audit_bundle,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_fixture(repo: Path, *, model_id: str = EXPECTED_MODEL) -> Path:
    segment = repo / "segments/gaze_heads_qwen3_8b"
    runs = segment / "runs"
    discovery_shard = runs / "gaze_discovery_seed42_0_1"
    discovery_merged = runs / "gaze_discovery_seed42_merged"
    ranking = [
        {"layer": index // 4, "head": index % 4, "score": 1.0 - index / 1000}
        for index in range(100)
    ]
    _write_json(
        discovery_shard / "summary.json",
        {
            "model_id": model_id,
            "comics_root": "segments/gaze_heads_qwen3_8b/data/discovery_comics",
            "start_comic_idx": 0,
            "max_comics": 1,
            "use_raw": True,
            "valid_samples": 1,
        },
    )
    _write_json(discovery_merged / "gaze_head_ranking.json", ranking)
    _write_json(discovery_merged / "summary.json", {"valid_samples": 1, "top_head": ranking[0]})
    (discovery_merged / "gaze_scores.npy").write_bytes(b"scores")

    shard = runs / "static_narration_seed42_top100_0_1"
    config = {
        "model_id": model_id,
        "comics_root": "segments/gaze_heads_qwen3_8b/data/eval_comics",
        "gaze_ranking": (
            "segments/gaze_heads_qwen3_8b/runs/"
            "gaze_discovery_seed42_merged/gaze_head_ranking.json"
        ),
        "start_comic_idx": 0,
        "max_comics": 1,
        "top_k_gaze": 100,
        "top_k_random": 100,
        "control_mode": "paper",
        "targets_per_strip": 6,
        "max_new_tokens": 100,
        "swap_bias": 10000.0,
        "decode_only": False,
        "seed": 42,
    }
    _write_json(shard / "experiment_config.json", config)

    merged = runs / "static_narration_seed42_top100_merged_0_1"
    rows = []
    for condition in ("gaze_top100", "non_gaze_58"):
        for panel in range(1, 7):
            rows.append(
                {
                    "strip_name": "strip-1",
                    "condition": condition,
                    "target_panel": panel,
                    "baseline_text": "baseline",
                    "generated_text": f"{condition}-{panel}",
                }
            )
    merged.mkdir(parents=True)
    (merged / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(
        merged / "summary.json",
        {"source_summaries": [{"conditions": ["gaze_top100", "non_gaze_58"]}]},
    )
    _write_json(merged / "validation.json", {"valid": True})
    return segment


def test_audit_proves_qwen3_configs_and_flags_unmatched_control(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".git").mkdir()
    segment = _make_fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.audit_neuronic_qwen3_artifacts._git",
        lambda _repo, *args: "abc123" if args[0] == "rev-parse" else "",
    )

    report = audit_artifacts(
        repo_root=tmp_path,
        segment_root=segment,
        base_seed=42,
        seeds=1,
        ranking_seed=42,
        top_ks=[100],
        shards=1,
        shard_size=1,
    )

    assert report["valid"] is True
    assert report["static"][0]["rows"] == 12
    assert report["static"][0]["control_heads"] == 58
    assert any("not a matched 100-head control" in warning for warning in report["warnings"])


def test_audit_rejects_qwen25_model_provenance(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    segment = _make_fixture(tmp_path, model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setattr(
        "scripts.audit_neuronic_qwen3_artifacts._git", lambda _repo, *args: ""
    )

    report = audit_artifacts(
        repo_root=tmp_path,
        segment_root=segment,
        base_seed=42,
        seeds=1,
        ranking_seed=42,
        top_ks=[100],
        shards=1,
        shard_size=1,
    )

    assert report["valid"] is False
    assert any("Qwen2.5-VL" in error for error in report["errors"])


def test_bundle_contains_audit_configs_and_top100_generations(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    segment = _make_fixture(tmp_path)
    audit_path = tmp_path / ".cache/audits/qwen3_artifact_audit.json"
    _write_json(audit_path, {"valid": True})
    bundle = tmp_path / ".cache/audits/qwen3_audit_bundle.tar.gz"

    result = create_audit_bundle(
        repo_root=tmp_path,
        segment_root=segment,
        audit_paths=[audit_path],
        bundle_path=bundle,
        base_seed=42,
        seeds=1,
        ranking_seed=42,
        top_ks=[100],
        shards=1,
        shard_size=1,
    )

    with tarfile.open(bundle) as archive:
        names = set(archive.getnames())
    assert result["files"] >= 8
    assert ".cache/audits/qwen3_artifact_audit.json" in names
    assert any(name.endswith("top100_merged_0_1/generations.jsonl") for name in names)
    assert any(name.endswith("top100_0_1/experiment_config.json") for name in names)
