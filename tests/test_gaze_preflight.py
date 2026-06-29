from __future__ import annotations

from pathlib import Path

import scripts.preflight_gaze_full_run as preflight_mod
from scripts.preflight_gaze_full_run import preflight


def test_preflight_passes_with_synthetic_segment(tmp_path: Path, monkeypatch) -> None:
    root = _make_segment(tmp_path, n_comics=3)
    monkeypatch.setattr(preflight_mod, "_check_python_dependencies", lambda: _ok("python_dependencies"))
    monkeypatch.setattr(preflight_mod, "_check_model_cache", lambda model_id: _ok("model_cache"))

    report = preflight(
        root=root,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        total_comics=3,
        shard_size=2,
        judge="baseline-only",
    )

    assert report["ok"]
    assert {check["name"] for check in report["checks"]} == {
        "segment_layout",
        "dataset",
        "python_dependencies",
        "model_cache",
        "judge",
        "shards",
        "discovery",
    }


def test_preflight_can_require_discovery(tmp_path: Path, monkeypatch) -> None:
    root = _make_segment(tmp_path, n_comics=3)
    monkeypatch.setattr(preflight_mod, "_check_python_dependencies", lambda: _ok("python_dependencies"))
    monkeypatch.setattr(preflight_mod, "_check_model_cache", lambda model_id: _ok("model_cache"))

    report = preflight(
        root=root,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        total_comics=3,
        shard_size=2,
        judge="baseline-only",
        require_discovery=True,
    )

    assert not report["ok"]
    discovery = next(check for check in report["checks"] if check["name"] == "discovery")
    assert discovery["required"]
    assert "gaze_head_ranking.json" in discovery["missing"]


def _make_segment(tmp_path: Path, *, n_comics: int) -> Path:
    root = tmp_path / "gaze"
    (root / "papers").mkdir(parents=True)
    (root / "papers" / "GazeHeads.pdf").write_bytes(b"pdf")
    (root / "runs").mkdir()
    (root / "reports").mkdir()
    for comic_idx in range(1, n_comics + 1):
        comic_dir = root / "data" / "comics" / f"comic{comic_idx}"
        comic_dir.mkdir(parents=True)
        for panel_idx in range(1, 7):
            (comic_dir / f"p{panel_idx}.png").write_bytes(b"png")
    return root


def _ok(name: str) -> dict:
    return {"name": name, "ok": True, "message": "ok"}
