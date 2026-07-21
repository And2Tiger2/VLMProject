from pathlib import Path

from PIL import Image

from vlm_eval.gaze_comics import build_strip_from_paths, sample_raw_comics_windows


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), "white").save(path)


def test_sample_raw_comics_windows_stays_within_page_and_is_reproducible(tmp_path: Path) -> None:
    comic = tmp_path / "comic_a"
    for page in range(1, 13):
        for panel in range(1, 7):
            _image(comic / f"{page}_{panel}.jpg")

    first = sample_raw_comics_windows(tmp_path, n_panels=6, n_samples=2, seed=7)
    second = sample_raw_comics_windows(tmp_path, n_panels=6, n_samples=2, seed=7)

    assert first == second
    assert len(first) == 2
    assert all(len({int(path.stem.split("_")[0]) for path in paths}) == 1 for _, paths in first)
    assert all(int(paths[0].stem.split("_")[0]) in {6, 7} for _, paths in first)
    strip = build_strip_from_paths(first[0][1], name=first[0][0], target_height=64)
    assert len(strip.panels) == 6
    assert strip.name == first[0][0]
