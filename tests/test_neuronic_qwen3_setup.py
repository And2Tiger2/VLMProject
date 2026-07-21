from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_neuronic_setup_pins_supported_python() -> None:
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup = (ROOT / "scripts/setup_neuronic_qwen3.sh").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.14"' in pyproject
    assert 'PYTHON_VERSION="3.12"' in setup
    assert 'UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/python"' in setup
    assert 'uv python install "$PYTHON_VERSION"' in setup
    assert 'uv sync --locked --python "$PYTHON_VERSION"' in setup
