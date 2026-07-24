from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHECK_OVERHEAT_DIR = Path("/n/fs/vl/scripts_group/check_overheat")


def maybe_pause() -> None:
    check_overheat = _load_check_overheat(required=_checker_is_required())
    if check_overheat is None:
        return
    if check_overheat.pause_needed():
        check_overheat.pause()


def require_check_overheat() -> Any:
    """Load and validate the checker before allocating time to model startup."""
    checker = _load_check_overheat(required=True)
    assert checker is not None
    return checker


def check_overheat_dir() -> Path:
    return Path(
        os.environ.get(
            "VLM_CHECK_OVERHEAT_DIR", str(DEFAULT_CHECK_OVERHEAT_DIR)
        )
    )


def _load_check_overheat(*, required: bool = False) -> Any | None:
    checker_dir = check_overheat_dir()
    try:
        available = checker_dir.is_dir()
    except OSError as exc:
        return _unavailable(checker_dir, required=required, cause=exc)
    if not available:
        return _unavailable(checker_dir, required=required)
    path = str(checker_dir)
    if path not in sys.path:
        sys.path.append(path)
    try:
        checker = importlib.import_module("check_overheat")
    except (ImportError, OSError) as exc:
        return _unavailable(checker_dir, required=required, cause=exc)
    missing = [
        name for name in ("pause_needed", "pause") if not callable(getattr(checker, name, None))
    ]
    if missing:
        message = (
            f"overheat checker at {checker_dir} is missing callable(s): "
            f"{', '.join(missing)}"
        )
        if required:
            raise RuntimeError(message)
        return None
    return checker


def _checker_is_required() -> bool:
    return os.environ.get("VLM_REQUIRE_OVERHEAT_CHECK", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _unavailable(
    checker_dir: Path,
    *,
    required: bool,
    cause: BaseException | None = None,
) -> None:
    if not required:
        return None
    detail = f": {type(cause).__name__}: {cause}" if cause is not None else ""
    raise RuntimeError(
        "Required Neuronic overheat checker is unavailable at "
        f"{checker_dir}{detail}. Set VLM_CHECK_OVERHEAT_DIR to an accessible "
        "directory containing check_overheat.py, or request access to the "
        "configured group-owned checker. Refusing to run the GPU experiment "
        "without the required safeguard."
    ) from cause
