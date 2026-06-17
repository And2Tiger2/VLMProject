from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


CHECK_OVERHEAT_DIR = Path("/n/fs/vl/scripts_group/check_overheat")


def maybe_pause() -> None:
    check_overheat = _load_check_overheat()
    if check_overheat is None:
        return
    if check_overheat.pause_needed():
        check_overheat.pause()


def _load_check_overheat() -> Any | None:
    if not CHECK_OVERHEAT_DIR.exists():
        return None
    path = str(CHECK_OVERHEAT_DIR)
    if path not in sys.path:
        sys.path.append(path)
    try:
        import check_overheat
    except ImportError:
        return None
    return check_overheat
