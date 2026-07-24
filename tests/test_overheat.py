from __future__ import annotations

import sys

import pytest

from vlm_eval import overheat


def test_inaccessible_optional_checker_is_ignored_but_required_checker_fails(
    monkeypatch, tmp_path,
) -> None:
    inaccessible = tmp_path / "not-accessible"

    def raise_permission_error(self) -> bool:
        if self == inaccessible:
            raise PermissionError("group-owned checker is not accessible")
        return False

    monkeypatch.setenv("VLM_CHECK_OVERHEAT_DIR", str(inaccessible))
    monkeypatch.setattr(overheat.Path, "is_dir", raise_permission_error)
    assert overheat._load_check_overheat(required=False) is None
    overheat.maybe_pause()
    monkeypatch.setenv("VLM_REQUIRE_OVERHEAT_CHECK", "1")
    with pytest.raises(RuntimeError, match="Refusing to run"):
        overheat.maybe_pause()


def test_accessible_required_checker_loads_and_validates(monkeypatch, tmp_path) -> None:
    checker_dir = tmp_path / "checker"
    checker_dir.mkdir()
    (checker_dir / "check_overheat.py").write_text(
        "def pause_needed():\n"
        "    return False\n\n"
        "def pause():\n"
        "    raise AssertionError('pause should not be called')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLM_CHECK_OVERHEAT_DIR", str(checker_dir))
    monkeypatch.setenv("VLM_REQUIRE_OVERHEAT_CHECK", "1")
    monkeypatch.delitem(sys.modules, "check_overheat", raising=False)

    checker = overheat.require_check_overheat()

    assert checker.pause_needed() is False
    overheat.maybe_pause()
