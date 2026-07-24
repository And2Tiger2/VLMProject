#!/usr/bin/env python3
from __future__ import annotations

import json

from vlm_eval.overheat import check_overheat_dir, require_check_overheat


def main() -> None:
    checker = require_check_overheat()
    print(
        json.dumps(
            {
                "valid": True,
                "checker_dir": str(check_overheat_dir()),
                "module": getattr(checker, "__file__", None),
                "pause_needed": bool(checker.pause_needed()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
