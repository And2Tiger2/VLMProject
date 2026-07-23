from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Kimi caption-calibration confusion and failed examples."
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path(
            "segments/gaze_heads_qwen3_8b/runs/kimi_judge_calibration_60/"
            "kimi_judge/judgments.jsonl"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = diagnose_calibration(args.judgments)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


def diagnose_calibration(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    confusion = {
        str(target): {
            str(predicted): 0
            for predicted in [0, 1, 2, 3, 4, 5, 6]
        }
        for target in range(1, 7)
    }
    predicted_counts: Counter[int] = Counter()
    failures: list[dict[str, Any]] = []
    for row in rows:
        target = int(row["target_panel"])
        judgment = row.get("judgment") or {}
        matched = judgment.get("matched_panel")
        predicted = int(matched) if matched is not None else 0
        confusion[str(target)][str(predicted)] += 1
        predicted_counts[predicted] += 1
        if predicted != target:
            failures.append(
                {
                    "strip_name": row.get("strip_name"),
                    "target_panel": target,
                    "matched_panel": matched,
                    "caption": row.get("generated_text"),
                    "is_junk": bool(judgment.get("is_junk")),
                    "raw_judge_text": judgment.get("raw_judge_text"),
                }
            )
    return {
        "n_rows": len(rows),
        "confusion_matrix": confusion,
        "predicted_panel_counts": {
            str(panel): predicted_counts[panel] for panel in range(0, 7)
        },
        "panel_4_failures": [
            failure for failure in failures if failure["target_panel"] == 4
        ],
        "all_failures": failures,
    }


if __name__ == "__main__":
    main()
