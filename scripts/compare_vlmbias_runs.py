from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image vs no-image VLMBias JSONL runs.")
    parser.add_argument("--image-run", required=True)
    parser.add_argument("--no-image-run", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    image_rows = _load_rows(Path(args.image_run))
    no_image_rows = _load_rows(Path(args.no_image_run))
    report = compare_runs(image_rows, no_image_rows)

    text = format_report(report)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def compare_runs(image_rows: list[dict], no_image_rows: list[dict]) -> dict:
    image_by_id = {row["example_id"]: row for row in image_rows}
    no_image_by_id = {row["example_id"]: row for row in no_image_rows}
    shared_ids = sorted(set(image_by_id) & set(no_image_by_id))

    report = {
        "image": _summary([image_by_id[row_id] for row_id in shared_ids]),
        "no_image": _summary([no_image_by_id[row_id] for row_id in shared_ids]),
        "by_topic": {},
        "transitions": defaultdict(int),
    }

    topics = sorted({image_by_id[row_id].get("topic") for row_id in shared_ids})
    for topic in topics:
        image_topic = [image_by_id[row_id] for row_id in shared_ids if image_by_id[row_id].get("topic") == topic]
        no_image_topic = [no_image_by_id[row_id] for row_id in shared_ids if no_image_by_id[row_id].get("topic") == topic]
        report["by_topic"][topic or "unknown"] = {
            "image": _summary(image_topic),
            "no_image": _summary(no_image_topic),
        }

    for row_id in shared_ids:
        report["transitions"][(_category(image_by_id[row_id]), _category(no_image_by_id[row_id]))] += 1
    report["transitions"] = {f"{src}->{dst}": count for (src, dst), count in sorted(report["transitions"].items())}
    return report


def format_report(report: dict) -> str:
    image = report["image"]
    no_image = report["no_image"]
    lines = [
        "VLMBias Image vs No-Image Comparison",
        "",
        "Headline:",
        _metric_line("Accuracy", image["accuracy"], no_image["accuracy"]),
        _metric_line("Error rate", image["error_rate"], no_image["error_rate"]),
        _metric_line("Bias-aligned fraction", image["bias_aligned_fraction"], no_image["bias_aligned_fraction"]),
        _metric_line("Bias-aligned error rate", image["bias_aligned_error_rate"], no_image["bias_aligned_error_rate"]),
        "",
        "Counts:",
        f"image:    n={image['n']} correct={image['correct']} bias_aligned={image['bias_aligned']} other_wrong={image['other_wrong']}",
        f"no-image: n={no_image['n']} correct={no_image['correct']} bias_aligned={no_image['bias_aligned']} other_wrong={no_image['other_wrong']}",
        "",
        "By topic:",
    ]
    for topic, values in report["by_topic"].items():
        img = values["image"]
        no = values["no_image"]
        lines.append(
            f"{topic:16s} n={img['n']:3d} "
            f"acc {img['accuracy']:.3f}->{no['accuracy']:.3f} "
            f"baf {img['bias_aligned_fraction']:.3f}->{no['bias_aligned_fraction']:.3f} "
            f"baer {img['bias_aligned_error_rate']:.3f}->{no['bias_aligned_error_rate']:.3f}"
        )
    lines.extend(["", "Transitions image -> no-image:"])
    for transition, count in report["transitions"].items():
        lines.append(f"{transition}: {count}")
    return "\n".join(lines)


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(row["is_correct"] for row in rows)
    bias_aligned = sum(row["is_bias_aligned_error"] for row in rows)
    other_wrong = sum((not row["is_correct"]) and (not row["is_bias_aligned_error"]) for row in rows)
    errors = n - correct
    return {
        "n": n,
        "correct": correct,
        "bias_aligned": bias_aligned,
        "other_wrong": other_wrong,
        "accuracy": correct / n if n else 0.0,
        "error_rate": errors / n if n else 0.0,
        "bias_aligned_fraction": bias_aligned / n if n else 0.0,
        "bias_aligned_error_rate": bias_aligned / errors if errors else 0.0,
    }


def _category(row: dict) -> str:
    if row["is_correct"]:
        return "correct"
    if row["is_bias_aligned_error"]:
        return "bias"
    return "other_wrong"


def _metric_line(name: str, image_value: float, no_image_value: float) -> str:
    return f"{name:25s} image={image_value:.4f} no_image={no_image_value:.4f} delta={no_image_value - image_value:+.4f}"


if __name__ == "__main__":
    main()

