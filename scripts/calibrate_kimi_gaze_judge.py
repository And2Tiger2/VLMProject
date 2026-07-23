from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from scripts.judge_kimi_gaze_generations import (
    DEFAULT_KIMI_MODEL,
    DEFAULT_KIMI_REVISION,
    judge_generations_kimi,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Kimi-VL panel matching against known OpenAI comic panel captions."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "segments/gaze_heads_qwen3_8b/data/openai_comic_strips_manifest.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("segments/gaze_heads_qwen3_8b/runs/kimi_judge_calibration_60"),
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument("--min-panel-accuracy", type=float, default=0.40)
    parser.add_argument("--model-id", default=DEFAULT_KIMI_MODEL)
    parser.add_argument("--revision", default=DEFAULT_KIMI_REVISION)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = calibrate_kimi(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        min_accuracy=args.min_accuracy,
        min_panel_accuracy=args.min_panel_accuracy,
        model_id=args.model_id,
        revision=args.revision,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def calibrate_kimi(
    *,
    manifest_path: Path,
    out_dir: Path,
    limit: int,
    seed: int,
    batch_size: int,
    min_accuracy: float,
    min_panel_accuracy: float,
    model_id: str,
    revision: str,
    resume: bool,
) -> dict[str, Any]:
    if limit <= 0 or limit % 6:
        raise ValueError("calibration limit must be a positive multiple of six")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = build_calibration_rows(manifest.get("rows") or [], limit=limit, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    generations = out_dir / "caption_generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    judge_dir = out_dir / "kimi_judge"
    judged = judge_generations_kimi(
        generations_path=generations,
        out_dir=judge_dir,
        model_id=model_id,
        revision=revision,
        batch_size=batch_size,
        seed=seed,
        resume=resume,
    )
    aggregate_payload = json.loads(
        Path(judged["aggregate"]).read_text(encoding="utf-8")
    )
    source = aggregate_payload["aggregate"]["source_caption"]
    overall = float(source["overall"]["accuracy"])
    per_panel = {
        panel: float(values["accuracy"])
        for panel, values in source["per_panel"].items()
    }
    errors = list(judged["errors"])
    if overall < min_accuracy:
        errors.append(
            f"caption calibration accuracy {overall:.1%} is below {min_accuracy:.1%}"
        )
    weak_panels = {
        panel: accuracy
        for panel, accuracy in per_panel.items()
        if accuracy < min_panel_accuracy
    }
    if weak_panels:
        errors.append(
            f"caption calibration panel accuracy is below {min_panel_accuracy:.1%}: "
            f"{weak_panels}"
        )
    result = {
        "valid": not errors,
        "stage": "kimi_caption_calibration",
        "n_rows": len(rows),
        "overall_accuracy": overall,
        "per_panel_accuracy": per_panel,
        "parse_failure_rate": judged["parse_failure_rate"],
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_panel_accuracy": min_panel_accuracy,
        },
        "aggregate": judged["aggregate"],
        "errors": errors,
        "warnings": judged["warnings"],
    }
    (out_dir / "calibration_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def build_calibration_rows(
    manifest_rows: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    candidates: dict[int, list[dict[str, Any]]] = {
        panel: [] for panel in range(1, 7)
    }
    for row in manifest_rows:
        captions = row.get("captions")
        if not isinstance(captions, list) or len(captions) != 6:
            continue
        for panel, caption in enumerate(captions, start=1):
            text = str(caption or "").strip()
            if text:
                candidates[panel].append(
                    {
                        "strip_name": Path(str(row["comic_dir"])).name,
                        "comic_dir": str(row["comic_dir"]),
                        "condition": "source_caption",
                        "target_panel": panel,
                        "generated_text": text,
                        "baseline_text": None,
                        "caption_provenance": "baulab/openai-comic-strips",
                    }
                )
    rng = random.Random(seed)
    for panel_rows in candidates.values():
        rng.shuffle(panel_rows)
    per_panel = limit // 6
    if any(len(candidates[panel]) < per_panel for panel in range(1, 7)):
        counts = {panel: len(rows) for panel, rows in candidates.items()}
        raise RuntimeError(
            f"manifest has insufficient non-empty captions for calibration: {counts}"
        )
    selected: list[dict[str, Any]] = []
    for index in range(per_panel):
        for panel in range(1, 7):
            selected.append(candidates[panel][index])
    return selected


if __name__ == "__main__":
    main()
