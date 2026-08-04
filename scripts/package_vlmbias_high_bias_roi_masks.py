#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from vlm_eval.qwen3_high_bias_roi_attention import (
    DEFAULT_ROI_ROOT,
    DEFAULT_RUNTIME_BUNDLE,
    TARGET_TOPICS,
    canonical_group_id,
)


DEFAULT_CANDIDATES = Path(
    "segments/vlm_bias_attention/data/vlmbias_high_bias_mask_candidates_v1"
)
DEFAULT_VLMBIAS_SOURCE = Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package manually reviewed high-bias VLMBias masks for Neuronic."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_RUNTIME_BUNDLE)
    parser.add_argument("--vlmbias", type=Path, default=DEFAULT_VLMBIAS_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.runtime_root.exists() and any(args.runtime_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Runtime directory is not empty: {args.runtime_root}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(args.runtime_root)
    candidate_rows = _read_jsonl(args.candidates / "manifest.jsonl")
    if len(candidate_rows) != 114:
        raise SystemExit(f"Expected 114 reviewed candidate rows, found {len(candidate_rows)}")
    if {str(row["topic"]) for row in candidate_rows} != set(TARGET_TOPICS):
        raise SystemExit("Candidate manifest contains unexpected topics")
    source_rows = _read_jsonl(args.vlmbias)
    source_by_id = {str(row["id"]): row for row in source_rows}

    (args.runtime_root / "masks_tight").mkdir(parents=True, exist_ok=True)
    (args.runtime_root / "masks_broad").mkdir(parents=True, exist_ok=True)
    (args.runtime_root / "dataset/images").mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        safe_id = _safe_name(str(row["id"]))
        artifacts = {}
        for variant in ("tight", "broad"):
            source = args.candidates / row["artifacts"][f"{variant}_mask"]
            destination = args.runtime_root / f"masks_{variant}" / f"{safe_id}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, destination)
            artifacts[f"{variant}_mask"] = str(destination.relative_to(args.runtime_root))
        accepted.append(
            {
                "id": row["id"],
                "group_id": canonical_group_id(str(row["id"])),
                "topic": row["topic"],
                "sub_topic": row.get("sub_topic"),
                "status": "accepted_manual_user_review_2026-08-04",
                "mask_methods": dict(row["methods"]),
                "mask_fractions": dict(row["mask_fraction"]),
                "artifacts": artifacts,
            }
        )
        source_row = source_by_id.get(str(row["id"]))
        if source_row is None:
            raise SystemExit(f"Candidate ID is absent from source VLMBias slice: {row['id']}")
        source_image = _resolve(args.vlmbias.parent, str(source_row["image_path"]))
        image_name = source_image.name
        shutil.copyfile(source_image, args.runtime_root / "dataset/images" / image_name)
        dataset_rows.append({**source_row, "image_path": f"images/{image_name}"})
    _write_jsonl(args.runtime_root / "accepted.jsonl", accepted)
    _write_jsonl(
        args.runtime_root / "dataset/vlmbias_high_bias_114.jsonl", dataset_rows
    )
    summary = {
        "valid": True,
        "n_rows": len(accepted),
        "n_groups": len({row["group_id"] for row in accepted}),
        "topics": sorted({str(row["topic"]) for row in accepted}),
        "mask_variants": ["tight", "broad"],
        "review_status": "accepted_manual_user_review_2026-08-04",
        "source_candidates": str(args.candidates),
        "source_vlmbias": str(args.vlmbias),
        "runtime_dataset": "dataset/vlmbias_high_bias_114.jsonl",
        "n_runtime_images": len(dataset_rows),
    }
    if summary["n_groups"] != 79:
        raise SystemExit(f"Expected 79 canonical visual groups, found {summary['n_groups']}")
    (args.runtime_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.bundle, "w:gz") as archive:
        for relative in (
            "accepted.jsonl",
            "summary.json",
            "masks_tight",
            "masks_broad",
            "dataset",
        ):
            archive.add(args.runtime_root / relative, arcname=relative)
    print(
        json.dumps(
            {**summary, "runtime_root": str(args.runtime_root), "bundle": str(args.bundle)},
            indent=2,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )[:160]


if __name__ == "__main__":
    main()
