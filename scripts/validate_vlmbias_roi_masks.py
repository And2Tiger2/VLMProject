#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("segments/vlm_bias_attention/data/vlmbias_roi_masks_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared VLMBias ROI masks.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    accepted = _read_jsonl(args.root / "accepted.jsonl")
    errors: list[str] = []

    if len(accepted) != summary.get("n_accepted"):
        errors.append(
            f"accepted row count {len(accepted)} != summary n_accepted {summary.get('n_accepted')}"
        )
    ids = [str(row.get("id")) for row in accepted]
    if len(ids) != len(set(ids)):
        errors.append("accepted IDs are not unique")
    edited_paths = [str(row.get("edited_path")) for row in accepted]
    if len(edited_paths) != len(set(edited_paths)):
        errors.append("accepted edited image paths are not unique")

    mask_fractions: list[float] = []
    for row in accepted:
        artifacts = row.get("artifacts") or {}
        mask_path = args.root / str(artifacts.get("mask_path") or "")
        overlay_path = args.root / str(artifacts.get("overlay_path") or "")
        original_path = args.root / str(artifacts.get("original_path") or "")
        for path in (mask_path, overlay_path, original_path):
            if not path.is_file():
                errors.append(f"missing artifact for {row.get('id')}: {path}")
        if not mask_path.is_file():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            errors.append(f"mask is not binary for {row.get('id')}: {sorted(values)}")
        fraction = float((mask > 0).mean())
        mask_fractions.append(fraction)
        recorded = float((row.get("mask_stats") or {}).get("clean_mask_fraction", -1.0))
        if abs(fraction - recorded) > 1e-12:
            errors.append(
                f"mask fraction disagrees with manifest for {row.get('id')}: {fraction} vs {recorded}"
            )

    result = {
        "valid": not errors,
        "stage": "vlmbias_roi_masks",
        "n_accepted_unique_images": len(accepted),
        "n_unique_ids": len(set(ids)),
        "n_unique_edited_paths": len(set(edited_paths)),
        "mask_fraction": {
            "minimum": min(mask_fractions) if mask_fractions else None,
            "mean": float(np.mean(mask_fractions)) if mask_fractions else None,
            "maximum": max(mask_fractions) if mask_fractions else None,
        },
        "requires_manual_review": bool(summary.get("requires_manual_review")),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
