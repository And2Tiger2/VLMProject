from __future__ import annotations

import argparse
from pathlib import Path


DATASET_ID = "baulab/openai-comic-strips"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAI comic strips to p1.png..p6.png folders.")
    parser.add_argument("--out", default="segments/gaze_heads_qwen25/data/comics")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_id, split=args.split)
    total = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)

    manifest = []
    exported = 0
    skipped = 0
    for idx, row in tqdm(enumerate(dataset), total=total, desc="Exporting comics"):
        if args.limit > 0 and idx >= args.limit:
            break
        comic_id = row.get("comic_id", idx)
        comic_dir = out_root / f"comic{comic_id}"
        comic_dir.mkdir(parents=True, exist_ok=True)
        panel_paths = [comic_dir / f"p{panel_idx}.png" for panel_idx in range(1, 7)]
        if not args.overwrite and all(path.exists() and path.stat().st_size > 0 for path in panel_paths):
            skipped += 1
            manifest.append({"comic_id": comic_id, "comic_dir": str(comic_dir), "panel_paths": [str(path) for path in panel_paths]})
            continue
        for panel_idx in range(1, 7):
            image = row[f"panel_{panel_idx}"]
            panel_path = panel_paths[panel_idx - 1]
            image.save(panel_path)
        exported += 1
        manifest.append({"comic_id": comic_id, "comic_dir": str(comic_dir), "panel_paths": [str(path) for path in panel_paths]})

    manifest_path = out_root.parent / "openai_comic_strips_manifest.json"
    import json

    manifest_path.write_text(json.dumps({"dataset_id": args.dataset_id, "split": args.split, "rows": manifest}, indent=2))
    print(f"Exported {exported} comics to {out_root}; skipped {skipped} already-complete comics.")
    print(f"Manifest contains {len(manifest)} comics.")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
