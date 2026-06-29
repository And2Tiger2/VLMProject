from __future__ import annotations

import argparse
from pathlib import Path


DATASET_ID = "baulab/openai-comic-strips"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAI comic strips to p1.png..p6.png folders.")
    parser.add_argument("--out", default="segments/gaze_heads_qwen25/data/comics")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    from datasets import load_dataset

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_id, split=args.split)

    manifest = []
    for idx, row in enumerate(dataset):
        comic_id = row.get("comic_id", idx)
        comic_dir = out_root / f"comic{comic_id}"
        comic_dir.mkdir(parents=True, exist_ok=True)
        panel_paths = []
        for panel_idx in range(1, 7):
            image = row[f"panel_{panel_idx}"]
            panel_path = comic_dir / f"p{panel_idx}.png"
            image.save(panel_path)
            panel_paths.append(str(panel_path))
        manifest.append({"comic_id": comic_id, "comic_dir": str(comic_dir), "panel_paths": panel_paths})

    manifest_path = out_root.parent / "openai_comic_strips_manifest.json"
    import json

    manifest_path.write_text(json.dumps({"dataset_id": args.dataset_id, "split": args.split, "rows": manifest}, indent=2))
    print(f"Exported {len(manifest)} comics to {out_root}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
