from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODES = {"paper", "layer_matched_random", "layer_matched_low"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and strictly validate one control-distribution shard set."
    )
    parser.add_argument(
        "--segment-root", type=Path, default=Path("segments/gaze_heads_qwen3_8b")
    )
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--starts", type=int, nargs="+", required=True)
    parser.add_argument("--shard-size", type=int, default=50)
    args = parser.parse_args()

    result = merge_control_shards(
        segment_root=args.segment_root,
        mode=args.mode,
        seed=args.seed,
        top_k=args.top_k,
        starts=args.starts,
        shard_size=args.shard_size,
    )
    print(json.dumps(result, indent=2))


def merge_control_shards(
    *,
    segment_root: Path,
    mode: str,
    seed: int,
    top_k: int,
    starts: list[int],
    shard_size: int,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported control mode: {mode}")
    runs = segment_root / "runs"
    shard_dirs = [
        runs
        / f"static_control_distribution_{mode}_seed{seed}_top{top_k}_{start}_{shard_size}"
        for start in starts
    ]
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        configs.append(_read_json(shard_dir / "experiment_config.json"))
        summaries.append(_read_json(shard_dir / "summary.json"))
        rows.extend(_read_jsonl(shard_dir / "generations.jsonl"))
    expected_condition = f"non_gaze_{mode}_{top_k}"
    for config in configs:
        _validate_config(
            config,
            mode=mode,
            top_k=top_k,
            expected_condition=expected_condition,
        )
    selected = configs[0]["selected_control_heads"]
    if any(config["selected_control_heads"] != selected for config in configs[1:]):
        raise RuntimeError("selected control heads differ across shards")

    expected_rows = len(starts) * shard_size * 6
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"merged {len(rows)} control rows; expected {expected_rows}"
        )
    keys = [
        (
            str(row.get("strip_name")),
            str(row.get("condition")),
            int(row.get("target_panel", 0)),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("control shards contain duplicate generation keys")
    if any(str(row.get("condition")) != expected_condition for row in rows):
        raise RuntimeError(
            f"control shards contain a condition other than {expected_condition}"
        )

    n_comics = len(starts) * shard_size
    out_dir = (
        runs
        / f"static_control_distribution_{mode}_seed{seed}_top{top_k}_merged_0_{n_comics}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "generations.jsonl"
    generations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    merged_config = {
        **configs[0],
        "start_comic_idx": 0,
        "max_comics": n_comics,
        "source_shards": [str(path) for path in shard_dirs],
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(merged_config, indent=2), encoding="utf-8"
    )
    summary = {
        "stage": "static",
        "control_distribution": True,
        "control_mode": mode,
        "seed": seed,
        "top_k": top_k,
        "n_comics": n_comics,
        "n_rows": len(rows),
        "conditions": [expected_condition],
        "selected_control_heads": selected,
        "source_shards": [str(path) for path in shard_dirs],
        "source_summaries": summaries,
        "generations": str(generations_path),
        "generations_sha256": _sha256(generations_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _validate_config(
    config: dict[str, Any], *, mode: str, top_k: int, expected_condition: str
) -> None:
    if config.get("control_mode") != mode:
        raise RuntimeError(
            f"shard control mode is {config.get('control_mode')!r}; expected {mode!r}"
        )
    if config.get("condition_set") != "control":
        raise RuntimeError("control-distribution shard is not control-only")
    if config.get("condition_labels") != [expected_condition]:
        raise RuntimeError(
            f"shard labels are {config.get('condition_labels')!r}; "
            f"expected {[expected_condition]!r}"
        )
    gaze = {
        (int(head[0]), int(head[1]))
        for head in config.get("selected_gaze_heads", [])
    }
    controls = [
        (int(head[0]), int(head[1]))
        for head in config.get("selected_control_heads", [])
    ]
    if len(controls) != top_k or len(set(controls)) != top_k:
        raise RuntimeError(f"control must contain exactly {top_k} unique heads")
    if gaze.intersection(controls):
        raise RuntimeError("control heads overlap selected gaze heads")
    if mode == "paper" and any(not (20 <= layer <= 35) for layer, _ in controls):
        raise RuntimeError("paper control contains a head outside layers 20--35")
    if mode.startswith("layer_matched"):
        gaze_layers = sorted(layer for layer, _ in gaze)
        control_layers = sorted(layer for layer, _ in controls)
        if gaze_layers != control_layers:
            raise RuntimeError(
                "layer-matched control does not reproduce the gaze layer histogram"
            )
    if mode == "layer_matched_low":
        scores_path = Path(str(config["gaze_ranking"])).parent / "gaze_scores.npy"
        scores = np.load(scores_path)
        by_layer: dict[int, int] = {}
        for layer, _ in gaze:
            by_layer[layer] = by_layer.get(layer, 0) + 1
        expected: set[tuple[int, int]] = set()
        for layer, count in by_layer.items():
            candidates = [
                (float(scores[layer, head]), (layer, head))
                for head in range(scores.shape[1])
                if (layer, head) not in gaze
            ]
            expected.update(
                head for _, head in sorted(candidates, key=lambda item: item[0])[:count]
            )
        if set(controls) != expected:
            raise RuntimeError(
                "layer-matched low control is not the lowest-score eligible set"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
