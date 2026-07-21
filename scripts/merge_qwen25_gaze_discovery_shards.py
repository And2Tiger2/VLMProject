from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.discover_qwen25_gaze_heads import gaze_routing_diagnostics, rank_heads_by_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded Qwen2.5-VL GazeHeads discovery outputs and report head stability."
    )
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen25/runs/gaze_discovery_merged")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    merge_discovery_shards([Path(path) for path in args.shard_dirs], Path(args.out_dir), top_k=args.top_k)


def merge_discovery_shards(shard_dirs: list[Path], out_dir: Path, *, top_k: int = 20) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_sum: np.ndarray | None = None
    total_valid = 0
    shard_records: list[dict[str, Any]] = []
    batch_rankings: list[list[dict[str, Any]]] = []

    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        scores_path = shard_dir / "gaze_scores.npy"
        ranking_path = shard_dir / "gaze_head_ranking.json"
        mean_path = shard_dir / "mean_panel_attention.npy"
        sum_path = shard_dir / "gaze_sum.npy"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        if not scores_path.exists() or not ranking_path.exists():
            raise FileNotFoundError(f"Missing ranking/scores under {shard_dir}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        valid_samples = int(summary["valid_samples"])
        if valid_samples <= 0:
            raise ValueError(f"{shard_dir} has no valid samples.")

        if sum_path.exists():
            shard_sum = np.load(sum_path)
        else:
            if not mean_path.exists():
                raise FileNotFoundError(f"Missing gaze_sum.npy or mean_panel_attention.npy under {shard_dir}")
            shard_sum = np.load(mean_path) * float(valid_samples)

        merged_sum = shard_sum.copy() if merged_sum is None else merged_sum + shard_sum
        total_valid += valid_samples
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        batch_rankings.append(ranking)
        shard_records.append(
            {
                "path": str(shard_dir),
                "valid_samples": valid_samples,
                "start_comic_idx": summary.get("start_comic_idx"),
                "max_comics": summary.get("max_comics"),
                "top_head": summary.get("top_head") or (ranking[0] if ranking else None),
            }
        )

    if merged_sum is None or total_valid <= 0:
        raise RuntimeError("No discovery shards were merged.")

    mean_panel_attention = merged_sum / float(total_valid)
    n_panels = int(mean_panel_attention.shape[0])
    n_layers = int(mean_panel_attention.shape[1])
    n_heads = int(mean_panel_attention.shape[2])
    gaze_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for panel_idx in range(n_panels):
        gaze_scores += mean_panel_attention[panel_idx, :, :, panel_idx]
    gaze_scores /= float(n_panels)
    gaze_selectivity, gaze_routing_accuracy = gaze_routing_diagnostics(mean_panel_attention)

    final_ranking = rank_heads_by_score(gaze_scores)
    stability = _stability(final_ranking, batch_rankings, top_k=top_k)

    np.save(out_dir / "gaze_sum.npy", merged_sum)
    np.save(out_dir / "mean_panel_attention.npy", mean_panel_attention)
    np.save(out_dir / "gaze_scores.npy", gaze_scores)
    np.save(out_dir / "gaze_selectivity.npy", gaze_selectivity)
    np.save(out_dir / "gaze_routing_accuracy.npy", gaze_routing_accuracy)
    (out_dir / "gaze_head_ranking.json").write_text(json.dumps(final_ranking, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "valid_samples": total_valid,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_panels": n_panels,
                "top_head": final_ranking[0],
                "top_head_selectivity": float(
                    gaze_selectivity[int(final_ranking[0]["layer"]), int(final_ranking[0]["head"])]
                ),
                "top_head_routing_accuracy": float(
                    gaze_routing_accuracy[int(final_ranking[0]["layer"]), int(final_ranking[0]["head"])]
                ),
                "max_head_routing_accuracy": float(gaze_routing_accuracy.max()),
                "merged_from": shard_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / f"top{top_k}_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    _write_stability_tsv(out_dir / f"top{top_k}_stability.tsv", stability)

    result = {
        "valid_samples": total_valid,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_panels": n_panels,
        "top_head": final_ranking[0],
        "merged_from": shard_records,
        "stability": stability,
    }
    print(f"Merged {len(shard_dirs)} shards with {total_valid} valid comics.")
    print(f"Wrote merged ranking to {out_dir / 'gaze_head_ranking.json'}")
    print(f"Wrote stability table to {out_dir / f'top{top_k}_stability.tsv'}")
    return result



def _stability(
    final_ranking: list[dict[str, Any]],
    batch_rankings: list[list[dict[str, Any]]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rank_maps = []
    for ranking in batch_rankings:
        rank_maps.append({_head_id(row): idx + 1 for idx, row in enumerate(ranking)})

    rows = []
    for final_rank, head in enumerate(final_ranking[:top_k], start=1):
        head_id = _head_id(head)
        batch_ranks = [rank_map.get(head_id) for rank_map in rank_maps]
        present_ranks = [rank for rank in batch_ranks if rank is not None]
        rows.append(
            {
                "head": head_id,
                "layer": int(head["layer"]),
                "head_index": int(head["head"]),
                "final_rank": final_rank,
                "final_score": float(head["score"]),
                "mean_batch_rank": float(np.mean(present_ranks)) if present_ranks else None,
                "median_batch_rank": float(np.median(present_ranks)) if present_ranks else None,
                "top5_batches": sum(1 for rank in present_ranks if rank <= 5),
                "top10_batches": sum(1 for rank in present_ranks if rank <= 10),
                "top20_batches": sum(1 for rank in present_ranks if rank <= 20),
                "top50_batches": sum(1 for rank in present_ranks if rank <= 50),
                "n_batches": len(batch_rankings),
            }
        )
    return rows


def _head_id(row: dict[str, Any]) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def _write_stability_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "head",
        "layer",
        "head_index",
        "final_rank",
        "final_score",
        "mean_batch_rank",
        "median_batch_rank",
        "top5_batches",
        "top10_batches",
        "top20_batches",
        "top50_batches",
        "n_batches",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join("" if row[field] is None else str(row[field]) for field in fields) + "\n")


if __name__ == "__main__":
    main()
