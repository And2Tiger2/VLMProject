#!/usr/bin/env python3
"""Render direct-versus-point behavioral and centroid reports as PNG/TSV/Markdown."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


def main()->None:
    parser=argparse.ArgumentParser(description="Render point-search behavioral reports.");add_standard_run_arguments(parser);args=parser.parse_args();config=load_json_config(args.config);table=args.output_dir/"direct_vs_point_behavior.tsv";figure_path=args.output_dir/"point_search_ood_accuracy.png";report=args.output_dir/"report.md";prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(table.name,figure_path.name,report.name))
    rows=[];inputs=[args.config]
    for condition,path_value in config["behavior_sources"].items():
        path=Path(path_value)
        if not path.is_file():continue
        inputs.append(path)
        for row in read_tsv(path):rows.append({**row,"condition":condition})
    if not rows:raise RuntimeError("no completed point-search behavior TSVs")
    aggregates=[]
    for (condition,count),group in sorted(groupby(rows,"condition","target_count").items(),key=lambda item:(item[0][0],int(item[0][1]))):aggregates.append({"condition":condition,"target_count":count,"n":len(group),"count_accuracy":sum(int(row["count_correct"]) for row in group)/len(group),"sequence_exact":sum(int(row["sequence_exact"]) for row in group)/len(group),"mean_point_rmse":mean_present(group,"point_rmse")})
    write_tsv(table,aggregates);render_ood(aggregates,figure_path)
    centroid_path=Path(config.get("centroid_summary",""));centroid_figure=None
    if centroid_path.is_file():inputs.append(centroid_path);centroid_figure=args.output_dir/"centroid_rmse_by_layer.png";render_centroids(read_tsv(centroid_path),centroid_figure)
    report.write_text("\n".join(["# Point-by-Point Search Report","","- Label: modified replication","- Point format: deterministic plain text; paper HTML boxes are not used.",f"- Behavioral rows: {len(rows)}","- Conditions: "+", ".join(sorted({row['condition'] for row in rows})),"- OOD target counts: 1, 2, 10, 30, 40, 50.","- Real Waldo data was not downloaded.","","## Files","",f"- `{table.name}`",f"- `{figure_path.name}`",*([f"- `{centroid_figure.name}`"] if centroid_figure else [])]),encoding="utf-8")
    outputs=[table,figure_path,report]+([centroid_figure] if centroid_figure else []);write_run_manifest(args.output_dir,config=config,seeds={"render":args.seed},inputs=inputs,outputs=outputs,status="complete",repo_root=Path.cwd());print(json.dumps({"valid":True,"rows":len(rows),"outputs":[str(path) for path in outputs]},indent=2))


def render_ood(rows,path):
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(8.5,5.2),constrained_layout=True)
    for condition in sorted({row["condition"] for row in rows}):
        group=sorted([row for row in rows if row["condition"]==condition],key=lambda row:int(row["target_count"]));axis.plot([int(row["target_count"]) for row in group],[float(row["count_accuracy"]) for row in group],marker="o",linewidth=2,label=condition.replace("_"," "))
    axis.set_xlabel("OOD number of targets");axis.set_ylabel("Count accuracy");axis.set_ylim(-.02,1.02);axis.set_title("Does point supervision improve extrapolative search?",loc="left",fontweight="bold");axis.legend(frameon=False,fontsize=8);figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)
def render_centroids(rows,path):
    import matplotlib.pyplot as plt
    clean=[row for row in rows if row.get("centroid_rmse_pixels") not in (None,"")];figure,axis=plt.subplots(figsize=(8.5,4.8),constrained_layout=True);axis.plot([int(row["layer"]) for row in clean],[float(row["centroid_rmse_pixels"]) for row in clean],marker="o",linewidth=2,color="#31688e");axis.set_xlabel("Language layer");axis.set_ylabel("Attention-centroid RMSE (pixels)");axis.set_title("Coordinate-token attention localization",loc="left",fontweight="bold");figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)
def groupby(rows,*keys):
    result=defaultdict(list)
    for row in rows:result[tuple(row[key] for key in keys)].append(row)
    return result
def mean_present(rows,key):
    values=[float(row[key]) for row in rows if row.get(key) not in (None,"")];return sum(values)/len(values) if values else ""
def read_tsv(path):
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))
def write_tsv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["condition"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
