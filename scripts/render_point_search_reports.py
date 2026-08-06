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
from vlm_eval.mechanistic_heads.preflight import require_calibration_report, require_current_artifact
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


def main()->None:
    parser=argparse.ArgumentParser(description="Render point-search behavioral reports.");add_standard_run_arguments(parser);args=parser.parse_args();config=load_json_config(args.config);table=args.output_dir/"direct_vs_point_behavior.tsv";figure_path=args.output_dir/"point_search_ood_accuracy.png";report=args.output_dir/"report.md";prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(table.name,figure_path.name,report.name,"point_head_family_heatmaps.png","causal_double_dissociation.tsv","causal_double_dissociation.png"))
    rows=[];inputs=[args.config]
    for condition,path_value in config["behavior_sources"].items():
        path=Path(path_value)
        if not path.is_file():raise RuntimeError(f"configured behavior source is missing: {path}")
        require_current_artifact(path)
        inputs.append(path)
        for row in read_tsv(path):rows.append({**row,"condition":condition})
    if not rows:raise RuntimeError("no completed point-search behavior TSVs")
    aggregates=[]
    for (condition,count),group in sorted(groupby(rows,"condition","target_count").items(),key=lambda item:(item[0][0],int(item[0][1]))):aggregates.append({"condition":condition,"target_count":count,"n":len(group),"count_accuracy":sum(int(row["count_correct"]) for row in group)/len(group),"sequence_exact":sum(int(row["sequence_exact"]) for row in group)/len(group),"mean_point_rmse":mean_present(group,"point_rmse")})
    write_tsv(table,aggregates);render_ood(aggregates,figure_path)
    centroid_path=Path(config.get("centroid_summary",""));centroid_figure=None
    if not centroid_path.is_file():raise RuntimeError(f"configured centroid summary is missing: {centroid_path}")
    require_current_artifact(centroid_path);inputs.append(centroid_path);centroid_figure=args.output_dir/"centroid_rmse_by_layer.png";render_centroids(read_tsv(centroid_path),centroid_figure)
    score_figure=args.output_dir/"point_head_family_heatmaps.png";render_head_heatmaps(config["head_score_sources"],score_figure,inputs)
    ablation_path=Path(config["head_ablation"]);require_current_artifact(ablation_path);inputs.append(ablation_path)
    ablation_summary=ablation_path.with_name("summary.json")
    if not args.smoke:
        require_calibration_report(ablation_summary)
        inputs.append(ablation_summary)
    double_table=args.output_dir/"causal_double_dissociation.tsv";double_figure=args.output_dir/"causal_double_dissociation.png";render_double_dissociation(read_tsv(ablation_path),double_table,double_figure)
    label = "instrumentation smoke test" if args.smoke else "modified replication"
    report.write_text("\n".join(["# Point-by-Point Search Report","",f"- Label: {label}","- Point format: deterministic plain text; paper HTML boxes are not used.",f"- Behavioral rows: {len(rows)}","- Conditions: "+", ".join(sorted({row['condition'] for row in rows})),"- OOD target counts: 1, 2, 10, 30, 40, 50.","- Search, verification, and distractor-suppression head sets are each ablated across all three locked tasks.","- Real Waldo data was not downloaded.","","## Files","",f"- `{table.name}`",f"- `{figure_path.name}`",f"- `{score_figure.name}`",f"- `{double_table.name}`",f"- `{double_figure.name}`",*([f"- `{centroid_figure.name}`"] if centroid_figure else [])]),encoding="utf-8")
    outputs=[table,figure_path,report,score_figure,double_table,double_figure]+([centroid_figure] if centroid_figure else []);write_run_manifest(args.output_dir,config={**config,"smoke":args.smoke},seeds={"render":args.seed},inputs=inputs,outputs=outputs,status="complete",repo_root=Path.cwd());print(json.dumps({"valid":True,"label":label,"rows":len(rows),"outputs":[str(path) for path in outputs]},indent=2))


def render_ood(rows,path):
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(8.5,5.2),constrained_layout=True)
    for condition in sorted({row["condition"] for row in rows}):
        group=sorted([row for row in rows if row["condition"]==condition],key=lambda row:int(row["target_count"]));axis.plot([int(row["target_count"]) for row in group],[float(row["count_accuracy"]) for row in group],marker="o",linewidth=2,label=condition.replace("_"," "))
    axis.set_xlabel("OOD number of targets");axis.set_ylabel("Count accuracy");axis.set_ylim(-.02,1.02);axis.set_title("Does point supervision improve extrapolative search?",loc="left",fontweight="bold");axis.legend(frameon=False,fontsize=8);figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)
def render_centroids(rows,path):
    import matplotlib.pyplot as plt
    clean=[row for row in rows if row.get("centroid_rmse_pixels") not in (None,"")];figure,axis=plt.subplots(figsize=(8.5,4.8),constrained_layout=True);axis.plot([int(row["layer"]) for row in clean],[float(row["centroid_rmse_pixels"]) for row in clean],marker="o",linewidth=2,color="#31688e");axis.set_xlabel("Language layer");axis.set_ylabel("Attention-centroid RMSE (pixels)");axis.set_title("Coordinate-token attention localization",loc="left",fontweight="bold");figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)
def render_head_heatmaps(sources,path,inputs):
    import matplotlib.pyplot as plt
    import numpy as np
    figure,axes=plt.subplots(len(sources),1,figsize=(10,3.2*len(sources)),squeeze=False,constrained_layout=True)
    for axis,(name,spec) in zip(axes[:,0],sources.items()):
        source=Path(spec["path"]);manifest=require_current_artifact(source);inputs.append(source);rows=read_tsv(source);n_layers,n_heads=architecture_shape(manifest,source);matrix=np.full((n_layers,n_heads),np.nan);grouped=defaultdict(list)
        for row in rows:grouped[(int(row["layer"]),int(row["head"]))].append(float(row[spec["column"]]))
        for (layer,head),values in grouped.items():
            if not (0 <= layer < n_layers and 0 <= head < n_heads):raise RuntimeError(f"head index {(layer,head)} is outside measured architecture {n_layers}x{n_heads}: {source}")
            matrix[layer,head]=sum(values)/len(values)
        scale=np.nanmax(np.abs(matrix)) or 1;image=axis.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=-scale,vmax=scale);axis.set_title(name.replace("_"," ").title(),loc="left",fontweight="bold");axis.set_xlabel("Head");axis.set_ylabel("Layer");figure.colorbar(image,ax=axis,label="Causal score")
    figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)
def architecture_shape(manifest,source):
    architecture=manifest.get("config",{}).get("architecture",{})
    try:n_layers=int(architecture["n_layers"]);n_heads=int(architecture["n_heads"])
    except (KeyError,TypeError,ValueError) as exc:raise RuntimeError(f"source manifest lacks measured Qwen3 architecture: {source}") from exc
    if n_layers <= 0 or n_heads <= 0:raise RuntimeError(f"invalid measured architecture {n_layers}x{n_heads}: {source}")
    return n_layers,n_heads
def render_double_dissociation(rows,table_path,figure_path):
    relevant=[row for row in rows if row["head_set"].endswith("_top")];aggregates=[]
    for (study,head_set),group in sorted(groupby(relevant,"study","head_set").items()):aggregates.append({"task":study,"head_set":head_set,"n":len(group),"mean_margin_change":sum(float(row["margin_change"]) for row in group)/len(group)})
    if not aggregates:raise RuntimeError("point-head ablation has no cross-task top-set rows")
    write_tsv(table_path,aggregates)
    import matplotlib.pyplot as plt
    tasks=sorted({row["task"] for row in aggregates});sets=sorted({row["head_set"] for row in aggregates});lookup={(row["task"],row["head_set"]):float(row["mean_margin_change"]) for row in aggregates};width=.8/max(1,len(sets));figure,axis=plt.subplots(figsize=(9,5),constrained_layout=True)
    for index,name in enumerate(sets):axis.bar([position-.4+width/2+index*width for position in range(len(tasks))],[lookup.get((task,name),0) for task in tasks],width=width,label=name.replace("_"," "))
    axis.axhline(0,color="#666",linewidth=.8);axis.set_xticks(range(len(tasks)),[task.replace("_"," ") for task in tasks]);axis.set_ylabel("Mean correct-vs-alternative margin change");axis.set_title("Causal task × head-set double dissociation",loc="left",fontweight="bold");axis.legend(frameon=False,fontsize=8);figure.savefig(figure_path,dpi=220,facecolor="white");plt.close(figure)
def groupby(rows,*keys):
    result=defaultdict(list)
    for row in rows:result[tuple(row[key] for key in keys)].append(row)
    return result
def mean_present(rows,key):
    values=[float(row[key]) for row in rows if row.get(key) not in (None,"")];return sum(values)/len(values) if values else ""
def read_tsv(path):
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))
def write_tsv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}) or ["condition"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
