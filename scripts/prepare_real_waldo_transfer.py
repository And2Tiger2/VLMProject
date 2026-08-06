#!/usr/bin/env python3
"""Prepare an explicitly downloaded real-Waldo dataset for a later locked transfer.

This script never downloads data. It accepts COCO annotations only and refuses
to proceed without a license file and recoverable original-page identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.splits import group_split


def main()->None:
    parser=argparse.ArgumentParser(description="Audit and prepare a user-downloaded real-Waldo locked transfer.");add_standard_run_arguments(parser);parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();config=load_json_config(args.config);output=args.output_dir/"real_waldo_transfer.jsonl";prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(output.name,"audit.json"));seed_everything(args.seed)
    annotation=args.input_dir/str(config["coco_annotation"]);license_paths=[args.input_dir/value for value in config.get("license_files",["LICENSE","LICENSE.txt","README.md"]) if (args.input_dir/value).is_file()]
    if not license_paths:raise RuntimeError("real-Waldo transfer refused: no license file found")
    if not annotation.is_file():raise RuntimeError(f"COCO annotation file is missing: {annotation}")
    coco=json.loads(annotation.read_text(encoding="utf-8"));categories={int(row["id"]):str(row["name"]) for row in coco.get("categories",[])};targets={cid for cid,name in categories.items() if name.casefold() in {value.casefold() for value in config.get("target_categories",["waldo"])}}
    if not targets:raise RuntimeError(f"no configured target category in annotation classes: {categories}")
    annotations={}
    for row in coco.get("annotations",[]):
        if int(row["category_id"]) in targets:annotations.setdefault(int(row["image_id"]),[]).append(row)
    pattern=re.compile(str(config["page_id_regex"]));records=[];limit=effective_limit(args);images=coco.get("images",[])[:limit]
    for row in images:
        match=pattern.search(str(row["file_name"]))
        if not match:raise RuntimeError(f"cannot recover original page ID from {row['file_name']!r}")
        page_id=match.group("page") if "page" in match.groupdict() else match.group(1);boxes=annotations.get(int(row["id"]),[])
        image_path=(args.input_dir/str(row["file_name"])).resolve()
        if not image_path.is_file():raise RuntimeError(f"missing image: {image_path}")
        for box_index,annotation_row in enumerate(boxes):
            x,y,width,height=(float(value) for value in annotation_row["bbox"]);records.append({"id":f"{row['id']}-{box_index}","page_id":page_id,"image_id":int(row["id"]),"image_path":str(image_path),"category":categories[int(annotation_row["category_id"])],"bbox_xywh":[x,y,width,height],"cell_10x10":bbox_cell(x,y,width,height,int(row["width"]),int(row["height"])),"annotation_id":annotation_row.get("id")})
    grouped=group_split(records,group_key=lambda item:item["page_id"],fractions={"train":.5,"validation":.2,"locked_test":.3},seed=args.seed);records=[{**row,"split":split,"zoom_conditions":render_zoom_conditions(row,args.output_dir,args.seed)} for split,rows in grouped.items() for row in rows]
    owners={}
    for row in records:
        previous=owners.setdefault(row["page_id"],row["split"])
        if previous!=row["split"]:raise RuntimeError("crop-level/page leakage detected")
    output.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in records),encoding="utf-8");audit={"valid":bool(records),"label":"dataset preparation","license_files":[{"path":str(path),"sha256":sha(path),"text_excerpt":path.read_text(encoding="utf-8",errors="replace")[:400]} for path in license_paths],"annotation_classes":categories,"target_categories":[categories[value] for value in sorted(targets)],"n_images":len({row["image_id"] for row in records}),"n_boxes":len(records),"n_pages":len(owners),"split_pages":{split:len({row["page_id"] for row in records if row["split"]==split}) for split in grouped},"split_policy":"original page ID only; crop-level random split refused","head_scoring_geometry":"exact boxes","behavioral_geometry":"10x10 cell only","errors":[]};audit_path=args.output_dir/"audit.json";audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    referenced=referenced_image_paths(records);output_root=args.output_dir.resolve();derived=[path for path in referenced if path.resolve().is_relative_to(output_root)];source_images=[path for path in referenced if not path.resolve().is_relative_to(output_root)]
    write_run_manifest(args.output_dir,config=config,seeds={"split":args.seed,"matched_crop":args.seed},inputs=[args.config,annotation,*license_paths,*source_images],outputs=[output,audit_path,*derived],status="complete",repo_root=Path.cwd());print(json.dumps(audit,indent=2))


def render_zoom_conditions(row:dict[str,Any],output_dir:Path,seed:int)->dict[str,str]:
    image=Image.open(row["image_path"]).convert("RGB");x,y,width,height=row["bbox_xywh"];box=(max(0,int(x)),max(0,int(y)),min(image.width,int(x+width)),min(image.height,int(y+height)));root=output_dir/"zoom_conditions"/row["id"];root.mkdir(parents=True,exist_ok=True);paths={"standard_full":row["image_path"]}
    high=image.resize((image.width*2,image.height*2),Image.Resampling.LANCZOS);paths["higher_resolution_full"]=save(high,root/"higher_resolution_full.png")
    uncluttered=Image.new("RGB",image.size,"white");uncluttered.paste(image.crop(box),box);paths["same_canvas_clutter_mask"]=save(uncluttered,root/"same_canvas_clutter_mask.png")
    crop=image.crop(box);tight=Image.new("RGB",image.size,"white");fitted=crop.copy();fitted.thumbnail(image.size,Image.Resampling.LANCZOS);tight.paste(fitted,((image.width-fitted.width)//2,(image.height-fitted.height)//2));paths["tight_crop_reencoded"]=save(tight,root/"tight_crop_reencoded.png")
    plus=Image.new("RGB",(image.width*2,image.height),"white");plus.paste(image,(0,0));large=crop.copy();large.thumbnail((image.width,image.height),Image.Resampling.LANCZOS);plus.paste(large,(image.width+(image.width-large.width)//2,(image.height-large.height)//2));paths["full_plus_crop"]=save(plus,root/"full_plus_crop.png")
    rng=random.Random(f"{seed}:{row['page_id']}:{row['id']}");cw,ch=max(1,box[2]-box[0]),max(1,box[3]-box[1]);candidates=[(rng.randrange(max(1,image.width-cw+1)),rng.randrange(max(1,image.height-ch+1))) for _ in range(100)];rx,ry=next(((a,b) for a,b in candidates if a+cw<=box[0] or a>=box[2] or b+ch<=box[1] or b>=box[3]),candidates[-1]);random_crop=image.crop((rx,ry,rx+cw,ry+ch));matched=Image.new("RGB",image.size,"white");random_crop.thumbnail(image.size,Image.Resampling.LANCZOS);matched.paste(random_crop,((image.width-random_crop.width)//2,(image.height-random_crop.height)//2));paths["random_matched_crop"]=save(matched,root/"random_matched_crop.png");return paths
def bbox_cell(x,y,w,h,image_width,image_height):return int(min(9,max(0,(y+h/2)/image_height*10)))*10+int(min(9,max(0,(x+w/2)/image_width*10)))
def save(image,path):image.save(path);return str(path.resolve())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=="__main__":main()
