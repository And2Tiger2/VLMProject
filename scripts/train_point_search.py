#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from adapters.qwen25_vl import _resolve_device_map, _resolve_torch_dtype
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.reproducibility import git_sha, hash_paths, referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs
from vlm_eval.mechanistic_heads.synthetic import length_matched_nonspatial_answer


CONDITIONS = {
    "base": "base",
    "direct_answer": "direct",
    "direct_length_matched": "direct_length_matched",
    "point_answer": "point",
    "shuffled_point_answer": "shuffled_point",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched Qwen3 point-search conditions.")
    add_standard_run_arguments(parser)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke and args.condition != "base":
        require_scientific_validation(validation_path_from_config(config))
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=("trainer_state.json", "adapter_config.json", "config.json"),
    )
    seed_everything(args.seed)
    rows = _read_jsonl(Path(config["dataset"]))
    rows = [row for row in rows if row.get("split") == "train"]
    limit = effective_limit(args)
    if limit is not None:
        rows = rows[:limit]
    training_inputs = [args.config, Path(config["dataset"]), *referenced_image_paths(rows)]
    if args.condition == "base":
        result = {
            "valid": True,
            "label": "instrumentation smoke test" if args.smoke else "base model evaluation condition",
            "trained": False,
            "n_rows": len(rows),
        }
    else:
        training_context = {
            "schema_version": 1,
            "git_sha": git_sha(Path.cwd()),
            "config": config,
            "condition": args.condition,
            "seed": args.seed,
            "smoke": args.smoke,
            "input_sha256": hash_paths(training_inputs),
        }
        context_path = args.output_dir / "training_context.json"
        validate_or_write_training_context(
            context_path, training_context, resume=args.resume
        )
        result = train_condition(
            rows,
            output_dir=args.output_dir,
            condition=args.condition,
            config=config,
            seed=args.seed,
            smoke=args.smoke,
            resume=args.resume,
            overwrite=args.overwrite,
            device_map=args.device_map,
        )
    summary = args.output_dir / "training_summary.json"
    summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint_outputs = checkpoint_manifest_inputs(
        {"adapter_path": str(args.output_dir)}
    )
    write_run_manifest(
        args.output_dir,
        config={**config, "condition": args.condition, "smoke": args.smoke},
        seeds={"global": args.seed},
        inputs=training_inputs,
        outputs=sorted(set([summary, *checkpoint_outputs])),
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(result, indent=2))


def train_condition(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    condition: str,
    config: dict[str, Any],
    seed: int,
    smoke: bool,
    resume: bool,
    overwrite: bool,
    device_map: str,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with `uv sync --extra qwen`.") from exc

    model_id = str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=_resolve_torch_dtype(torch),
        device_map=_resolve_device_map(device_map, torch),
    )
    training_mode = str(config.get("training_mode", "lora"))
    if training_mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA training requires `uv sync --extra mechanistic`.") from exc
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(config.get("lora_r", 16)),
                lora_alpha=int(config.get("lora_alpha", 32)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=list(config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
                task_type="CAUSAL_LM",
            ),
        )
        replication_label = "modified replication"
    elif training_mode == "full_weight":
        replication_label = "methods-based reproduction"
    else:
        raise ValueError(f"unknown training_mode: {training_mode}")

    answer_key = CONDITIONS[condition]

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return rows[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = []
        for row in batch:
            answer = str(row["answers"][answer_key])
            if condition == "direct_length_matched":
                answer = length_matched_nonspatial_answer(
                    processor.tokenizer,
                    direct_answer=str(row["target_count"]),
                    point_answer=str(row["answers"]["point"]),
                )
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(row["image_path"]).convert("RGB")},
                            {"type": "text", "text": str(row["prompt"])},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            )
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            return_assistant_tokens_mask=True,
        )
        assistant_mask = encoded.pop("assistant_masks", None)
        if assistant_mask is None:
            assistant_mask = encoded.pop("assistant_tokens_mask", None)
        if assistant_mask is None:
            raise RuntimeError(
                "Qwen chat template did not return an assistant-token mask; refusing to train on prompt tokens"
            )
        if not torch.is_tensor(assistant_mask):
            assistant_mask = torch.as_tensor(assistant_mask, device=encoded["input_ids"].device)
        encoded["labels"] = encoded["input_ids"].clone()
        encoded["labels"][assistant_mask.eq(0)] = -100
        encoded["labels"][encoded["attention_mask"].eq(0)] = -100
        return encoded

    max_steps = 2 if smoke else int(config.get("max_steps", -1))
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=overwrite,
        learning_rate=float(config.get("learning_rate", 1e-5)),
        lr_scheduler_type="cosine",
        warmup_steps=min(1, int(config.get("warmup_steps", 200))) if smoke else int(config.get("warmup_steps", 200)),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(config.get("epochs", 1)),
        max_steps=max_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=int(config.get("save_steps", 50)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        logging_steps=1,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=Dataset(), data_collator=collate)
    train_output = trainer.train(
        resume_from_checkpoint=args_resume_checkpoint(output_dir) if resume else None
    )
    trainer.save_model(str(output_dir))
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else replication_label,
        "condition": condition,
        "training_mode": training_mode,
        "n_rows": len(rows),
        "metrics": train_output.metrics,
        "deviation": (
            "LoRA pilot; never label as full-weight paper replication"
            if training_mode == "lora"
            else "deterministic text point format replaces paper HTML boxes"
        ),
    }


def args_resume_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    return str(checkpoints[-1]) if checkpoints else None


def validate_or_write_training_context(path: Path, expected: dict[str, Any], *, resume: bool) -> None:
    checkpoints = list(path.parent.glob("checkpoint-*"))
    if resume and checkpoints:
        if not path.is_file():
            raise RuntimeError("training checkpoints exist without a reproducibility context")
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("training checkpoint context does not match this run")
    path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
