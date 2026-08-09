#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
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
from vlm_eval.mechanistic_heads.synthetic import length_matched_nonspatial_answer, point_condition_prompt


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
        known_outputs=(
            "trainer_state.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "config.json",
            "training_context.json",
            "training_summary.json",
        ),
    )
    if args.overwrite:
        remove_declared_training_checkpoints(args.output_dir)
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

    # Qwen3-VL-8B fits comfortably for inference on the 48 GB Neuronic GPUs,
    # but retaining every decoder activation for LoRA backpropagation does
    # not.  Cache tensors are training-incompatible, and non-reentrant
    # checkpointing keeps the activation footprint bounded without changing
    # the effective batch size or optimizer schedule.
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    training_examples = build_training_examples(
        rows,
        condition=condition,
        auxiliary_examples_per_task=int(
            config.get("spatial_contract_examples_per_task", 500)
        ),
        image_size=int(config.get("search_image_size", 224)),
    )

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(training_examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return training_examples[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = []
        prompt_messages = []
        for example in batch:
            row = example["row"]
            answer = str(example["answer"])
            if example["format"] == "direct_length_matched":
                answer = length_matched_nonspatial_answer(
                    processor.tokenizer,
                    direct_answer=str(row["target_count"]),
                    point_answer=str(row["answers"]["point"]),
                )
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = str(example["prompt"])
            user = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
            messages.append(
                [
                    user,
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            )
            prompt_messages.append([user])
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        prompt_encoded = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        encoded["labels"] = labels_after_prompt_prefix(
            encoded["input_ids"],
            encoded["attention_mask"],
            prompt_encoded["input_ids"],
            prompt_encoded["attention_mask"],
        )
        return encoded

    if not training_examples:
        raise RuntimeError("point training dataset has no rows")
    probe = collate(training_examples[:1])
    probe_supervised_tokens = int(probe["labels"].ne(-100).sum().item())
    if probe_supervised_tokens <= 0:
        raise RuntimeError("point training produced no supervised assistant tokens")
    del probe

    max_steps = 2 if smoke else int(config.get("max_steps", -1))
    arguments = make_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        seed=seed,
        config=config,
        smoke=smoke,
        max_steps=max_steps,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=Dataset(), data_collator=collate)
    train_output = trainer.train(
        resume_from_checkpoint=args_resume_checkpoint(output_dir) if resume else None
    )
    train_loss = float(train_output.metrics.get("train_loss", float("nan")))
    if not math.isfinite(train_loss) or train_loss <= 0:
        raise RuntimeError(
            f"point training returned invalid train_loss={train_loss}; "
            "refusing an unsupervised checkpoint"
        )
    trainer.save_model(str(output_dir))
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else replication_label,
        "condition": condition,
        "training_mode": training_mode,
        "n_rows": len(training_examples),
        "n_source_rows": len(rows),
        "training_formats": {
            name: sum(example["format"] == name for example in training_examples)
            for name in sorted({example["format"] for example in training_examples})
        },
        "metrics": train_output.metrics,
        "label_policy": "verified multimodal prompt-prefix boundary",
        "probe_supervised_tokens": probe_supervised_tokens,
        "memory_policy": {
            "use_cache": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
        },
        "deviation": (
            "LoRA pilot; never label as full-weight paper replication"
            if training_mode == "lora"
            else "deterministic text point format replaces paper HTML boxes"
        ),
    }


def build_training_examples(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    auxiliary_examples_per_task: int,
    image_size: int,
) -> list[dict[str, Any]]:
    """Build matched point training plus training-only output-contract tasks.

    Waldo locked-test images remain a true domain transfer. Instead, teach the
    normalized-point, grid-cell, and presence response contracts on ordinary
    point-search training scenes. The shuffled-point control receives the same
    prompts and number of examples, but uses distractor locations for positive
    spatial labels.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"unknown point training condition: {condition}")
    if auxiliary_examples_per_task < 0:
        raise ValueError("spatial_contract_examples_per_task must be nonnegative")
    if image_size <= 1:
        raise ValueError("search_image_size must exceed one pixel")
    answer_key = CONDITIONS[condition]
    examples = [
        {
            "row": row,
            "prompt": point_condition_prompt(row, condition),
            "answer": str(row["answers"][answer_key]),
            "format": (
                "direct_length_matched"
                if condition == "direct_length_matched"
                else "standard"
            ),
        }
        for row in rows
    ]
    auxiliary_rows = rows[: min(len(rows), auxiliary_examples_per_task)]
    if condition not in {"point_answer", "shuffled_point_answer"}:
        # Keep optimizer exposure matched across all trained conditions. The
        # direct controls repeat ordinary examples while the spatial
        # conditions use the same number of contract-alignment examples.
        if examples:
            examples.extend(
                dict(examples[index % len(examples)])
                for index in range(3 * len(auxiliary_rows))
            )
        return examples

    for task in ("normalized_point", "grid_cell", "presence"):
        examples.extend(
            spatial_contract_example(
                row,
                condition=condition,
                task=task,
                image_size=image_size,
            )
            for row in auxiliary_rows
        )
    return examples


def spatial_contract_example(
    row: dict[str, Any],
    *,
    condition: str,
    task: str,
    image_size: int,
) -> dict[str, Any]:
    """Materialize one domain-independent spatial output-contract example."""

    if row.get("split") != "train":
        raise RuntimeError("spatial contract supervision may use only training rows")
    targets = [value for value in row["objects"] if value.get("class") == "target"]
    if len(targets) != int(row["target_count"]) or len(targets) > 1:
        raise RuntimeError("spatial contract rows require zero or one declared target")
    point = tuple(targets[0]["center"]) if targets else None
    if condition == "shuffled_point_answer" and point is not None:
        distractors = [
            value for value in row["objects"] if value.get("class") != "target"
        ]
        if not distractors:
            raise RuntimeError("shuffled spatial contract requires a distractor")
        point = tuple(distractors[0]["center"])
    color = str(row["target"]["color"])
    shape = str(row["target"]["shape"])

    if task == "normalized_point":
        prompt = (
            f"Give the normalized center of the {color} {shape}. "
            "Answer point=(x,y), or absent."
        )
        answer = (
            "absent"
            if point is None
            else f"point=({point[0] / (image_size - 1):.3f},"
            f"{point[1] / (image_size - 1):.3f})"
        )
    elif task == "grid_cell":
        prompt = (
            f"Which 10x10 cell contains the {color} {shape}? "
            "Answer cell=NN, or absent."
        )
        if point is None:
            answer = "absent"
        else:
            column = min(9, int(point[0]) * 10 // image_size)
            row_index = min(9, int(point[1]) * 10 // image_size)
            answer = f"cell={row_index * 10 + column:02d}"
    elif task == "presence":
        prompt = f"Is a {color} {shape} present? Answer present or absent."
        answer = "present" if targets else "absent"
    else:
        raise ValueError(f"unsupported spatial contract task: {task}")
    return {"row": row, "prompt": prompt, "answer": answer, "format": task}


def labels_after_prompt_prefix(
    full_input_ids: Any,
    full_attention_mask: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
) -> Any:
    """Mask the multimodal prompt and supervise only assistant-response tokens.

    Qwen3-VL's current chat template does not expose Jinja generation spans, so
    ``return_assistant_tokens_mask=True`` returns an all-zero mask. Derive the
    boundary by applying the identical template to the user turn with an
    assistant generation prompt, then verify that it is an exact token prefix
    of the full user+assistant conversation. This works with either padding
    side and refuses empty or mismatched supervision.
    """

    import torch

    tensors = (
        full_input_ids,
        full_attention_mask,
        prompt_input_ids,
        prompt_attention_mask,
    )
    if any(not torch.is_tensor(value) or value.ndim != 2 for value in tensors):
        raise ValueError("point-training token IDs and masks must be rank-two tensors")
    if full_input_ids.shape != full_attention_mask.shape:
        raise ValueError("full input IDs and attention mask must align")
    if prompt_input_ids.shape != prompt_attention_mask.shape:
        raise ValueError("prompt input IDs and attention mask must align")
    if full_input_ids.shape[0] != prompt_input_ids.shape[0]:
        raise ValueError("full and prompt batches must have equal size")

    labels = torch.full_like(full_input_ids, -100)
    for batch_index in range(full_input_ids.shape[0]):
        full_positions = full_attention_mask[batch_index].ne(0).nonzero().flatten()
        prompt_positions = prompt_attention_mask[batch_index].ne(0).nonzero().flatten()
        full_tokens = full_input_ids[batch_index].index_select(0, full_positions)
        prompt_tokens = prompt_input_ids[batch_index].index_select(0, prompt_positions)
        if prompt_tokens.numel() >= full_tokens.numel():
            raise RuntimeError("assistant response has no supervised tokens")
        if not torch.equal(full_tokens[: prompt_tokens.numel()], prompt_tokens):
            raise RuntimeError(
                "generation-prompt tokens are not a prefix of the training conversation"
            )
        answer_positions = full_positions[prompt_tokens.numel() :]
        if answer_positions.numel() == 0:
            raise RuntimeError("assistant response has no supervised tokens")
        labels[batch_index, answer_positions] = full_input_ids[batch_index].index_select(
            0, answer_positions
        )
    return labels


def make_training_arguments(
    training_arguments_cls: Any,
    *,
    output_dir: str,
    seed: int,
    config: dict[str, Any],
    smoke: bool,
    max_steps: int,
    bf16: bool,
) -> Any:
    """Construct arguments supported by the locked Transformers release.

    Output replacement is owned by ``prepare_output_directory`` and
    ``remove_declared_training_checkpoints``. Transformers 5.10 removed its
    older ``overwrite_output_dir`` argument, so passing it here both duplicated
    that policy and caused every point-training condition to fail before the
    first optimizer step.
    """

    return training_arguments_cls(
        output_dir=output_dir,
        learning_rate=float(config.get("learning_rate", 1e-5)),
        lr_scheduler_type="cosine",
        warmup_steps=(
            min(1, int(config.get("warmup_steps", 200)))
            if smoke
            else int(config.get("warmup_steps", 200))
        ),
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
    )


def args_resume_checkpoint(output_dir: Path) -> str | None:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    latest = max(
        checkpoints,
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        default=None,
    )
    return str(latest) if latest is not None else None


def remove_declared_training_checkpoints(output_dir: Path) -> None:
    """Remove only numeric Transformers checkpoints for explicit overwrite."""

    for checkpoint in output_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir() or not checkpoint.name.removeprefix("checkpoint-").isdigit():
            continue
        shutil.rmtree(checkpoint)


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
