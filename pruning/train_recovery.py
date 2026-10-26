from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .common import atomic_write_json, checkpoint_depth
from .data import ProgressiveDataset, build_anchor_batch, make_data_window
from .losses import (
    boundary_hidden_mse,
    hard_accept_length,
    verifier_corrected_kl,
    verifier_cross_entropy,
)
from .modeling import load_drafter, student_outputs, teacher_outputs
from models.token_ids import checkpoint_mask_id

# Fixed coefficients used for every progressive recovery stage.
LAMBDA_AR = 1.0
LAMBDA_CE = 1.0
LAMBDA_KL = 1.0
LAMBDA_H = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-budget recovery after one physical layer removal."
    )
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-cursor", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--anchors-per-example",
        type=int,
        default=0,
        help="Defaults to 2048 / block_size.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizer", choices=["adamw8bit", "adamw"], default="adamw8bit"
    )
    parser.add_argument("--fsdp-cpu-offload", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.max_steps,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.min_lr,
        args.save_steps,
        args.log_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Step counts and learning rates must be positive")
    if not 0 <= args.data_cursor or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid data cursor or warmup ratio")
    if args.candidate_multiplier < 1:
        raise ValueError("candidate-multiplier must be at least one")
    if args.min_lr > args.learning_rate:
        raise ValueError("min-lr cannot exceed learning-rate")
    if args.optimizer == "adamw8bit" and args.fsdp_cpu_offload:
        raise ValueError("adamw8bit does not support FSDP CPU parameter offload")


def freeze_input_output_layers(model: Any) -> None:
    model.model.embed_tokens.requires_grad_(False)
    model.lm_head.requires_grad_(False)


def shard_teacher(model: Any, dtype: torch.dtype, world_size: int) -> Any:
    if world_size == 1:
        return model
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=dtype, reduce_dtype=dtype, output_dtype=dtype
    )
    for layer in model.model.layers:
        fully_shard(layer, reshard_after_forward=True, mp_policy=policy)
    fully_shard(model, reshard_after_forward=True, mp_policy=policy)
    return model


def build_optimizer(args: argparse.Namespace, model: Any) -> torch.optim.Optimizer:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.optimizer == "adamw":
        return AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=not args.fsdp_cpu_offload,
        )
    from torchao.optim import AdamW8bit

    return AdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def learning_rate(
    step: int, total_steps: int, base: float, minimum: float, warmup_ratio: float
) -> float:
    warmup = round(total_steps * warmup_ratio) if warmup_ratio else 0
    if warmup and step <= warmup:
        progress = 1.0 if warmup == 1 else (step - 1) / (warmup - 1)
        return minimum + (base - minimum) * progress
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (base - minimum) * cosine


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(value)
        else:
            group["lr"] = value


def distributed_mean(value: torch.Tensor) -> float:
    value = value.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return float(value.item())


def next_recovery_example(data_iterator: Any, *, device: torch.device) -> Any:
    """Fetch on all ranks or fail together, before any model/loss collective."""
    example = None
    error = None
    try:
        example = next(data_iterator)
    except StopIteration:
        error = (
            "candidate window exhausted after filtering; increase "
            "--candidate-multiplier or check dataset lengths"
        )
    except Exception as exc:
        # Malformed rows/read errors must not strand peers in FSDP either.
        error = f"{type(exc).__name__}: {exc}"

    errors = [error]
    if dist.is_available() and dist.is_initialized():
        ready = torch.tensor(int(error is None), device=device, dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if ready.item():
            return example
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(message is not None for message in errors):
        details = "; ".join(
            f"rank {rank}: {message}"
            for rank, message in enumerate(errors)
            if message is not None
        )
        raise RuntimeError(
            "Recovery data unavailable before the requested step budget "
            f"completed ({details})"
        )
    return example


def autoregressive_loss(student: Any, batch: Any) -> torch.Tensor:
    positions = torch.arange(
        batch.prompt_length - 1,
        batch.clean_input_ids.shape[1] - 1,
        device=batch.clean_input_ids.device,
    )
    output = student(
        batch.clean_input_ids,
        attention_mask=batch.clean_attention_mask,
        position_ids=torch.arange(
            batch.clean_input_ids.shape[1], device=batch.clean_input_ids.device
        ).unsqueeze(0),
        use_cache=False,
        store_kv=False,
        logits_to_keep=positions,
    )
    targets = batch.clean_input_ids[:, positions + 1]
    return verifier_cross_entropy(output.logits, targets, torch.ones_like(targets))


def save_checkpoint(
    *,
    accelerator: Any,
    student: Any,
    tokenizer: Any,
    initial_checkpoint: Path,
    output_dir: Path,
    step: int,
    run_config: dict[str, Any],
) -> Path:
    state_dict = accelerator.get_state_dict(student)
    checkpoint = output_dir / f"step-{step}"
    if accelerator.is_main_process:
        if checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite {checkpoint}")
        with tempfile.TemporaryDirectory(
            prefix=f".{checkpoint.name}.", dir=output_dir
        ) as temporary_name:
            temporary = Path(temporary_name)
            accelerator.unwrap_model(student).save_pretrained(
                temporary,
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size="2GB",
            )
            tokenizer.save_pretrained(temporary)
            shutil.copy2(
                initial_checkpoint / "layer_prune.json",
                temporary / "layer_prune.json",
            )
            atomic_write_json(
                temporary / "recovery_info.json",
                {**run_config, "optimizer_step": step, "checkpoint": str(checkpoint)},
            )
            os.replace(temporary, checkpoint)
    accelerator.wait_for_everyone()
    return checkpoint


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    student_path = args.student.resolve()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    metadata = json.loads((student_path / "layer_prune.json").read_text())
    block_size = int(metadata["block_size"])
    child_depth = int(metadata["child_depth"])
    teacher_path = Path(metadata["teacher_checkpoint"]).resolve()
    mask_token_id = checkpoint_mask_id(student_path)
    checkpoint_mask_id(teacher_path, mask_token_id)
    retained_original_layers = [
        int(layer) for layer in metadata["retained_original_layers"]
    ]
    if checkpoint_depth(student_path) != child_depth:
        raise ValueError("Student config and layer-pruning metadata disagree")
    if len(retained_original_layers) != child_depth:
        raise ValueError("Retained teacher-layer mapping has the wrong length")
    anchors = args.anchors_per_example or 2048 // block_size
    if anchors <= 0 or anchors * block_size > 2048:
        raise ValueError("anchors-per-example exceeds the 2048-token draft budget")
    position_decay = {16: 7.75, 32: 15.5}.get(block_size)
    if position_decay is None:
        raise ValueError("The reference implementation supports block size 16 or 32")

    from accelerate import Accelerator, FullyShardedDataParallelPlugin

    fsdp = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["D3DDecoderLayer"],
        mixed_precision_policy="bf16",
        cpu_offload=args.fsdp_cpu_offload,
        state_dict_type="FULL_STATE_DICT",
        activation_checkpointing=True,
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=fsdp,
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    with dataset_path.open("rb") as stream:
        dataset_rows = sum(bool(line.strip()) for line in stream)
    window = make_data_window(
        cursor=args.data_cursor,
        steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=world_size,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
        dataset_rows=dataset_rows,
    )
    tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    teacher = load_drafter(str(teacher_path), torch.bfloat16).to(device).eval()
    teacher.requires_grad_(False)
    teacher = shard_teacher(teacher, torch.bfloat16, world_size)
    student = load_drafter(str(student_path), torch.float32)
    student.train()
    student.config.use_cache = False
    student.config.mask_token_id = mask_token_id
    freeze_input_output_layers(student)
    optimizer = build_optimizer(args, student)
    student, optimizer = accelerator.prepare(student, optimizer)

    run_config = {
        "format_version": 1,
        "method": "progressive_layer_pruning_fixed_budget_recovery",
        "mask_token_id": mask_token_id,
        "student": str(student_path),
        "teacher": str(teacher_path),
        "child_depth": child_depth,
        "retained_original_layers": retained_original_layers,
        "block_size": block_size,
        "loss_weights": {
            "autoregressive": LAMBDA_AR,
            "cross_entropy": LAMBDA_CE,
            "kl": LAMBDA_KL,
            "hidden": LAMBDA_H,
        },
        "frozen_modules": ["model.embed_tokens", "lm_head"],
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "warmup_ratio": args.warmup_ratio,
        "optimizer": args.optimizer,
        "data_window": window.as_dict(),
        "dataset": str(dataset_path),
        "seed": args.seed,
    }
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True)
        atomic_write_json(output_dir / "run_config.json", run_config)
        print(
            "Loss weights: "
            f"AR={LAMBDA_AR:g}, CE={LAMBDA_CE:g}, KL={LAMBDA_KL:g}, H={LAMBDA_H:g}",
            flush=True,
        )
    accelerator.wait_for_everyone()

    dataset = ProgressiveDataset(
        dataset_path,
        tokenizer,
        window=window,
        rank=rank,
        max_prompt_tokens=args.max_prompt_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
        minimum_response_tokens=block_size,
    )
    data_iterator = iter(DataLoader(dataset, batch_size=None, num_workers=0))
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    metrics_path = output_dir / "training_metrics.jsonl"
    started = time.time()
    step_started = time.perf_counter()
    last_checkpoint: Path | None = None
    progress = (
        tqdm(
            total=args.max_steps, desc=f"[recovery D{child_depth}]", dynamic_ncols=True
        )
        if accelerator.is_main_process
        else None
    )

    while optimizer_step < args.max_steps:
        virtual_position, example = next_recovery_example(data_iterator, device=device)
        batch = build_anchor_batch(
            example,
            block_size=block_size,
            max_anchors=anchors,
            namespace=f"d3d-prune-{args.seed}-{virtual_position}",
            device=device,
            mask_token_id=mask_token_id,
        )
        if batch is None:
            continue

        with accelerator.accumulate(student):
            with accelerator.autocast():
                ar_loss = autoregressive_loss(student, batch)
            accelerator.backward(LAMBDA_AR * ar_loss)

            with accelerator.autocast():
                teacher_logits, teacher_hidden = teacher_outputs(
                    teacher, batch, retained_original_layers
                )
                student_logits, student_hidden = student_outputs(
                    student, batch, child_depth
                )
                offsets = torch.arange(
                    block_size - 1, device=device, dtype=torch.float32
                )
                token_weights = torch.exp(-offsets / position_decay)[None, :]
                token_weights = token_weights * batch.active_token_mask
                ce_loss = verifier_cross_entropy(
                    student_logits, batch.target_ids, token_weights
                )
                kl_loss = verifier_corrected_kl(
                    student_logits,
                    teacher_logits,
                    batch.target_ids,
                    token_weights,
                )
                hidden_loss = boundary_hidden_mse(
                    student_hidden, teacher_hidden, token_weights
                )
                recovery_loss = (
                    LAMBDA_CE * ce_loss + LAMBDA_KL * kl_loss + LAMBDA_H * hidden_loss
                )
            accelerator.backward(recovery_loss)

            if accelerator.sync_gradients:
                gradient_norm = accelerator.clip_grad_norm_(
                    student.parameters(), args.gradient_clip
                )
                optimizer_step += 1
                current_lr = learning_rate(
                    optimizer_step,
                    args.max_steps,
                    args.learning_rate,
                    args.min_lr,
                    args.warmup_ratio,
                )
                set_learning_rate(optimizer, current_lr)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                student_accept = (
                    hard_accept_length(
                        student_logits.detach().argmax(-1),
                        batch.target_ids,
                        batch.active_token_mask,
                    )
                    .float()
                    .mean()
                )
                teacher_accept = (
                    hard_accept_length(
                        teacher_logits.argmax(-1),
                        batch.target_ids,
                        batch.active_token_mask,
                    )
                    .float()
                    .mean()
                )
                total_loss = LAMBDA_AR * ar_loss.detach() + recovery_loss.detach()
                record = {
                    "optimizer_step": optimizer_step,
                    "loss": distributed_mean(total_loss),
                    "autoregressive_loss": distributed_mean(ar_loss.detach()),
                    "cross_entropy": distributed_mean(ce_loss.detach()),
                    "kl": distributed_mean(kl_loss.detach()),
                    "hidden": distributed_mean(hidden_loss.detach()),
                    "student_accept_length": distributed_mean(student_accept),
                    "teacher_accept_length": distributed_mean(teacher_accept),
                    "learning_rate": current_lr,
                    "gradient_norm": float(gradient_norm),
                    "step_seconds": time.perf_counter() - step_started,
                    "elapsed_seconds": time.time() - started,
                }
                if accelerator.is_main_process:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    assert progress is not None
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{record['loss']:.3f}",
                        accept=f"{record['student_accept_length']:.2f}",
                        grad=f"{record['gradient_norm']:.2f}",
                        lr=f"{current_lr:.2e}",
                        refresh=optimizer_step % args.log_steps == 0,
                    )

                if (
                    optimizer_step % args.save_steps == 0
                    or optimizer_step == args.max_steps
                ):
                    last_checkpoint = save_checkpoint(
                        accelerator=accelerator,
                        student=student,
                        tokenizer=tokenizer,
                        initial_checkpoint=student_path,
                        output_dir=output_dir,
                        step=optimizer_step,
                        run_config=run_config,
                    )
                step_started = time.perf_counter()

    if progress is not None:
        progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        assert last_checkpoint is not None
        atomic_write_json(
            output_dir / "training_summary.json",
            {
                **run_config,
                "completed_steps": optimizer_step,
                "next_data_cursor": window.cursor_end,
                "final_checkpoint": str(last_checkpoint),
                "elapsed_seconds": time.time() - started,
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
