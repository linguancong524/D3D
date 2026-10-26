from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, PretrainedConfig
from models.token_ids import DEFAULT_MASK_TOKEN_ID, checkpoint_mask_id

from .trajectory import block_accept_lengths, distributed_block_mean_loss, relational_trajectory_loss, trajectory_selection_mask
from .runtime import DistContext, WandbRun, all_reduce_sum_tensor, accumulation_window_size, build_d3d_fsdp_plugin_from_args, distributed_context_from_accelerator, dummy_zero_backward, dtype_from_arg, load_model, lr_for_step, mixed_precision_from_dtype, resolve_mask_token_id, save_checkpoint, set_optimizer_lr, should_save_final_checkpoint, should_step_optimizer
from .dataset import PackedBatch, build_loader, count_jsonl_rows
from .types import ParallelOfflineBatch
from .losses import _accept_len_sum_from_blocks, acceptance_target_ids, active_loss_weight_sum, active_token_loss_weights, clean_ar_loss_sum, normalize_objective, objective_requires_target, per_token_ce_aux_losses, weighted_objective_loss_sum
from .model_utils import capture_final_norm_hidden, dummy_clean_cache_forward, dummy_zero_forward_loss, enable_causal_flash_attention_compatibility, enable_gradient_checkpointing, load_draft_checkpoint, load_random_model, packed_draft_forward, prefill_clean_kv_cache, teacher_logits_at_positions, tokenizer_load_kwargs
from .arguments import parse_args
from .blocks import to_device

class TensorboardRun:
    """Small SummaryWriter wrapper with the same scalar payload shape as wandb."""

    def __init__(self, enabled: bool, *, log_dir: Path, config: dict[str, Any]):
        self.enabled = False
        self._writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[tensorboard] tensorboard not installed; logging disabled", flush=True)
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._writer.add_text(
            "config/json",
            json.dumps(config, indent=2, sort_keys=True),
            global_step=0,
        )
        self.enabled = True

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=step)

    def finish(self) -> None:
        if self.enabled:
            self._writer.flush()
            self._writer.close()


def train_packed_block_batch(
    *,
    draft_ddp,
    draft_module,
    verifier,
    batch: ParallelOfflineBatch | None,
    global_objective_weight_sum: torch.Tensor,
    global_ar_token_count: torch.Tensor,
    ctx: DistContext,
    objective: str,
    mask_token_id: int,
    device: torch.device,
    loss_scale: float = 1.0,
    ce_weight: float = 1.0,
    hnce_weight: float = 0.0,
    hnce_top_k: int = 16,
    hnce_margin: float = 0.0,
    margin_weight: float = 0.0,
    margin_top_k: int = 4,
    margin_value: float = 0.5,
    margin_smooth: bool = False,
    loss_decay_gamma: float = 15.5,
    gradient_checkpointing: bool = False,
    ar_loss_weight: float = 0.0,
    ar_logit_chunk_size: int = 512,
    trajectory_teacher=None,
    trajectory_weight: float = 0.0,
    trajectory_shuffle_teacher_blocks: bool = False,
    trajectory_shuffle_seed: int = 0,
    trajectory_gradient_probe: bool = False,
    block_forward_context=nullcontext,
) -> dict[str, float]:
    """Run one packed offline D3D training micro-batch and return metrics."""

    objective = normalize_objective(objective)
    trajectory_enabled = float(trajectory_weight) > 0.0
    if trajectory_enabled and trajectory_teacher is None:
        raise ValueError("trajectory_teacher is required when trajectory_weight > 0")
    if batch is None:
        if float(ar_loss_weight) > 0.0:
            dummy_zero_backward(draft_ddp, mask_token_id=mask_token_id, device=device)
        dummy_clean_cache_forward(
            draft_module,
            mask_token_id=mask_token_id,
            device=device,
        )
        if trajectory_enabled:
            # Match the valid rank's student block forward, participate in the
            # selected-block collective, then enter backward in the same order.
            dummy_loss = dummy_zero_forward_loss(
                draft_ddp,
                mask_token_id=mask_token_id,
                device=device,
            )
            all_reduce_sum_tensor(
                torch.zeros((), dtype=torch.float32, device=device)
            )
            (dummy_loss * float(loss_scale)).backward()
        else:
            dummy_zero_backward(draft_ddp, mask_token_id=mask_token_id, device=device)
        return {
            "weighted_loss_sum": 0.0,
            "objective_weight_sum": 0.0,
            "tokens": 0.0,
            "blocks": 0.0,
            "top1_match_sum": 0.0,
            "accept_len_sum": 0.0,
            "weighted_ce_sum": 0.0,
            "weighted_hnce_sum": 0.0,
            "weighted_margin_sum": 0.0,
            "ar_loss_sum": 0.0,
            "ar_tokens": 0.0,
            "trajectory_loss_sum": 0.0,
            "trajectory_relation_abs_error_sum": 0.0,
            "trajectory_selected_blocks": 0.0,
            "trajectory_teacher_better_blocks": 0.0,
            "trajectory_both_full_blocks": 0.0,
            "trajectory_teacher_accept_sum": 0.0,
            "trajectory_student_accept_sum": 0.0,
            "trajectory_ce_hidden_grad_sq_sum": 0.0,
            "trajectory_unit_hidden_grad_sq_sum": 0.0,
        }

    ar_loss_sum_value = 0.0
    ar_tokens = 0
    if float(ar_loss_weight) > 0.0:
        # Run AR training without a live KV cache so gradient-checkpointed
        # activations and the FSDP root all-gather do not overlap with clean KV.
        ar_loss_sum, ar_tokens = clean_ar_loss_sum(
            draft_module,
            batch.clean_input_ids,
            batch.prompt_lens,
            batch.clean_lens,
            ar_logit_chunk_size=int(ar_logit_chunk_size),
        )
        ar_denom = global_ar_token_count.to(device=device, dtype=torch.float32).clamp_min(1.0)
        scaled_ar_loss = (
            ar_loss_sum
            * (float(ctx.world_size) / ar_denom)
            * float(ar_loss_weight)
            * float(loss_scale)
        )
        scaled_ar_loss.backward()
        ar_loss_sum_value = float(ar_loss_sum.detach())
        del scaled_ar_loss
        del ar_loss_sum
        # Build reusable KV only after AR backward has released its activations.
        clean_cache = prefill_clean_kv_cache(
            draft_module,
            batch.clean_input_ids,
            batch.clean_lens,
        )
    else:
        clean_cache = prefill_clean_kv_cache(
            draft_module,
            batch.clean_input_ids,
            batch.clean_lens,
        )
    teacher_logits = None
    if verifier is not None:
        # Teacher logits are evaluated on clean AR positions, not packed blocks.
        teacher_logits = teacher_logits_at_positions(
            verifier,
            batch.clean_input_ids,
            batch.clean_lens,
            batch.teacher_logit_positions,
            batch.active_token_mask,
        ).detach()

    trajectory_teacher_logits_by_keep = None
    trajectory_teacher_hidden = None
    if trajectory_enabled:
        teacher_clean_cache = prefill_clean_kv_cache(
            trajectory_teacher,
            batch.clean_input_ids,
            batch.clean_lens,
        )
        with torch.no_grad():
            with capture_final_norm_hidden(trajectory_teacher) as teacher_capture:
                trajectory_teacher_out = trajectory_teacher(
                    batch.draft_input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    past_key_values=teacher_clean_cache,
                    use_cache=True,
                    store_kv=False,
                    logits_to_keep=batch.logit_keep_indices,
                )
            trajectory_teacher_logits_by_keep = (
                trajectory_teacher_out.logits.detach()
            )
            trajectory_teacher_hidden = teacher_capture["hidden"].detach()
        del teacher_clean_cache

    if trajectory_enabled:
        with capture_final_norm_hidden(draft_ddp) as student_capture:
            draft_out = packed_draft_forward(
                draft_ddp,
                batch,
                clean_cache=clean_cache,
                gradient_checkpointing=gradient_checkpointing,
                block_forward_context=block_forward_context,
            )
        trajectory_student_hidden = student_capture["hidden"]
    else:
        draft_out = packed_draft_forward(
            draft_ddp,
            batch,
            clean_cache=clean_cache,
            gradient_checkpointing=gradient_checkpointing,
            block_forward_context=block_forward_context,
        )
        trajectory_student_hidden = None
    student_logits_by_keep = draft_out.logits
    # Flatten only active kept positions for the objective.
    student_logits = student_logits_by_keep[batch.active_token_mask]
    target_ids = batch.target_ids[batch.active_token_mask]
    loss_weights = active_token_loss_weights(
        batch,
        loss_decay_gamma=loss_decay_gamma,
    )
    if objective == "ce":
        per_token, ce_per_token, hnce_per_token, margin_per_token = per_token_ce_aux_losses(
            student_logits,
            target_ids,
            ce_weight=float(ce_weight),
            hnce_weight=float(hnce_weight),
            hnce_top_k=int(hnce_top_k),
            hnce_margin=float(hnce_margin),
            margin_weight=float(margin_weight),
            margin_top_k=int(margin_top_k),
            margin_value=float(margin_value),
            margin_smooth=bool(margin_smooth),
        )
        loss_weights = loss_weights.to(device=per_token.device, dtype=per_token.dtype)
        weighted_loss_sum = (per_token * loss_weights).sum()
        weighted_ce_sum = (ce_per_token * loss_weights).sum()
        weighted_hnce_sum = (hnce_per_token * loss_weights).sum()
        weighted_margin_sum = (margin_per_token * loss_weights).sum()
    else:
        weighted_loss_sum = weighted_objective_loss_sum(
            student_logits,
            target_ids,
            objective=objective,
            loss_weights=loss_weights,
            teacher_logits=teacher_logits,
        )
        weighted_ce_sum = weighted_loss_sum.new_zeros(())
        weighted_hnce_sum = weighted_loss_sum.new_zeros(())
        weighted_margin_sum = weighted_loss_sum.new_zeros(())
    denom = global_objective_weight_sum.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    scaled_block_loss = weighted_loss_sum * (float(ctx.world_size) / denom)

    trajectory_metrics = {
        "trajectory_loss_sum": 0.0,
        "trajectory_relation_abs_error_sum": 0.0,
        "trajectory_selected_blocks": 0.0,
        "trajectory_teacher_better_blocks": 0.0,
        "trajectory_both_full_blocks": 0.0,
        "trajectory_teacher_accept_sum": 0.0,
        "trajectory_student_accept_sum": 0.0,
        "trajectory_ce_hidden_grad_sq_sum": 0.0,
        "trajectory_unit_hidden_grad_sq_sum": 0.0,
    }
    if trajectory_enabled:
        block_size = int(
            batch.draft_input_ids.shape[1] // batch.anchor_positions.shape[1]
        )
        with torch.no_grad():
            student_pred_by_keep = student_logits_by_keep.float().argmax(dim=-1)
            teacher_pred_by_keep = (
                trajectory_teacher_logits_by_keep.float().argmax(dim=-1)
            )
            student_accept = block_accept_lengths(
                student_pred_by_keep,
                batch.target_ids,
                keep_index_for_q=batch.keep_index_for_q,
                active_lengths=batch.active_lengths,
                block_keep_mask=batch.block_keep_mask,
                block_size=block_size,
            )
            teacher_accept = block_accept_lengths(
                teacher_pred_by_keep,
                batch.target_ids,
                keep_index_for_q=batch.keep_index_for_q,
                active_lengths=batch.active_lengths,
                block_keep_mask=batch.block_keep_mask,
                block_size=block_size,
            )
            selected_blocks = trajectory_selection_mask(
                teacher_accept,
                student_accept,
                active_lengths=batch.active_lengths,
                block_keep_mask=batch.block_keep_mask,
            )
            teacher_better = (
                (teacher_accept > student_accept) & batch.block_keep_mask
            )
            both_full = (
                (teacher_accept == batch.active_lengths)
                & (student_accept == batch.active_lengths)
                & batch.block_keep_mask
            )

        trajectory_result = relational_trajectory_loss(
            trajectory_student_hidden,
            trajectory_teacher_hidden,
            active_lengths=batch.active_lengths,
            block_keep_mask=batch.block_keep_mask,
            selected_blocks=selected_blocks,
            block_size=block_size,
            shuffle_teacher_blocks=bool(trajectory_shuffle_teacher_blocks),
            shuffle_seed=int(trajectory_shuffle_seed),
        )
        global_selected = all_reduce_sum_tensor(
            trajectory_result["selected_block_count"].to(
                device=device,
                dtype=torch.float32,
            ).clone()
        )
        scaled_trajectory_unit = distributed_block_mean_loss(
            trajectory_result["loss_sum"],
            global_block_count=global_selected,
            world_size=ctx.world_size,
        )
        if trajectory_gradient_probe:
            ce_hidden_grad = torch.autograd.grad(
                scaled_block_loss,
                trajectory_student_hidden,
                retain_graph=True,
            )[0]
            trajectory_hidden_grad = torch.autograd.grad(
                scaled_trajectory_unit,
                trajectory_student_hidden,
                retain_graph=True,
            )[0]
            ce_hidden_grad_sq_sum = float(
                ce_hidden_grad.float().square().sum().detach()
            )
            trajectory_hidden_grad_sq_sum = float(
                trajectory_hidden_grad.float().square().sum().detach()
            )
            del ce_hidden_grad
            del trajectory_hidden_grad
        else:
            ce_hidden_grad_sq_sum = 0.0
            trajectory_hidden_grad_sq_sum = 0.0
        scaled_block_loss = scaled_block_loss + (
            scaled_trajectory_unit * float(trajectory_weight)
        )
        trajectory_metrics = {
            "trajectory_loss_sum": float(trajectory_result["loss_sum"].detach()),
            "trajectory_relation_abs_error_sum": float(
                trajectory_result["relation_abs_error_sum"]
            ),
            "trajectory_selected_blocks": float(
                trajectory_result["selected_block_count"]
            ),
            "trajectory_teacher_better_blocks": float(
                teacher_better.float().sum()
            ),
            "trajectory_both_full_blocks": float(both_full.float().sum()),
            "trajectory_teacher_accept_sum": float(
                (teacher_accept.float() * batch.block_keep_mask.float()).sum()
            ),
            "trajectory_student_accept_sum": float(
                (student_accept.float() * batch.block_keep_mask.float()).sum()
            ),
            "trajectory_ce_hidden_grad_sq_sum": ce_hidden_grad_sq_sum,
            "trajectory_unit_hidden_grad_sq_sum": (
                trajectory_hidden_grad_sq_sum
            ),
        }
    (scaled_block_loss * float(loss_scale)).backward()

    with torch.no_grad():
        pred_ids = student_logits.float().argmax(dim=-1)
        pred_ids_by_keep = student_logits_by_keep.float().argmax(dim=-1)
        metric_target_ids, metric_target_ids_by_keep = acceptance_target_ids(
            batch.target_ids,
            batch.active_token_mask,
            teacher_logits=teacher_logits,
        )
        matches = pred_ids.eq(metric_target_ids)
        return {
            "weighted_loss_sum": float(weighted_loss_sum.detach()),
            "objective_weight_sum": float(loss_weights.detach().sum()),
            "tokens": float(target_ids.numel()),
            "blocks": float(batch.block_keep_mask.float().sum().detach()),
            "top1_match_sum": float(matches.float().sum().detach()),
            "accept_len_sum": _accept_len_sum_from_blocks(
                pred_ids_by_keep,
                metric_target_ids_by_keep,
                batch.active_lengths,
                batch.keep_index_for_q,
                block_size=int(batch.draft_input_ids.shape[1] // batch.anchor_positions.shape[1]),
            ),
            "weighted_ce_sum": float(weighted_ce_sum.detach()),
            "weighted_hnce_sum": float(weighted_hnce_sum.detach()),
            "weighted_margin_sum": float(weighted_margin_sum.detach()),
            "ar_loss_sum": ar_loss_sum_value,
            "ar_tokens": float(ar_tokens),
        } | trajectory_metrics


def main(
    *,
    default_fix_mistral_regex: bool = False,
    tokenizer_validator: Callable[[argparse.Namespace, Any, Any], None] | None = None,
) -> None:
    """Distributed offline training entry point."""

    args = parse_args(default_fix_mistral_regex=default_fix_mistral_regex)
    args.draft_code_model = args.draft_code_model or str(Path(__file__).resolve().parents[1] / "models/d3d")
    args.objective = normalize_objective(args.objective)
    if args.block_size <= 1:
        raise ValueError("block_size must be > 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.parallel_anchors_per_example <= 0:
        raise ValueError("parallel_anchors_per_example must be > 0")
    if args.workers < 0:
        raise ValueError("workers must be >= 0")
    if args.workers > 0 and args.prefetch <= 0:
        raise ValueError("prefetch must be > 0 when workers are enabled")
    if args.loss_decay_gamma <= 0.0:
        raise ValueError("loss-decay-gamma must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if args.min_lr_ratio < 0.0 or args.min_lr_ratio > 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if args.ce_weight < 0.0:
        raise ValueError("ce-weight must be >= 0")
    if args.hnce_weight < 0.0:
        raise ValueError("hnce-weight must be >= 0")
    if args.hnce_top_k <= 0:
        raise ValueError("hnce-top-k must be > 0")
    if args.margin_weight < 0.0:
        raise ValueError("margin-weight must be >= 0")
    if args.margin_top_k <= 0:
        raise ValueError("margin-top-k must be > 0")
    if args.ar_loss_weight < 0.0:
        raise ValueError("ar-loss-weight must be >= 0")
    if args.trajectory_weight < 0.0:
        raise ValueError("trajectory-weight must be >= 0")
    if args.trajectory_gradient_probe_steps < 0:
        raise ValueError("trajectory-gradient-probe-steps must be >= 0")
    if (
        args.trajectory_gradient_probe_steps > 0
        and args.trajectory_weight <= 0.0
    ):
        raise ValueError(
            "trajectory-gradient-probe-steps requires trajectory-weight > 0"
        )
    if args.trajectory_weight > 0.0 and not args.trajectory_teacher_model:
        raise ValueError(
            "trajectory-weight > 0 requires --trajectory-teacher-model"
        )
    if args.trajectory_weight > 0.0 and args.objective != "ce":
        raise ValueError("trajectory distillation currently requires objective=ce")
    if args.trajectory_shuffle_teacher_blocks and args.trajectory_weight <= 0.0:
        raise ValueError(
            "trajectory-shuffle-teacher-blocks requires trajectory-weight > 0"
        )
    if args.ar_logit_chunk_size < 0:
        raise ValueError("ar-logit-chunk-size must be >= 0")
    if (args.margin_weight > 0.0 or args.hnce_weight > 0.0) and args.objective != "ce":
        raise ValueError(
            f"margin/hnce terms apply only to objective=ce; got objective={args.objective}"
        )
    if objective_requires_target(args.objective) and not args.target_model:
        raise ValueError(f"objective={args.objective} requires --target-model")

    from accelerate import Accelerator

    output_dir = Path(args.output_dir)
    fsdp_plugin = build_d3d_fsdp_plugin_from_args(args)
    accelerator = Accelerator(
        mixed_precision=mixed_precision_from_dtype(args.dtype),
        project_dir=str(output_dir / "logs"),
        fsdp_plugin=fsdp_plugin,
    )
    ctx = distributed_context_from_accelerator(accelerator)
    device = accelerator.device
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    random.seed(args.seed + ctx.rank)
    torch.manual_seed(args.seed + ctx.rank)
    torch.cuda.manual_seed_all(args.seed + ctx.rank)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args) | {"world_size": ctx.world_size}, f, indent=2, sort_keys=True)
    accelerator.wait_for_everyone()

    dtype = dtype_from_arg(args.dtype)
    tokenizer_model = args.tokenizer_model or args.target_model or args.draft_model
    tokenizer_kwargs = tokenizer_load_kwargs(
        fix_mistral_regex=bool(args.fix_mistral_regex)
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **tokenizer_kwargs)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model, **tokenizer_kwargs
    )
    if tokenizer_validator is not None:
        tokenizer_validator(args, tokenizer, draft_tokenizer)
    draft_config, _ = PretrainedConfig.get_config_dict(args.draft_model)
    configured_mask = draft_config.get("mask_token_id")
    explicit_mask = getattr(args, "mask_token_id", None)
    if explicit_mask is not None and configured_mask is not None and explicit_mask != configured_mask:
        raise ValueError("--mask-token-id conflicts with the trained checkpoint's mask_token_id")
    requested_mask = explicit_mask if explicit_mask is not None else configured_mask
    if requested_mask is None and draft_config.get("model_type") == "qwen3":
        requested_mask = DEFAULT_MASK_TOKEN_ID
    mask_token_id = resolve_mask_token_id(
        draft_tokenizer, args.mask_token, mask_token_id=requested_mask,
        vocab_size=int(draft_config["vocab_size"]),
    )

    if ctx.is_main:
        print(
            "[config] "
            f"objective={args.objective} draft={args.draft_model} target={args.target_model or '<not-loaded>'} "
            f"draft_code={args.draft_code_model or '<checkpoint>'} "
            f"trajectory_teacher={args.trajectory_teacher_model or '<not-loaded>'} "
            f"trajectory_teacher_code={args.trajectory_teacher_code_model or '<checkpoint>'} "
            f"trajectory_weight={args.trajectory_weight} "
            f"trajectory_shuffle={args.trajectory_shuffle_teacher_blocks} "
            f"trajectory_gradient_probe_steps={args.trajectory_gradient_probe_steps} "
            f"tokenizer={tokenizer_model} data={args.data_path} block={args.block_size} "
            f"batch_size={args.batch_size} anchors_per_example={args.parallel_anchors_per_example} "
            f"workers={args.workers} "
            f"prefetch={args.prefetch} "
            f"persistent_workers={args.persistent_workers} "
            f"pin_memory={args.pin_memory} "
            f"grad_accum={args.gradient_accumulation_steps} lr={args.lr} min_lr_ratio={args.min_lr_ratio} "
            f"max_prompt_len={args.max_prompt_len} max_seq_len={args.max_seq_len} "
            f"thinking_template={args.enable_thinking_template} mask_id={mask_token_id} "
            f"fix_mistral_regex={args.fix_mistral_regex} "
            f"ce_weight={args.ce_weight} hnce_weight={args.hnce_weight} "
            f"hnce_top_k={args.hnce_top_k} hnce_margin={args.hnce_margin} "
            f"margin_weight={args.margin_weight} margin_top_k={args.margin_top_k} "
            f"margin_value={args.margin_value} margin_smooth={args.margin_smooth} "
            f"ar_loss_weight={args.ar_loss_weight} "
            f"ar_logit_chunk_size={args.ar_logit_chunk_size} "
            f"loss_decay_gamma={args.loss_decay_gamma} "
            f"gradient_checkpointing={args.gradient_checkpointing} "
            f"random_init={args.random_init}",
            flush=True,
        )

    verifier = None
    if objective_requires_target(args.objective):
        verifier = load_model(
            args.target_model,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(device)
        verifier.eval()
        verifier.requires_grad_(False)

    trajectory_teacher = None
    if args.trajectory_weight > 0.0:
        checkpoint_mask_id(args.trajectory_teacher_model, mask_token_id)
        trajectory_teacher = load_draft_checkpoint(
            args.trajectory_teacher_model,
            draft_code_model=args.trajectory_teacher_code_model,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(device)
        teacher_causal_layers = enable_causal_flash_attention_compatibility(
            trajectory_teacher
        )
        trajectory_teacher.config.use_cache = True
        if hasattr(trajectory_teacher.config, "fuse_cross_entropy"):
            trajectory_teacher.config.fuse_cross_entropy = False
        trajectory_teacher.eval()
        trajectory_teacher.requires_grad_(False)
        if ctx.is_main:
            print(
                "[trajectory-teacher] "
                f"model={args.trajectory_teacher_model} "
                f"causal_attention_layers={teacher_causal_layers} "
                "mode=eval no_grad",
                flush=True,
            )

    if args.random_init:
        draft = load_random_model(
            args.draft_model,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            seed=args.seed,
            draft_code_model=args.draft_code_model,
        )
    else:
        draft = load_draft_checkpoint(
            args.draft_model,
            draft_code_model=args.draft_code_model,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
    causal_attention_layers = enable_causal_flash_attention_compatibility(draft)
    if ctx.is_main:
        print(
            "[causal-attention] "
            f"mode=trainer_attention_wrapper layers={causal_attention_layers} "
            "checkpoint_recompute_safe=True",
            flush=True,
        )
    draft.config.mask_token_id = mask_token_id
    draft.config.use_cache = True
    if hasattr(draft.config, "fuse_cross_entropy"):
        draft.config.fuse_cross_entropy = False
    draft.train()
    if args.gradient_checkpointing:
        checkpointed_layers = enable_gradient_checkpointing(draft)
        if ctx.is_main:
            print(
                "[gradient-checkpointing] "
                f"mode=trainer_layer_wrapper layers={checkpointed_layers} "
                "preserve_past_key_value=True",
                flush=True,
            )

    total_rows = count_jsonl_rows(args.data_path, max_examples=args.max_examples)
    global_batch = max(ctx.world_size * int(args.batch_size), 1)
    steps_per_epoch = max(math.ceil(total_rows / global_batch), 1)
    total_steps = steps_per_epoch * int(args.epochs)
    if args.max_micro_steps > 0:
        total_steps = min(total_steps, int(args.max_micro_steps))
    total_optimizer_steps = max(
        math.ceil(total_steps / max(int(args.gradient_accumulation_steps), 1)),
        1,
    )

    optimizer = AdamW(draft.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    draft, optimizer = accelerator.prepare(draft, optimizer)
    wandb_run = WandbRun(
        ctx.is_main and bool(args.wandb_project) and not args.disable_wandb,
        project=args.wandb_project,
        name=args.wandb_name or output_dir.name,
        config=vars(args) | {"world_size": ctx.world_size},
    )
    tensorboard_run = TensorboardRun(
        ctx.is_main,
        log_dir=output_dir / "tensorboard",
        config=vars(args) | {"world_size": ctx.world_size},
    )
    if ctx.is_main:
        print(
            f"[data] rows={total_rows} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps} optimizer_steps~={total_optimizer_steps}",
            flush=True,
        )
    data_loader = build_loader(
        args=args,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        ctx=ctx,
    )

    global_step = 0
    optimizer_steps = 0
    accumulated_backward = False
    for epoch in range(int(args.epochs)):
        data_iter = iter(data_loader)
        for _ in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            global_step += 1
            step_start = time.perf_counter()
            if (global_step - 1) % int(args.gradient_accumulation_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
                accumulated_backward = False
            lr = lr_for_step(
                global_step,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=args.min_lr_ratio,
            )
            set_optimizer_lr(optimizer, lr)

            packed_data = next(
                data_iter,
                PackedBatch(
                    parallel_batch=None,
                    errors=[],
                    prompt_count=0,
                ),
            )

            if ctx.is_main:
                for row_index, exc_name, message in packed_data.errors:
                    print(f"[skip] row={row_index} error={exc_name}: {message}", flush=True)

            parallel_batch = to_device(
                packed_data.parallel_batch,
                device=device,
            )

            local_tokens = (
                float(parallel_batch.active_token_mask.float().sum().detach())
                if parallel_batch is not None
                else 0.0
            )
            local_objective_weight_sum = (
                float(
                    active_loss_weight_sum(
                        parallel_batch,
                        loss_decay_gamma=args.loss_decay_gamma,
                    ).detach()
                )
                if parallel_batch is not None
                else 0.0
            )
            local_block_count = (
                float(parallel_batch.block_keep_mask.float().sum().detach())
                if parallel_batch is not None
                else 0.0
            )
            local_ar_tokens = (
                float(
                    (parallel_batch.clean_lens - parallel_batch.prompt_lens)
                    .sum()
                    .detach()
                )
                if parallel_batch is not None and args.ar_loss_weight > 0.0
                else 0.0
            )
            local_metrics = torch.tensor(
                [
                    local_objective_weight_sum,
                    local_tokens,
                    local_block_count,
                    float(packed_data.prompt_count),
                    local_ar_tokens,
                ],
                dtype=torch.float32,
                device=device,
            )
            global_metrics = all_reduce_sum_tensor(local_metrics.clone())
            global_objective_weight_sum = global_metrics[0].detach()
            global_ar_token_count = global_metrics[4].detach()

            train_metrics = {
                "weighted_loss_sum": 0.0,
                "objective_weight_sum": 0.0,
                "tokens": 0.0,
                "blocks": 0.0,
                "top1_match_sum": 0.0,
                "accept_len_sum": 0.0,
                "weighted_ce_sum": 0.0,
                "weighted_hnce_sum": 0.0,
                "weighted_margin_sum": 0.0,
                "ar_loss_sum": 0.0,
                "ar_tokens": 0.0,
                "trajectory_loss_sum": 0.0,
                "trajectory_relation_abs_error_sum": 0.0,
                "trajectory_selected_blocks": 0.0,
                "trajectory_teacher_better_blocks": 0.0,
                "trajectory_both_full_blocks": 0.0,
                "trajectory_teacher_accept_sum": 0.0,
                "trajectory_student_accept_sum": 0.0,
                "trajectory_ce_hidden_grad_sq_sum": 0.0,
                "trajectory_unit_hidden_grad_sq_sum": 0.0,
            }
            if float(global_objective_weight_sum.item()) > 0.0:
                accum_window = accumulation_window_size(
                    global_step=global_step,
                    total_steps=total_steps,
                    accumulation_steps=args.gradient_accumulation_steps,
                )
                train_metrics = train_packed_block_batch(
                    draft_ddp=draft,
                    draft_module=draft,
                    verifier=verifier,
                    batch=parallel_batch,
                    global_objective_weight_sum=global_objective_weight_sum,
                    global_ar_token_count=global_ar_token_count,
                    ctx=ctx,
                    objective=args.objective,
                    mask_token_id=mask_token_id,
                    device=device,
                    loss_scale=1.0 / float(accum_window),
                    ce_weight=args.ce_weight,
                    hnce_weight=args.hnce_weight,
                    hnce_top_k=args.hnce_top_k,
                    hnce_margin=args.hnce_margin,
                    margin_weight=args.margin_weight,
                    margin_top_k=args.margin_top_k,
                    margin_value=args.margin_value,
                    margin_smooth=args.margin_smooth,
                    loss_decay_gamma=args.loss_decay_gamma,
                    gradient_checkpointing=args.gradient_checkpointing,
                    ar_loss_weight=args.ar_loss_weight,
                    ar_logit_chunk_size=args.ar_logit_chunk_size,
                    trajectory_teacher=trajectory_teacher,
                    trajectory_weight=args.trajectory_weight,
                    trajectory_shuffle_teacher_blocks=(
                        args.trajectory_shuffle_teacher_blocks
                    ),
                    trajectory_shuffle_seed=args.seed + global_step,
                    trajectory_gradient_probe=(
                        global_step <= args.trajectory_gradient_probe_steps
                    ),
                )
                accumulated_backward = True

            if should_step_optimizer(
                global_step=global_step,
                total_steps=total_steps,
                accumulation_steps=args.gradient_accumulation_steps,
            ) and accumulated_backward:
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(draft.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                accumulated_backward = False

            local_train_metrics = torch.tensor(
                [
                    train_metrics["weighted_loss_sum"],
                    train_metrics["objective_weight_sum"],
                    train_metrics["tokens"],
                    train_metrics["blocks"],
                    train_metrics["top1_match_sum"],
                    train_metrics["accept_len_sum"],
                    train_metrics["weighted_ce_sum"],
                    train_metrics["weighted_hnce_sum"],
                    train_metrics["weighted_margin_sum"],
                    train_metrics["ar_loss_sum"],
                    train_metrics["ar_tokens"],
                    train_metrics["trajectory_loss_sum"],
                    train_metrics["trajectory_relation_abs_error_sum"],
                    train_metrics["trajectory_selected_blocks"],
                    train_metrics["trajectory_teacher_better_blocks"],
                    train_metrics["trajectory_both_full_blocks"],
                    train_metrics["trajectory_teacher_accept_sum"],
                    train_metrics["trajectory_student_accept_sum"],
                    train_metrics["trajectory_ce_hidden_grad_sq_sum"],
                    train_metrics["trajectory_unit_hidden_grad_sq_sum"],
                ],
                dtype=torch.float32,
                device=device,
            )
            global_train_metrics = all_reduce_sum_tensor(local_train_metrics.clone())

            if ctx.is_main and (
                global_step == 1
                or global_step % int(args.log_every) == 0
                or global_step == total_steps
            ):
                tokens = float(global_train_metrics[2].item())
                blocks = float(global_train_metrics[3].item())
                objective_weight_sum = float(global_train_metrics[1].clamp_min(1e-8).item())
                loss_value = float((global_train_metrics[0] / global_train_metrics[1].clamp_min(1e-8)).item())
                top1 = float(global_train_metrics[4].item() / max(tokens, 1.0))
                accept_len = float(global_train_metrics[5].item() / max(blocks, 1.0))
                ce_mean = float(global_train_metrics[6].item() / objective_weight_sum)
                hnce_mean = float(global_train_metrics[7].item() / objective_weight_sum)
                margin_mean = float(global_train_metrics[8].item() / objective_weight_sum)
                ar_loss_mean = float(
                    global_train_metrics[9].item()
                    / max(float(global_train_metrics[10].item()), 1.0)
                )
                trajectory_selected = float(global_train_metrics[13].item())
                trajectory_loss_mean = float(
                    global_train_metrics[11].item()
                    / max(trajectory_selected, 1.0)
                )
                trajectory_relation_error = float(
                    global_train_metrics[12].item()
                    / max(trajectory_selected, 1.0)
                )
                trajectory_teacher_accept = float(
                    global_train_metrics[16].item() / max(blocks, 1.0)
                )
                trajectory_student_accept = float(
                    global_train_metrics[17].item() / max(blocks, 1.0)
                )
                trajectory_ce_hidden_grad_norm = math.sqrt(
                    max(float(global_train_metrics[18].item()), 0.0)
                )
                trajectory_unit_hidden_grad_norm = math.sqrt(
                    max(float(global_train_metrics[19].item()), 0.0)
                )
                trajectory_unit_to_ce_hidden_grad_ratio = (
                    trajectory_unit_hidden_grad_norm
                    / max(trajectory_ce_hidden_grad_norm, 1e-20)
                )
                log_payload = {
                    "train/tokens": tokens,
                    "train/blocks": blocks,
                    "train/prompts": float(global_metrics[3].item()),
                    "train/top1_match": top1,
                    "train/accept_len_mean": accept_len,
                    "train/block_step_len_mean": accept_len + 1.0 if blocks > 0 else 0.0,
                    "train/lr": lr,
                    "train/optimizer_step": float(optimizer_steps),
                    "train/step_s": time.perf_counter() - step_start,
                }
                if args.trajectory_weight > 0.0:
                    log_payload.update(
                        {
                            "train/trajectory_loss": trajectory_loss_mean,
                            "train/trajectory_relation_abs_error": (
                                trajectory_relation_error
                            ),
                            "train/trajectory_selected_blocks": (
                                trajectory_selected
                            ),
                            "train/trajectory_selected_rate": (
                                trajectory_selected / max(blocks, 1.0)
                            ),
                            "train/trajectory_teacher_better_rate": (
                                float(global_train_metrics[14].item())
                                / max(blocks, 1.0)
                            ),
                            "train/trajectory_both_full_rate": (
                                float(global_train_metrics[15].item())
                                / max(blocks, 1.0)
                            ),
                            "train/trajectory_teacher_accept_len": (
                                trajectory_teacher_accept
                            ),
                            "train/trajectory_student_accept_len": (
                                trajectory_student_accept
                            ),
                        }
                    )
                    if global_step <= args.trajectory_gradient_probe_steps:
                        log_payload.update(
                            {
                                "train/trajectory_ce_hidden_grad_norm": (
                                    trajectory_ce_hidden_grad_norm
                                ),
                                "train/trajectory_unit_hidden_grad_norm": (
                                    trajectory_unit_hidden_grad_norm
                                ),
                                "train/trajectory_unit_to_ce_hidden_grad_ratio": (
                                    trajectory_unit_to_ce_hidden_grad_ratio
                                ),
                            }
                        )
                if args.ar_loss_weight > 0.0:
                    log_payload["train/block_loss"] = loss_value
                    log_payload["train/ar_loss"] = ar_loss_mean
                    print(
                        "step {step} | opt_step {opt_step} | obj {obj} "
                        "| block_loss {block_loss:.6f} | ar_loss {ar_loss:.6f} "
                        "| blocks {blocks:.0f} | tokens {tokens:.0f} "
                        "| accept_len {accept:.3f} | block_step {dstep:.3f} "
                        "| top1 {top1:.4f} | lr {lr:.2e} | step_s {step_s:.2f}".format(
                            step=global_step,
                            opt_step=optimizer_steps,
                            obj=args.objective,
                            block_loss=loss_value,
                            ar_loss=ar_loss_mean,
                            blocks=log_payload["train/blocks"],
                            tokens=log_payload["train/tokens"],
                            accept=log_payload["train/accept_len_mean"],
                            dstep=log_payload["train/block_step_len_mean"],
                            top1=log_payload["train/top1_match"],
                            lr=lr,
                            step_s=log_payload["train/step_s"],
                        ),
                        flush=True,
                    )
                else:
                    log_payload.update(
                        {
                            "train/loss": loss_value,
                            "train/ce_mean": ce_mean,
                            "train/hnce_mean": hnce_mean,
                            "train/margin_mean": margin_mean,
                        }
                    )
                    print(
                        "step {step} | opt_step {opt_step} | obj {obj} | loss {loss:.4f} | ce {ce:.4f} "
                        "| margin {margin:.4f} | hnce {hnce:.4f} | blocks {blocks:.0f} "
                        "| tokens {tokens:.0f} | accept_len {accept:.2f} | block_step {dstep:.2f} "
                        "| top1 {top1:.4f} | lr {lr:.2e} | step_s {step_s:.2f}".format(
                            step=global_step,
                            opt_step=optimizer_steps,
                            obj=args.objective,
                            loss=log_payload["train/loss"],
                            ce=log_payload["train/ce_mean"],
                            margin=log_payload["train/margin_mean"],
                            hnce=log_payload["train/hnce_mean"],
                            blocks=log_payload["train/blocks"],
                            tokens=log_payload["train/tokens"],
                            accept=log_payload["train/accept_len_mean"],
                            dstep=log_payload["train/block_step_len_mean"],
                            top1=log_payload["train/top1_match"],
                            lr=lr,
                            step_s=log_payload["train/step_s"],
                        ),
                        flush=True,
                    )
                if args.trajectory_weight > 0.0:
                    print(
                        "  trajectory | loss {loss:.6f} | relation_mae {mae:.6f} "
                        "| selected {selected:.0f}/{blocks:.0f} ({rate:.3f}) "
                        "| teacher_accept {teacher:.3f} | student_accept {student:.3f}".format(
                            loss=trajectory_loss_mean,
                            mae=trajectory_relation_error,
                            selected=trajectory_selected,
                            blocks=blocks,
                            rate=trajectory_selected / max(blocks, 1.0),
                            teacher=trajectory_teacher_accept,
                            student=trajectory_student_accept,
                        ),
                        flush=True,
                    )
                    if global_step <= args.trajectory_gradient_probe_steps:
                        print(
                            "  trajectory_grad | ce_hidden {ce:.6e} "
                            "| unit_traj_hidden {trajectory:.6e} "
                            "| unit/ce {ratio:.4f}".format(
                                ce=trajectory_ce_hidden_grad_norm,
                                trajectory=trajectory_unit_hidden_grad_norm,
                                ratio=trajectory_unit_to_ce_hidden_grad_ratio,
                            ),
                            flush=True,
                        )
                wandb_run.log(log_payload, step=global_step)
                tensorboard_run.log(log_payload, step=global_step)

            if args.save_every > 0 and global_step % int(args.save_every) == 0:
                save_checkpoint(
                    draft_module=draft,
                    tokenizer=draft_tokenizer,
                    output_dir=output_dir,
                    step=global_step,
                    ctx=ctx,
                    accelerator=accelerator,
                    code_model=args.draft_code_model or args.draft_model,
                )
        if global_step >= total_steps:
            break
    if should_save_final_checkpoint(global_step=global_step, save_every=args.save_every):
        save_checkpoint(
            draft_module=draft,
            tokenizer=draft_tokenizer,
            output_dir=output_dir,
            step=global_step,
            ctx=ctx,
            accelerator=accelerator,
            code_model=args.draft_code_model or args.draft_model,
        )
    # Remote logger uploads must not hold other ranks inside a collective.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb_run.finish()
    tensorboard_run.finish()
