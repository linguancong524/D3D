from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .objective_kernels import forward_kl_per_token, reverse_kl_per_token, topk_hard_negative_ce_per_token, topk_target_margin_per_token
from .types import ParallelOfflineBatch

OBJECTIVE_ALIASES = {
    "ce": "ce",
    "hard_ce": "ce",
    "kl": "forward_kl",
    "fkl": "forward_kl",
    "forward_kl": "forward_kl",
    "forward-kl": "forward_kl",
    "rkl": "reverse_kl",
    "reverse_kl": "reverse_kl",
    "reverse-kl": "reverse_kl",
}


def normalize_objective(objective: str) -> str:
    """Map CLI aliases to canonical objective names."""

    key = str(objective).lower()
    if key not in OBJECTIVE_ALIASES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVE_ALIASES)}, got {objective!r}")
    return OBJECTIVE_ALIASES[key]


def objective_requires_target(objective: str) -> bool:
    """Return whether the objective needs teacher/verifier logits."""

    return normalize_objective(objective) in {"forward_kl", "reverse_kl"}


def kept_logit_loss_weights(
    logit_keep_indices: torch.LongTensor,
    *,
    block_size: int,
    loss_decay_gamma: float = 15.5,
) -> torch.Tensor:
    """Compute position-decayed loss weights for kept draft logits."""

    offsets = logit_keep_indices.to(dtype=torch.long) % int(block_size)
    return torch.exp(
        -(offsets - 1).clamp(min=0).float() / float(loss_decay_gamma)
    )


def active_token_loss_weights(
    batch: ParallelOfflineBatch,
    *,
    loss_decay_gamma: float = 15.5,
) -> torch.Tensor:
    """Return loss weights only for active prediction tokens."""

    block_size = int(batch.draft_input_ids.shape[1] // batch.anchor_positions.shape[1])
    weights_by_keep = kept_logit_loss_weights(
        batch.logit_keep_indices,
        block_size=block_size,
        loss_decay_gamma=loss_decay_gamma,
    ).to(device=batch.active_token_mask.device)
    return weights_by_keep.view(1, -1).expand_as(batch.active_token_mask)[
        batch.active_token_mask
    ]


def active_loss_weight_sum(
    batch: ParallelOfflineBatch,
    *,
    loss_decay_gamma: float = 15.5,
) -> torch.Tensor:
    """Sum the weighted token count used to normalize distributed loss."""

    block_size = int(batch.draft_input_ids.shape[1] // batch.anchor_positions.shape[1])
    weights_by_keep = kept_logit_loss_weights(
        batch.logit_keep_indices,
        block_size=block_size,
        loss_decay_gamma=loss_decay_gamma,
    ).to(device=batch.active_token_mask.device)
    return (batch.active_token_mask.float() * weights_by_keep.view(1, -1)).sum()


def weighted_objective_loss_sum(
    student_logits: torch.Tensor,
    target_ids: torch.LongTensor,
    *,
    objective: str,
    loss_weights: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the weighted block loss for CE or KL objectives."""

    objective = normalize_objective(objective)
    if objective == "ce":
        per_token = F.cross_entropy(
            student_logits.float(),
            target_ids.long(),
            reduction="none",
        )
    else:
        if teacher_logits is None:
            raise ValueError(f"{objective} requires teacher_logits")
        if objective == "forward_kl":
            per_token = forward_kl_per_token(student_logits, teacher_logits)
        elif objective == "reverse_kl":
            per_token = reverse_kl_per_token(student_logits, teacher_logits)
        else:
            raise AssertionError(f"unhandled objective {objective}")

    loss_weights = loss_weights.to(device=per_token.device, dtype=per_token.dtype)
    if loss_weights.shape != per_token.shape:
        raise ValueError(
            f"loss_weights shape must match per-token loss; got "
            f"{tuple(loss_weights.shape)} vs {tuple(per_token.shape)}"
        )
    return (per_token * loss_weights).sum()


def per_token_ce_aux_losses(
    student_logits: torch.Tensor,
    target_ids: torch.LongTensor,
    *,
    ce_weight: float,
    hnce_weight: float,
    hnce_top_k: int,
    hnce_margin: float,
    margin_weight: float,
    margin_top_k: int,
    margin_value: float,
    margin_smooth: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine CE, hard-negative CE, and target-margin token losses."""

    ce = F.cross_entropy(student_logits.float(), target_ids.long(), reduction="none")
    if float(hnce_weight) != 0.0:
        hnce = topk_hard_negative_ce_per_token(
            student_logits,
            target_ids,
            top_k=int(hnce_top_k),
            margin=float(hnce_margin),
        )
    else:
        hnce = torch.zeros_like(ce)
    if float(margin_weight) != 0.0:
        margin = topk_target_margin_per_token(
            student_logits,
            target_ids,
            top_k=int(margin_top_k),
            margin=float(margin_value),
            smooth=bool(margin_smooth),
        )
    else:
        margin = torch.zeros_like(ce)
    return float(ce_weight) * ce + float(hnce_weight) * hnce + float(margin_weight) * margin, ce, hnce, margin


def acceptance_target_ids(
    target_ids_by_keep: torch.LongTensor,
    active_token_mask: torch.BoolTensor,
    *,
    teacher_logits: torch.Tensor | None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Choose metric targets from ground truth or teacher argmax logits."""

    if teacher_logits is None:
        return target_ids_by_keep[active_token_mask], target_ids_by_keep
    flat_targets = teacher_logits.float().argmax(dim=-1).long()
    targets_by_keep = target_ids_by_keep.clone()
    targets_by_keep[active_token_mask] = flat_targets
    return flat_targets, targets_by_keep


def response_prediction_positions_and_mask(
    prompt_lens: torch.LongTensor,
    clean_lens: torch.LongTensor,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Select clean AR logits that predict response tokens."""

    prompt_lens = prompt_lens.to(dtype=torch.long)
    clean_lens = clean_lens.to(device=prompt_lens.device, dtype=torch.long)
    if prompt_lens.ndim != 1 or clean_lens.ndim != 1 or prompt_lens.shape != clean_lens.shape:
        raise ValueError("prompt_lens and clean_lens must be matching rank-1 tensors")
    if bool((prompt_lens <= 0).any().item()):
        raise ValueError("prompt_lens must be positive")
    if bool((clean_lens <= prompt_lens).any().item()):
        raise ValueError("every clean sequence must contain at least one response token")

    source_positions = torch.arange(
        int(clean_lens.max().item()) - 1,
        device=prompt_lens.device,
        dtype=torch.long,
    )
    active_by_position = (
        (source_positions.view(1, -1) >= (prompt_lens - 1).view(-1, 1))
        & (source_positions.view(1, -1) < (clean_lens - 1).view(-1, 1))
    )
    keep_positions = torch.nonzero(
        active_by_position.any(dim=0),
        as_tuple=False,
    ).flatten()
    return keep_positions, active_by_position[:, keep_positions]


def chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.LongTensor,
    *,
    chunk_size: int,
    weight_owner: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Compute summed CE while checkpointing each full-vocabulary logits chunk."""

    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}"
        )
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [vocab, hidden], got {tuple(weight.shape)}")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError(
            f"hidden size mismatch: {hidden_states.shape[1]} vs {weight.shape[1]}"
        )
    if targets.ndim != 1 or targets.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"targets must have shape [{hidden_states.shape[0]}], got {tuple(targets.shape)}"
        )
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be > 0")
    if hidden_states.shape[0] <= 0:
        raise ValueError("chunked linear cross entropy requires at least one token")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError(
            f"bias must have shape [{weight.shape[0]}], got {tuple(bias.shape)}"
        )

    if weight_owner is None:
        if bias is None:
            def chunk_loss(
                hidden_chunk: torch.Tensor,
                linear_weight: torch.Tensor,
                target_chunk: torch.LongTensor,
            ) -> torch.Tensor:
                return F.cross_entropy(
                    F.linear(hidden_chunk, linear_weight),
                    target_chunk,
                    reduction="sum",
                )
        else:
            def chunk_loss(
                hidden_chunk: torch.Tensor,
                linear_weight: torch.Tensor,
                linear_bias: torch.Tensor,
                target_chunk: torch.LongTensor,
            ) -> torch.Tensor:
                return F.cross_entropy(
                    F.linear(hidden_chunk, linear_weight, linear_bias),
                    target_chunk,
                    reduction="sum",
                )
    else:
        def chunk_loss(
            hidden_chunk: torch.Tensor,
            target_chunk: torch.LongTensor,
        ) -> torch.Tensor:
            live_weight = weight_owner.weight
            live_bias = getattr(weight_owner, "bias", None)
            return F.cross_entropy(
                F.linear(hidden_chunk, live_weight, live_bias),
                target_chunk,
                reduction="sum",
            )

    # Non-reentrant checkpointing keeps ordinary parameter autograd edges (and
    # therefore FSDP's hooks) while discarding logits until backward recomputes them.
    loss_sum = hidden_states.new_zeros(())
    for start in range(0, int(hidden_states.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(hidden_states.shape[0]))
        hidden_chunk = hidden_states[start:end]
        target_chunk = targets[start:end]
        if weight_owner is not None:
            chunk_loss_sum = checkpoint(
                chunk_loss,
                hidden_chunk,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        elif bias is None:
            chunk_loss_sum = checkpoint(
                chunk_loss,
                hidden_chunk,
                weight,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_loss_sum = checkpoint(
                chunk_loss,
                hidden_chunk,
                weight,
                bias,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        loss_sum = loss_sum + chunk_loss_sum
    return loss_sum


def _get_output_embedding_module(model) -> torch.nn.Module:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output_embeddings = getter()
        if output_embeddings is not None:
            return output_embeddings
    module = getattr(model, "module", None)
    getter = getattr(module, "get_output_embeddings", None)
    if callable(getter):
        output_embeddings = getter()
        if output_embeddings is not None:
            return output_embeddings
    raise TypeError("draft model must expose get_output_embeddings() for chunked AR CE")


@contextmanager
def _chunked_ar_lm_head(
    model,
    *,
    targets_by_keep: torch.LongTensor,
    active_mask: torch.BoolTensor,
    chunk_size: int,
):
    """Temporarily make lm_head return a scalar chunked AR CE loss."""

    lm_head = _get_output_embedding_module(model)
    if getattr(lm_head, "weight", None) is None:
        raise TypeError("draft output embedding must expose a weight parameter")
    had_instance_forward = "forward" in lm_head.__dict__
    original_instance_forward = lm_head.__dict__.get("forward")

    def chunked_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[:2] != targets_by_keep.shape:
            raise RuntimeError(
                "lm_head hidden positions do not match AR targets: "
                f"{tuple(hidden_states.shape[:2])} vs {tuple(targets_by_keep.shape)}"
            )
        active_hidden = hidden_states[active_mask].contiguous()
        active_targets = targets_by_keep[active_mask].contiguous()
        # FSDP2 parameters are DTensors outside root forward and become full
        # tensors in its pre-forward hook, so read them only at execution time.
        weight = lm_head.weight
        bias = getattr(lm_head, "bias", None)
        return chunked_linear_cross_entropy(
            active_hidden,
            weight,
            bias,
            active_targets,
            chunk_size=int(chunk_size),
            weight_owner=lm_head,
        )

    lm_head.forward = chunked_forward
    try:
        yield
    finally:
        if had_instance_forward:
            lm_head.forward = original_instance_forward
        else:
            del lm_head.forward


def clean_ar_loss_sum(
    draft_module,
    clean_input_ids: torch.LongTensor,
    prompt_lens: torch.LongTensor,
    clean_lens: torch.LongTensor,
    *,
    ar_logit_chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Compute clean causal AR loss without retaining a KV cache."""

    clean_lens = clean_lens.to(device=clean_input_ids.device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=clean_input_ids.device, dtype=torch.long)
    batch_size = int(clean_input_ids.shape[0])
    max_clean_len = int(clean_lens.max().item())
    clean_input_ids = clean_input_ids[:, :max_clean_len]
    device = clean_input_ids.device

    query_indices = torch.arange(max_clean_len, device=device).view(1, max_clean_len, 1)
    key_indices = torch.arange(max_clean_len, device=device).view(1, 1, max_clean_len)
    lens = clean_lens.view(batch_size, 1, 1)
    valid_causal = (query_indices < lens) & (key_indices < lens) & (key_indices <= query_indices)
    padding_self = (query_indices >= lens) & (query_indices == key_indices)
    attention_mask = (valid_causal | padding_self).view(
        batch_size,
        1,
        max_clean_len,
        max_clean_len,
    )
    use_flash_causal = (
        bool(getattr(draft_module, "supports_draft_causal", False))
        and bool((clean_lens == max_clean_len).all().item())
    )
    if use_flash_causal:
        attention_mask = None
    position_ids = torch.arange(
        max_clean_len,
        dtype=torch.long,
        device=device,
    ).view(1, -1).expand(batch_size, -1)
    keep_positions, active_mask = response_prediction_positions_and_mask(
        prompt_lens,
        clean_lens,
    )
    targets_by_keep = clean_input_ids[:, keep_positions + 1]

    if not bool(getattr(draft_module, "training", False)):
        raise RuntimeError("clean_ar_loss_sum requires the draft model to be in training mode")
    if int(ar_logit_chunk_size) > 0:
        with _chunked_ar_lm_head(
            draft_module,
            targets_by_keep=targets_by_keep,
            active_mask=active_mask,
            chunk_size=int(ar_logit_chunk_size),
        ):
            out = draft_module(
                clean_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                store_kv=False,
                logits_to_keep=keep_positions,
                draft_causal=use_flash_causal,
            )
    else:
        out = draft_module(
            clean_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            store_kv=False,
            logits_to_keep=keep_positions,
            draft_causal=use_flash_causal,
        )
    if out.logits is None:
        raise RuntimeError("clean AR forward requires materialized logits or a scalar chunked loss")

    ar_targets = targets_by_keep[active_mask]
    if int(ar_logit_chunk_size) > 0:
        if out.logits.ndim != 0:
            raise RuntimeError(
                f"chunked AR lm_head must return a scalar loss, got {tuple(out.logits.shape)}"
            )
        ar_loss_sum = out.logits
    else:
        ar_logits = out.logits[active_mask]
        ar_loss_sum = F.cross_entropy(ar_logits, ar_targets, reduction="sum")
    return ar_loss_sum, int(ar_targets.numel())


def _accept_len_sum_from_blocks(
    pred_ids_by_keep: torch.LongTensor,
    target_ids_by_keep: torch.LongTensor,
    active_lengths: torch.LongTensor,
    keep_index_for_q: torch.LongTensor,
    *,
    block_size: int,
) -> float:
    """Compute summed prefix acceptance length over all packed blocks."""

    accept_len_sum = 0.0
    pred = pred_ids_by_keep.detach().cpu()
    target = target_ids_by_keep.detach().cpu()
    lengths = active_lengths.detach().cpu()
    keep_map = keep_index_for_q.detach().cpu()
    for batch_idx in range(int(lengths.shape[0])):
        for block_idx in range(int(lengths.shape[1])):
            active_len = int(lengths[batch_idx, block_idx].item())
            for offset in range(1, active_len + 1):
                query_pos = block_idx * int(block_size) + offset
                keep_idx = int(keep_map[query_pos].item())
                if keep_idx < 0:
                    break
                if int(pred[batch_idx, keep_idx].item()) != int(target[batch_idx, keep_idx].item()):
                    break
                accept_len_sum += 1.0
    return accept_len_sum
