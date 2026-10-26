"""Tensor utilities for D3D hidden-trajectory distillation.

The teacher and student may have different hidden widths.  We therefore
compare the geometry within each packed block rather than matching hidden
vectors directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def distributed_block_mean_loss(
    local_loss_sum: torch.Tensor,
    *,
    global_block_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Scale a local block-loss sum for DDP/FSDP's averaged gradients."""
    if local_loss_sum.ndim != 0 or global_block_count.numel() != 1:
        raise ValueError("loss sum and global block count must be scalars")
    count = global_block_count.to(
        device=local_loss_sum.device,
        dtype=local_loss_sum.dtype,
    )
    has_blocks = (count > 0).to(dtype=local_loss_sum.dtype)
    return (
        local_loss_sum
        * (float(world_size) / count.clamp_min(1.0))
        * has_blocks
    )


def packed_block_valid_mask(
    active_lengths: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Return validity for the anchor and active future tokens of each block."""
    if active_lengths.ndim != 2 or block_keep_mask.shape != active_lengths.shape:
        raise ValueError("active_lengths and block_keep_mask must have shape [B, A]")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    offsets = torch.arange(block_size, device=active_lengths.device)
    valid = offsets.view(1, 1, -1) <= active_lengths.unsqueeze(-1)
    return valid & block_keep_mask.to(dtype=torch.bool).unsqueeze(-1)


def reshape_packed_hidden(
    hidden: torch.Tensor,
    *,
    active_lengths: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Reshape packed queries [B, A*K, D] into [B, A, K, D]."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B, Q, D]")
    if active_lengths.ndim != 2 or hidden.shape[0] != active_lengths.shape[0]:
        raise ValueError("hidden batch and active_lengths must agree")

    batch_size, num_blocks = active_lengths.shape
    expected_query_length = num_blocks * block_size
    if hidden.shape[1] != expected_query_length:
        raise ValueError(
            "hidden query length does not match active_lengths * block_size: "
            f"{hidden.shape[1]} != {expected_query_length}"
        )
    return hidden.reshape(batch_size, num_blocks, block_size, hidden.shape[-1])


def dense_future_token_ids(
    token_ids_by_keep: torch.Tensor,
    *,
    keep_index_for_q: torch.Tensor,
    active_lengths: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
    fill_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map kept future-token ids back to dense packed-block coordinates."""
    if token_ids_by_keep.ndim != 2:
        raise ValueError("token_ids_by_keep must have shape [B, M]")
    if keep_index_for_q.ndim != 1:
        raise ValueError("keep_index_for_q must have shape [A*K]")

    batch_size, num_blocks = active_lengths.shape
    expected_query_length = num_blocks * block_size
    if keep_index_for_q.numel() != expected_query_length:
        raise ValueError("keep_index_for_q query length does not match packed blocks")
    if token_ids_by_keep.shape[0] != batch_size:
        raise ValueError("token ids and active_lengths batch sizes must agree")

    keep_by_block = keep_index_for_q.reshape(num_blocks, block_size)[:, 1:]
    keep_by_block = keep_by_block.to(device=token_ids_by_keep.device)
    gather_index = keep_by_block.clamp_min(0)
    gathered = token_ids_by_keep[:, gather_index]

    future_offsets = torch.arange(
        1,
        block_size,
        device=active_lengths.device,
    )
    valid = future_offsets.view(1, 1, -1) <= active_lengths.unsqueeze(-1)
    valid = valid & block_keep_mask.to(dtype=torch.bool).unsqueeze(-1)
    valid = valid & (keep_by_block >= 0).unsqueeze(0)
    dense = torch.where(
        valid.to(device=gathered.device),
        gathered,
        torch.full_like(gathered, fill_value),
    )
    return dense, valid.to(device=gathered.device)


def block_accept_lengths(
    predicted_ids_by_keep: torch.Tensor,
    target_ids_by_keep: torch.Tensor,
    *,
    keep_index_for_q: torch.Tensor,
    active_lengths: torch.Tensor,
    block_keep_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute contiguous future-token acceptance length for every block."""
    predicted, valid = dense_future_token_ids(
        predicted_ids_by_keep,
        keep_index_for_q=keep_index_for_q,
        active_lengths=active_lengths,
        block_keep_mask=block_keep_mask,
        block_size=block_size,
    )
    targets, target_valid = dense_future_token_ids(
        target_ids_by_keep,
        keep_index_for_q=keep_index_for_q,
        active_lengths=active_lengths,
        block_keep_mask=block_keep_mask,
        block_size=block_size,
    )
    valid = valid & target_valid
    correct = (predicted == targets) & valid
    prefix_correct = correct.to(dtype=torch.long).cumprod(dim=-1)
    return (prefix_correct * valid.to(dtype=torch.long)).sum(dim=-1)


def trajectory_selection_mask(
    teacher_accept_lengths: torch.Tensor,
    student_accept_lengths: torch.Tensor,
    *,
    active_lengths: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Select teacher-better blocks and blocks both models solve fully."""
    expected = active_lengths.shape
    for name, tensor in (
        ("teacher_accept_lengths", teacher_accept_lengths),
        ("student_accept_lengths", student_accept_lengths),
        ("block_keep_mask", block_keep_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {tuple(expected)}")

    teacher_better = teacher_accept_lengths > student_accept_lengths
    both_full = (
        (teacher_accept_lengths == active_lengths)
        & (student_accept_lengths == active_lengths)
    )
    return (teacher_better | both_full) & block_keep_mask.to(dtype=torch.bool)


def _shuffle_selected_teacher_relations(
    teacher_relation: torch.Tensor,
    *,
    selected_blocks: torch.Tensor,
    active_lengths: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Shuffle selected teacher targets only among equal-length blocks."""
    flat_relation = teacher_relation.flatten(0, 1)
    flat_selected = selected_blocks.flatten()
    flat_lengths = active_lengths.flatten()
    shuffled = flat_relation.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for length in torch.unique(flat_lengths[flat_selected]).tolist():
        indices = torch.nonzero(
            flat_selected & (flat_lengths == length),
            as_tuple=False,
        ).flatten()
        count = indices.numel()
        if count < 2:
            continue
        permutation = torch.randperm(count, generator=generator)
        if torch.equal(permutation, torch.arange(count)):
            permutation = torch.roll(permutation, shifts=1)
        shuffled[indices] = flat_relation[indices[permutation.to(indices.device)]]
    return shuffled.reshape_as(teacher_relation)


def relational_trajectory_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    active_lengths: torch.Tensor,
    block_keep_mask: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int,
    shuffle_teacher_blocks: bool = False,
    shuffle_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compare within-block cosine-relation matrices.

    Each selected block contributes its mean off-diagonal SmoothL1 error,
    independently of its active length.  The returned loss is a *sum* of
    these block means so the caller can normalize it across distributed
    workers by the global selected-block count.
    """
    student = reshape_packed_hidden(
        student_hidden,
        active_lengths=active_lengths,
        block_size=block_size,
    )
    teacher = reshape_packed_hidden(
        teacher_hidden.detach(),
        active_lengths=active_lengths,
        block_size=block_size,
    )
    if student.shape[:3] != teacher.shape[:3]:
        raise ValueError("teacher and student packed trajectory geometry must agree")
    if selected_blocks.shape != active_lengths.shape:
        raise ValueError("selected_blocks must have shape [B, A]")

    valid = packed_block_valid_mask(
        active_lengths,
        block_keep_mask,
        block_size,
    )
    selected = selected_blocks.to(dtype=torch.bool) & block_keep_mask.to(
        dtype=torch.bool
    )

    student_normalized = F.normalize(student.float(), dim=-1)
    teacher_normalized = F.normalize(teacher.float(), dim=-1)
    student_relation = torch.matmul(
        student_normalized,
        student_normalized.transpose(-1, -2),
    )
    teacher_relation = torch.matmul(
        teacher_normalized,
        teacher_normalized.transpose(-1, -2),
    )
    if shuffle_teacher_blocks:
        teacher_relation = _shuffle_selected_teacher_relations(
            teacher_relation,
            selected_blocks=selected,
            active_lengths=active_lengths,
            seed=shuffle_seed,
        )

    eye = torch.eye(block_size, dtype=torch.bool, device=valid.device)
    pair_mask = (
        valid.unsqueeze(-1)
        & valid.unsqueeze(-2)
        & ~eye.view(1, 1, block_size, block_size)
    )
    pair_count = pair_mask.sum(dim=(-1, -2))
    per_pair_loss = F.smooth_l1_loss(
        student_relation,
        teacher_relation,
        reduction="none",
    )
    per_pair_abs_error = (student_relation - teacher_relation).abs()
    denominator = pair_count.clamp_min(1).to(dtype=per_pair_loss.dtype)
    per_block_loss = (
        per_pair_loss * pair_mask.to(dtype=per_pair_loss.dtype)
    ).sum(dim=(-1, -2)) / denominator
    per_block_abs_error = (
        per_pair_abs_error * pair_mask.to(dtype=per_pair_abs_error.dtype)
    ).sum(dim=(-1, -2)) / denominator
    selected = selected & (pair_count > 0)
    selected_float = selected.to(dtype=per_block_loss.dtype)

    # Multiplication by a zero mask deliberately preserves the student graph
    # when no local block is selected.
    return {
        "loss_sum": (per_block_loss * selected_float).sum(),
        "relation_abs_error_sum": (
            per_block_abs_error * selected_float
        ).sum().detach(),
        "selected_block_count": selected_float.sum().detach(),
    }
