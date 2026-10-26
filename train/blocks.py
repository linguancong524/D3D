from __future__ import annotations

import random
from typing import Any

import torch

from .types import OfflineExample, ParallelOfflineBatch

def pin_batch(batch: ParallelOfflineBatch | None) -> ParallelOfflineBatch | None:
    """Pin CPU tensors for faster non-blocking H2D copies."""

    if batch is None:
        return None
    return ParallelOfflineBatch(
        clean_input_ids=batch.clean_input_ids.pin_memory(),
        prompt_lens=batch.prompt_lens.pin_memory(),
        clean_lens=batch.clean_lens.pin_memory(),
        draft_input_ids=batch.draft_input_ids.pin_memory(),
        attention_mask=batch.attention_mask.pin_memory(),
        position_ids=batch.position_ids.pin_memory(),
        logit_keep_indices=batch.logit_keep_indices.pin_memory(),
        active_token_mask=batch.active_token_mask.pin_memory(),
        teacher_logit_positions=batch.teacher_logit_positions.pin_memory(),
        target_ids=batch.target_ids.pin_memory(),
        anchor_positions=batch.anchor_positions.pin_memory(),
        block_keep_mask=batch.block_keep_mask.pin_memory(),
        active_lengths=batch.active_lengths.pin_memory(),
        keep_index_for_q=batch.keep_index_for_q.pin_memory(),
    )


def to_device(
    batch: ParallelOfflineBatch | None,
    *,
    device: torch.device,
) -> ParallelOfflineBatch | None:
    """Move a packed CPU batch to the training device."""

    if batch is None:
        return None
    return ParallelOfflineBatch(
        clean_input_ids=batch.clean_input_ids.to(device=device, non_blocking=True),
        prompt_lens=batch.prompt_lens.to(device=device, non_blocking=True),
        clean_lens=batch.clean_lens.to(device=device, non_blocking=True),
        draft_input_ids=batch.draft_input_ids.to(device=device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device=device, non_blocking=True),
        position_ids=batch.position_ids.to(device=device, non_blocking=True),
        logit_keep_indices=batch.logit_keep_indices.to(device=device, non_blocking=True),
        active_token_mask=batch.active_token_mask.to(device=device, non_blocking=True),
        teacher_logit_positions=batch.teacher_logit_positions.to(device=device, non_blocking=True),
        target_ids=batch.target_ids.to(device=device, non_blocking=True),
        anchor_positions=batch.anchor_positions.to(device=device, non_blocking=True),
        block_keep_mask=batch.block_keep_mask.to(device=device, non_blocking=True),
        active_lengths=batch.active_lengths.to(device=device, non_blocking=True),
        keep_index_for_q=batch.keep_index_for_q.to(device=device, non_blocking=True),
    )


def _select_anchor_starts(
    response_len: int,
    *,
    max_anchors: int,
) -> list[int]:
    """Select response token offsets that can serve as block anchors."""

    starts = list(range(0, max(int(response_len) - 1, 0)))
    if max_anchors > 0 and len(starts) > int(max_anchors):
        starts = sorted(random.sample(starts, int(max_anchors)))
    return starts


def build_packed_block_batch(
    examples: OfflineExample | list[OfflineExample],
    *,
    block_size: int,
    mask_token_id: int,
    max_anchors: int,
    device: torch.device,
) -> ParallelOfflineBatch | None:
    """Pack multiple anchor+MASK blocks so they can be trained in one forward pass."""

    if int(max_anchors) <= 0:
        raise ValueError("max_anchors must be > 0")
    if not isinstance(examples, list):
        examples = [examples]
    if not examples:
        return None

    prepared: list[dict[str, Any]] = []
    for example in examples:
        prompt_ids = example.prompt_ids.to(device=device, dtype=torch.long).view(-1)
        response_ids = example.response_ids.to(device=device, dtype=torch.long).view(-1)
        # Each start selects response[start] as the visible anchor token.
        starts = _select_anchor_starts(
            int(response_ids.numel()),
            max_anchors=max_anchors,
        )
        if not starts:
            continue
        clean_ids = torch.cat([prompt_ids, response_ids], dim=0)
        prepared.append(
            {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "starts": starts,
                "clean_ids": clean_ids,
            }
        )
    if not prepared:
        return None

    batch_size = len(prepared)
    block_size = int(block_size)
    clean_lens = torch.tensor(
        [int(item["clean_ids"].numel()) for item in prepared],
        dtype=torch.long,
        device=device,
    )
    prompt_lens = torch.tensor(
        [int(item["prompt_ids"].numel()) for item in prepared],
        dtype=torch.long,
        device=device,
    )
    clean_len = int(clean_lens.max().item())
    num_blocks = max(len(item["starts"]) for item in prepared)
    query_len = num_blocks * block_size
    clean_input_ids = torch.zeros((batch_size, clean_len), dtype=torch.long, device=device)
    for batch_idx, item in enumerate(prepared):
        ids = item["clean_ids"]
        clean_input_ids[batch_idx, : int(ids.numel())] = ids

    draft_input_ids = torch.full(
        (batch_size, query_len),
        int(mask_token_id),
        dtype=torch.long,
        device=device,
    )
    position_ids_flat = torch.zeros((batch_size, query_len), dtype=torch.long, device=device)
    anchor_positions = torch.zeros((batch_size, num_blocks), dtype=torch.long, device=device)
    block_keep_mask = torch.zeros((batch_size, num_blocks), dtype=torch.bool, device=device)
    active_lengths = torch.zeros((batch_size, num_blocks), dtype=torch.long, device=device)
    active_by_q = torch.zeros((batch_size, query_len), dtype=torch.bool, device=device)
    teacher_pos_by_q = torch.zeros((batch_size, query_len), dtype=torch.long, device=device)
    target_ids_by_q = torch.zeros((batch_size, query_len), dtype=torch.long, device=device)

    for batch_idx, item in enumerate(prepared):
        prompt_len = int(item["prompt_ids"].numel())
        response_ids = item["response_ids"]
        for block_idx, response_start in enumerate(item["starts"]):
            anchor_pos = prompt_len + int(response_start)
            block_start = block_idx * block_size
            # Predict at most block_size - 1 tokens after the anchor.
            active_len = min(
                block_size - 1,
                int(response_ids.numel()) - int(response_start) - 1,
            )
            if active_len <= 0:
                continue
            anchor_positions[batch_idx, block_idx] = anchor_pos
            block_keep_mask[batch_idx, block_idx] = True
            active_lengths[batch_idx, block_idx] = active_len
            # Block layout: [anchor, MASK, MASK, ...].
            draft_input_ids[batch_idx, block_start] = response_ids[int(response_start)]
            position_ids_flat[batch_idx, block_start : block_start + block_size] = torch.arange(
                anchor_pos,
                anchor_pos + block_size,
                dtype=torch.long,
                device=device,
            )
            for offset in range(1, active_len + 1):
                query_pos = block_start + offset
                active_by_q[batch_idx, query_pos] = True
                # AR logit at position t-1 predicts token t.
                teacher_pos_by_q[batch_idx, query_pos] = anchor_pos + offset - 1
                target_ids_by_q[batch_idx, query_pos] = response_ids[
                    int(response_start) + offset
                ]

    if not bool(active_by_q.any().item()):
        return None

    logit_keep_indices = torch.nonzero(active_by_q.any(dim=0), as_tuple=False).flatten()
    # Keep only query positions that have at least one active target in the batch.
    active_token_mask = active_by_q[:, logit_keep_indices]
    teacher_logit_positions = teacher_pos_by_q[:, logit_keep_indices]
    target_ids = target_ids_by_q[:, logit_keep_indices]
    keep_index_for_q = torch.full((query_len,), -1, dtype=torch.long, device=device)
    keep_index_for_q[logit_keep_indices] = torch.arange(
        int(logit_keep_indices.numel()),
        dtype=torch.long,
        device=device,
    )

    q_indices = torch.arange(query_len, device=device).view(1, 1, query_len, 1)
    kv_indices = torch.arange(clean_len + query_len, device=device).view(
        1, 1, 1, clean_len + query_len
    )
    clean_len_expanded = clean_lens.view(batch_size, 1, 1, 1)
    q_block_ids = q_indices // block_size
    anchor_expanded = anchor_positions.view(batch_size, 1, num_blocks, 1).repeat_interleave(
        block_size,
        dim=2,
    )
    mask_context = (
        (kv_indices < clean_len)
        & (kv_indices < clean_len_expanded)
        & (kv_indices < anchor_expanded)
    )
    is_draft = kv_indices >= clean_len
    kv_block_ids = (kv_indices - clean_len) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    valid_block = block_keep_mask.view(batch_size, 1, num_blocks, 1).repeat_interleave(
        block_size,
        dim=2,
    )
    draft_self_mask = is_draft & ((kv_indices - clean_len) == q_indices)
    attention_mask = (
        ((mask_context | mask_draft) & valid_block)
        | (draft_self_mask & ~valid_block)
    ).bool()
    # Shape: [batch, 1, draft_query_len, clean_kv_len + draft_kv_len].

    return ParallelOfflineBatch(
        clean_input_ids=clean_input_ids,
        prompt_lens=prompt_lens,
        clean_lens=clean_lens,
        draft_input_ids=draft_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_flat,
        logit_keep_indices=logit_keep_indices,
        active_token_mask=active_token_mask,
        teacher_logit_positions=teacher_logit_positions,
        target_ids=target_ids.long(),
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        active_lengths=active_lengths,
        keep_index_for_q=keep_index_for_q,
    )
