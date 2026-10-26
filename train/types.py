from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OfflineExample:
    """A tokenized prompt-response pair used by offline D3D training."""

    prompt_id: str
    prompt_ids: torch.LongTensor
    response_ids: torch.LongTensor


@dataclass(frozen=True)
class ParallelOfflineBatch:
    """Packed clean context plus parallel anchor-conditioned draft blocks."""

    clean_input_ids: torch.LongTensor
    prompt_lens: torch.LongTensor
    clean_lens: torch.LongTensor
    draft_input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    position_ids: torch.LongTensor
    logit_keep_indices: torch.LongTensor
    active_token_mask: torch.BoolTensor
    teacher_logit_positions: torch.LongTensor
    target_ids: torch.LongTensor
    anchor_positions: torch.LongTensor
    block_keep_mask: torch.BoolTensor
    active_lengths: torch.LongTensor
    keep_index_for_q: torch.LongTensor


@dataclass(frozen=True)
class RowItem:
    """One row after CPU-side parsing/tokenization."""

    example: OfflineExample | None
    error: tuple[int, str, str] | None = None
