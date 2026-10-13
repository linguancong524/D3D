"""Pure-CPU planner for a D3DV2 pending-replay and branch-packed query."""

from __future__ import annotations

from typing import Any, Literal

import msgspec
import numpy as np

from .protocol import Branch


class PackedSegment(msgspec.Struct, frozen=True):
    branch: Branch
    start: int
    stop: int


class PackedLayout(msgspec.Struct, frozen=True):
    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    # uint8 [query_len, base_kv_len + query_len], one row per query token
    # including the cached KV columns. Materialized with vectorized slices:
    # a Python nested list costs ~2.7M element writes per packed batch, which
    # exceeded the draft model forward itself.
    attention_mask: Any
    base_kv_len: int
    pending_len: int
    segments: tuple[PackedSegment, ...]

    @property
    def query_len(self) -> int:
        return len(self.input_ids)

    @property
    def kv_len(self) -> int:
        return self.base_kv_len + self.query_len

    @property
    def retained_len(self) -> int:
        """KV length published after candidate scratch is discarded."""
        return self.base_kv_len + self.pending_len


def group_branches_by_accept_count(
    branches: tuple[Branch, ...], *, max_group_size: int = 3
) -> tuple[tuple[Branch, ...], ...]:
    """Group sibling bonus candidates without mixing accept positions."""
    if max_group_size < 1:
        raise ValueError("packed branch group size must be positive")
    groups = []
    start = 0
    while start < len(branches):
        accept_count = branches[start].num_accept_tokens
        stop = start
        while (
            stop < len(branches)
            and branches[stop].num_accept_tokens == accept_count
            and stop - start < max_group_size
        ):
            stop += 1
        groups.append(branches[start:stop])
        start = stop
    return tuple(groups)


def _fill_causal_rows(
    mask: np.ndarray, rows: np.ndarray, base_kv_len: int
) -> None:
    """Mark keys [0, base_kv_len + row] visible for each causal query row."""
    if rows.size == 0:
        return
    keys = np.arange(mask.shape[1], dtype=np.int64)[None, :]
    mask[rows, :] = keys <= (base_kv_len + rows)[:, None]


def build_packed_layout(
    *,
    base_kv_len: int,
    pending_tokens: tuple[int, ...],
    branches: tuple[Branch, ...],
    branch_token_count: int,
    mask_token_id: int,
    attention_mode: Literal["causal", "block_causal"] = "block_causal",
    causal_prefix_len: int = 0,
) -> PackedLayout:
    """Build the exact tokens, logical positions, and visibility of D3DV2.

    Physical KV slots are append-packed. Logical positions restart for every
    candidate branch, and the explicit mask prevents sibling visibility.
    """
    if base_kv_len < 0 or branch_token_count < 1:
        raise ValueError("invalid packed query dimensions")
    if not pending_tokens or not branches:
        raise ValueError("packed replay requires pending tokens and branches")
    if not 0 <= causal_prefix_len < len(pending_tokens):
        raise ValueError("causal replay prefix must precede a pending proposal")
    if attention_mode not in ("causal", "block_causal"):
        raise ValueError(f"unsupported attention mode: {attention_mode}")
    pending_len = len(pending_tokens)
    proposal_len = pending_len - causal_prefix_len
    for branch in branches:
        if not 1 <= branch.num_accept_tokens <= proposal_len:
            raise ValueError("branch acceptance is outside the pending proposal")

    query_len = pending_len + len(branches) * branch_token_count
    kv_len = base_kv_len + query_len
    input_ids = list(pending_tokens) + [mask_token_id] * (
        query_len - pending_len
    )
    position_ids = list(range(base_kv_len, base_kv_len + pending_len))
    mask = np.zeros((query_len, kv_len), dtype=np.uint8)

    if causal_prefix_len:
        _fill_causal_rows(mask, np.arange(causal_prefix_len), base_kv_len)
    if attention_mode == "block_causal":
        mask[causal_prefix_len:pending_len, : base_kv_len + pending_len] = 1
    else:
        _fill_causal_rows(
            mask, np.arange(causal_prefix_len, pending_len), base_kv_len
        )

    segments = []
    for index, branch in enumerate(branches):
        start = pending_len + index * branch_token_count
        stop = start + branch_token_count
        accepted_pending = causal_prefix_len + branch.num_accept_tokens
        logical_start = base_kv_len + accepted_pending
        position_ids.extend(range(logical_start, logical_start + branch_token_count))
        if branch.bonus_token is not None:
            input_ids[start] = branch.bonus_token
        visible_pending_stop = base_kv_len + accepted_pending
        own_start = base_kv_len + start
        own_stop = base_kv_len + stop
        mask[start:stop, :visible_pending_stop] = 1
        mask[start:stop, own_start:own_stop] = 1
        segments.append(PackedSegment(branch=branch, start=start, stop=stop))

    return PackedLayout(
        input_ids=tuple(input_ids),
        position_ids=tuple(position_ids),
        attention_mask=mask,
        base_kv_len=base_kv_len,
        pending_len=pending_len,
        segments=tuple(segments),
    )
