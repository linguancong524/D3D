"""Scheduler-owned transient KV state for D3DV2 packed drafting.

This module owns allocator metadata only. It never inserts token/KV pairs into
radix; callers publish a state only after a complete model forward succeeds.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch

from sglang.srt.utils.common import ceil_align

from .protocol import Commit


class TransientDraftState(msgspec.Struct):
    session: str
    revision: int
    proposal_id: str
    tokens: tuple[int, ...]
    pending_base_len: int
    req_pool_idx: int
    kv_allocated_len: int
    pending_causal: bool = False
    inflight_transaction: str | None = None
    transaction_base_len: int = 0


class TransientDraftStateManager:
    """Own non-radix req/token-pool slots across decoupled revisions."""

    def __init__(self, *, req_to_token_pool: Any, token_allocator: Any) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_allocator = token_allocator
        self.page_size = token_allocator.page_size
        self.states: dict[str, TransientDraftState] = {}

    def prefix_indices(self, *, session: str) -> torch.Tensor:
        """Materialize prefix indices for alloc_for_extend's int64 pointer ABI."""
        state = self._get(session)
        return self.req_to_token_pool.req_to_token[
            state.req_pool_idx, : len(state.tokens)
        ].to(dtype=torch.int64)

    def register(
        self,
        *,
        session: str,
        revision: int,
        proposal_id: str,
        tokens: tuple[int, ...],
        pending_base_len: int,
        req_pool_idx: int,
        kv_allocated_len: int,
        pending_causal: bool = False,
    ) -> None:
        if session in self.states:
            raise ValueError("transient draft session already exists")
        self._validate_lengths(
            tokens=tokens,
            pending_base_len=pending_base_len,
            kv_allocated_len=kv_allocated_len,
        )
        self.states[session] = TransientDraftState(
            session=session,
            revision=revision,
            proposal_id=proposal_id,
            tokens=tokens,
            pending_base_len=pending_base_len,
            req_pool_idx=req_pool_idx,
            kv_allocated_len=kv_allocated_len,
            pending_causal=pending_causal,
        )

    def begin(
        self,
        *,
        session: str,
        revision: int,
        transaction: str,
        proposal_id: str | None = None,
    ) -> None:
        state = self._get(session)
        if state.revision != revision:
            raise ValueError("packed transaction revision mismatch")
        if proposal_id is not None and state.proposal_id and proposal_id != state.proposal_id:
            raise ValueError("packed proposal does not match session revision")
        if state.inflight_transaction is not None:
            raise ValueError("transient draft session already has an inflight forward")
        if not transaction:
            raise ValueError("packed transaction identity is required")
        state.inflight_transaction = transaction
        state.transaction_base_len = state.kv_allocated_len

    def publish_pending(
        self,
        *,
        session: str,
        transaction: str,
        proposal_id: str,
        tokens: tuple[int, ...],
        pending_base_len: int,
        allocated_len: int,
        pending_causal: bool = False,
    ) -> None:
        """Publish pending replay and discard all candidate-branch scratch."""
        state = self._transaction(session=session, transaction=transaction)
        self._validate_lengths(
            tokens=tokens,
            pending_base_len=pending_base_len,
            kv_allocated_len=allocated_len,
        )
        retained_len = len(tokens)
        self._free_tail(state=state, target=retained_len, end=allocated_len)
        state.tokens = tokens
        state.pending_base_len = pending_base_len
        state.proposal_id = proposal_id
        state.kv_allocated_len = retained_len
        state.pending_causal = pending_causal
        state.inflight_transaction = None
        state.transaction_base_len = 0

    def abort(self, *, session: str, transaction: str, allocated_len: int) -> None:
        """Rollback scratch from a failed optional forward without publishing."""
        state = self._transaction(session=session, transaction=transaction)
        self._free_tail(
            state=state,
            target=state.transaction_base_len,
            end=allocated_len,
        )
        state.inflight_transaction = None
        state.transaction_base_len = 0

    def validate_commit(self, item: Commit) -> int:
        """Validate without mutation and return the safe reusable KV length.

        Pending rows are block-causal and therefore depend on every token in
        that pending block.  A proper prefix of the block is not reusable after
        rejection; only the immutable causal base is.  The whole block is safe
        when Target committed it exactly.
        """
        state = self._get(item.session)
        if state.inflight_transaction is not None:
            raise ValueError("cannot commit while optional forward is inflight")
        if item.revision != state.revision + 1:
            raise ValueError("commit revision mismatch")
        if state.proposal_id and item.proposal_id != state.proposal_id:
            raise ValueError("commit proposal does not match session revision")
        committed = item.committed_prefix[:-1]
        common = 0
        for left, right in zip(state.tokens, committed):
            if left != right:
                break
            common += 1
        if common < state.pending_base_len:
            raise ValueError("commit diverged inside immutable committed KV")
        if state.pending_causal:
            return common
        return common if common == len(state.tokens) else state.pending_base_len

    def commit(self, item: Commit) -> int:
        """Crop to actual reusable KV and return its logical token length.

        A correction may diverge inside block-causal pending tokens.  A partial
        LCP is context-polluted by the rejected suffix, so crop to the immutable
        causal base and replay every accepted token on the next packed draft.
        """
        state = self._get(item.session)
        common = self.validate_commit(item)
        self._free_tail(
            state=state,
            target=common,
            end=state.kv_allocated_len,
        )
        state.tokens = state.tokens[:common]
        state.pending_base_len = common
        state.kv_allocated_len = common
        state.revision = item.revision
        state.proposal_id = ""
        state.pending_causal = True
        return common

    def release_all(self) -> None:
        for session in tuple(self.states):
            self.release(session=session)

    def release(self, *, session: str) -> None:
        state = self.states.pop(session, None)
        if state is None:
            return
        self._free_tail(state=state, target=0, end=state.kv_allocated_len)
        self.req_to_token_pool.free_slots.append(state.req_pool_idx)

    def _free_tail(
        self, *, state: TransientDraftState, target: int, end: int
    ) -> None:
        if end <= target:
            return
        start = target if self.page_size == 1 else ceil_align(target, self.page_size)
        if start >= end:
            return
        indices = self.req_to_token_pool.req_to_token[
            state.req_pool_idx, start:end
        ]
        self.token_allocator.free(indices)

    def _get(self, session: str) -> TransientDraftState:
        try:
            return self.states[session]
        except KeyError as exc:
            raise ValueError("unknown transient draft session") from exc

    def _transaction(
        self, *, session: str, transaction: str
    ) -> TransientDraftState:
        state = self._get(session)
        if state.inflight_transaction != transaction:
            raise ValueError("stale packed transaction")
        return state

    @staticmethod
    def _validate_lengths(
        *, tokens: tuple[int, ...], pending_base_len: int, kv_allocated_len: int
    ) -> None:
        if not 0 <= pending_base_len <= len(tokens) <= kv_allocated_len:
            raise ValueError("invalid transient KV lengths")
