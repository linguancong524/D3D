"""Greedy one-step SDAR backend with bounded packed outcome generation.

Algorithm reference: D3D caa257f, D3DV2/eval_async_packed.py. This adapter
implements the model-facing contract independently; no dependency on D3D's
evaluation scripts, paths, or dataset loaders is required.
"""

from __future__ import annotations

import importlib
from typing import Any

import msgspec
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .packed_layout import build_packed_layout
from .protocol import (
    Branch,
    Commit,
    Proposal,
    proposal_identity,
    select_branches,
    vocabulary_fingerprint,
)


class CacheState(msgspec.Struct):
    tokens: tuple[int, ...]
    cache: Any  # Transformers cache protocol, owned exclusively by this backend.
    revision: int = -1
    proposal_id: str = ""
    pending_base_len: int = 0


class SDARBackend:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str | None = None,
        model_class: str | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        mask_token_id: int,
        block_size: int = 16,
        num_anchors: int = 0,
        accept_counts: tuple[int, ...] = (),
        max_branches: int = 16,
        attention_backend: str = "sdpa",
        trust_remote_code: bool = False,
        prefill_chunk_size: int = 1024,
    ) -> None:
        if block_size < 2 or num_anchors < 0 or max_branches < 1:
            raise ValueError("invalid block size or packed budget")
        if prefill_chunk_size < 1:
            raise ValueError("prefill chunk size must be positive")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path, trust_remote_code=trust_remote_code
        )
        self.fingerprint = vocabulary_fingerprint(tokenizer.get_vocab())
        cls = AutoModelForCausalLM
        if model_class:
            module, name = model_class.rsplit(":", 1)
            # Explicit CLI-selected local loader; never supplied by a peer.
            cls = vars(importlib.import_module(module))[name]
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = (
            cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attention_backend,
                trust_remote_code=trust_remote_code,
            )
            .to(device)
            .eval()
        )
        self.vocab_size = self.model.config.vocab_size
        if not 0 <= mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id is outside model vocabulary")
        if max(tokenizer.get_vocab().values()) >= self.vocab_size:
            raise ValueError("tokenizer exceeds draft model vocabulary")
        self.device = torch.device(device)
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.accept_counts = accept_counts or tuple(range(1, block_size + 1))
        if any(not 1 <= c <= block_size for c in self.accept_counts):
            raise ValueError("accept counts outside block")
        self.max_branches = max_branches
        self.prefill_chunk_size = prefill_chunk_size
        self.states: dict[str, CacheState] = {}

    def _state(
        self, *, session: str, revision: int, prefix: tuple[int, ...]
    ) -> CacheState:
        state = self.states.get(session)
        if state is None:
            state = CacheState((), DynamicCache(), revision=revision)
            self.states[session] = state
        elif state.revision != revision:
            raise ValueError(
                f"draft revision mismatch: state={state.revision}, request={revision}"
            )
        common = 0
        for left, right in zip(state.tokens, prefix):
            if left != right:
                break
            common += 1
        state.cache.crop(common)
        state.tokens = state.tokens[:common]
        return state

    def _forward(
        self,
        *,
        state: CacheState,
        tokens: list[int],
        positions: list[int],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            torch.tensor([tokens], dtype=torch.long, device=self.device),
            position_ids=torch.tensor([positions], device=self.device),
            attention_mask=mask[None, None],
            past_key_values=state.cache,
            use_cache=True,
            store_kv=True,
            logits_to_keep=len(tokens),
        )
        state.cache = output.past_key_values
        return output.logits[0]

    def _proposal(self, *, logits: torch.Tensor, bonus_token: int | None) -> Proposal:
        tokens = logits.argmax(-1).tolist()
        if bonus_token is not None:
            tokens[0] = bonus_token
        width = min(self.num_anchors + 1, self.vocab_size)
        top_tokens = torch.topk(logits, k=width, dim=-1).indices.tolist()
        return Proposal(tuple(tokens), tuple(tuple(row) for row in top_tokens))

    def _prefill(self, *, state: CacheState, committed: tuple[int, ...]) -> None:
        # Bound the query dimension of long-prompt masks and activations. Keep
        # the final suffix for the fused commit + draft forward below.
        while len(committed) - len(state.tokens) > self.prefill_chunk_size:
            cached = len(state.tokens)
            end = cached + self.prefill_chunk_size
            positions = torch.arange(cached, end, device=self.device)
            keys = torch.arange(end, device=self.device)
            self._forward(
                state=state,
                tokens=list(committed[cached:end]),
                positions=list(range(cached, end)),
                mask=keys[None, :] <= positions[:, None],
            )
            state.tokens = committed[:end]

    @torch.inference_mode()
    def draft(
        self, *, session: str, revision: int, prefix: tuple[int, ...]
    ) -> Proposal:
        committed, bonus_token = prefix[:-1], prefix[-1]
        state = self._state(session=session, revision=revision, prefix=committed)
        self._prefill(state=state, committed=committed)
        cached = len(state.tokens)
        pending = committed[cached:]
        block = [bonus_token] + [self.mask_token_id] * (self.block_size - 1)
        tokens = list(pending) + block
        size = len(tokens)
        mask = torch.ones((size, cached + size), dtype=torch.bool, device=self.device)
        # Real committed tokens use causal attention, so truncation preserves KV.
        # The denoising block uses full within-block attention.
        for row in range(len(pending)):
            mask[row, cached + row + 1 :] = False
        logits = self._forward(
            state=state,
            tokens=tokens,
            positions=list(range(cached, cached + size)),
            mask=mask,
        )
        state.cache.crop(len(committed))
        state.tokens = committed
        state.pending_base_len = len(committed)
        proposal = self._proposal(
            logits=logits[-self.block_size :], bonus_token=bonus_token
        )
        state.proposal_id = proposal_identity(proposal)
        return proposal

    @torch.inference_mode()
    def speculate(
        self,
        *,
        session: str,
        revision: int,
        prefix: tuple[int, ...],
        proposal: Proposal,
    ) -> dict[Branch, Proposal]:
        branches = select_branches(
            proposal=proposal,
            num_anchors=self.num_anchors,
            accept_counts=self.accept_counts,
            max_branches=self.max_branches,
        )
        if not branches:
            return {}
        committed = prefix[:-1]
        state = self._state(session=session, revision=revision, prefix=committed)
        if state.proposal_id != proposal_identity(proposal):
            raise ValueError("packed proposal does not match session revision")
        self._prefill(state=state, committed=committed)
        # Selected packed blocks may not yet have real-token KV. Reconstruct
        # only the missing committed suffix together with this pending block.
        cached = len(state.tokens)
        pending = list(committed[cached:]) + list(proposal.tokens)
        layout = build_packed_layout(
            base_kv_len=cached,
            pending_tokens=tuple(pending),
            branches=branches,
            branch_token_count=self.block_size,
            mask_token_id=self.mask_token_id,
            attention_mode="causal",
            causal_prefix_len=len(committed) - cached,
        )
        logits = self._forward(
            state=state,
            tokens=list(layout.input_ids),
            positions=list(layout.position_ids),
            mask=torch.tensor(
                layout.attention_mask, dtype=torch.bool, device=self.device
            ),
        )
        state.cache.crop(layout.retained_len)
        state.tokens = committed + proposal.tokens
        state.pending_base_len = len(committed)
        return {
            segment.branch: self._proposal(
                logits=logits[segment.start : segment.stop],
                bonus_token=segment.branch.bonus_token,
            )
            for segment in layout.segments
        }

    def draft_many(
        self, *, items: list[tuple[str, int, tuple[int, ...]]]
    ) -> dict[str, Proposal]:
        # Serial fallback: the transformers backend has no native batch loop.
        return {
            key: self.draft(session=key, revision=revision, prefix=prefix)
            for key, revision, prefix in items
        }

    def commit_many(self, *, items: list[Commit]) -> None:
        for item in items:
            state = self.states.get(item.session)
            # Optional failure may deliberately discard backend KV while the
            # service keeps the published proposal. The next required draft
            # reconstructs from committed_prefix, so a missing state is valid.
            if state is None:
                continue
            if item.revision != state.revision + 1:
                raise ValueError(
                    f"commit revision mismatch: state={state.revision}, "
                    f"request={item.revision}"
                )
            if state.proposal_id != item.proposal_id:
                raise ValueError("commit proposal does not match session revision")
            committed = item.committed_prefix[:-1]
            common = 0
            for left, right in zip(state.tokens, committed):
                if left != right:
                    break
                common += 1
            if common < state.pending_base_len:
                raise ValueError("commit diverged inside immutable committed KV")
            reusable = (
                common if common == len(state.tokens) else state.pending_base_len
            )
            state.cache.crop(reusable)
            state.tokens = state.tokens[:reusable]
            state.pending_base_len = reusable
            state.revision = item.revision
            state.proposal_id = ""

    def close(self, *, session: str) -> None:
        self.states.pop(session, None)

    def shutdown(self) -> None:
        self.states.clear()
