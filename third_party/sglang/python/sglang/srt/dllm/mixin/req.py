# pyright: reportGeneralTypeIssues=false
# The ``self: Req`` annotations below are sglang's mixin idiom: the host class
# (``schedule_batch.Req``) supplies every attribute at runtime. Pyright cannot
# prove a mixin's self type against its host class and flags every such method,
# so that rule is disabled for this module while host attributes are declared
# explicitly under TYPE_CHECKING.
from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Any

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    if TYPE_CHECKING:
        # Supplied by the host class (``Req``).
        origin_input_ids: Any
        output_ids: Any
        prefix_indices: Any
        full_untruncated_fill_ids: Any
        dllm_initialized: bool
        dllm_incomplete_ids: Any
        dllm_algo_state: Any
        dllm_block_offset: int
        dllm_phase: DllmReqPhase | None
        # Set for the whole lifetime of a dLLM request (see init_diffusion_llm);
        # every helper below assumes it is present.
        dllm_config: DllmConfig

    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: DllmReqPhase | None = None
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)
        if self.dllm_config.anchor_conditioned_block:
            committed_length = len(self.origin_input_ids) + len(self.output_ids)
            # SDAR's prefix is causal, but the final anchor must be recomputed
            # together with the masks so its K/V is conditioned on the whole
            # denoising block. Prefill every committed token except that anchor.
            if prefix_length < committed_length - 1:
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
            else:
                self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        fill_length = len(self.full_untruncated_fill_ids)
        min_required_length = min(
            prefix_length + self.dllm_config.block_size, fill_length
        )
        input_block = self.full_untruncated_fill_ids[prefix_length:min_required_length]
        if not input_block:
            # still incoming stage
            return
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_incomplete_ids:
            prefix_len = len(self.prefix_indices)
            assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
            self.full_untruncated_fill_ids = (
                self.full_untruncated_fill_ids[:prefix_len] + self.dllm_incomplete_ids
            )
            # extend_range is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        self.dllm_block_offset = (
            0
            if not self.dllm_initialized
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        # OneStep uses the trailing ``block_size`` tokens as
        # [anchor] + [mask]*(block_size-1). The anchor is the last committed
        # token (origin_input_ids + output_ids); it stays the final real token
        # of the prefix and the block is extended from it, so the anchor enters
        # at its true position instead of being duplicated. Iterative dLLM
        # algorithms retain their upstream all-mask, fixed-width block.
        num_masks = self.dllm_config.block_size - (
            1 if self.dllm_config.anchor_conditioned_block else 0
        )
        committed_ids = self.origin_input_ids + self.output_ids
        if (
            self.dllm_config.anchor_conditioned_block
            and len(self.prefix_indices) < len(committed_ids) - 1
        ):
            # Keep prefix staging from crossing the anchor boundary. Once the
            # prefix reaches this truncated fill, the next round exposes the
            # complete anchor-conditioned denoising block.
            self.full_untruncated_fill_ids = committed_ids[:-1]
        else:
            self.full_untruncated_fill_ids = committed_ids + array(
                "q", [self.dllm_config.mask_id] * num_masks
            )
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        if not self.dllm_config.anchor_conditioned_block:
            assert prefix_len % self.dllm_config.block_size == 0, (
                f"Unexpected prefix len: {prefix_len}"
            )
        # OneStep may match any token inside the pending denoising block. Its
        # block offset follows that exact accepted prefix rather than rounding
        # to a dLLM block boundary.
        self.dllm_block_offset = max(self.dllm_block_offset, prefix_len)
