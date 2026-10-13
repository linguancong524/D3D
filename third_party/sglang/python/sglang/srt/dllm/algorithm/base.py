from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu

_is_npu = is_npu()

DllmRunOutput = Tuple[
    Union[LogitsProcessorOutput, torch.Tensor],
    List,
    Optional[List[int]],
    Optional[List[Any]],
    bool,
]


class DllmAlgorithm:
    """dLLM algorithm: subclasses implement ``step``; the base owns the
    synchronous and FDFO (``--dllm-fdfo``) execution loops in ``run``.
    """

    def __init__(self, config: DllmConfig):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        self.fdfo = config.first_done_first_out_mode
        self.anchor_conditioned_block = getattr(
            config, "anchor_conditioned_block", False
        )

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [None] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return block_size + 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        """One denoise step, advancing ``forward_batch.input_ids``/``states`` in
        place. Returns, per block, whether it was already complete *on entry* --
        i.e. this forward persisted its final KV cache and it can be emitted.
        """
        raise NotImplementedError

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        if self.fdfo:
            return self._run_fdfo(model_runner, forward_batch, algo_states)
        return self._run_sync(model_runner, forward_batch)

    def _block_lengths(self, forward_batch: ForwardBatch) -> List[int]:
        """Per-request token count forwarded in this batch."""
        cpu_lens = forward_batch.extend_seq_lens_cpu
        if cpu_lens is not None:
            return list(cpu_lens)
        seq_lens = forward_batch.extend_seq_lens
        if seq_lens is None:
            return []
        return [int(length) for length in seq_lens.tolist()]

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        """Per-request count of leading real (non-mask) tokens in its segment.

        The denoising block is the trailing ``block_size`` tokens of the fill
        ([anchor] + [mask]*(block_size-1)), so a segment ends with masks unless
        the extend was truncated by the token budget.
        """
        input_ids = forward_batch.input_ids
        lengths = self._block_lengths(forward_batch)
        if not lengths:
            return []
        device = input_ids.device
        is_mask = (input_ids == self.mask_id).to(torch.int32)
        seg_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=device, dtype=torch.int64),
            torch.tensor(lengths, device=device, dtype=torch.int64),
        )
        mask_counts = torch.zeros(len(lengths), device=device, dtype=torch.int32)
        mask_counts.scatter_add_(0, seg_ids, is_mask)
        return [
            length - int(mask_count)
            for length, mask_count in zip(lengths, mask_counts.tolist())
        ]

    def _run_sync(
        self, model_runner: ModelRunner, forward_batch: ForwardBatch
    ) -> DllmRunOutput:
        start_list = self._block_start_list(forward_batch)
        lengths = self._block_lengths(forward_batch)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # No mask to denoise: return empty so process_batch_result_dllm skips the
        # stream branch (matches the pre-refactor behavior).
        if all(start == length for start, length in zip(start_list, lengths)):
            return out.logits_output, [], None, None, out.can_run_graph

        states = self.init_step_state(forward_batch)
        # NPU: attention metadata is stable across a block's denoise steps (the
        # first forward above already planned it), so mark it ready once and let
        # every later forward skip re-planning.
        if _is_npu:
            forward_batch.mark_forward_metadata_ready()
        for _ in range(self.max_steps(self.block_size)):
            done = self.step(forward_batch, out.logits_output.full_logits, states)
            if all(done):
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        # Each request's segment is [real tokens][masks]; the denoised tokens are
        # exactly the mask positions that were filled in this forward.
        input_ids = forward_batch.input_ids
        next_token_ids_list: List[torch.Tensor] = []
        offset = 0
        for start, length in zip(start_list, lengths):
            next_token_ids_list.append(input_ids[offset + start : offset + length])
            offset += length
        return out.logits_output, next_token_ids_list, None, None, out.can_run_graph

    def _run_fdfo(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]],
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        # Capture the mask boundary before ``step`` mutates input_ids. OneStep
        # fills every mask in place, so recomputing it afterwards would make all
        # generated tails look empty.
        start_list = self._block_start_list(forward_batch)
        lengths = self._block_lengths(forward_batch)
        if not self.anchor_conditioned_block:
            # Preserve the upstream FDFO contract for iterative, fixed-width
            # algorithms: every round returns the complete block, including the
            # final no-mask round that commits its KV. The scheduler already
            # filters real prompt positions before emitting output tokens.
            start_list = [0] * batch_size

        if algo_states is None:
            algo_states = [None] * batch_size
        fresh: Optional[List[Any]] = None
        states: List[Any] = []
        for i, carried in enumerate(algo_states):
            if carried is None:
                if fresh is None:
                    fresh = self.init_step_state(forward_batch)
                states.append(fresh[i])
            else:
                states.append(carried)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        done = self.step(forward_batch, out.logits_output.full_logits, states)

        if self.anchor_conditioned_block:
            accept_length_per_req_cpu = [
                length - start if d else 0
                for start, length, d in zip(start_list, lengths, done)
            ]
        else:
            accept_length_per_req_cpu = [self.block_size if d else 0 for d in done]
        # Per-request segments: the denoising block is the trailing block_size
        # tokens of the fill, and the matched prefix is block-aligned while the
        # block starts at the anchor, so segments carry a variable token count.
        # Only the mask positions were filled by ``step``.
        input_ids = forward_batch.input_ids
        next_token_ids_list: List[List[int]] = []
        offset = 0
        for start, length in zip(start_list, lengths):
            next_token_ids_list.append(
                input_ids[offset + start : offset + length].tolist()
            )
            offset += length
        states_out = [None if done[i] else states[i] for i in range(batch_size)]

        return (
            out.logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            states_out,
            out.can_run_graph,
        )
