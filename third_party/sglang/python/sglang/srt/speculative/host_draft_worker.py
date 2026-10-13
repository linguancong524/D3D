from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func as assign_extend_cache_locs_func,
)
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_schedule
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs

_is_cpu = is_cpu()

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


def _derive_tree_links(
    mask: np.ndarray, bs: int, draft_token_num: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side (retrive_next_token, retrive_next_sibling), matching what
    reconstruct_indices_from_tree_mask produces on device.

    ``mask[b, i, j]`` marks node j as an ancestor of node i, so i's immediate parent
    is the largest such j < i, and both links follow from the parents alone.
    """
    tree = mask.reshape(bs, draft_token_num, draft_token_num)
    node_order = np.arange(draft_token_num)
    ancestors = tree & (node_order < node_order[:, None])
    parents = np.where(ancestors.any(-1), (ancestors * node_order).argmax(-1), -1)

    next_token = np.full((bs, draft_token_num), -1, dtype=np.int64)
    next_sibling = np.full((bs, draft_token_num), -1, dtype=np.int64)
    for b in range(bs):
        # Descending scan, so every k > i is already recorded when i is reached.
        earliest_child_of = {}
        for i in reversed(range(draft_token_num)):
            next_token[b, i] = earliest_child_of.get(i, -1)
            parent = int(parents[b, i])
            if parent >= 0:
                next_sibling[b, i] = earliest_child_of.get(parent, -1)
                earliest_child_of[parent] = i
    return torch.from_numpy(next_token), torch.from_numpy(next_sibling)


class HostDraftWorker(BaseSpecWorker):
    """Shared V2 target verification for candidates supplied by a host source.

    NgramVerifyInput is the existing generic host-tree layout. No local draft
    model or draft KV pool is allocated by this worker.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        super().__init__()
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = ps.tp_rank
        self.page_size = get_schedule().page_size
        self.draft_token_num = server_args.speculative_num_draft_tokens
        self.speculative_num_draft_tokens = self.draft_token_num
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.device = server_args.device
        self.grammar_tree_host = None

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _after_verify(self, batch: ScheduleBatch) -> None:
        pass

    def alloc_memory_pool(self, **kwargs) -> None:
        # The target memory pool does not exist yet when __init__ runs.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        # Candidates arrive from a host provider; no local draft model exists.
        return None

    def _init_preallocated_tensors(self) -> None:
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch) -> None:
        # Decode-only: extend goes through the plain target forward, and an
        # IDLE batch must keep its forward_mode instead of being rewritten to
        # TARGET_VERIFY below (relevant once DP attention support lands).
        if not batch.forward_mode.is_decode():
            return

        bs = len(batch.reqs)

        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # Pre-sync: only needs seq_lens_cpu, so build before the blocking
        # accept sync inside _prepare_draft_tokens.
        ones_masks = None
        if USE_FULL_MASK and not _is_cpu:
            ones_masks = [
                torch.ones(
                    (self.draft_token_num, batch.seq_lens_cpu[i]), device=self.device
                )
                for i in range(bs)
            ]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # Staged for the grammar bitmask, derived after the verify launch below.
        self.grammar_tree_host = (mask, req_drafts) if batch.has_grammar else None

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK and not _is_cpu:
            tree_mask = []
            mask = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            # TODO(siyuan): the for loop here leads to significant overhead in large batch size. Can be written into a kernel.
            for i in range(bs):
                req_mask = torch.cat(
                    (
                        ones_masks[i],
                        torch.from_numpy(mask[i]).to(
                            device=self.device, non_blocking=True
                        ),
                    ),
                    dim=1,
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + self.draft_token_num,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=self.device,
        )

        prepare_mamba_track_for_verify(batch)

        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=self.draft_token_num,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: Callable[[torch.Tensor], None] | None = None,
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input: NgramVerifyInput = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        start = time.perf_counter()
        if batch.forward_mode.is_target_verify():
            verify_start = None
            verify_end = None
            if self.server_args.decoupled_timing_sync:
                verify_start = torch.cuda.Event(enable_timing=True)
                verify_end = torch.cuda.Event(enable_timing=True)
                verify_start.record()
            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True
            )
            if self.server_args.decoupled_timing_sync:
                verify_end.record()
                verify_end.synchronize()
                verify_seconds = verify_start.elapsed_time(verify_end) / 1000
            else:
                verify_seconds = time.perf_counter() - start
            if (
                batch.spec_algorithm.is_decoupled()
                and self.server_args.decoupled_timing_sync
            ):
                logger.info(
                    "DECOUPLED target verify timing items=%d seconds=%.6f synchronized=%s",
                    bs,
                    verify_seconds,
                    self.server_args.decoupled_timing_sync,
                )

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = batch.spec_info
            grammar_mask = None
            if batch.has_grammar:
                # From the host tree rather than the device output: no readback to
                # wait on, and deriving here keeps it under the verify forward.
                mask, req_drafts = self.grammar_tree_host
                retrieve_next_token_cpu, retrieve_next_sibling_cpu = _derive_tree_links(
                    mask, bs, self.draft_token_num
                )
                grammar_mask = build_grammar_vocab_mask(
                    reqs=batch.reqs,
                    tree=GrammarTree.from_host(
                        retrieve_next_token_cpu,
                        retrieve_next_sibling_cpu,
                        torch.from_numpy(req_drafts).to(torch.int64).view(bs, -1),
                    ),
                    sampling_info=batch.sampling_info,
                    device=verify_input.retrieve_next_token.device,
                    # Host corpus lookup, so NGRAM stays synchronous: nothing pending.
                    barrier=None,
                )

            # Sample
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            (
                predict,
                accept_lens,
                accept_index,
            ) = eagle_sample(verify_input, batch, logits_output, grammar_mask)
            new_seq_lens = batch.seq_lens + accept_lens
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.draft_token_num,
            )
            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens

            # The KV mover expects drafts-only counts. NGRAM's
            # accept_lens includes the bonus token, matching scheduler output.
            num_correct_drafts_per_req = accept_lens - 1
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
            if batch.return_logprob:
                compute_spec_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index=accept_index,
                )

            if on_publish is not None:
                on_publish(new_seq_lens)

            self._after_verify(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            new_seq_lens = batch.seq_lens.clone()

            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict

            if on_publish is not None:
                on_publish(new_seq_lens)

        # Construct the next draft input
        next_draft_input = NgramVerifyInput(
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            # Consumed by the non-overlap V2 scheduler branch to advance
            # batch.seq_lens after the isolation restore; overlap mode relays
            # it via on_publish instead.
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
