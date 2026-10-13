"""SGLang target half: host proposals, standard target-only tree verification."""

from __future__ import annotations

import atexit
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sglang.srt.speculative.host_draft_worker import HostDraftWorker

from .client import DraftClient
from .local_drafter import LocalSDARDrafter
from .protocol import Proposal

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DecoupledWorker(HostDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        self.local_drafter = LocalSDARDrafter(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.client = DraftClient(
            endpoint=server_args.decoupled_draft_endpoint,
            rank=gpu_id,
            fingerprint=self.local_drafter.fingerprint,
            block_size=self.draft_token_num,
            num_anchors=server_args.decoupled_num_anchors,
            accept_counts=(
                ()
                if server_args.decoupled_accept_counts is None
                else tuple(server_args.decoupled_accept_counts)
            ),
            timeout_seconds=server_args.decoupled_draft_timeout,
            lookup_wait_seconds=server_args.decoupled_lookup_wait,
            wait_for_covered_result=server_args.decoupled_wait_for_covered_result,
            poll_interval_seconds=server_args.decoupled_poll_interval,
        )
        self._vocab_size = self.model_runner.model_config.vocab_size
        self.client.metrics.update(
            local_required_forward_ct=0,
            local_required_cuda_graph_ct=0,
            local_required_eager_ct=0,
            local_graph_check_ct=0,
            local_graph_check_tokens_exact_ct=0,
            local_graph_check_topk_exact_ct=0,
            local_graph_check_logit_max_abs=0,
            local_graph_check_flip_regret_max=0,
            local_graph_check_nonfinite_ct=0,
            local_required_req_ct=0,
            local_required_seconds=0,
            local_required_build_seconds=0,
            local_required_forward_dispatch_seconds=0,
            local_required_postprocess_seconds=0,
            local_required_release_seconds=0,
            local_required_thread_cpu_seconds=0,
            local_required_input_tokens=0,
            remote_optional_skip_local_ct=0,
            remote_optional_full_accept_skip_local_ct=0,
            draft_prepare_seconds=0,
        )
        self._chain_mask = np.tril(
            np.ones((self.draft_token_num, self.draft_token_num), dtype=np.bool_)
        )
        self.shadow_local = server_args.decoupled_shadow_local
        self.shadow_metrics = {
            "remote_local_compare_ct": 0,
            "remote_local_exact_ct": 0,
            "remote_local_match_token_ct": 0,
            "remote_local_token_ct": 0,
        }
        self._closed = False
        self._atexit_callback = self.close
        atexit.register(self._atexit_callback)

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        prepare_start = time.perf_counter()
        live = {req.rid for req in batch.reqs}
        for rid in set(self.client.active_rids()) - live:
            self.note_request_finished(rid=rid, natural_stop=False)
        selected: dict[str, Proposal] = {}
        misses = []
        for req in batch.reqs:
            prompt = tuple(req.origin_input_ids)
            outputs = tuple(req.output_ids)
            if not outputs:
                raise RuntimeError("decoupled verify requires a prefill bonus token")
            full_hits = self.client.metrics["remote_optional_full_accept_hit_ct"]
            proposal = self.client.lookup(rid=req.rid, prompt=prompt, outputs=outputs)
            if self._valid(proposal, anchor=outputs[-1]):
                selected[req.rid] = proposal
                self.client.metrics["remote_optional_skip_local_ct"] += 1
                self.client.metrics["remote_optional_full_accept_skip_local_ct"] += (
                    self.client.metrics["remote_optional_full_accept_hit_ct"] - full_hits
                )
            else:
                misses.append(req)
        if self.shadow_local and selected:
            shadow_reqs = [req for req in batch.reqs if req.rid in selected]
            shadow = self.local_drafter.propose_many(reqs=shadow_reqs)
            for rid, remote in selected.items():
                local = shadow[rid]
                self.shadow_metrics["remote_local_compare_ct"] += 1
                self.shadow_metrics["remote_local_token_ct"] += len(remote.tokens)
                num_match = sum(
                    left == right for left, right in zip(remote.tokens, local.tokens)
                )
                self.shadow_metrics["remote_local_match_token_ct"] += num_match
                if remote.tokens == local.tokens:
                    self.shadow_metrics["remote_local_exact_ct"] += 1
                else:
                    first = next(
                        index
                        for index, (left, right) in enumerate(
                            zip(remote.tokens, local.tokens)
                        )
                        if left != right
                    )
                    key = f"remote_local_first_diff_{first}_ct"
                    self.shadow_metrics[key] = self.shadow_metrics.get(key, 0) + 1
            logger.info("DECOUPLED shadow proposal metrics: %s", self.shadow_metrics)
        if misses:
            start = time.perf_counter()
            selected.update(self.local_drafter.propose_many(
                reqs=misses, phase_metrics=self.client.metrics,
            ))
            elapsed = time.perf_counter() - start
            self.client.metrics["local_required_forward_ct"] += 1
            self.client.metrics["local_required_req_ct"] += len(misses)
            self.client.metrics["local_required_seconds"] += elapsed
            bucket = next((bound for bound in (16, 24, 32, 48, 64, 128)
                           if elapsed * 1000 <= bound), "inf")
            key = f"local_required_wall_bucket_{bucket}ms_ct"
            self.client.metrics[key] = self.client.metrics.get(key, 0) + 1
            if self.server_args.decoupled_timing_sync:
                logger.info(
                    "DECOUPLED local fallback timing items=%d seconds=%.6f client_metrics=%s",
                    len(misses),
                    time.perf_counter() - start,
                    self.client.metrics,
                )

        rows = []
        for req in batch.reqs:
            proposal = selected.get(req.rid)
            outputs = tuple(req.output_ids)
            if not self._valid(proposal, anchor=outputs[-1]):
                raise RuntimeError("local SDAR returned an invalid required proposal")
            self.client.publish(
                rid=req.rid,
                prompt=tuple(req.origin_input_ids),
                outputs=outputs,
                proposal=proposal,
            )
            rows.append(proposal.tokens)
        drafts = np.asarray(rows, dtype=np.int64).reshape(-1)
        mask = np.tile(self._chain_mask, (len(rows), 1, 1)).reshape(-1)
        self.client.metrics["draft_prepare_seconds"] += time.perf_counter() - prepare_start
        return drafts, mask

    def _valid(self, proposal: Proposal | None, *, anchor: int) -> bool:
        return bool(
            proposal is not None
            and len(proposal.tokens) == self.draft_token_num
            and proposal.tokens[0] == anchor
            and all(0 <= token < self._vocab_size for token in proposal.tokens)
        )

    @property
    def draft_worker(self):
        return self.local_drafter.worker

    def _draft_model_runners(self):
        # Local SDAR is driven by a plain TpModelWorker, like DFLASH, rather
        # than an EagleDraftWorkerBase exposing ``draft_runners``.
        return (self.local_drafter.model_runner,)

    def alloc_combined_memory_pools(self) -> None:
        self.local_drafter.allocate_with_target(self.target_worker)

    def alloc_memory_pool(self, **kwargs) -> None:
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def init_attention_backends(self) -> None:
        self.local_drafter.init_attention_backends()

    def init_cuda_graphs(self) -> None:
        self.local_drafter.init_cuda_graphs()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self.local_drafter.release_request(rid=rid)
        self.client.finish(rid=rid)
        if self.server_args.decoupled_timing_sync:
            logger.info(
                "DECOUPLED request finish client_metrics=%s", self.client.metrics
            )

    def clear_cache_pool(self) -> None:
        self.local_drafter.clear_prefix_states()
        for rid in self.client.active_rids():
            self.client.finish(rid=rid)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.client.close()
        finally:
            atexit.unregister(self._atexit_callback)

    def update_weights_from_disk(self, recv_req) -> tuple[bool, str]:
        # This request names the Target checkpoint. The independently configured
        # SDAR checkpoint must never be replaced with Target weights; arbitrary
        # local proposals remain exact because Target verification is authoritative.
        self.clear_cache_pool()
        return True, "Target updated; local SDAR weights are unchanged."

    def update_weights_from_ipc(self, recv_req) -> tuple[bool, str]:
        self.clear_cache_pool()
        return True, "Target updated; local SDAR weights are unchanged."

    def update_weights_from_tensor(self, recv_req) -> tuple[bool, str]:
        self.clear_cache_pool()
        return self.target_worker.update_weights_from_tensor(recv_req)
