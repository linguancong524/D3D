from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.host_draft_worker import HostDraftWorker

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class NGRAMWorker(HostDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)

        self.max_trie_depth = server_args.speculative_ngram_max_trie_depth
        self.adaptive_controller = None
        self._prev_decode_rids: set = set()

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            corpus_path = server_args.speculative_ngram_external_corpus_path
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    def clear_cache_pool(self):
        self.ngram_corpus.reset()
        self._prev_decode_rids = set()

    def update_weights_from_tensor(self, recv_req):
        # NGRAM has no draft weights of its own — the n-gram corpus is a CPU
        # lookup structure built from request token streams — and its
        # `model_runner` is shared with the target worker. The scheduler
        # mixin dispatches via `self.draft_worker or self.tp_worker`, so
        # without this method any caller of `update_weights_from_tensor`
        # under `--speculative-algorithm NGRAM` raises AttributeError.
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        # Signature must match BaseSpecWorker.on_verify_complete_cpu; the
        # result processor calls it with batch_size as a keyword argument.
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        stride = self.draft_token_num

        # Overlap mode processes results one iteration behind, so the last
        # round's accepted tokens are not yet in req.output_ids and must be
        # spliced in from spec_info. Sync mode and grammar batches process
        # results before the next draft prep, so output_ids is already
        # complete and splicing would duplicate the tail.
        use_prev_tokens = self.enable_overlap and not batch.grammar_needs_sync()

        # Accept-independent prep, hoisted above the blocking .cpu() below.
        req_ids = [req.rid for req in batch.reqs]
        # Only the last max_trie_depth tokens can match; the ids are
        # array.array, so list() the tails for list concat below.
        input_tails = [
            list(req.origin_input_ids[-self.max_trie_depth :]) for req in batch.reqs
        ]
        output_tails = [
            list(req.output_ids[-self.max_trie_depth :]) for req in batch.reqs
        ]
        base_lens = [
            len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs
        ]

        if use_prev_tokens:
            prev_token_ids, prev_accept_lens = (
                batch.spec_info.accept_tokens,
                batch.spec_info.accept_lens,
            )
            if not prev_token_ids.is_cpu:
                prev_token_ids = prev_token_ids.cpu()
                prev_accept_lens = prev_accept_lens.cpu()
            # Worker-level staging: written here at draft prep, consumed by
            # _update_ngram_corpus after verify within the same forward call.
            self.prev_token_ids = prev_token_ids.tolist()
            self.prev_accept_lens = prev_accept_lens.tolist()
            assert bs == len(self.prev_accept_lens)
        else:
            # _update_ngram_corpus still reads the staging; fill it empty.
            self.prev_token_ids = []
            self.prev_accept_lens = [0] * bs

        self.ngram_corpus.synchronize()
        batch_tokens = []
        total_lens = []
        for i in range(bs):
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            check_token = self._efficient_concat_last_n(
                input_tails[i],
                output_tails[i] + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(check_token)
            total_lens.append(base_lens[i] + len(prev_tokens))
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert total_draft_token_num == bs * self.draft_token_num, (
            f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        )
        return req_drafts, mask

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        batch_tokens = []
        i, stride = 0, self.draft_token_num
        # Same splice condition as _prepare_draft_tokens: only overlap mode
        # has accepted tokens missing from req.output_ids.
        use_prev_tokens = self.enable_overlap and not batch.grammar_needs_sync()
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            put_ids = self._efficient_concat_last_n(
                list(req.origin_input_ids[-self.max_trie_depth :]),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(put_ids)
            i += 1
        self.ngram_corpus.batch_put(batch_tokens)

    def _after_verify(self, batch):
        self._update_ngram_corpus(batch)
        cur_rids = {req.rid for req in batch.reqs}
        departed_rids = self._prev_decode_rids - cur_rids
        if departed_rids:
            self.ngram_corpus.erase_match_state(list(departed_rids))
        self._prev_decode_rids = cur_rids
