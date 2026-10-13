# pyright: reportMissingImports=false
"""CPU regressions for the anchor-conditioned native dLLM draft block."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.algorithm.one_step import OneStep
from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.layers.attention.flashinfer_backend import (
    _build_dllm_custom_mask,
    _resolve_dllm_custom_mask,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_dllm_packed_token_buckets,
    dllm_graph_batch_is_supported,
    dllm_packed_graph_batch_is_supported,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def dllm_config(*, fdfo: bool = True):
    return SimpleNamespace(
        block_size=4,
        mask_id=31,
        first_done_first_out_mode=fdfo,
        anchor_conditioned_block=True,
    )


class FakeModelRunner:
    def __init__(self, predicted_ids):
        self.predicted_ids = predicted_ids
        self.forward_ct = 0

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.forward_ct += 1
        vocab_size = 32
        logits = torch.full(
            (len(self.predicted_ids), vocab_size), -10.0, dtype=torch.float32
        )
        logits[torch.arange(len(self.predicted_ids)), self.predicted_ids] = 10.0
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits),
            can_run_graph=False,
        )


def variable_forward_batch():
    # block_size=4. First request carries a two-token radix gap plus
    # [anchor, mask, mask, mask]; the second starts directly at its anchor.
    return SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([1, 2, 3, 31, 31, 31, 4, 31, 31, 31], dtype=torch.int64),
        extend_seq_lens_cpu=[6, 4],
        extend_seq_lens=None,
    )


class FakeReq(ReqDllmMixin):
    pass


class TestAnchorConditionedBlock(CustomTestCase):
    def test_packed_graph_uses_geometric_token_buckets(self):
        self.assertEqual(
            build_dllm_packed_token_buckets(block_size=16, max_num_tokens=1000),
            [32, 64, 128, 256, 512, 1000],
        )
        self.assertEqual(
            build_dllm_packed_token_buckets(
                block_size=16, max_num_tokens=720, token_step=240
            ),
            [32, 64, 128, 240, 256, 480, 512, 720],
        )
        with self.assertRaisesRegex(ValueError, "below one block"):
            build_dllm_packed_token_buckets(block_size=16, max_num_tokens=15)

    def test_packed_graph_admission_is_bounded_by_requests_and_tokens(self):
        batch = SimpleNamespace(
            dllm_packed_cuda_graph=True,
            dllm_self_attention_custom_mask=torch.ones(16, dtype=torch.uint8),
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[2, 2],
            batch_size=2,
            input_ids=torch.zeros(4, dtype=torch.int64),
        )
        self.assertTrue(
            dllm_packed_graph_batch_is_supported(
                batch, max_requests=2, token_buckets=[4, 8]
            )
        )
        batch.batch_size = 3
        self.assertFalse(
            dllm_packed_graph_batch_is_supported(
                batch, max_requests=2, token_buckets=[4, 8]
            )
        )
        batch.batch_size = 2
        batch.input_ids = torch.zeros(9, dtype=torch.int64)
        self.assertFalse(
            dllm_packed_graph_batch_is_supported(
                batch, max_requests=2, token_buckets=[4, 8]
            )
        )

    def test_dllm_graph_admits_only_fixed_width_denoising_batches(self):
        batch = SimpleNamespace(
            batch_size=2,
            input_ids=torch.zeros(8, dtype=torch.int64),
            dllm_is_denoising_cpu=[True, True],
            extend_seq_lens_cpu=[4, 4],
        )
        self.assertTrue(dllm_graph_batch_is_supported(batch, 4))

        batch.dllm_is_denoising_cpu = [True, False]
        self.assertFalse(dllm_graph_batch_is_supported(batch, 4))
        batch.dllm_is_denoising_cpu = [True, True]
        batch.extend_seq_lens_cpu = [4, 3]
        self.assertFalse(dllm_graph_batch_is_supported(batch, 4))
        batch.extend_seq_lens_cpu = [4, 4]
        batch.input_ids = torch.zeros(7, dtype=torch.int64)
        self.assertFalse(dllm_graph_batch_is_supported(batch, 4))

    def test_dllm_attention_mask_preserves_request_phases(self):
        mask = _build_dllm_custom_mask(
            extend_prefix_lens_cpu=[3, 2],
            extend_seq_lens_cpu=[3, 4],
            dllm_is_denoising_cpu=[False, True],
            device=torch.device("cpu"),
        )

        prefix_mask = mask[:18].view(3, 6)
        denoising_mask = mask[18:].view(4, 6)
        self.assertEqual(
            prefix_mask.tolist(),
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
        )
        self.assertTrue(bool(denoising_mask.all()))

    def test_explicit_packed_mask_overrides_phase_derived_visibility(self):
        explicit = torch.tensor(
            [1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.bool
        )
        batch = SimpleNamespace(
            dllm_self_attention_custom_mask=explicit,
            dllm_is_denoising_cpu=[True],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[2],
            input_ids=torch.tensor([31, 31]),
        )
        # A 2-token, zero-prefix rectangle has four entries, so first prove the
        # planner rejects an accidentally concatenated sibling mask.
        with self.assertRaisesRegex(ValueError, "8 elements; expected 4"):
            _resolve_dllm_custom_mask(batch)
        batch.dllm_self_attention_custom_mask = explicit[:4]
        self.assertEqual(
            _resolve_dllm_custom_mask(batch).tolist(), [1, 0, 1, 1]
        )

    def test_explicit_packed_mask_must_share_the_input_device(self):
        batch = SimpleNamespace(
            dllm_self_attention_custom_mask=torch.empty(0, device="meta"),
            dllm_is_denoising_cpu=None,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[0],
            input_ids=torch.empty(0, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "share the input device"):
            _resolve_dllm_custom_mask(batch)

    def test_dllm_attention_mask_rejects_mismatched_metadata(self):
        with self.assertRaisesRegex(ValueError, "metadata lengths must match"):
            _build_dllm_custom_mask(
                extend_prefix_lens_cpu=[3],
                extend_seq_lens_cpu=[4, 4],
                dllm_is_denoising_cpu=[True],
                device=torch.device("cpu"),
            )

    def test_fill_places_existing_anchor_before_three_masks(self):
        req = FakeReq()
        req.origin_input_ids = array("q", [10, 11, 12])
        req.output_ids = array("q", [13])
        req.prefix_indices = array("q")
        req.full_untruncated_fill_ids = array("q")
        req.dllm_initialized = False
        req.dllm_incomplete_ids = array("q")
        req.dllm_block_offset = 0
        req.dllm_config = dllm_config()

        req._init_fill_ids_for_dllm()
        self.assertEqual(req.full_untruncated_fill_ids.tolist(), [10, 11, 12])

        req.prefix_indices = array("q", [10, 11, 12])
        req._init_fill_ids_for_dllm()
        self.assertEqual(
            req.full_untruncated_fill_ids.tolist(), [10, 11, 12, 13, 31, 31, 31]
        )
        self.assertEqual(req.full_untruncated_fill_ids[-4:].tolist(), [13, 31, 31, 31])

    def test_phase_prefills_every_committed_token_except_the_anchor(self):
        req = FakeReq()
        req.dllm_config = dllm_config()
        req.dllm_incomplete_ids = array("q")
        req.origin_input_ids = array("q", [10, 11, 12])
        req.output_ids = array("q", [13])
        req.full_untruncated_fill_ids = array("q", [10, 11, 12, 13, 31, 31, 31])
        req.dllm_phase = DllmReqPhase.INCOMING_PREFILL

        req.prefix_indices = array("q", [10, 11])
        req.determine_dllm_phase()
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_PREFILL)

        req.prefix_indices = array("q", [10, 11, 12])
        req.determine_dllm_phase()
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)

    def test_radix_match_never_consumes_the_anchor(self):
        req = Req.__new__(Req)
        req.dllm_config = dllm_config()
        req.origin_input_ids = array("q", [10, 11, 12])
        req.output_ids = array("q", [13])
        req.return_logprob = False
        req.logprob_start_len = -1

        # Four committed tokens plus three masks: the generic cap would be 6,
        # while anchor-conditioned matching must stop before token 13 at index 3.
        self.assertEqual(req._compute_max_prefix_len(input_len=7), 3)

        req.dllm_config.anchor_conditioned_block = False
        self.assertEqual(req._compute_max_prefix_len(input_len=8), 7)

    def test_anchor_conditioned_radix_match_accepts_token_granular_prefix(self):
        req = FakeReq()
        req.prefix_indices = array("q", [10, 11, 12])
        req.dllm_block_offset = 0
        req.dllm_config = dllm_config()

        req._update_block_offset_for_dllm()
        self.assertEqual(req.dllm_block_offset, 3)

        req.dllm_config.anchor_conditioned_block = False
        with self.assertRaisesRegex(AssertionError, "Unexpected prefix len: 3"):
            req._update_block_offset_for_dllm()

    def test_non_one_step_config_keeps_legacy_all_mask_block(self):
        req = FakeReq()
        req.origin_input_ids = array("q", [10, 11, 12])
        req.output_ids = array("q", [13])
        req.prefix_indices = array("q")
        req.full_untruncated_fill_ids = array("q")
        req.dllm_initialized = False
        req.dllm_incomplete_ids = array("q")
        req.dllm_block_offset = 0
        req.dllm_config = SimpleNamespace(
            block_size=4,
            mask_id=31,
            anchor_conditioned_block=False,
        )

        req._init_fill_ids_for_dllm()

        self.assertEqual(
            req.full_untruncated_fill_ids.tolist(),
            [10, 11, 12, 13, 31, 31, 31, 31],
        )

    def test_fdfo_slices_generated_tails_using_pre_step_masks(self):
        batch = variable_forward_batch()
        runner = FakeModelRunner(list(range(10, 20)))
        algorithm = OneStep(dllm_config(fdfo=True))

        _, next_ids, accept_lens, states, can_run_graph = algorithm.run(runner, batch)

        self.assertEqual(next_ids, [[13, 14, 15], [17, 18, 19]])
        self.assertEqual(accept_lens, [3, 3])
        self.assertEqual(states, [None, None])
        self.assertFalse(can_run_graph)
        self.assertEqual(runner.forward_ct, 1)

    def test_sync_slices_each_variable_request_segment(self):
        batch = variable_forward_batch()
        runner = FakeModelRunner(list(range(10, 20)))
        algorithm = OneStep(dllm_config(fdfo=False))

        _, next_ids, accept_lens, states, _ = algorithm.run(runner, batch)

        self.assertEqual(
            [tokens.tolist() for tokens in next_ids], [[13, 14, 15], [17, 18, 19]]
        )
        self.assertIsNone(accept_lens)
        self.assertIsNone(states)

    def test_legacy_fdfo_keeps_complete_block_through_commit_round(self):
        config = SimpleNamespace(
            block_size=4,
            mask_id=31,
            first_done_first_out_mode=True,
            anchor_conditioned_block=False,
            algorithm_config={"threshold": 0.5},
        )
        algorithm = LowConfidence(config)
        batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.tensor([31, 31, 31, 31], dtype=torch.int64),
            extend_seq_lens_cpu=[4],
            extend_seq_lens=None,
        )
        runner = FakeModelRunner([10, 11, 12, 13])

        _, first_ids, first_accept, states, _ = algorithm.run(runner, batch, [None])
        _, final_ids, final_accept, _, _ = algorithm.run(runner, batch, states)

        self.assertEqual(first_ids, [[10, 11, 12, 13]])
        self.assertEqual(first_accept, [0])
        self.assertEqual(final_ids, [[10, 11, 12, 13]])
        self.assertEqual(final_accept, [4])

    def test_top_logprobs_use_only_each_segment_mask_tail(self):
        vocab_size = 16
        logits = torch.full((10, vocab_size), -10.0)
        top_ids = list(range(10))
        logits[torch.arange(10), top_ids] = 10.0
        reqs = [
            SimpleNamespace(
                return_logprob=True,
                logprob=SimpleNamespace(
                    top_logprobs_num=2,
                    output_top_logprobs_val=[],
                    output_top_logprobs_idx=[],
                ),
            )
            for _ in range(2)
        ]
        scheduler = SimpleNamespace(dllm_config=dllm_config())
        batch = SimpleNamespace(
            return_logprob=True,
            extend_lens=[6, 4],
            reqs=reqs,
        )
        result = SimpleNamespace(logits_output=SimpleNamespace(full_logits=logits))

        SchedulerDllmMixin._populate_dllm_output_top_logprobs(scheduler, batch, result)

        self.assertEqual(
            [row[0] for row in reqs[0].logprob.output_top_logprobs_idx],
            [3, 4, 5],
        )
        self.assertEqual(
            [row[0] for row in reqs[1].logprob.output_top_logprobs_idx],
            [7, 8, 9],
        )

    def test_scheduler_prefills_to_the_anchor_then_admits_one_complete_block(self):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder._init_dllm_meta(
            SimpleNamespace(
                block_size=4,
                max_running_requests=3,
                anchor_conditioned_block=True,
            )
        )
        req = SimpleNamespace(
            origin_input_ids=array("q", [10, 11, 12]),
            output_ids=array("q", [13]),
            full_untruncated_fill_ids=array("q", [10, 11, 12]),
            is_dllm_prefill=lambda: True,
        )

        self.assertEqual(adder.dllm_extend_max, 7)
        self.assertEqual(adder.rem_dllm_tokens, 21)
        self.assertEqual(adder._dllm_extend_len(req, prefix_len=0, rem_tokens=7), 3)

        req.full_untruncated_fill_ids = array("q", [10, 11, 12, 13, 31, 31, 31])
        req.is_dllm_prefill = lambda: False
        self.assertEqual(adder._dllm_extend_len(req, prefix_len=3, rem_tokens=4), 4)
        self.assertEqual(adder._dllm_extend_len(req, prefix_len=3, rem_tokens=3), 0)

    def test_scheduler_never_restores_exhausted_real_token_budget(self):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.rem_dllm_tokens = 7
        adder.dllm_extend_max = 7
        adder.is_all_swa = False
        adder.is_hybrid_swa = False
        adder.is_hybrid_ssm_cache = False
        adder.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 7)
        adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)

        adder.rem_total_token_offset = 7
        adder.cur_rem_token_offset = 0
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 8
        self.assertEqual(adder._get_dllm_remain_tokens(), 0)

        # A later retry can proceed only after both real budgets recover.
        adder.cur_rem_token_offset = 2
        self.assertEqual(adder._get_dllm_remain_tokens(), 5)


if __name__ == "__main__":
    unittest.main()
