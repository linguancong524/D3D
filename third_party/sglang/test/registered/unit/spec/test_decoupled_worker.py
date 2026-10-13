"""Host candidate validation and shared verify bookkeeping, without CUDA kernels."""

import sys
import unittest
from collections import defaultdict
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sglang.srt.speculative.decoupled.protocol import Proposal
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecoupledWorker(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # This suite never calls the native tree-index builder. Fail explicitly
        # if a future change starts doing so instead of silently emulating CUDA.
        native = ModuleType("sgl_kernel.speculative")
        native.reconstruct_indices_from_tree_mask = Mock(
            side_effect=AssertionError("this test does not execute CUDA kernels")
        )
        with patch.dict(sys.modules, {"sgl_kernel.speculative": native}):
            import sglang.srt.speculative.host_draft_worker as host
            from sglang.srt.speculative.decoupled.worker import DecoupledWorker
            from sglang.srt.speculative.ngram_worker import NGRAMWorker
        cls.worker_class = DecoupledWorker
        cls.ngram_class = NGRAMWorker
        cls.host_module = host

    def worker(self):
        from sglang.srt.server_args import ServerArgs

        worker = self.worker_class.__new__(self.worker_class)
        worker.server_args = ServerArgs(model_path="dummy")
        worker.shadow_local = False
        worker.draft_token_num = 4
        worker.speculative_num_draft_tokens = 4
        worker._vocab_size = 32
        worker._chain_mask = np.tril(np.ones((4, 4), dtype=np.bool_))
        worker.client = SimpleNamespace(
            metrics=defaultdict(float),
            sessions={},
            active_rids=Mock(return_value=()),
            lookup=Mock(return_value=None),
            publish=Mock(),
            finish=Mock(),
            close=Mock(),
        )
        worker.local_drafter = SimpleNamespace(
            states={}, propose_many=Mock(), release_request=Mock(), clear_prefix_states=Mock()
        )
        return worker

    def test_optional_miss_immediately_uses_local_required_proposal(self):
        worker = self.worker()
        worker.local_drafter.propose_many.return_value = {
            "a": Proposal((7, 8, 9, 10))
        }
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a", origin_input_ids=[1, 2], output_ids=[7])]
        )
        tokens, mask = worker._prepare_draft_tokens(batch)
        self.assertEqual(tokens.tolist(), [7, 8, 9, 10])
        worker.local_drafter.propose_many.assert_called_once()
        worker.client.publish.assert_called_once()
        self.assertTrue(
            np.array_equal(mask.reshape(4, 4), np.tril(np.ones((4, 4), dtype=bool)))
        )

    def test_departed_request_is_closed_and_new_request_retains_anchor(self):
        worker = self.worker()
        worker.client.sessions = {"aborted": object()}
        worker.client.active_rids.return_value = ("aborted",)
        worker.client.lookup.return_value = Proposal((7, 8, 9, 10))
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a", origin_input_ids=[1], output_ids=[7])]
        )
        tokens, _ = worker._prepare_draft_tokens(batch)
        worker.client.finish.assert_called_once_with(rid="aborted")
        self.assertEqual(tokens.tolist(), [7, 8, 9, 10])

    def test_shared_verify_publishes_bonus_inclusive_lengths_and_draft_only_kv_count(
        self,
    ):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        host = self.host_module

        self.assertIs(
            self.ngram_class.forward_batch_generation,
            host.HostDraftWorker.forward_batch_generation,
        )
        worker = self.worker()
        worker.device = "cpu"
        worker._prepare_for_speculative_decoding = Mock()
        worker._after_verify = Mock()
        worker.token_to_kv_pool_allocator = object()
        worker._target_worker = Mock()
        worker._target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.ones(3, 32)),
            can_run_cuda_graph=False,
        )
        batch = SimpleNamespace(
            reqs=[object(), object(), object()],
            spec_algorithm=SpeculativeAlgorithm.DECOUPLED,
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([10, 20, 30]),
            spec_info=object(),
            has_grammar=False,
            return_logprob=False,
        )
        lengths = torch.tensor([1, 2, 4], dtype=torch.int32)
        indices = torch.tensor([[0, -1, -1, -1], [4, 5, -1, -1], [8, 9, 10, 11]])
        publish = Mock()
        with (
            patch.object(host, "record_stream_for_v2_verify"),
            patch.object(host, "set_time_batch"),
            patch.object(host, "maybe_detect_nan"),
            patch.object(host, "maybe_detect_inf"),
            patch.object(host, "commit_mamba_states_after_verify") as commit,
            patch.object(host, "move_accept_tokens_to_target_kvcache") as move,
            patch.object(
                host, "eagle_sample", return_value=(torch.arange(12), lengths, indices)
            ),
            patch.object(
                host.torch,
                "get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: None),
            ),
        ):
            result = worker.forward_batch_generation(batch, on_publish=publish)
        self.assertEqual(result.new_seq_lens.tolist(), [11, 22, 34])
        self.assertEqual(result.accept_lens.tolist(), [1, 2, 4])
        self.assertEqual(move.call_args.args[2].tolist(), [0, 1, 3])
        self.assertEqual(commit.call_args.args[2].tolist(), [1, 2, 4])
        self.assertEqual(publish.call_args.args[0].tolist(), [11, 22, 34])
        self.assertEqual(result.next_draft_input.new_seq_lens.tolist(), [11, 22, 34])
        worker._after_verify.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
