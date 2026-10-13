"""Packed full-replay padding must preserve real masks and reject overflow."""

import unittest
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    PackedDllmGraphConfig,
    build_dllm_packed_token_buckets,
    dllm_packed_graph_batch_is_supported,
)
from sglang.srt.speculative.decoupled.local_drafter import plan_required_draft_layout

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLocalGraphLayout(CustomTestCase):
    def test_explicit_mask_does_not_require_target_dllm_config(self):
        """Colocated draft capture used to reject the non-diffusion Target config."""
        batch = self.batch(4)
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.dllm_config = None
        backend.indices_updater_prefill = Mock()
        backend.prefill_cuda_graph_metadata = {1: []}
        backend.use_sliding_window_kv_pool = False
        backend.init_forward_metadata_out_graph(batch)
        kwargs = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertTrue(torch.equal(
            kwargs["self_attention_custom_mask"], batch.dllm_self_attention_custom_mask
        ))
        self.assertTrue(backend.forward_metadata.uses_custom_mask)
        batch.dllm_self_attention_custom_mask = None
        with self.assertRaisesRegex(RuntimeError, "requires dLLM config"):
            backend.init_forward_metadata_out_graph(batch)

    def batch(self, num_prefix):
        layout = plan_required_draft_layout(
            committed_prefix=tuple(range(1, num_prefix + 2)),
            block_size=4,
            mask_token=99,
        )
        n = len(layout.input_tokens)
        return ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=torch.tensor(layout.input_tokens),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([n]),
            seq_lens_cpu=torch.tensor([n]),
            out_cache_loc=torch.arange(100, 100 + n),
            seq_lens_sum=n,
            positions=torch.tensor(layout.positions),
            extend_num_tokens=n,
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([n], dtype=torch.int32),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[n],
            dllm_packed_cuda_graph=True,
            dllm_self_attention_custom_mask=layout.attention_mask.flatten().to(
                torch.uint8
            ),
        )

    def test_padding_preserves_causal_prefix_and_real_kv_destinations(self):
        """A padded graph must not turn full replay into all-denoising attention."""
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.dllm_packed_graph_max_requests = 1
        runner.max_bs = 2
        for num_prefix in (1, 4, 7):
            with self.subTest(num_prefix=num_prefix):
                batch = self.batch(num_prefix)
                n = batch.input_ids.numel()
                mask = batch.dllm_self_attention_custom_mask.clone()
                padded = runner._prepare_dllm_packed_graph_batch(batch, 16)
                self.assertEqual(padded.extend_seq_lens_cpu, [n, 16 - n])
                self.assertTrue(torch.equal(padded.out_cache_loc[:n], batch.out_cache_loc))
                self.assertTrue(torch.equal(padded.positions[:n], batch.positions))
                self.assertTrue(torch.equal(
                    padded.dllm_self_attention_custom_mask[: n * n], mask
                ))
                self.assertEqual(
                    padded.dllm_self_attention_custom_mask.numel(),
                    n * n + (16 - n) ** 2,
                )
                self.assertEqual(batch.batch_size, 1)
                self.assertEqual(batch.extend_seq_lens_cpu, [n])
                self.assertTrue(torch.equal(batch.dllm_self_attention_custom_mask, mask))

    def test_capacity_and_mask_admission_fail_closed(self):
        batch = self.batch(4)

        def admitted():
            return dllm_packed_graph_batch_is_supported(
                batch, max_requests=1, token_buckets=[8]
            )

        self.assertTrue(admitted())
        batch.input_ids = torch.arange(9)
        self.assertFalse(admitted())
        batch.input_ids = torch.arange(8)
        batch.batch_size = 2
        self.assertFalse(admitted())
        batch.batch_size = 1
        batch.dllm_self_attention_custom_mask = None
        self.assertFalse(admitted())

    def test_bucket_boundary_and_invalid_capacity(self):
        buckets = build_dllm_packed_token_buckets(
            block_size=16, max_num_tokens=1024, token_step=128
        )
        self.assertEqual(DecodeCudaGraphRunner._pad_to_bucket(256, buckets), 256)
        self.assertEqual(DecodeCudaGraphRunner._pad_to_bucket(257, buckets), 384)
        for changes in ({"max_tokens": 15}, {"max_requests": 0}, {"token_step": 8}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PackedDllmGraphConfig(**(
                    {"block_size": 16, "max_requests": 1, "max_tokens": 1024}
                    | changes
                ))


if __name__ == "__main__":
    unittest.main()
