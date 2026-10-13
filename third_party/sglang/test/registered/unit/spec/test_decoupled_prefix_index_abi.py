"""Regression for retained int32 KV mappings passed to an int64 pointer ABI."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.decoupled.transient_state import TransientDraftStateManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefixIndexABI(CustomTestCase):
    def test_retained_prefix_is_widened_before_pointer_based_allocation(self):
        """Int64 loads from an int32 table skip entries and corrupt retained KV."""
        table = torch.tensor([[0, 0, 0, 0], [31, 47, 53, 999]], dtype=torch.int32)
        manager = TransientDraftStateManager(
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            token_allocator=SimpleNamespace(page_size=1),
        )
        manager.register(
            session="retained", revision=0, proposal_id="", tokens=(10, 11, 12),
            pending_base_len=3, req_pool_idx=1, kv_allocated_len=3,
            pending_causal=True,
        )
        prefix = manager.prefix_indices(session="retained")
        self.assertEqual(prefix.dtype, torch.int64)
        self.assertEqual(prefix.tolist(), [31, 47, 53])
        # Prefix copies may be consumed after the request mapping is rewritten.
        table[1, :3] = 0
        self.assertEqual(prefix.tolist(), [31, 47, 53])


if __name__ == "__main__":
    unittest.main()
