"""Transactional ownership tests for non-radix D3DV2 draft KV."""

import unittest

import torch
from sglang.srt.speculative.decoupled.protocol import Commit
from sglang.srt.speculative.decoupled.transient_state import (
    TransientDraftStateManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeReqPool:
    def __init__(self):
        self.req_to_token = torch.arange(64, dtype=torch.int32).reshape(2, 32)
        self.free_slots = []


class FakeAllocator:
    def __init__(self, page_size=1):
        self.page_size = page_size
        self.freed = []

    def free(self, indices):
        self.freed.append(tuple(indices.tolist()))


def manager(page_size=1):
    req_pool = FakeReqPool()
    allocator = FakeAllocator(page_size)
    return (
        TransientDraftStateManager(
            req_to_token_pool=req_pool, token_allocator=allocator
        ),
        req_pool,
        allocator,
    )


class TestTransientDraftState(CustomTestCase):
    def test_publish_keeps_pending_and_frees_candidate_scratch(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=2,
        )
        states.begin(session="a", revision=0, transaction="txn")
        states.publish_pending(
            session="a",
            transaction="txn",
            proposal_id="p0",
            tokens=(1, 2, 7, 8, 9, 10),
            pending_base_len=2,
            allocated_len=14,
        )
        state = states.states["a"]
        self.assertEqual(state.tokens, (1, 2, 7, 8, 9, 10))
        self.assertEqual(state.kv_allocated_len, 6)
        self.assertIsNone(state.inflight_transaction)
        self.assertEqual(allocator.freed, [tuple(range(38, 46))])

    def test_partial_commit_keeps_causal_pending_lcp(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2, 7, 8, 9, 10),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=6,
            pending_causal=True,
        )
        retained = states.commit(
            Commit(
                session="a",
                revision=1,
                proposal_id="p0",
                committed_prefix=(1, 2, 7, 8, 99),
            )
        )
        self.assertEqual(retained, 4)
        self.assertEqual(states.states["a"].tokens, (1, 2, 7, 8))
        self.assertTrue(states.states["a"].pending_causal)
        self.assertEqual(allocator.freed, [(36, 37)])

    def test_partial_commit_crops_block_causal_pending_to_immutable_base(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2, 7, 8, 9, 10),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=6,
        )
        retained = states.commit(
            Commit(
                session="a",
                revision=1,
                proposal_id="p0",
                committed_prefix=(1, 2, 7, 8, 99),
            )
        )
        self.assertEqual(retained, 2)
        self.assertEqual(states.states["a"].tokens, (1, 2))
        self.assertEqual(states.states["a"].revision, 1)
        self.assertEqual(allocator.freed, [(34, 35, 36, 37)])

    def test_commit_without_materialized_optional_proposal_keeps_safe_base(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=1,
            proposal_id="",
            tokens=(1, 2, 7),
            pending_base_len=3,
            req_pool_idx=1,
            kv_allocated_len=3,
        )
        retained = states.commit(
            Commit(
                session="a",
                revision=2,
                proposal_id="required-proposal",
                committed_prefix=(1, 2, 7, 8, 9),
            )
        )
        self.assertEqual(retained, 3)
        self.assertEqual(states.states["a"].revision, 2)
        self.assertEqual(allocator.freed, [])

    def test_stale_commit_and_transaction_leave_published_state_unchanged(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=3,
            proposal_id="p3",
            tokens=(1, 2, 7),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=3,
        )
        with self.assertRaisesRegex(ValueError, "proposal does not match"):
            states.commit(
                Commit(
                    session="a",
                    revision=4,
                    proposal_id="stale",
                    committed_prefix=(1, 2, 7, 9),
                )
            )
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            states.begin(session="a", revision=2, transaction="old")
        self.assertEqual(states.states["a"].tokens, (1, 2, 7))
        self.assertEqual(allocator.freed, [])

    def test_conflicting_same_revision_proposal_cannot_begin_transaction(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=2,
        )
        with self.assertRaisesRegex(ValueError, "proposal does not match"):
            states.begin(
                session="a",
                revision=0,
                transaction="conflict",
                proposal_id="different",
            )
        self.assertIsNone(states.states["a"].inflight_transaction)
        self.assertEqual(allocator.freed, [])

    def test_failed_forward_rolls_back_only_transaction_allocation(self):
        states, _, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=2,
        )
        states.begin(session="a", revision=0, transaction="txn")
        states.abort(session="a", transaction="txn", allocated_len=10)
        self.assertEqual(states.states["a"].kv_allocated_len, 2)
        self.assertEqual(allocator.freed, [tuple(range(34, 42))])

    def test_release_all_returns_every_owned_slot(self):
        states, req_pool, allocator = manager()
        for session, slot in (("a", 0), ("b", 1)):
            states.register(
                session=session,
                revision=0,
                proposal_id="",
                tokens=(1, 2),
                pending_base_len=2,
                req_pool_idx=slot,
                kv_allocated_len=2,
            )
        states.release_all()
        self.assertFalse(states.states)
        self.assertEqual(req_pool.free_slots, [0, 1])
        self.assertEqual(allocator.freed, [(0, 1), (32, 33)])

    def test_release_is_idempotent_and_returns_req_slot(self):
        states, req_pool, allocator = manager()
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2, 7),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=3,
        )
        states.release(session="a")
        states.release(session="a")
        self.assertEqual(req_pool.free_slots, [1])
        self.assertEqual(allocator.freed, [(32, 33, 34)])

    def test_page_aligned_crop_does_not_free_a_partially_retained_page(self):
        states, _, allocator = manager(page_size=4)
        states.register(
            session="a",
            revision=0,
            proposal_id="p0",
            tokens=(1, 2, 7, 8, 9, 10),
            pending_base_len=2,
            req_pool_idx=1,
            kv_allocated_len=12,
        )
        states.commit(
            Commit(
                session="a",
                revision=1,
                proposal_id="p0",
                committed_prefix=(1, 2, 7, 99),
            )
        )
        self.assertEqual(states.states["a"].kv_allocated_len, 2)
        self.assertEqual(allocator.freed, [tuple(range(36, 44))])


if __name__ == "__main__":
    unittest.main()
