"""Target-local required layout and remote-down fallback regressions."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.decoupled.client import DraftClient
from sglang.srt.speculative.decoupled.local_drafter import (
    LocalContinuousPrefixState,
    LocalSDARDrafter,
    plan_required_draft_layout,
    proposal_from_logits,
    validate_vocabulary_compatibility,
)
from sglang.srt.speculative.decoupled.packed_layout import (
    group_branches_by_accept_count,
)
from sglang.srt.speculative.decoupled.protocol import (
    Branch,
    Outcome,
    Proposal,
    Reply,
    classify_outcome_miss,
    outcome_branch_is_covered,
    proposal_identity,
)


class TestLocalRequiredLayout(unittest.TestCase):
    def test_groups_three_sibling_candidates_without_crossing_positions(self):
        branches = tuple(
            Branch(count, bonus)
            for count in (1, 2)
            for bonus in (10, 11, 12)
        ) + (Branch(4, None),)
        groups = group_branches_by_accept_count(branches)
        self.assertEqual([len(group) for group in groups], [3, 3, 1])
        self.assertEqual(
            [[branch.num_accept_tokens for branch in group] for group in groups],
            [[1, 1, 1], [2, 2, 2], [4]],
        )

    def test_full_replay_is_causal_before_the_bidirectional_block(self):
        layout = plan_required_draft_layout(
            committed_prefix=(1, 2, 3, 4, 9),
            block_size=4,
            mask_token=99,
        )
        self.assertEqual(layout.input_tokens, (1, 2, 3, 4, 9, 99, 99, 99))
        self.assertEqual(layout.positions, tuple(range(8)))
        self.assertEqual(layout.replay_len, 4)
        mask = layout.attention_mask
        self.assertEqual(tuple(mask.shape), (8, 8))
        for row in range(4):
            self.assertTrue(mask[row, : row + 1].all())
            self.assertFalse(mask[row, row + 1 :].any())
        self.assertTrue(mask[4:].all())

    def test_layout_requires_an_anchor_and_mask_position(self):
        with self.assertRaisesRegex(ValueError, "Target bonus"):
            plan_required_draft_layout(
                committed_prefix=(),
                block_size=4,
                mask_token=99,
            )
        with self.assertRaisesRegex(ValueError, "anchor and MASK"):
            plan_required_draft_layout(
                committed_prefix=(9,), block_size=1, mask_token=99
            )

    def test_proposal_keeps_target_anchor_and_aligns_topk(self):
        logits = torch.zeros((4, 8))
        logits[1, 3] = 5
        logits[2, 4] = 5
        logits[3, 5] = 5
        proposal = proposal_from_logits(
            anchor=7,
            full_logits=logits,
            replay_len=0,
            block_size=4,
            topk=2,
        )
        self.assertEqual(proposal.tokens, (7, 3, 4, 5))
        self.assertEqual(proposal.top_tokens[0], (7,))
        self.assertEqual(len(proposal.top_tokens), 4)


class FakeLocalReqPool:
    def __init__(self):
        self.req_to_token = torch.arange(128, dtype=torch.int32).reshape(4, 32)
        self.free_slots = [1, 2, 3]


class FakeLocalAllocator:
    page_size = 1

    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(tuple(indices.tolist()))


class FakeLocalBatch:
    def __init__(self, reqs, req_pool):
        self.reqs = reqs
        self.req_pool = req_pool
        self.device = torch.device("cpu")
        self.prefill_input_ids_cpu = None

    def prepare_for_extend(self):
        for req in self.reqs:
            if req.req_pool_idx is None:
                req.req_pool_idx = self.req_pool.free_slots.pop()
            req.kv = SimpleNamespace(kv_allocated_len=req.extend_range.end)


class TestLocalRequiredTransactions(unittest.TestCase):
    def drafter(self, forward):
        result = LocalSDARDrafter.__new__(LocalSDARDrafter)
        result.block_size = 4
        result.mask_token = 31
        result.topk = 4
        result._proposal_log_budget = 0
        result.prefix_reuse = False
        result.graph_check = False
        result.continuous_prefix_states = {}
        result.req_to_token_pool = FakeLocalReqPool()
        result.token_allocator = FakeLocalAllocator()
        result.tree_cache = object()
        result.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_eos_token_id=None, vocab_size=32
            )
        )
        result.worker = SimpleNamespace(forward_dllm_packed=forward)
        return result

    @staticmethod
    def sources(count):
        return [
            SimpleNamespace(rid=f"r{i}", origin_input_ids=[1, 2], output_ids=[7])
            for i in range(count)
        ]

    def run_with_fake_batch(self, drafter, sources):
        def init_new(**kwargs):
            return FakeLocalBatch(kwargs["reqs"], drafter.req_to_token_pool)

        with patch(
            "sglang.srt.managers.schedule_batch.ScheduleBatch.init_new",
            side_effect=init_new,
        ):
            return drafter.propose_many(reqs=sources)

    def test_release_request_drops_continuous_prefix_state(self):
        drafter = self.drafter(Mock())
        drafter.prefix_reuse = True
        drafter.continuous_prefix_states["r0"] = LocalContinuousPrefixState(
            tokens=(1, 2), layer_kv=[]
        )
        drafter.release_request(rid="r0")
        self.assertNotIn("r0", drafter.continuous_prefix_states)

    def test_forward_failure_releases_kv_and_request_slots(self):
        drafter = self.drafter(Mock(side_effect=RuntimeError("forward failed")))
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.run_with_fake_batch(drafter, self.sources(1))
        self.assertEqual(
            drafter.token_allocator.freed, [(96, 97, 98, 99, 100, 101)]
        )
        self.assertIn(3, drafter.req_to_token_pool.free_slots)

    def test_second_postprocess_failure_rolls_back_whole_batch(self):
        output = SimpleNamespace(full_logits=torch.zeros((12, 32)))
        drafter = self.drafter(Mock(return_value=output))
        proposal = Proposal((7, 8, 9, 10))
        with patch(
            "sglang.srt.speculative.decoupled.local_drafter.proposal_from_logits",
            side_effect=[proposal, RuntimeError("second failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "second failed"):
                self.run_with_fake_batch(drafter, self.sources(2))
        self.assertEqual(len(drafter.token_allocator.freed), 2)
        self.assertEqual(set(drafter.req_to_token_pool.free_slots), {1, 2, 3})


class TestVocabularyCompatibility(unittest.TestCase):
    def test_requires_equal_model_sizes_target_ids_and_one_draft_mask(self):
        target = Mock(get_vocab=Mock(return_value={"a": 0, "b": 1}))
        compatible = Mock(
            get_vocab=Mock(return_value={"a": 0, "b": 1, "<mask>": 2})
        )
        different = Mock(
            get_vocab=Mock(return_value={"a": 1, "b": 0, "<mask>": 2})
        )
        self.assertTrue(
            validate_vocabulary_compatibility(
                target_tokenizer=target,
                target_vocab_size=8,
                draft_tokenizer=compatible,
                draft_vocab_size=8,
                mask_token=2,
            )
        )
        for draft_size in (7, 9):
            with self.assertRaisesRegex(ValueError, "model vocabulary sizes"):
                validate_vocabulary_compatibility(
                    target_tokenizer=target,
                    target_vocab_size=8,
                    draft_tokenizer=compatible,
                    draft_vocab_size=draft_size,
                    mask_token=2,
                )
        with self.assertRaisesRegex(ValueError, "identical Target token IDs"):
            validate_vocabulary_compatibility(
                target_tokenizer=target,
                target_vocab_size=8,
                draft_tokenizer=different,
                draft_vocab_size=8,
                mask_token=2,
            )
        wrong_extra = Mock(
            get_vocab=Mock(return_value={"a": 0, "b": 1, "<other>": 3})
        )
        with self.assertRaisesRegex(ValueError, "Draft-only MASK"):
            validate_vocabulary_compatibility(
                target_tokenizer=target,
                target_vocab_size=8,
                draft_tokenizer=wrong_extra,
                draft_vocab_size=8,
                mask_token=2,
            )


class TestRemoteOptionalClient(unittest.TestCase):
    def test_remote_absent_never_blocks_local_required_path(self):
        client = DraftClient(
            endpoint="tcp://127.0.0.1:1",
            rank=0,
            fingerprint="vocab",
            block_size=4,
            queue_size=2,
        )
        self.addCleanup(client.close)
        start = time.monotonic()
        self.assertIsNone(client.lookup(rid="r", prompt=(1,), outputs=(7,)))
        client.publish(
            rid="r",
            prompt=(1,),
            outputs=(7,),
            proposal=Proposal((7, 8, 9, 10)),
        )
        self.assertLess(time.monotonic() - start, 0.1)

    def test_outcome_miss_classification_separates_coverage_failures(self):
        previous = Proposal(
            (7, 8, 9, 10),
            ((7,), (8, 18, 28), (9, 19, 29), (10, 20, 30)),
        )
        outcomes = {Branch(2, 99): Proposal((99, 1, 2, 3))}
        self.assertEqual(
            classify_outcome_miss(previous=previous, delta=(8, 77), outcomes=outcomes),
            "candidate_rank",
        )
        self.assertEqual(
            classify_outcome_miss(
                previous=previous, delta=(8, 9, 10, 11), outcomes=outcomes
            ),
            "full_accept",
        )
        self.assertEqual(
            classify_outcome_miss(previous=previous, delta=(99, 77), outcomes=outcomes),
            "prefix",
        )

    def test_branch_coverage_matches_remote_anchor_selection(self):
        previous = Proposal(
            (7, 8, 9, 10),
            ((7,), (8, 18, 28, 38, 48), (9, 19), (10, 20)),
        )
        self.assertTrue(
            outcome_branch_is_covered(
                previous=previous,
                delta=(18,),
                num_anchors=3,
                accept_counts=(1, 2, 3, 4),
            )
        )
        self.assertFalse(
            outcome_branch_is_covered(
                previous=previous,
                delta=(48,),
                num_anchors=3,
                accept_counts=(1, 2, 3, 4),
            )
        )

    def test_non_inflight_lookup_does_not_use_publication_deadline(self):
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            lookup_wait_seconds=1,
        )
        proposal = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=proposal)
        start = time.monotonic()
        self.assertIsNone(
            client.lookup(rid="r", prompt=(1,), outputs=(7, 8, 99))
        )
        self.assertLess(time.monotonic() - start, 0.005)
        client.close()

    def test_expired_publication_deadline_does_not_wait_again(self):
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            lookup_wait_seconds=0.01,
        )
        proposal = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=proposal)
        client.sessions["r"].published_at -= 1
        start = time.monotonic()
        self.assertIsNone(
            client.lookup(rid="r", prompt=(1,), outputs=(7, 8, 99))
        )
        self.assertLess(time.monotonic() - start, 0.005)
        client.close()

    def test_bounded_lookup_wait_wakes_when_result_arrives(self):
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            lookup_wait_seconds=0.1,
        )
        previous = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=previous)
        state = client.sessions["r"]
        outcome = Proposal((99, 20, 21, 22))
        client._record(
            Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision,
                proposal_id=state.proposal_id,
                inflight=True,
                expected_seconds=0.05,
            )
        )
        producer = threading.Thread(
            target=lambda: (
                time.sleep(0.01),
                client._record(
                    Reply(
                        rpc="",
                        session=state.nonce,
                        revision=state.revision,
                        proposal_id=state.proposal_id,
                        outcomes=(Outcome(Branch(2, 99), outcome),),
                    )
                ),
            )
        )
        producer.start()
        start = time.monotonic()
        selected = client.lookup(
            rid="r", prompt=(1,), outputs=(7, 8, 99)
        )
        producer.join()
        self.assertEqual(selected.tokens, outcome.tokens)
        self.assertLess(time.monotonic() - start, 0.05)
        client.close()

    def test_exact_ready_result_is_consumed_and_stale_result_is_dropped(self):
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
        )
        previous = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=previous)
        state = client.sessions["r"]
        outcome = Proposal((99, 20, 21, 22))
        client._record(
            Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision,
                proposal_id=state.proposal_id,
                outcomes=(Outcome(Branch(2, 99), outcome),),
            )
        )
        selected = client.lookup(rid="r", prompt=(1,), outputs=(7, 8, 99))
        self.assertEqual(selected.tokens, (99, 20, 21, 22))
        client._record(
            Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision - 1,
                proposal_id=proposal_identity(previous),
                outcomes=(Outcome(Branch(1, 99), outcome),),
            )
        )
        self.assertEqual(client.metrics["remote_optional_stale_result_ct"], 1)
        client.close()

    def test_finish_and_record_share_one_atomic_state_boundary(self):
        client = DraftClient(
            endpoint=None, rank=0, fingerprint="vocab", block_size=4
        )
        for index in range(50):
            proposal = Proposal((7, 8, 9, 10))
            client.publish(
                rid="r", prompt=(index,), outputs=(7,), proposal=proposal
            )
            state = client.sessions["r"]
            reply = Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision,
                proposal_id=state.proposal_id,
                outcomes=(Outcome(Branch(1, 99), proposal),),
            )
            finish = threading.Thread(target=client.finish, kwargs={"rid": "r"})
            record = threading.Thread(target=client._record, args=(reply,))
            finish.start()
            record.start()
            finish.join()
            record.join()
            self.assertFalse(client.sessions)
            self.assertFalse(client._ready)
        client.close()

    def test_epoch_change_clears_ready_and_republishes_full_snapshot(self):
        client = DraftClient(
            endpoint=None, rank=0, fingerprint="vocab", block_size=4
        )
        proposal = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=proposal)
        state = client.sessions["r"]
        client._record(
            Reply(
                rpc="h1",
                fingerprint="vocab",
                block_size=4,
                service_epoch="epoch-1",
            )
        )
        sent = []
        client._enqueue = lambda request: sent.append(request) or True
        client._ready[(state.nonce, state.revision, state.proposal_id)] = (
            time.monotonic(),
            {},
        )
        client._record(
            Reply(
                rpc="h2",
                fingerprint="vocab",
                block_size=4,
                service_epoch="epoch-2",
            )
        )
        self.assertFalse(client._ready)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].committed_prefix, (1, 7))
        self.assertEqual(sent[0].service_epoch, "epoch-2")
        client.close()

    def test_retraction_reopens_with_a_fresh_session(self):
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
        )
        first = Proposal((7, 8, 9, 10))
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=first)
        old_nonce = client.sessions["r"].nonce
        client.publish(
            rid="r",
            prompt=(2,),
            outputs=(11,),
            proposal=Proposal((11, 12, 13, 14)),
        )
        self.assertNotEqual(old_nonce, client.sessions["r"].nonce)
        client.close()


if __name__ == "__main__":
    unittest.main()
