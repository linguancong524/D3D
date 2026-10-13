"""KV rollback and packed branch isolation, using a CPU attention oracle."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.speculative.decoupled.packed_layout import build_packed_layout
from sglang.srt.speculative.decoupled.protocol import (
    Branch,
    Commit,
    proposal_identity,
)
from sglang.srt.speculative.decoupled.sdar_backend import SDARBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TokenCache:
    def __init__(self):
        self.tokens = []
        self.context_values = []

    def crop(self, count):
        self.tokens = self.tokens[:count]
        self.context_values = self.context_values[:count]


class AttentionOracle:
    """Visible-token sums expose leakage between branches or stale suffixes."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        input_ids,
        *,
        position_ids,
        attention_mask,
        past_key_values,
        use_cache,
        store_kv,
        logits_to_keep,
    ):
        if not isinstance(past_key_values, TokenCache):
            past_key_values = TokenCache()
        input_tokens = input_ids[0].tolist()
        mask = attention_mask[0, 0]
        self.calls.append((input_ids.clone(), position_ids.clone(), mask.clone()))
        raw_values = past_key_values.tokens + input_tokens
        weights = torch.arange(1, len(raw_values) + 1)
        # A two-layer cache oracle: layer-one values depend on visible raw
        # context, and layer two consumes those cached values. Causal chunking
        # remains equivalent, while a block-causal rejected suffix pollutes KV.
        layer_one = (mask * torch.tensor(raw_values) * weights).sum(dim=-1)
        values = past_key_values.context_values + layer_one.tolist()
        visible = (mask * torch.tensor(values) * weights).sum(dim=-1)
        prediction = (visible + position_ids[0]) % 31
        logits = torch.zeros((1, input_ids.shape[1], 32))
        logits[0].scatter_(1, prediction[:, None], 100)
        past_key_values.tokens.extend(input_tokens)
        # Cached representations depend on all visible context, unlike the old
        # token-only oracle that could not expose block-causal KV pollution.
        past_key_values.context_values.extend(layer_one.tolist())
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def backend():
    # Bypass only external weight loading; exercise production masks/cache code.
    result = SDARBackend.__new__(SDARBackend)
    result.model = AttentionOracle()
    result.device = torch.device("cpu")
    result.mask_token_id = 31
    result.block_size = 4
    result.num_anchors = 0
    result.accept_counts = (1, 2, 3, 4)
    result.max_branches = 4
    result.vocab_size = 32
    result.states = {}
    result.prefill_chunk_size = 1024
    return result


class TestPackedLayout(CustomTestCase):
    def test_single_row_reuses_pending_without_changing_branch_visibility(self):
        """A shared pending row must match isolated branches at every query."""
        branches = (Branch(1, 19), Branch(3, 20), Branch(4, None))
        options = dict(
            base_kv_len=2,
            pending_tokens=(7, 8, 9, 10),
            branch_token_count=4,
            mask_token_id=31,
            attention_mode="causal",
        )
        shared = build_packed_layout(branches=branches, **options)
        self.assertEqual(shared.query_len, 4 + len(branches) * 4)
        for segment in shared.segments:
            isolated = build_packed_layout(branches=(segment.branch,), **options)
            shared_start = segment.start
            shared_stop = segment.stop
            self.assertEqual(
                shared.position_ids[shared_start:shared_stop],
                isolated.position_ids[4:8],
            )
            self.assertEqual(
                shared.input_ids[shared_start:shared_stop],
                isolated.input_ids[4:8],
            )
            self.assertTrue(
                (
                    shared.attention_mask[shared_start:shared_stop, :6]
                    == isolated.attention_mask[4:8, :6]
                ).all()
            )
            self.assertTrue(
                (
                    shared.attention_mask[
                        shared_start:shared_stop, 2 + shared_start : 2 + shared_stop
                    ]
                    == isolated.attention_mask[4:8, 6:10]
                ).all()
            )
            for other in shared.segments:
                if other is segment:
                    continue
                self.assertFalse(
                    shared.attention_mask[
                        shared_start:shared_stop,
                        2 + other.start : 2 + other.stop,
                    ].any()
                )

    def test_missing_committed_replay_precedes_branch_acceptance_positions(self):
        layout = build_packed_layout(
            base_kv_len=2,
            pending_tokens=(3, 4, 7, 8, 9, 10),
            branches=(Branch(2, 19),),
            branch_token_count=4,
            mask_token_id=31,
            attention_mode="block_causal",
            causal_prefix_len=2,
        )
        # The replayed committed suffix is causal, while proposal positions can
        # use block-causal visibility. Branch positions restart after two replay
        # tokens plus two actually accepted proposal tokens.
        self.assertEqual(layout.position_ids[-4:], (6, 7, 8, 9))
        self.assertTrue(layout.attention_mask[0, :3].all())
        self.assertFalse(layout.attention_mask[0, 3:].any())
        branch_start = 6
        self.assertTrue(layout.attention_mask[branch_start, :6].all())
        self.assertFalse(layout.attention_mask[branch_start, 6:8].any())
        self.assertTrue(layout.attention_mask[branch_start, 8:12].all())

    def test_full_accept_unanchored_branch_starts_with_mask(self):
        layout = build_packed_layout(
            base_kv_len=3,
            pending_tokens=(7, 8, 9, 10),
            branches=(Branch(4, None),),
            branch_token_count=4,
            mask_token_id=31,
            attention_mode="block_causal",
        )
        self.assertEqual(layout.input_ids, (7, 8, 9, 10, 31, 31, 31, 31))
        self.assertEqual(layout.position_ids, (3, 4, 5, 6, 7, 8, 9, 10))
        self.assertEqual(layout.retained_len, 7)
        # Pending replay is block-causal; the branch sees all accepted pending
        # tokens and only its own physical scratch segment.
        self.assertTrue(layout.attention_mask[:4, :7].all())
        self.assertTrue(layout.attention_mask[4:8, :7].all())
        self.assertTrue(layout.attention_mask[4:8, 7:11].all())


class TestSDARCache(CustomTestCase):
    def test_chunked_prefill_matches_full_prefill(self):
        chunked = backend()
        chunked.prefill_chunk_size = 3
        prefix = tuple(range(1, 17))
        result = chunked.draft(session="a", revision=0, prefix=prefix)
        reference = backend().draft(session="b", revision=0, prefix=prefix)
        self.assertEqual(result, reference)
        self.assertEqual(chunked.states["a"].cache.tokens, list(prefix[:-1]))
        self.assertEqual(len(chunked.model.calls), 5)
        self.assertTrue(all(call[0].shape[1] <= 7 for call in chunked.model.calls))

    def test_partial_rejection_matches_fresh_prefix_reconstruction(self):
        warm = backend()
        prefix = (1, 2, 3, 7)
        proposal = warm.draft(session="a", revision=0, prefix=prefix)
        warm.speculate(session="a", revision=0, prefix=prefix, proposal=proposal)
        corrected = prefix + (29,)
        warm.commit_many(
            items=[
                Commit(
                    session="a",
                    revision=1,
                    proposal_id=proposal_identity(proposal),
                    committed_prefix=corrected,
                )
            ]
        )
        replay = warm.draft(session="a", revision=1, prefix=corrected)
        cold_backend = backend()
        cold = cold_backend.draft(session="b", revision=0, prefix=corrected)
        self.assertEqual(replay.tokens, cold.tokens)
        self.assertEqual(replay.top_tokens, cold.top_tokens)
        self.assertEqual(warm.states["a"].tokens, corrected[:-1])
        self.assertEqual(warm.states["a"].cache.tokens, list(corrected[:-1]))
        self.assertEqual(
            warm.states["a"].cache.context_values,
            cold_backend.states["b"].cache.context_values,
        )

    def test_packed_masks_hide_sibling_and_rejected_prefix_tokens(self):
        draft = backend()
        prefix = (1, 2, 3, 7)
        proposal = draft.draft(session="a", revision=0, prefix=prefix)
        draft.speculate(session="a", revision=0, prefix=prefix, proposal=proposal)
        inputs, positions, mask = draft.model.calls[-1]
        cached = len(prefix) - 1
        for index, count in enumerate((1, 2, 3, 4)):
            start, end = 4 + index * 4, 8 + index * 4
            # A branch sees exactly the real prefix and its own candidate block.
            self.assertTrue(mask[start:end, : cached + count].all())
            self.assertFalse(mask[start:end, cached + count : cached + start].any())
            self.assertTrue(mask[start:end, cached + start : cached + end].all())
            self.assertFalse(mask[start:end, cached + end :].any())
            self.assertEqual(positions[0, start].item(), cached + count)
        self.assertEqual(
            draft.states["a"].cache.tokens, list(prefix[:-1] + proposal.tokens)
        )

    def test_stale_proposal_cannot_mutate_session_kv(self):
        draft = backend()
        prefix = (1, 2, 3, 7)
        proposal = draft.draft(session="a", revision=0, prefix=prefix)
        before = tuple(draft.states["a"].cache.tokens)
        with self.assertRaisesRegex(ValueError, "proposal does not match"):
            draft.commit_many(
                items=[
                    Commit(
                        session="a",
                        revision=1,
                        proposal_id="stale",
                        committed_prefix=prefix + (29,),
                    )
                ]
            )
        self.assertEqual(tuple(draft.states["a"].cache.tokens), before)
        self.assertEqual(draft.states["a"].proposal_id, proposal_identity(proposal))

    def test_sessions_never_share_mutable_kv(self):
        draft = backend()
        first = draft.draft(session="a", revision=0, prefix=(1, 2, 3))
        draft.draft(session="b", revision=0, prefix=(10, 11, 12))
        second = draft.draft(session="a", revision=0, prefix=(1, 2, 3))
        self.assertEqual(first, second)
        draft.close(session="a")
        self.assertEqual(set(draft.states), {"b"})


if __name__ == "__main__":
    unittest.main()
