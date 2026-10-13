"""Optional-only native packed Engine adapter tests."""

import unittest

from sglang.srt.managers.io_struct import (
    DllmKvControlReqOutput,
    DllmPackedItemResult,
    DllmPackedOutcome,
    DllmPackedReqOutput,
)
from sglang.srt.speculative.decoupled.engine_backend import DllmEngineBackend
from sglang.srt.speculative.decoupled.protocol import Branch, Commit, Proposal


class FakeEngine:
    def __init__(self):
        self.packed = []
        self.controls = []
        self.shutdown_ct = 0

    def dllm_packed_replay(self, request):
        self.packed.append(request)
        items = []
        for item in request.items:
            items.append(
                DllmPackedItemResult(
                    session=item.session,
                    outcomes=[
                        DllmPackedOutcome(
                            branch=branch,
                            tokens=[branch.bonus_token or 11, 12, 13, 14],
                            top_tokens=[[11], [12], [13], [14]],
                        )
                        for branch in item.branches
                    ],
                )
            )
        return DllmPackedReqOutput(
            success=True, physical_forward_count=1, items=items
        )

    def dllm_kv_control(self, request):
        self.controls.append(request)
        return DllmKvControlReqOutput(success=True)

    def shutdown(self):
        self.shutdown_ct += 1


def backend():
    result = DllmEngineBackend.__new__(DllmEngineBackend)
    result.block_size = 4
    result.num_anchors = 1
    result.accept_counts = (1, 2, 3, 4)
    result.max_branches = 8
    result.candidate_topk = 4
    result.engine = FakeEngine()
    result.packed_physical_forward_count = 0
    result.packed_model_forward_seconds = 0.0
    result.packed_build_seconds = 0.0
    result.packed_prepare_seconds = 0.0
    result.packed_postprocess_seconds = 0.0
    result.packed_protocol_seconds = 0.0
    return result


class TestDllmEngineBackend(unittest.TestCase):
    def test_multiple_sessions_share_exactly_one_native_packed_forward(self):
        draft = backend()
        proposal_a = Proposal(
            (7, 8, 9, 10), ((7,), (8, 18), (9, 19), (10, 20), (11, 21))
        )
        proposal_b = Proposal(
            (9, 10, 11, 12), ((9,), (10, 20), (11, 21), (12, 22), (13, 23))
        )
        result = draft.speculate_many(
            items=[
                ("a", 0, (1, 2, 7), proposal_a),
                ("b", 0, (3, 4, 9), proposal_b),
            ]
        )
        self.assertEqual(len(draft.engine.packed), 1)
        self.assertEqual({item.session for item in draft.engine.packed[0].items}, {"a", "b"})
        self.assertEqual(set(result), {"a", "b"})
        self.assertEqual(draft.packed_physical_forward_count, 1)

    def test_backend_has_no_remote_required_generation_api(self):
        draft = backend()
        self.assertFalse(hasattr(draft, "draft"))
        self.assertFalse(hasattr(draft, "draft_many"))
        self.assertFalse(hasattr(draft, "_generate"))

    def test_commit_and_close_preserve_identity(self):
        draft = backend()
        draft.commit_many(
            items=[Commit(
                session="a",
                revision=3,
                proposal_id="p2",
                committed_prefix=(1, 2, 7, 8),
            )]
        )
        draft.close(session="a")
        self.assertEqual(draft.engine.controls[0].commits[0].proposal_id, "p2")
        self.assertEqual(draft.engine.controls[1].close_sessions, ["a"])

    def test_native_packed_is_mandatory(self):
        draft = backend()
        draft.num_anchors = 0
        draft.engine = object()
        with self.assertRaisesRegex(RuntimeError, "requires native packed replay"):
            draft.speculate_many(
                items=[("a", 0, (1, 7), Proposal((7, 8, 9, 10)))]
            )


if __name__ == "__main__":
    unittest.main()
