import unittest
from types import SimpleNamespace
from d3d_sglang.service import StrictDraftService
from sglang.srt.speculative.decoupled.protocol import Branch, Proposal


class FakeBackend:
    def __init__(self):
        self.closed = []
        self.commits = []

    def close(self, *, session):
        self.closed.append(session)

    def commit_many(self, *, items):
        self.commits.extend(items)

    def speculate_many(self, *, items):
        return {
            session: {Branch(2, 99): Proposal((99, 7, 8, 9))}
            for session, revision, prefix, proposal in items
        }


class RemoteServiceTests(unittest.TestCase):
    def service(self):
        service = StrictDraftService(FakeBackend())
        service.required = lambda prefix: Proposal((prefix[-1], 2, 3, 4))
        return service

    def test_hit_uses_committed_delta_and_never_invokes_required(self):
        service = self.service()
        reply, pending = service.step([
            {"rid": "request", "prompt": [10, 11], "outputs": [1]}
        ])
        service.speculate(pending)

        def forbidden(prefix):
            raise AssertionError("covered branch must skip required replay")

        service.required = forbidden
        reply, pending = service.step([
            {"rid": "request", "prompt": [10, 11], "outputs": [1, 2, 99]}
        ])
        self.assertEqual(reply["proposals"]["request"]["tokens"], [99, 7, 8, 9])
        self.assertEqual(service.metrics["remote_hit_count"], 1)
        self.assertEqual(service.backend.commits[0].revision, 1)

    def test_branch_miss_still_generates_on_remote_backend(self):
        service = self.service()
        reply, pending = service.step([
            {"rid": "request", "prompt": [10, 11], "outputs": [1]}
        ])
        service.speculate(pending)
        reply, pending = service.step([
            {"rid": "request", "prompt": [10, 11], "outputs": [1, 77]}
        ])
        self.assertEqual(reply["proposals"]["request"]["tokens"], [77, 2, 3, 4])
        self.assertEqual(service.metrics["remote_miss_count"], 2)

    def test_divergent_reused_request_drops_stale_session(self):
        service = self.service()
        reply, pending = service.step([
            {"rid": "request", "prompt": [10, 11], "outputs": [1]}
        ])
        session = pending[0][0]
        service.speculate(pending)
        reply, pending = service.step([
            {"rid": "request", "prompt": [20, 21], "outputs": [5]}
        ])
        self.assertIn(session, service.backend.closed)
        self.assertEqual(reply["proposals"]["request"]["tokens"][0], 5)
        self.assertNotEqual(pending[0][0], session)

    def test_required_replay_places_anchor_after_causal_prefix(self):
        backend = FakeBackend()
        backend.block_size = 4
        backend.candidate_topk = 2
        requests = []

        def replay(request):
            requests.append(request)
            return SimpleNamespace(
                success=True,
                items=[SimpleNamespace(outcomes=[SimpleNamespace(
                    tokens=[12, 7, 8, 9], top_tokens=[[12], [7], [8], [9]]
                )])],
            )

        backend.engine = SimpleNamespace(dllm_packed_replay=replay)
        service = StrictDraftService(backend)
        proposal = service.required((10, 11, 12))
        item = requests[0].items[0]
        self.assertEqual(item.committed_tokens, [10])
        self.assertEqual(item.proposal_tokens, [11])
        self.assertEqual(item.branches[0].num_accept_tokens, 1)
        self.assertEqual(item.branches[0].bonus_token, 12)
        self.assertEqual(proposal.tokens, (12, 7, 8, 9))
        self.assertEqual(backend.closed, [item.session])


class PluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.plugins import load_plugins
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        load_plugins()
        cls.algorithm = SpeculativeAlgorithm.from_string("D3D_REMOTE")

    def args(self):
        return SimpleNamespace(
            decoupled_draft_endpoint="tcp://127.0.0.1:17891",
            speculative_draft_model_path=None, tp_size=1, dp_size=1,
            speculative_num_draft_tokens=16,
        )

    def test_scheduler_lifecycle_has_no_local_draft_state(self):
        args = self.args()
        self.algorithm.handle_server_args(args)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertFalse(args.enable_mixed_chunk)
        self.assertTrue(self.algorithm.is_host_draft())
        self.assertFalse(self.algorithm.has_draft_kv())
        self.assertFalse(self.algorithm.carries_draft_hidden_states())
        self.assertFalse(self.algorithm.need_topk())
        self.assertTrue(callable(self.algorithm.create_future_map))

    def test_local_checkpoint_is_rejected_before_loading(self):
        args = self.args()
        args.speculative_draft_model_path = "forbidden-local-draft"
        with self.assertRaisesRegex(ValueError, "Target-local"):
            self.algorithm.handle_server_args(args)


if __name__ == "__main__":
    unittest.main()
