"""Optional-publication protocol and service regressions."""

import time
import unittest

import msgspec
import zmq
from sglang.srt.speculative.decoupled.client import DraftClient
from sglang.srt.speculative.decoupled.protocol import (
    Branch,
    Proposal,
    Request,
    proposal_identity,
)
from sglang.srt.speculative.decoupled.service import DraftService


class TestOptionalClientPump(unittest.TestCase):
    """The pump must deliver immediately without spinning.

    The idle poll interval is deliberately far longer than the assertion
    window: only the wakeup socket can deliver a publication in time, so a
    spinning pump would pass the latency check but is not needed.
    """

    @staticmethod
    def _bind_router(context):
        router = context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind("tcp://127.0.0.1:0")
        return router, router.getsockopt_string(zmq.LAST_ENDPOINT).replace(
            "0.0.0.0", "127.0.0.1"
        )

    def _router(self, context):
        router, endpoint = self._bind_router(context)
        self.addCleanup(router.close)
        return router, endpoint

    @staticmethod
    def _drain(router, decoder, count, timeout=3.0):
        deadline = time.monotonic() + timeout
        requests = []
        while len(requests) < count and time.monotonic() < deadline:
            if not router.poll(50):
                continue
            while True:
                try:
                    _, payload = router.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                requests.append(decoder.decode(payload))
        return requests

    def test_enqueue_wakes_an_idle_pump(self):
        decoder = msgspec.msgpack.Decoder(Request)
        context = zmq.Context()
        self.addCleanup(context.term)
        router, endpoint = self._router(context)
        client = DraftClient(
            endpoint=endpoint,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            poll_interval_seconds=2.0,
        )
        self.addCleanup(client.close)
        self.assertEqual(self._drain(router, decoder, 1)[0].op, "hello")
        time.sleep(0.05)

        start = time.perf_counter()
        client.publish(
            rid="r",
            prompt=(1, 2, 3, 7),
            outputs=(7,),
            proposal=Proposal((7, 8, 9, 10), ((7,), (8,), (9,), (10,))),
        )
        ops = [request.op for request in self._drain(router, decoder, 2)]
        elapsed = time.perf_counter() - start
        self.assertIn("publish", ops)
        self.assertLess(elapsed, 0.5)
        self.assertGreater(client.metrics["remote_optional_pump_wakeup_ct"], 0)

    def test_close_joins_a_blocked_pump(self):
        context = zmq.Context()
        self.addCleanup(context.term)
        router, endpoint = self._router(context)
        del router
        client = DraftClient(
            endpoint=endpoint,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            poll_interval_seconds=2.0,
        )
        time.sleep(0.1)
        client.close()
        self.assertFalse(client._thread.is_alive())


class OptionalBackend:
    fingerprint = "vocab"
    vocab_size = 128
    block_size = 4

    def __init__(self):
        self.stage_calls = []
        self.speculate_calls = []
        self.commit_calls = []
        self.closed = []
        self.packed_physical_forward_count = 0

    def stage_many(self, *, items):
        self.stage_calls.extend(items)

    def speculate_many(self, *, items):
        self.speculate_calls.append(items)
        self.packed_physical_forward_count += 1
        return {
            session: {Branch(1, 99): Proposal((99, 31, 32, 33))}
            for session, _, _, _ in items
        }

    def commit_many(self, *, items):
        self.commit_calls.extend(items)

    def close(self, *, session):
        self.closed.append(session)

    def shutdown(self):
        pass


class TestOptionalService(unittest.TestCase):
    def setUp(self):
        self.backend = OptionalBackend()
        self.service = DraftService(
            backend=self.backend, batching_window_seconds=0, speculation_batch_size=8
        )
        self.addCleanup(self.service.close)
        for peer in (b"a", b"b"):
            reply = self.service.handle(
                peer=peer,
                request=Request(
                    rpc="hello", op="hello", fingerprint="vocab", block_size=4
                ),
            )
            self.assertFalse(reply.error)

    def publish(self, *, peer=b"a", session="same", revision=0, anchor=7, previous=""):
        proposal = Proposal((anchor, 20, 21, 22), ((anchor,), (20, 30), (21, 31), (22, 32)))
        request = Request(
            rpc=f"p-{peer!r}-{revision}",
            op="publish",
            session=session,
            revision=revision,
            committed_prefix=(1, 2, anchor),
            proposal=proposal,
            proposal_id=proposal_identity(proposal),
            previous_proposal_id=previous,
        )
        return proposal, self.service.handle(peer=peer, request=request)

    def test_prefix_stage_reuses_backend_key_for_first_publication(self):
        reply = self.service.handle(
            peer=b"a",
            request=Request(
                rpc="stage",
                op="stage",
                session="staged",
                committed_prefix=(1, 2),
            ),
        )
        self.assertFalse(reply.error)
        self.assertEqual(len(self.backend.stage_calls), 1)
        backend_key = self.backend.stage_calls[0][0]
        self.publish(peer=b"a", session="staged")
        self.service.speculate_batch(force=True)
        self.assertEqual(self.backend.speculate_calls[0][0][0], backend_key)
        self.service.handle(
            peer=b"a",
            request=Request(rpc="close", op="close", session="staged"),
        )
        self.assertEqual(self.backend.closed, [backend_key])

    def test_staged_only_session_expires_and_releases_backend_state(self):
        self.service.lease_seconds = 0.001
        self.service.handle(
            peer=b"a",
            request=Request(
                rpc="stage",
                op="stage",
                session="orphan",
                committed_prefix=(1, 2),
            ),
        )
        backend_key = self.backend.stage_calls[-1][0]
        time.sleep(0.002)
        self.service.expire()
        self.assertNotIn((b"a", "orphan"), self.service.staged)
        self.assertIn(backend_key, self.backend.closed)

    def test_batch_selection_takes_one_oldest_item_per_peer_first(self):
        service = DraftService(
            backend=OptionalBackend(), batching_window_seconds=0, speculation_batch_size=2
        )
        self.addCleanup(service.close)
        for peer in (b"x", b"y"):
            service.handle(
                peer=peer,
                request=Request(
                    rpc="h", op="hello", fingerprint="vocab", block_size=4
                ),
            )
        proposal = Proposal((7, 8, 9, 10))
        for peer, session in ((b"x", "x1"), (b"x", "x2"), (b"y", "y1")):
            service.handle(
                peer=peer,
                request=Request(
                    rpc=session,
                    op="publish",
                    session=session,
                    committed_prefix=(1, 7),
                    proposal=proposal,
                    proposal_id=proposal_identity(proposal),
                ),
            )
        service.speculate_batch(force=True)
        selected_sessions = {
            session for session, _, _, _ in service.backend.speculate_calls[0]
        }
        backend_sessions = {
            service.sessions[(b"x", "x1")].backend_key,
            service.sessions[(b"y", "y1")].backend_key,
        }
        self.assertEqual(selected_sessions, backend_sessions)

    def test_inflight_event_precedes_optional_outcome(self):
        self.publish()
        started = []
        replies = self.service.speculate_batch(
            force=True, on_start=lambda items: started.extend(items)
        )
        self.assertEqual(len(started), 1)
        self.assertTrue(started[0][1].inflight)
        self.assertFalse(started[0][1].outcomes)
        self.assertTrue(replies[0][1].outcomes)

    def test_same_rid_from_multiple_targets_is_isolated(self):
        self.publish(peer=b"a")
        self.publish(peer=b"b", anchor=9)
        self.assertEqual(len(self.service.sessions), 2)
        first_key = self.service.sessions[(b"a", "same")].backend_key
        second_key = self.service.sessions[(b"b", "same")].backend_key
        self.assertNotEqual(first_key, second_key)

    def test_latest_revision_supersedes_queued_work(self):
        first, _ = self.publish()
        second = Proposal((8, 40, 41, 42))
        reply = self.service.handle(
            peer=b"a",
            request=Request(
                rpc="p1",
                op="publish",
                session="same",
                revision=1,
                committed_prefix=(1, 2, 7, 8),
                proposal=second,
                proposal_id=proposal_identity(second),
                previous_proposal_id=proposal_identity(first),
            ),
        )
        self.assertFalse(reply.error)
        replies = self.service.speculate_batch(force=True)
        self.assertEqual(len(replies), 1)
        self.assertEqual(self.backend.speculate_calls[0][0][1], 1)
        self.assertEqual(self.service.metrics["remote_optional_supersede_ct"], 1)

    def test_stale_and_conflicting_publications_fail_closed(self):
        first, _ = self.publish()
        _, duplicate = self.publish()
        self.assertFalse(duplicate.error)
        wrong = Proposal((7, 1, 2, 3))
        reply = self.service.handle(
            peer=b"a",
            request=Request(
                rpc="bad",
                op="publish",
                session="same",
                revision=0,
                committed_prefix=(1, 2, 7),
                proposal=wrong,
                proposal_id=proposal_identity(wrong),
            ),
        )
        self.assertIn("conflicting", reply.error)
        self.assertEqual(
            self.service.sessions[(b"a", "same")].proposal_id,
            proposal_identity(first),
        )

    def test_batching_crosses_target_peers_and_returns_exact_identity(self):
        pa, _ = self.publish(peer=b"a")
        pb, _ = self.publish(peer=b"b", anchor=9)
        replies = self.service.speculate_batch(force=True)
        self.assertEqual(len(self.backend.speculate_calls[0]), 2)
        self.assertEqual({peer for peer, _ in replies}, {b"a", b"b"})
        ids = {reply.proposal_id for _, reply in replies}
        self.assertEqual(ids, {proposal_identity(pa), proposal_identity(pb)})
        self.assertTrue(all(reply.outcomes for _, reply in replies))

    def test_close_cancels_queue_and_releases_backend_state(self):
        self.publish()
        key = self.service.sessions[(b"a", "same")].backend_key
        self.service.handle(
            peer=b"a", request=Request(rpc="close", op="close", session="same")
        )
        self.assertFalse(self.service.speculate_batch(force=True))
        self.assertEqual(self.backend.closed, [key])

    def test_per_peer_quota_preserves_capacity_for_another_target(self):
        service = DraftService(
            backend=OptionalBackend(),
            max_sessions=4,
            max_sessions_per_peer=2,
            batching_window_seconds=0,
        )
        self.addCleanup(service.close)
        for peer in (b"x", b"y"):
            service.handle(
                peer=peer,
                request=Request(
                    rpc="h", op="hello", fingerprint="vocab", block_size=4
                ),
            )
        proposal = Proposal((7, 8, 9, 10))
        for session in ("x1", "x2"):
            reply = service.handle(
                peer=b"x",
                request=Request(
                    rpc=session,
                    op="publish",
                    session=session,
                    committed_prefix=(1, 7),
                    proposal=proposal,
                    proposal_id=proposal_identity(proposal),
                ),
            )
            self.assertFalse(reply.error)
        blocked = service.handle(
            peer=b"x",
            request=Request(
                rpc="x3",
                op="publish",
                session="x3",
                committed_prefix=(1, 7),
                proposal=proposal,
                proposal_id=proposal_identity(proposal),
            ),
        )
        self.assertIn("per-peer", blocked.error)
        admitted = service.handle(
            peer=b"y",
            request=Request(
                rpc="y1",
                op="publish",
                session="y1",
                committed_prefix=(1, 7),
                proposal=proposal,
                proposal_id=proposal_identity(proposal),
            ),
        )
        self.assertFalse(admitted.error)

    def test_expire_removes_idle_peer_identity(self):
        service = DraftService(backend=OptionalBackend(), lease_seconds=0.001)
        self.addCleanup(service.close)
        service.handle(
            peer=b"idle",
            request=Request(
                rpc="h", op="hello", fingerprint="vocab", block_size=4
            ),
        )
        service.peers[b"idle"] = time.monotonic() - 1
        service.expire()
        self.assertNotIn(b"idle", service.peers)

    def test_batching_window_does_not_run_early(self):
        service = DraftService(
            backend=OptionalBackend(), batching_window_seconds=10
        )
        self.addCleanup(service.close)
        service.handle(
            peer=b"x",
            request=Request(rpc="h", op="hello", fingerprint="vocab", block_size=4),
        )
        proposal = Proposal((7, 8, 9, 10))
        service.handle(
            peer=b"x",
            request=Request(
                rpc="p",
                op="publish",
                session="s",
                committed_prefix=(1, 7),
                proposal=proposal,
                proposal_id=proposal_identity(proposal),
            ),
        )
        self.assertEqual(service.speculate_batch(), [])


if __name__ == "__main__":
    unittest.main()
