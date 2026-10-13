"""Regressions for waiting only on an inflight, covered Draft branch."""

import threading
import time
import unittest

from sglang.srt.speculative.decoupled.client import DraftClient
from sglang.srt.speculative.decoupled.protocol import Branch, Outcome, Proposal, Reply
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCoveredResultWait(CustomTestCase):
    def _client(self) -> tuple[DraftClient, Proposal]:
        client = DraftClient(
            endpoint=None,
            rank=0,
            fingerprint="vocab",
            block_size=4,
            num_anchors=1,
            lookup_wait_seconds=0.08,
            wait_for_covered_result=True,
        )
        self.addCleanup(client.close)
        proposal = Proposal(
            (7, 8, 9, 10),
            ((7,), (8, 18), (9, 99), (10, 20)),
        )
        client.publish(rid="r", prompt=(1,), outputs=(7,), proposal=proposal)
        return client, proposal

    def test_covered_result_arriving_after_estimate_is_used(self):
        """An EWMA underestimate must not force a second local Draft forward."""
        client, _ = self._client()
        state = client.sessions["r"]
        client._record(
            Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision,
                proposal_id=state.proposal_id,
                inflight=True,
                expected_seconds=0.001,
            )
        )
        outcome = Proposal((99, 20, 21, 22))

        def deliver_result():
            time.sleep(0.015)
            client._record(
                Reply(
                    rpc="",
                    session=state.nonce,
                    revision=state.revision,
                    proposal_id=state.proposal_id,
                    outcomes=(Outcome(Branch(2, 99), outcome),),
                )
            )

        producer = threading.Thread(target=deliver_result)
        producer.start()
        selected = client.lookup(rid="r", prompt=(1,), outputs=(7, 8, 99))
        producer.join()
        self.assertEqual(selected, outcome)
        self.assertEqual(client.metrics["remote_optional_lookup_hit_ct"], 1)

    def test_uncovered_anchor_does_not_block(self):
        """A correction outside the anchor candidates must fall back immediately."""
        client, _ = self._client()
        state = client.sessions["r"]
        client._record(
            Reply(
                rpc="",
                session=state.nonce,
                revision=state.revision,
                proposal_id=state.proposal_id,
                inflight=True,
                expected_seconds=1,
            )
        )
        start = time.monotonic()
        self.assertIsNone(client.lookup(rid="r", prompt=(1,), outputs=(7, 8, 77)))
        self.assertLess(time.monotonic() - start, 0.01)
        self.assertEqual(client.metrics["remote_optional_inflight_wait_ct"], 0)


if __name__ == "__main__":
    unittest.main()
