"""Nonblocking optional-result client for the shared packed drafter.

Only the pump thread touches ZMQ. Scheduler-facing methods mutate bounded
in-memory queues/caches and therefore never wait for the remote GPU.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid

import msgspec
import zmq

from .protocol import (
    VERSION,
    Branch,
    Proposal,
    Reply,
    Request,
    classify_outcome_miss,
    correction_token_rank,
    outcome_branch_is_covered,
    proposal_identity,
    select_outcome,
)


class RemoteSession(msgspec.Struct):
    nonce: str
    prompt: tuple[int, ...]
    outputs: tuple[int, ...]
    revision: int = -1
    proposal: Proposal | None = None
    proposal_id: str = ""
    published_at: float = 0


class DraftClient:
    """Best-effort optional cache; failures never disable local drafting."""

    def __init__(
        self,
        *,
        endpoint: str | None,
        rank: int,
        fingerprint: str,
        block_size: int,
        num_anchors: int = 0,
        accept_counts: tuple[int, ...] = (),
        timeout_seconds: float = 30,
        lookup_wait_seconds: float = 0,
        wait_for_covered_result: bool = False,
        poll_interval_seconds: float = 0.001,
        queue_size: int = 1024,
    ) -> None:
        if (
            timeout_seconds <= 0
            or lookup_wait_seconds < 0
            or poll_interval_seconds <= 0
            or queue_size <= 0
        ):
            raise ValueError("optional client limits must be positive")
        if wait_for_covered_result and lookup_wait_seconds == 0:
            raise ValueError("covered-result wait requires a positive lookup limit")
        self.endpoint = endpoint
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.accept_counts = accept_counts or tuple(range(1, block_size + 1))
        self.fingerprint = fingerprint
        self.timeout_seconds = timeout_seconds
        self.lookup_wait_seconds = lookup_wait_seconds
        self.wait_for_covered_result = wait_for_covered_result
        self.poll_interval_seconds = poll_interval_seconds
        self.identity = f"{rank}:{uuid.uuid4().hex}"
        self.sessions: dict[str, RemoteSession] = {}
        self._outbound: queue.Queue[tuple[float, Request]] = queue.Queue(queue_size)
        self._ready: dict[
            tuple[str, int, str], tuple[float, dict[Branch, Proposal]]
        ] = {}
        self._inflight: dict[tuple[str, int, str], tuple[float, float]] = {}
        # One boundary protects mutable sessions and ready outcomes. Queue I/O
        # deliberately stays outside it so Scheduler methods remain bounded.
        self._lock = threading.Lock()
        self._result_ready = threading.Condition(self._lock)
        self._service_epoch = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: zmq.Context | None = None
        self._wakeup_rx: zmq.Socket | None = None
        self._wakeup_tx: zmq.Socket | None = None
        self.metrics = {
            "remote_prefix_stage_drop_ct": 0,
            "remote_optional_publish_drop_ct": 0,
            "remote_optional_result_ct": 0,
            "remote_optional_inflight_ct": 0,
            "remote_optional_inflight_wait_ct": 0,
            "remote_optional_ready_lookup_ct": 0,
            "remote_optional_lookup_hit_ct": 0,
            "remote_optional_lookup_ct": 0,
            "remote_optional_publish_to_lookup_seconds": 0,
            "remote_optional_publish_to_result_seconds": 0,
            "remote_optional_full_accept_lookup_ct": 0,
            "remote_optional_full_accept_ready_ct": 0,
            "remote_optional_full_accept_hit_ct": 0,
            "remote_optional_branch_miss_ct": 0,
            "remote_optional_branch_miss_length_ct": 0,
            "remote_optional_branch_miss_prefix_ct": 0,
            "remote_optional_branch_miss_full_accept_ct": 0,
            "remote_optional_branch_miss_candidate_rank_ct": 0,
            "remote_optional_branch_miss_missing_branch_ct": 0,
            "remote_optional_lookup_miss_ct": 0,
            "remote_optional_ready_wait_seconds": 0,
            "remote_optional_lookup_block_seconds": 0,
            "remote_optional_stale_result_ct": 0,
            "remote_optional_decode_error_ct": 0,
            "remote_optional_pump_wakeup_ct": 0,
        }
        if endpoint:
            # A wakeup PAIR socket lets the Scheduler thread interrupt an idle
            # blocking poll. Sub-millisecond polling without it costs about 5%
            # of Target throughput: 10k wakeups/s contend for the GIL with the
            # Scheduler thread. Each socket below is used by exactly one
            # thread after this construction.
            self._context = zmq.Context()
            address = f"inproc://decoupled-wakeup-{self.identity}"
            self._wakeup_rx = self._context.socket(zmq.PAIR)
            self._wakeup_rx.setsockopt(zmq.LINGER, 0)
            self._wakeup_rx.bind(address)
            self._wakeup_tx = self._context.socket(zmq.PAIR)
            self._wakeup_tx.setsockopt(zmq.LINGER, 0)
            self._wakeup_tx.connect(address)
            self._thread = threading.Thread(
                target=self._pump, name="decoupled-optional-pump", daemon=True
            )
            self._thread.start()

    def _signal_wakeup(self) -> None:
        if self._wakeup_tx is None:
            return
        try:
            self._wakeup_tx.send(b"\x00", zmq.NOBLOCK)
            self.metrics["remote_optional_pump_wakeup_ct"] += 1
        except zmq.ZMQError:
            pass

    def _enqueue(self, request: Request) -> bool:
        if not self.endpoint or self._stop.is_set():
            return False
        try:
            self._outbound.put_nowait((time.monotonic(), request))
        except queue.Full:
            self.metrics["remote_optional_publish_drop_ct"] += 1
            return False
        self._signal_wakeup()
        return True

    def lookup(
        self, *, rid: str, prompt: tuple[int, ...], outputs: tuple[int, ...]
    ) -> Proposal | None:
        """Consume an exact ready outcome for the previous local proposal."""
        divergent = False
        with self._lock:
            state = self.sessions.get(rid)
            if state is None or state.proposal is None:
                return None
            if prompt != state.prompt or outputs[: len(state.outputs)] != state.outputs:
                divergent = True
            else:
                delta = outputs[len(state.outputs) :]
                if not delta:
                    return None
                key = (state.nonce, state.revision, state.proposal_id)
                ready = self._ready.pop(key, None)
                previous = state.proposal
                self.metrics["remote_optional_lookup_ct"] += 1
                self.metrics["remote_optional_publish_to_lookup_seconds"] += (
                    time.monotonic() - state.published_at
                )
                count_key = f"remote_optional_accept_{len(delta)}_ct"
                self.metrics[count_key] = self.metrics.get(count_key, 0) + 1
        if divergent:
            self.finish(rid=rid)
            return None
        full_accept = (
            len(delta) == len(previous.tokens)
            and delta[:-1] == previous.tokens[1:]
        )
        if full_accept:
            self.metrics["remote_optional_full_accept_lookup_ct"] += 1
        if ready is None and self.lookup_wait_seconds > 0:
            wait_start = time.monotonic()
            inflight = self._inflight.get(key)
            can_use_branch = outcome_branch_is_covered(
                previous=previous,
                delta=delta,
                num_anchors=self.num_anchors,
                accept_counts=self.accept_counts,
            )
            deadline = wait_start
            if inflight is not None and can_use_branch:
                deadline = state.published_at + self.lookup_wait_seconds
                if not self.wait_for_covered_result:
                    deadline = min(deadline, inflight[0] + inflight[1])
            with self._result_ready:
                if deadline > wait_start:
                    self.metrics["remote_optional_inflight_wait_ct"] += 1
                while ready is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._result_ready.wait(remaining)
                    state = self.sessions.get(rid)
                    if state is None or state.proposal_id != key[2]:
                        break
                    ready = self._ready.pop(key, None)
            self.metrics["remote_optional_lookup_block_seconds"] += (
                time.monotonic() - wait_start
            )
        if ready is None:
            self.metrics["remote_optional_lookup_miss_ct"] += 1
            return None
        created, outcomes = ready
        if time.monotonic() - created > self.timeout_seconds:
            self.metrics["remote_optional_lookup_miss_ct"] += 1
            return None
        self.metrics["remote_optional_ready_lookup_ct"] += 1
        if full_accept:
            self.metrics["remote_optional_full_accept_ready_ct"] += 1
        self.metrics["remote_optional_ready_wait_seconds"] += (
            time.monotonic() - created
        )
        outcome = select_outcome(
            previous=previous, delta=delta, outcomes=outcomes
        )
        if outcome is None:
            self.metrics["remote_optional_branch_miss_ct"] += 1
            reason = classify_outcome_miss(
                previous=previous, delta=delta, outcomes=outcomes
            )
            self.metrics[f"remote_optional_branch_miss_{reason}_ct"] += 1
            rank_info = correction_token_rank(previous=previous, delta=delta)
            if rank_info is not None:
                rank, observed_topk = rank_info
                key = f"remote_optional_correction_rank_{rank}_of_{observed_topk}_ct"
                self.metrics[key] = self.metrics.get(key, 0) + 1
        else:
            self.metrics["remote_optional_lookup_hit_ct"] += 1
            if full_accept:
                self.metrics["remote_optional_full_accept_hit_ct"] += 1
        return outcome

    def publish(
        self,
        *,
        rid: str,
        prompt: tuple[int, ...],
        outputs: tuple[int, ...],
        proposal: Proposal,
    ) -> None:
        """Publish a Target-local proposal without waiting for acknowledgement."""
        close_request = None
        with self._lock:
            state = self.sessions.get(rid)
            if state is not None and (
                prompt != state.prompt
                or outputs[: len(state.outputs)] != state.outputs
            ):
                self.sessions.pop(rid)
                self._drop_ready_locked(state.nonce)
                close_request = self._close_request(state)
                state = None
            stage_request = None
            if state is None:
                state = RemoteSession(
                    nonce=uuid.uuid4().hex, prompt=prompt, outputs=()
                )
                self.sessions[rid] = state
                safe_prefix = prompt + outputs[:-1]
                if safe_prefix:
                    stage_request = Request(
                        rpc=uuid.uuid4().hex,
                        op="stage",
                        session=state.nonce,
                        revision=0,
                        committed_prefix=safe_prefix,
                        fingerprint=self.fingerprint,
                        block_size=self.block_size,
                        service_epoch=self._service_epoch,
                    )
            revision = state.revision + 1
            previous_id = state.proposal_id
            proposal_id = proposal_identity(proposal)
            request = self._publication_request(
                state=state,
                revision=revision,
                proposal=proposal,
                proposal_id=proposal_id,
                previous_id=previous_id,
                committed=prompt + outputs,
            )
            state.prompt = prompt
            state.outputs = outputs
            state.revision = revision
            state.proposal = proposal
            state.proposal_id = proposal_id
            state.published_at = time.monotonic()
        if close_request is not None:
            self._enqueue(close_request)
        if stage_request is not None and not self._enqueue(stage_request):
            self.metrics["remote_prefix_stage_drop_ct"] += 1
        self._enqueue(request)

    def finish(self, *, rid: str) -> None:
        with self._lock:
            state = self.sessions.pop(rid, None)
            if state is None:
                return
            self._drop_ready_locked(state.nonce)
            request = self._close_request(state)
        self._enqueue(request)

    def active_rids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self.sessions)

    def _drop_ready_locked(self, nonce: str) -> None:
        for key in [key for key in self._ready if key[0] == nonce]:
            self._ready.pop(key, None)
        self._inflight = {
            key: inflight
            for key, inflight in self._inflight.items()
            if key[0] != nonce
        }

    @staticmethod
    def _close_request(state: RemoteSession) -> Request:
        return Request(
            rpc=uuid.uuid4().hex,
            op="close",
            session=state.nonce,
            revision=state.revision,
        )

    def _publication_request(
        self,
        *,
        state: RemoteSession,
        revision: int,
        proposal: Proposal,
        proposal_id: str,
        previous_id: str,
        committed: tuple[int, ...],
    ) -> Request:
        return Request(
            rpc=uuid.uuid4().hex,
            op="publish",
            session=state.nonce,
            revision=revision,
            committed_prefix=committed,
            proposal=proposal,
            proposal_id=proposal_id,
            previous_proposal_id=previous_id,
            fingerprint=self.fingerprint,
            block_size=self.block_size,
            service_epoch=self._service_epoch,
        )

    def _pump(self) -> None:
        assert self._context is not None and self._wakeup_rx is not None
        context = self._context
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, self.identity.encode())
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, self._outbound.maxsize)
        socket.setsockopt(zmq.RCVHWM, self._outbound.maxsize)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.connect(self.endpoint)
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(Reply)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        poller.register(self._wakeup_rx, zmq.POLLIN)
        # This is only an idle safety net. Real work (publications, replies) is
        # delivered through the wakeup socket, so a long floor costs no
        # latency while avoiding thousands of GIL-contending wakeups per
        # second. It still bounds hello cadence and loop stop checking.
        idle_ms = max(50, int(round(self.poll_interval_seconds * 1000)))
        hello_at = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= hello_at:
                    self._send(
                        socket,
                        encoder,
                        Request(
                            rpc=uuid.uuid4().hex,
                            op="hello",
                            fingerprint=self.fingerprint,
                            block_size=self.block_size,
                        ),
                    )
                    hello_at = now + 1.0
                for _ in range(64):
                    try:
                        created, request = self._outbound.get_nowait()
                    except queue.Empty:
                        break
                    expired = (
                        request.op in ("stage", "publish")
                        and time.monotonic() - created > self.timeout_seconds
                    )
                    if expired or not self._send(socket, encoder, request):
                        self.metrics["remote_optional_publish_drop_ct"] += 1
                # Any progress must loop back to drain the outbound queue
                # before blocking again; otherwise a request enqueued between
                # the drain and the poll waits for the whole idle timeout.
                progressed = False
                events = dict(poller.poll(0))
                if self._wakeup_rx in events:
                    self._drain_wakeups()
                    progressed = True
                if socket in events:
                    self._drain_replies(socket, decoder)
                    progressed = True
                if progressed:
                    continue
                if self._stop.is_set():
                    break
                events = dict(poller.poll(idle_ms))
                if self._wakeup_rx in events:
                    self._drain_wakeups()
                if socket in events:
                    self._drain_replies(socket, decoder)
        finally:
            socket.close()
            self._wakeup_rx.close()

    def _drain_wakeups(self) -> None:
        assert self._wakeup_rx is not None
        while True:
            try:
                self._wakeup_rx.recv(zmq.NOBLOCK)
            except zmq.Again:
                return

    def _drain_replies(self, socket, decoder) -> None:
        while True:
            try:
                reply = decoder.decode(socket.recv(zmq.NOBLOCK))
            except zmq.Again:
                return
            except (msgspec.DecodeError, ValueError):
                self.metrics["remote_optional_decode_error_ct"] += 1
                return
            self._record(reply)

    @staticmethod
    def _send(socket, encoder, request: Request) -> bool:
        try:
            socket.send(encoder.encode(request), zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def _record(self, reply: Reply) -> None:
        if reply.version != VERSION:
            return
        republish = []
        # Only a negotiated hello can replace an epoch. Result events from an
        # older service are stale and must not roll the client back.
        is_hello = bool(reply.service_epoch and reply.fingerprint)
        with self._lock:
            if is_hello and reply.service_epoch != self._service_epoch:
                self._service_epoch = reply.service_epoch
                self._ready.clear()
                # Full snapshots make this safe both for first contact after
                # an unavailable startup and for a service restart.
                republish = [
                    self._publication_request(
                        state=state,
                        revision=state.revision,
                        proposal=state.proposal,
                        proposal_id=state.proposal_id,
                        previous_id="",
                        committed=state.prompt + state.outputs,
                    )
                    for state in self.sessions.values()
                    if state.proposal is not None
                ]
            if not reply.outcomes and not reply.inflight:
                pass
            elif reply.inflight:
                key = (reply.session, reply.revision, reply.proposal_id)
                self._inflight[key] = (
                    time.monotonic(), reply.expected_seconds
                )
                self.metrics["remote_optional_inflight_ct"] += 1
                self._result_ready.notify_all()
            elif (
                reply.service_epoch
                and self._service_epoch
                and reply.service_epoch != self._service_epoch
            ):
                self.metrics["remote_optional_stale_result_ct"] += 1
            else:
                key = (reply.session, reply.revision, reply.proposal_id)
                self._inflight.pop(key, None)
                live = next(
                    (
                        state
                        for state in self.sessions.values()
                        if state.nonce == reply.session
                        and state.revision == reply.revision
                        and state.proposal_id == reply.proposal_id
                    ),
                    None,
                )
                if live is None:
                    self.metrics["remote_optional_stale_result_ct"] += 1
                else:
                    now = time.monotonic()
                    self.metrics["remote_optional_publish_to_result_seconds"] += (
                        now - live.published_at
                    )
                    age_ms = (now - live.published_at) * 1000
                    bucket = next((bound for bound in (8, 10, 12, 16, 24, 32, 64)
                                   if age_ms <= bound), "inf")
                    metric = f"remote_optional_result_age_bucket_{bucket}ms_ct"
                    self.metrics[metric] = self.metrics.get(metric, 0) + 1
                    self._ready[key] = (
                        now,
                        {
                            outcome.branch: outcome.proposal
                            for outcome in reply.outcomes
                        },
                    )
                    self._result_ready.notify_all()
                    self.metrics["remote_optional_result_ct"] += 1
        for request in republish:
            self._enqueue(request)

    def close(self) -> None:
        if self._stop.is_set():
            return
        for rid in self.active_rids():
            self.finish(rid=rid)
        self._stop.set()
        self._signal_wakeup()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._wakeup_tx is not None:
            self._wakeup_tx.close()
            self._wakeup_tx = None
        if self._context is not None:
            self._context.term()
            self._context = None
