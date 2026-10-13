"""Optional-only packed draft service shared by Target replicas."""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict

import msgspec
import zmq

from .packed_layout import group_branches_by_accept_count
from .protocol import (
    VERSION,
    Backend,
    Branch,
    Commit,
    Outcome,
    Proposal,
    Reply,
    Request,
    proposal_identity,
)

logger = logging.getLogger(__name__)


class OptionalSession(msgspec.Struct, kw_only=True):
    backend_key: str
    prefix: tuple[int, ...]
    revision: int
    proposal: Proposal
    proposal_id: str
    touched: float


class OptionalDraftService:
    """Latest-revision packed speculation with no required drafting path."""

    def __init__(
        self,
        *,
        backend: Backend,
        max_sessions: int = 512,
        max_sessions_per_peer: int | None = None,
        max_context: int = 32768,
        lease_seconds: float = 300,
        batch_size: int = 64,
        speculation_batch_size: int = 8,
        batching_window_seconds: float = 0.001,
        **legacy,
    ) -> None:
        if legacy.get("mode") not in (None, "packed"):
            raise ValueError("remote required/sync mode was removed in protocol v4")
        if min(max_sessions, max_context, lease_seconds, batch_size, speculation_batch_size) <= 0:
            raise ValueError("resource limits must be positive")
        if batching_window_seconds < 0:
            raise ValueError("batching window must be nonnegative")
        self.backend = backend
        self.max_sessions = max_sessions
        self.max_sessions_per_peer = max_sessions_per_peer or max(
            1, max_sessions // 4
        )
        if not 0 < self.max_sessions_per_peer <= max_sessions:
            raise ValueError("per-peer session limit must fit global capacity")
        self.max_context = max_context
        self.lease_seconds = lease_seconds
        self.batch_size = batch_size
        self.speculation_batch_size = speculation_batch_size
        self.batching_window_seconds = batching_window_seconds
        self.service_epoch = uuid.uuid4().hex
        self.sessions: dict[tuple[bytes, str], OptionalSession] = {}
        self.staged: dict[tuple[bytes, str], tuple[str, tuple[int, ...], float]] = {}
        self.peers: dict[bytes, float] = {}
        self.speculation: OrderedDict[tuple[bytes, str], tuple[int, float]] = OrderedDict()
        self._shutdown = False
        self._forward_seconds_ewma = 0.0
        self.metrics: dict[str, float] = {
            "remote_prefix_stage_ct": 0,
            "remote_prefix_stage_seconds": 0,
            "remote_optional_publish_ct": 0,
            "remote_optional_supersede_ct": 0,
            "remote_optional_queue_seconds": 0,
            "remote_optional_batch_ct": 0,
            "remote_optional_batch_item_ct": 0,
            "remote_optional_batch_max": 0,
            "remote_optional_branch_ct": 0,
            "remote_optional_branch_group_ct": 0,
            "remote_optional_stale_completion_ct": 0,
            "remote_optional_seconds": 0,
            "error_ct": 0,
        }

    def handle(self, *, peer: bytes, request: Request) -> Reply:
        try:
            if request.version != VERSION:
                raise ValueError("protocol version mismatch")
            if request.op == "hello":
                if request.fingerprint != self.backend.fingerprint:
                    raise ValueError("target/draft tokenizer mappings differ")
                if request.block_size != self.backend.block_size:
                    raise ValueError("target/draft block sizes differ")
                self.peers[peer] = time.monotonic()
                return Reply(
                    rpc=request.rpc,
                    fingerprint=self.backend.fingerprint,
                    block_size=self.backend.block_size,
                    service_epoch=self.service_epoch,
                )
            if peer not in self.peers:
                raise ValueError("hello negotiation required")
            self.peers[peer] = time.monotonic()
            if request.op == "stats":
                return Reply(rpc=request.rpc, metrics=self.snapshot_metrics(), service_epoch=self.service_epoch)
            if not request.session:
                raise ValueError("session nonce is required")
            key = (peer, request.session)
            if request.op == "close":
                self._close(key)
                return Reply(rpc=request.rpc, session=request.session, revision=request.revision)
            if request.op == "commit":
                self._commit(key, request)
                return Reply(rpc=request.rpc, session=request.session, revision=request.revision)
            if request.op == "stage":
                self._stage(key, request)
                return Reply(rpc=request.rpc, session=request.session, revision=request.revision)
            if request.op != "publish":
                raise ValueError("unsupported optional operation")
            self._publish(key, request)
            return Reply(rpc=request.rpc, session=request.session, revision=request.revision)
        except Exception as exc:
            self.metrics["error_ct"] += 1
            return Reply(
                rpc=request.rpc,
                session=request.session,
                revision=request.revision,
                error=str(exc),
                service_epoch=self.service_epoch,
            )

    def _stage(self, key: tuple[bytes, str], request: Request) -> None:
        tokens = request.committed_prefix
        if request.revision != 0 or not tokens:
            raise ValueError("prefix stage requires revision zero and nonempty tokens")
        if any(token < 0 or token >= self.backend.vocab_size for token in tokens):
            raise ValueError("token outside draft vocabulary")
        if len(tokens) + self.backend.block_size > self.max_context:
            raise ValueError("prefix stage context budget exceeded")
        existing = self.staged.get(key)
        if existing is not None:
            if existing[1] != tokens:
                raise ValueError("conflicting duplicate prefix stage")
            return
        if key in self.sessions:
            return
        backend_key = uuid.uuid4().hex
        start = time.monotonic()
        self.backend.stage_many(items=[(backend_key, tokens)])
        self.staged[key] = (backend_key, tokens, time.monotonic())
        self.metrics["remote_prefix_stage_ct"] += 1
        self.metrics["remote_prefix_stage_seconds"] += time.monotonic() - start

    def _validate_publication(self, request: Request) -> Proposal:
        proposal = request.proposal
        if proposal is None:
            raise ValueError("publish requires a proposal")
        if proposal_identity(proposal) != request.proposal_id:
            raise ValueError("proposal identity mismatch")
        if not request.committed_prefix or proposal.tokens[0] != request.committed_prefix[-1]:
            raise ValueError("proposal anchor does not match committed bonus")
        if len(proposal.tokens) != self.backend.block_size:
            raise ValueError("proposal width mismatch")
        tokens = request.committed_prefix + proposal.tokens
        if any(token < 0 or token >= self.backend.vocab_size for token in tokens):
            raise ValueError("token outside draft vocabulary")
        if len(request.committed_prefix) + self.backend.block_size > self.max_context:
            raise ValueError("optional context budget exceeded")
        return proposal

    def _publish(self, key: tuple[bytes, str], request: Request) -> None:
        proposal = self._validate_publication(request)
        state = self.sessions.get(key)
        if state is not None and request.revision == state.revision:
            if request.proposal_id != state.proposal_id or request.committed_prefix != state.prefix:
                raise ValueError("conflicting duplicate revision")
            state.touched = time.monotonic()
            return
        if state is not None and request.revision == state.revision + 1:
            if request.previous_proposal_id != state.proposal_id:
                raise ValueError("previous proposal identity mismatch")
            self.backend.commit_many(
                items=[
                    Commit(
                        session=state.backend_key,
                        revision=request.revision,
                        proposal_id=state.proposal_id,
                        committed_prefix=request.committed_prefix,
                    )
                ]
            )
            backend_key = state.backend_key
        else:
            staged = self.staged.pop(key, None)
            if state is not None:
                self.backend.close(session=state.backend_key)
            elif staged is not None and request.revision == 0:
                backend_key, staged_tokens, _ = staged
                committed = request.committed_prefix[:-1]
                if committed[: len(staged_tokens)] != staged_tokens:
                    self.backend.close(session=backend_key)
                    raise ValueError("published prefix diverges from staged KV")
            else:
                peer_sessions = sum(
                    existing_peer == key[0]
                    for existing_peer, _ in (*self.sessions, *self.staged)
                )
                if peer_sessions >= self.max_sessions_per_peer:
                    raise ValueError("optional per-peer session capacity reached")
                if len(self.sessions) + len(self.staged) >= self.max_sessions:
                    raise ValueError("optional session capacity reached")
                backend_key = uuid.uuid4().hex
        now = time.monotonic()
        self.sessions[key] = OptionalSession(
            backend_key=backend_key,
            prefix=request.committed_prefix,
            revision=request.revision,
            proposal=proposal,
            proposal_id=request.proposal_id,
            touched=now,
        )
        if key in self.speculation:
            self.metrics["remote_optional_supersede_ct"] += 1
        self.speculation[key] = (request.revision, now)
        self.speculation.move_to_end(key)
        self.metrics["remote_optional_publish_ct"] += 1

    def _commit(self, key: tuple[bytes, str], request: Request) -> None:
        state = self.sessions.get(key)
        if state is None:
            return
        if request.revision != state.revision + 1 or request.proposal_id != state.proposal_id:
            raise ValueError("stale optional commit")
        self.backend.commit_many(
            items=[Commit(
                session=state.backend_key,
                revision=request.revision,
                proposal_id=request.proposal_id,
                committed_prefix=request.committed_prefix,
            )]
        )
        self.speculation.pop(key, None)
        state.revision = request.revision
        state.prefix = request.committed_prefix
        state.touched = time.monotonic()

    def _select_speculation_keys(
        self, *, limit: int
    ) -> list[tuple[tuple[bytes, str], int, float]]:
        """Select oldest work fairly across Target peers, then fill capacity."""
        pending = [
            (key, revision, queued_at)
            for key, (revision, queued_at) in self.speculation.items()
        ]
        selected = []
        selected_keys = set()
        selected_peers = set()
        for key, revision, queued_at in pending:
            if key[0] in selected_peers:
                continue
            selected.append((key, revision, queued_at))
            selected_keys.add(key)
            selected_peers.add(key[0])
            if len(selected) == limit:
                return selected
        for key, revision, queued_at in pending:
            if key in selected_keys:
                continue
            selected.append((key, revision, queued_at))
            if len(selected) == limit:
                break
        return selected

    def speculate_batch(
        self, *, force: bool = False, on_start=None
    ) -> list[tuple[bytes, Reply]]:
        if not self.speculation:
            return []
        now = time.monotonic()
        first_revision, first_at = next(iter(self.speculation.values()))
        del first_revision
        if not force and now - first_at < self.batching_window_seconds:
            return []
        selected: list[tuple[tuple[bytes, str], int, OptionalSession]] = []
        queue_seconds = 0.0
        for key, revision, queued_at in self._select_speculation_keys(
            limit=self.speculation_batch_size
        ):
            self.speculation.pop(key, None)
            state = self.sessions.get(key)
            if state is not None and state.revision == revision:
                selected.append((key, revision, state))
                queue_seconds += now - queued_at
        if not selected:
            return []
        self.metrics["remote_optional_batch_ct"] += 1
        self.metrics["remote_optional_queue_seconds"] += queue_seconds
        self.metrics["remote_optional_batch_item_ct"] += len(selected)
        self.metrics["remote_optional_batch_max"] = max(
            self.metrics["remote_optional_batch_max"], len(selected)
        )
        start = time.monotonic()
        if on_start is not None:
            expected_seconds = self._forward_seconds_ewma
            on_start(
                [
                    (
                        key[0],
                        Reply(
                            rpc="",
                            session=key[1],
                            revision=revision,
                            proposal_id=state.proposal_id,
                            inflight=True,
                            expected_seconds=expected_seconds,
                            service_epoch=self.service_epoch,
                        ),
                    )
                    for key, revision, state in selected
                ]
            )
        results = self.backend.speculate_many(
            items=[
                (state.backend_key, revision, state.prefix, state.proposal)
                for _, revision, state in selected
            ]
        )
        replies: list[tuple[bytes, Reply]] = []
        for key, revision, state in selected:
            current = self.sessions.get(key)
            if current is not state or current.revision != revision:
                self.metrics["remote_optional_stale_completion_ct"] += 1
                continue
            outcomes = results.get(state.backend_key, {})
            self.metrics["remote_optional_branch_ct"] += len(outcomes)
            self.metrics["remote_optional_branch_group_ct"] += len(
                group_branches_by_accept_count(tuple(outcomes))
            )
            replies.append(
                (
                    key[0],
                    Reply(
                        rpc="",
                        session=key[1],
                        revision=revision,
                        proposal_id=state.proposal_id,
                        outcomes=tuple(Outcome(branch, proposal) for branch, proposal in outcomes.items()),
                        service_epoch=self.service_epoch,
                    ),
                )
            )
        elapsed = time.monotonic() - start
        self.metrics["remote_optional_seconds"] += elapsed
        self._forward_seconds_ewma = (
            elapsed
            if self._forward_seconds_ewma == 0
            else 0.8 * self._forward_seconds_ewma + 0.2 * elapsed
        )
        return replies

    def snapshot_metrics(self) -> dict[str, float]:
        metrics = dict(self.metrics)
        batches = metrics["remote_optional_batch_ct"]
        metrics.update(
            active_sessions=float(len(self.sessions) + len(self.staged)),
            remote_prefix_staged=float(len(self.staged)),
            remote_optional_queue=float(len(self.speculation)),
            remote_optional_batch_mean=(
                metrics["remote_optional_batch_item_ct"] / batches if batches else 0.0
            ),
            packed_model_forward_seconds=float(
                getattr(self.backend, "packed_model_forward_seconds", 0)
            ),
            packed_build_seconds=float(
                getattr(self.backend, "packed_build_seconds", 0)
            ),
            packed_prepare_seconds=float(
                getattr(self.backend, "packed_prepare_seconds", 0)
            ),
            packed_postprocess_seconds=float(
                getattr(self.backend, "packed_postprocess_seconds", 0)
            ),
            packed_protocol_seconds=float(
                getattr(self.backend, "packed_protocol_seconds", 0)
            ),
            packed_physical_forward_count=float(
                getattr(self.backend, "packed_physical_forward_count", 0)
            ),
        )
        return metrics

    def expire(self) -> None:
        cutoff = time.monotonic() - self.lease_seconds
        for key, state in list(self.sessions.items()):
            if state.touched < cutoff:
                self._close(key)
        for key, (_, _, touched) in list(self.staged.items()):
            if touched < cutoff:
                self._close(key)
        active_peers = {peer for peer, _ in self.sessions} | {
            peer for peer, _ in self.staged
        }
        for peer, touched in list(self.peers.items()):
            if peer not in active_peers and touched < cutoff:
                self.peers.pop(peer, None)

    def _close(self, key: tuple[bytes, str]) -> None:
        self.speculation.pop(key, None)
        staged = self.staged.pop(key, None)
        state = self.sessions.pop(key, None)
        if state is not None:
            self.backend.close(session=state.backend_key)
        elif staged is not None:
            self.backend.close(session=staged[0])

    def close(self) -> None:
        if self._shutdown:
            return
        for key in list(self.sessions) + list(self.staged):
            self._close(key)
        self.backend.shutdown()
        self._shutdown = True


DraftService = OptionalDraftService


def serve(
    *,
    service: OptionalDraftService,
    endpoint: str,
    stop=None,
    metrics_interval: float = 10.0,
) -> None:
    """Serve control frames and emit unsolicited optional result events."""
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(Request)
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
    socket.setsockopt(zmq.SNDHWM, service.batch_size * 4)
    socket.setsockopt(zmq.RCVHWM, service.batch_size * 4)
    socket.bind(endpoint)
    next_metrics = time.monotonic() + metrics_interval

    def send_all(items: list[tuple[bytes, Reply]]) -> None:
        for peer, reply in items:
            try:
                socket.send_multipart([peer, encoder.encode(reply)], zmq.NOBLOCK)
            except zmq.Again:
                pass

    try:
        while stop is None or not stop.is_set():
            service.expire()
            now = time.monotonic()
            if metrics_interval > 0 and now >= next_metrics:
                logger.info("Optional draft service metrics: %s", service.snapshot_metrics())
                next_metrics = now + metrics_interval
            replies: list[tuple[bytes, Reply]] = []
            if socket.poll(1):
                for _ in range(service.batch_size):
                    try:
                        peer, payload = socket.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        request = decoder.decode(payload)
                    except (msgspec.DecodeError, ValueError) as exc:
                        replies.append((peer, Reply(rpc="", error=f"invalid envelope: {exc}")))
                        continue
                    replies.append((peer, service.handle(peer=peer, request=request)))
            send_all(replies)
            send_all(service.speculate_batch(on_start=send_all))
    finally:
        logger.info("Optional draft service metrics: %s", service.snapshot_metrics())
        service.close()
        socket.close()
        context.term()
