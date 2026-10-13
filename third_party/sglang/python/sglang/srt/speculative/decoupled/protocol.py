"""Versioned request/reply envelope for the decoupled draft service.

Target publications carry full committed snapshots and authoritative local
proposals. A session nonce, RPC id, service epoch, and monotonic revision make
delayed replies, retries, service restarts, and reused request IDs unambiguous.
The optional ROUTER/DEALER service never participates in required drafting.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

import msgspec

# v4 makes the remote service optional-only. The Target publishes a proposal
# produced by its colocated SDAR worker; the service may asynchronously return
# packed outcomes, but it never produces a correctness-critical proposal.
VERSION = 4


class Request(msgspec.Struct, frozen=True, kw_only=True):
    rpc: str
    op: Literal["hello", "stage", "publish", "commit", "close", "stats"]
    session: str = ""
    revision: int = 0
    # Full snapshots deliberately make reconnect/service-epoch recovery
    # unambiguous. committed_prefix includes the current Target bonus/anchor.
    committed_prefix: tuple[int, ...] = ()
    proposal: Proposal | None = None
    proposal_id: str = ""
    previous_proposal_id: str = ""
    fingerprint: str = ""
    block_size: int = 0
    service_epoch: str = ""
    version: int = VERSION


class Proposal(msgspec.Struct, frozen=True):
    tokens: tuple[int, ...]
    top_tokens: tuple[tuple[int, ...], ...] = ()
    # Reserved for exact p/q sampling. v1 only negotiates greedy verification.
    support_tokens: tuple[tuple[int, ...], ...] = ()
    support_probs: tuple[tuple[float, ...], ...] = ()


class Outcome(msgspec.Struct, frozen=True):
    branch: Branch
    proposal: Proposal


class Reply(msgspec.Struct, frozen=True, kw_only=True):
    rpc: str
    session: str = ""
    revision: int = 0
    proposal_id: str = ""
    outcomes: tuple[Outcome, ...] = ()
    inflight: bool = False
    expected_seconds: float = 0
    error: str = ""
    fingerprint: str = ""
    block_size: int = 0
    service_epoch: str = ""
    metrics: dict[str, float] = {}
    version: int = VERSION


class Branch(msgspec.Struct, frozen=True):
    num_accept_tokens: int
    bonus_token: int | None


class Commit(msgspec.Struct, frozen=True, kw_only=True):
    session: str
    revision: int
    proposal_id: str
    committed_prefix: tuple[int, ...]


class Backend(Protocol):
    """Optional packed backend; required proposal generation is Target-local."""

    fingerprint: str
    vocab_size: int
    block_size: int

    def stage_many(self, *, items: list[tuple[str, tuple[int, ...]]]) -> None: ...

    def commit_many(self, *, items: list[Commit]) -> None: ...

    def speculate(
        self,
        *,
        session: str,
        revision: int,
        prefix: tuple[int, ...],
        proposal: Proposal,
    ) -> dict[Branch, Proposal]: ...

    def close(self, *, session: str) -> None: ...

    def shutdown(self) -> None: ...


def vocabulary_fingerprint(vocab: dict[str, int]) -> str:
    """Hash mappings, not just their size; equal vocab sizes are insufficient."""
    return hashlib.sha256(msgspec.json.encode(sorted(vocab.items()))).hexdigest()


def proposal_identity(proposal: Proposal) -> str:
    """Return a stable identity for revision-bound Target verification."""
    return hashlib.sha256(msgspec.msgpack.encode(proposal)).hexdigest()


def select_branches(
    *,
    proposal: Proposal,
    num_anchors: int,
    accept_counts: tuple[int, ...],
    max_branches: int,
) -> tuple[Branch, ...]:
    """Bound expansion before allocating an attention mask or model buffers."""
    if num_anchors < 0 or max_branches < 1:
        raise ValueError("num_anchors must be nonnegative; max_branches positive")
    size = len(proposal.tokens)
    branches: list[Branch] = []
    for count in dict.fromkeys(accept_counts):
        if not 1 <= count <= size:
            raise ValueError("accept counts must be within the verification block")
        if num_anchors == 0:
            branches.append(Branch(count, None))
        else:
            # A proposal contains ``size`` verification tokens. Native Engine
            # backends can additionally retain the logits for the token after
            # a fully accepted block, hence ``size + 1`` rows. Legacy SDAR
            # proposals only expose ``size`` rows and keep their full-accept
            # branch unanchored for compatibility.
            if count == size and len(proposal.top_tokens) == size:
                branches.append(Branch(count, None))
                if len(branches) >= max_branches:
                    break
                continue
            if len(proposal.top_tokens) not in (size, size + 1):
                raise ValueError("anchored speculation requires draft top tokens")
            proposed_token = proposal.tokens[count] if count < size else None
            alternatives = [
                t
                for t in dict.fromkeys(proposal.top_tokens[count])
                if t != proposed_token
            ][:num_anchors]
            branches.extend(Branch(count, t) for t in alternatives)
        if len(branches) >= max_branches:
            break
    return tuple(branches[:max_branches])


def correction_token_rank(
    *, previous: Proposal, delta: tuple[int, ...]
) -> tuple[int, int] | None:
    """Rank Target's correction in the observed Draft top-k at a rejection.

    The D3D count includes the previous Target anchor.  Consequently
    ``delta[:-1]`` is the accepted draft tail and ``delta[-1]`` is the next
    Target bonus/correction.  Full acceptance and non-contiguous commits do
    not describe an incorrect draft-token distribution and return ``None``.
    A rank of ``observed_topk + 1`` means the correction was outside the
    returned Draft candidates; its exact vocabulary rank is unknown.
    """
    count = len(delta)
    size = len(previous.tokens)
    if not 1 <= count < size or delta[:-1] != previous.tokens[1:count]:
        return None
    correction = delta[-1]
    if correction == previous.tokens[count] or len(previous.top_tokens) <= count:
        return None
    candidates = tuple(dict.fromkeys(previous.top_tokens[count]))
    if not candidates:
        return None
    try:
        return candidates.index(correction) + 1, len(candidates)
    except ValueError:
        return len(candidates) + 1, len(candidates)


def outcome_branch_is_covered(
    *,
    previous: Proposal,
    delta: tuple[int, ...],
    num_anchors: int,
    accept_counts: tuple[int, ...],
) -> bool:
    count = len(delta)
    if count not in accept_counts or not 1 <= count <= len(previous.tokens):
        return False
    if delta[:-1] != previous.tokens[1:count]:
        return False
    if num_anchors == 0:
        return True
    if count == len(previous.tokens) and len(previous.top_tokens) == len(
        previous.tokens
    ):
        return True
    if count >= len(previous.top_tokens):
        return False
    proposed_token = previous.tokens[count] if count < len(previous.tokens) else None
    alternatives = [
        token
        for token in dict.fromkeys(previous.top_tokens[count])
        if token != proposed_token
    ][:num_anchors]
    return delta[-1] in alternatives


def classify_outcome_miss(
    *, previous: Proposal, delta: tuple[int, ...], outcomes: dict[Branch, Proposal]
) -> str:
    count = len(delta)
    if not 1 <= count <= len(previous.tokens):
        return "length"
    if delta[:-1] != previous.tokens[1:count]:
        return "prefix"
    if Branch(count, delta[-1]) in outcomes or Branch(count, None) in outcomes:
        return "none"
    if count == len(previous.tokens):
        return "full_accept"
    candidates = (
        previous.top_tokens[count]
        if count < len(previous.top_tokens)
        else ()
    )
    return "candidate_rank" if delta[-1] not in candidates else "missing_branch"


def select_outcome(
    *, previous: Proposal, delta: tuple[int, ...], outcomes: dict[Branch, Proposal]
) -> Proposal | None:
    """A commit advances by correct tail tokens plus the next target bonus.

    D3D's key counts the *previous* anchor plus correct tail. Its numeric value
    equals len(delta), although delta contains the *next* bonus instead.
    Rejecting a divergent prefix must invalidate every optimistic branch.
    """
    count = len(delta)
    if not 1 <= count <= len(previous.tokens):
        return None
    if delta[:-1] != previous.tokens[1:count]:
        return None
    outcome = outcomes.get(Branch(count, delta[-1]))
    if outcome is None:
        outcome = outcomes.get(Branch(count, None))
    if outcome is None:
        return None
    return msgspec.structs.replace(outcome, tokens=(delta[-1],) + outcome.tokens[1:])
