from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpeculativeSampleResult:
    accepted_draft_tokens: int
    correction_token: torch.Tensor
    all_accepted: bool


def verify_speculative_block(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> SpeculativeSampleResult:
    """Verify a sampled draft block with standard p/q rejection sampling.

    ``draft_tokens`` has shape [B, K]. ``target_probs`` has shape [B, K+1, V],
    where the last row is the target bonus distribution. ``draft_probs`` has
    shape [B, K, V] and is the exact proposal distribution used for sampling
    each draft token.
    """
    if draft_tokens.ndim != 2 or target_probs.ndim != 3 or draft_probs.ndim != 3:
        raise ValueError("expected draft_tokens [B,K], target_probs/draft_probs [B,K,V]")
    batch, draft_len = draft_tokens.shape
    if target_probs.shape[0] != batch or target_probs.shape[1] != draft_len + 1:
        raise ValueError("target_probs must contain K draft rows plus one bonus row")
    if draft_probs.shape[0] != batch or draft_probs.shape[1] != draft_len:
        raise ValueError("draft_probs must contain one row per draft token")
    if target_probs.shape[-1] != draft_probs.shape[-1]:
        raise ValueError("target and draft vocabularies must match")

    p = target_probs.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = draft_probs.float()
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # D3DV2 currently supports batch size one. Keeping the implementation
    # tensorized over the vocabulary makes it directly reusable by async.
    if batch != 1:
        raise ValueError("standard verification currently supports batch size 1")

    token_ids = draft_tokens[0].long()
    positions = torch.arange(draft_len, device=token_ids.device)
    p_token = p[0, positions, token_ids]
    q_token = q[0, positions, token_ids]
    acceptance_probability = torch.minimum(
        torch.ones_like(p_token),
        p_token / q_token.clamp_min(1e-12),
    )
    uniforms = torch.rand_like(acceptance_probability)
    accepted = uniforms <= acceptance_probability
    rejected = (~accepted).nonzero(as_tuple=True)[0]

    if rejected.numel() == 0:
        accepted_count = draft_len
        correction_probs = p[0, draft_len]
        all_accepted = True
    else:
        accepted_count = int(rejected[0].item())
        position = accepted_count
        residual = torch.clamp(p[0, position] - q[0, position], min=0.0)
        mass = residual.sum()
        correction_probs = p[0, position] if float(mass.item()) <= 1e-12 else residual / mass
        all_accepted = False

    correction_token = torch.multinomial(correction_probs.reshape(1, -1), 1).reshape(1)
    return SpeculativeSampleResult(
        accepted_draft_tokens=accepted_count,
        correction_token=correction_token,
        all_accepted=all_accepted,
    )


def verify_speculative_block_sparse(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_token_probs: torch.Tensor,
) -> SpeculativeSampleResult:
    """Verify against an exact sparse proposal distribution."""
    if draft_tokens.ndim != 2 or target_probs.ndim != 3:
        raise ValueError("expected draft_tokens [B,K] and target_probs [B,K+1,V]")
    if draft_token_ids.ndim != 3 or draft_token_probs.ndim != 3:
        raise ValueError("expected sparse draft ids/probs [B,K,S]")
    batch, draft_len = draft_tokens.shape
    if batch != 1:
        raise ValueError("standard verification currently supports batch size 1")
    if target_probs.shape[:2] != (batch, draft_len + 1):
        raise ValueError("target_probs must contain K draft rows plus one bonus row")
    if draft_token_ids.shape[:2] != (batch, draft_len):
        raise ValueError("sparse draft support must contain one row per draft token")
    if draft_token_probs.shape != draft_token_ids.shape:
        raise ValueError("sparse draft ids and probs must have identical shapes")

    p = target_probs.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q_probs = draft_token_probs.float()
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q_ids = draft_token_ids.long()

    token_ids = draft_tokens[0].long()
    positions = torch.arange(draft_len, device=token_ids.device)
    p_token = p[0, positions, token_ids]
    support_matches = q_ids[0] == token_ids.unsqueeze(-1)
    q_token = (q_probs[0] * support_matches).sum(dim=-1)
    acceptance_probability = torch.minimum(
        torch.ones_like(p_token),
        p_token / q_token.clamp_min(1e-12),
    )
    accepted = torch.rand_like(acceptance_probability) <= acceptance_probability
    rejected = (~accepted).nonzero(as_tuple=True)[0]

    if rejected.numel() == 0:
        accepted_count = draft_len
        correction_probs = p[0, draft_len]
        all_accepted = True
    else:
        accepted_count = int(rejected[0].item())
        correction_probs = p[0, accepted_count].clone()
        correction_probs.scatter_add_(
            0,
            q_ids[0, accepted_count],
            -q_probs[0, accepted_count],
        )
        correction_probs.clamp_min_(0.0)
        mass = correction_probs.sum()
        if float(mass.item()) <= 1e-12:
            correction_probs = p[0, accepted_count]
        else:
            correction_probs /= mass
        all_accepted = False

    correction_token = torch.multinomial(correction_probs.reshape(1, -1), 1).reshape(1)
    return SpeculativeSampleResult(
        accepted_draft_tokens=accepted_count,
        correction_token=correction_token,
        all_accepted=all_accepted,
    )
