"""Confidence-gated CE/KL distillation loss for AR->block-diffusion OPD.

All functions operate on the *flattened masked-token* tensors produced by the
training step (one row per supervised response token):

    student_logits : [N, V]   (grad)
    teacher_logits : [N, V]   (no grad, from the AR teacher's clean forward)

The gate uses the teacher's top-1 softmax probability `c` per token to blend a
hard cross-entropy term (teacher confident -> distill its hard label) with a
forward-KL term (teacher uncertain -> cover its full distribution):

    c >= hi              -> CE only
    c <= lo              -> forward-KL only
    lo < c < hi          -> w_ce * CE + w_kl * KL, w_ce ramps lo..hi

Gate weights are a detached function of the teacher; gradient flows only
through `student_logits`.
"""

import torch
import torch.nn.functional as F


def forward_kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Full-vocab forward KL ``KL(p_teacher || p_student)`` per token, no temperature, no T^2.

    Returns a [N] tensor (not reduced). Mode-covering: spreads student mass over
    the teacher's full distribution. The teacher distribution is a frozen target,
    so it is detached here; gradient flows only through log p_s (student). This
    makes the term equal to soft-label CE against the teacher up to a
    student-independent constant.
    """
    with torch.no_grad():
        teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    # sum_v p_t * (log p_t - log p_s); p_t and log p_t are constants w.r.t. student.
    per_token = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1)
    return per_token


def reverse_kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Full-vocab reverse KL ``KL(p_student || p_teacher)`` per token.

    Mode-seeking: penalizes student mass on tokens the teacher assigns low
    probability to. The teacher distribution is detached; gradients flow through
    both ``p_student`` and ``log p_student``.
    """
    with torch.no_grad():
        teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    student_prob = student_log_prob.exp()
    return (student_prob * (student_log_prob - teacher_log_prob)).sum(dim=-1)


def hard_ce_per_token(student_logits: torch.Tensor, teacher_argmax: torch.Tensor) -> torch.Tensor:
    """Hard cross-entropy against the teacher's argmax token. Returns [N]."""
    return F.cross_entropy(student_logits.float(), teacher_argmax.long(), reduction="none")


def topk_target_margin_per_token(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int = 16,
    margin: float = 1.0,
    temperature: float = 1.0,
    smooth: bool = True,
) -> torch.Tensor:
    """Pairwise top-k margin loss against the strongest non-target logits.

    For each supervised token, take the top-k non-target logits and penalize
    competitors that are too close to or above the target logit:

        other_j - target + margin

    ``smooth=True`` uses a softplus penalty, which spreads gradient across the
    selected top-k competitors without the hard kink of hinge loss. ``smooth=False``
    uses ReLU hinge. The function returns one unreduced loss value per token.
    """
    if student_logits.ndim != 2:
        raise ValueError(f"student_logits must be [N, V], got {tuple(student_logits.shape)}")
    if target_ids.ndim != 1 or target_ids.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"target_ids must be [N] aligned to student_logits; got "
            f"{tuple(target_ids.shape)} vs {tuple(student_logits.shape)}"
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    logits = student_logits.float()
    n_tokens, vocab_size = int(logits.shape[0]), int(logits.shape[1])
    if n_tokens == 0:
        return logits.sum(dim=-1)
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1 for non-target margin, got {vocab_size}")

    target_ids = target_ids.to(device=logits.device).long()
    target_logits = logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    k = min(int(top_k), vocab_size - 1)
    # top-k+1 is enough to recover the exact top-k non-target logits without
    # materializing a full [N, V] one-hot mask or cloned masked-logit tensor.
    candidate_k = min(k + 1, vocab_size)
    candidate_values, candidate_ids = torch.topk(logits, k=candidate_k, dim=-1)
    candidate_values = candidate_values.masked_fill(
        candidate_ids == target_ids.unsqueeze(-1),
        -torch.inf,
    )
    topk_other = torch.topk(candidate_values, k=k, dim=-1).values
    violations = topk_other - target_logits.unsqueeze(-1) + float(margin)
    if smooth:
        per_competitor = F.softplus(violations / float(temperature)) * float(temperature)
    else:
        per_competitor = F.relu(violations)
    return per_competitor.mean(dim=-1)


def topk_hard_negative_ce_per_token(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int = 16,
    margin: float = 0.0,
) -> torch.Tensor:
    """Hard-negative CE over target plus top-k non-target student logits.

    This is the local-softmax loss

        -log exp(z_y) / (exp(z_y) + sum_{v in TopK negatives} exp(z_v + margin))

    computed as ``logsumexp([0, z_v - z_y + margin])`` for stability. It operates
    on raw logits, but is shift-invariant like a softmax probability loss.
    """
    if student_logits.ndim != 2:
        raise ValueError(f"student_logits must be [N, V], got {tuple(student_logits.shape)}")
    if target_ids.ndim != 1 or target_ids.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"target_ids must be [N] aligned to student_logits; got "
            f"{tuple(target_ids.shape)} vs {tuple(student_logits.shape)}"
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    logits = student_logits.float()
    n_tokens, vocab_size = int(logits.shape[0]), int(logits.shape[1])
    if n_tokens == 0:
        return logits.sum(dim=-1)
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be > 1 for hard-negative CE, got {vocab_size}")

    target_ids = target_ids.to(device=logits.device).long()
    target_logits = logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    k = min(int(top_k), vocab_size - 1)
    candidate_k = min(k + 1, vocab_size)
    candidate_values, candidate_ids = torch.topk(logits, k=candidate_k, dim=-1)
    candidate_values = candidate_values.masked_fill(
        candidate_ids == target_ids.unsqueeze(-1),
        -torch.inf,
    )
    topk_other = torch.topk(candidate_values, k=k, dim=-1).values
    negative_terms = topk_other - target_logits.unsqueeze(-1) + float(margin)
    target_term = torch.zeros((n_tokens, 1), dtype=negative_terms.dtype, device=negative_terms.device)
    return torch.logsumexp(torch.cat([target_term, negative_terms], dim=-1), dim=-1)


def survival_accept_weights(
    teacher_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    student_logits: torch.Tensor | None = None,
    response_positions: torch.Tensor | None,
    block_size: int,
    mode: str = "ratio",
    weight_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-grad ratio survival weights for accept-length-aware KL.

    ``alpha_j`` is the speculative ratio ``min(1, p_teacher(d_j)/p_student(d_j))``.
    The returned weight for position ``i`` is ``s_{i-1}=prod_{j<i} alpha_j``
    within each draft block. Blocks reset when the response-internal position
    wraps back to 0.
    """
    mode = str(mode).lower()
    if mode != "ratio":
        raise ValueError(f"survival weight mode is fixed to 'ratio', got {mode!r}")
    if teacher_logits.shape[0] != draft_token_ids.shape[0]:
        raise ValueError(
            f"teacher logits and draft token ids must align in N; got "
            f"{teacher_logits.shape[0]} vs {draft_token_ids.shape[0]}"
        )
    if response_positions is not None and response_positions.shape[0] != draft_token_ids.shape[0]:
        raise ValueError(
            f"response_positions and draft token ids must align in N; got "
            f"{response_positions.shape[0]} vs {draft_token_ids.shape[0]}"
        )
    if student_logits is None:
        raise ValueError("student_logits are required for ratio survival weights")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must match for ratio survival; got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    floor = float(weight_floor)
    if floor < 0.0 or floor > 1.0:
        raise ValueError(f"survival weight floor must be in [0, 1], got {floor}")

    with torch.no_grad():
        draft_token_ids = draft_token_ids.to(teacher_logits.device).long()
        p = F.softmax(teacher_logits.float(), dim=-1).gather(
            dim=-1,
            index=draft_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        q = F.softmax(student_logits.detach().float(), dim=-1).gather(
            dim=-1,
            index=draft_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        alpha = (p / q.clamp_min(1e-12)).clamp(max=1.0)

        n = int(alpha.shape[0])
        if n == 0:
            return alpha.clone(), alpha
        if response_positions is None:
            response_positions = torch.arange(n, device=teacher_logits.device)
        else:
            response_positions = response_positions.to(teacher_logits.device)
        draft_span = max(int(block_size) - 1, 1)
        local_pos = torch.remainder(response_positions.long(), draft_span)
        reset = torch.ones(n, dtype=torch.bool, device=teacher_logits.device)
        if n > 1:
            reset[1:] = local_pos[1:] <= local_pos[:-1]
        starts = torch.nonzero(reset, as_tuple=False).flatten().tolist()
        starts.append(n)

        weights = torch.empty_like(alpha)
        for i in range(len(starts) - 1):
            start, end = int(starts[i]), int(starts[i + 1])
            seg_alpha = alpha[start:end]
            seg_weights = torch.ones_like(seg_alpha)
            if seg_alpha.numel() > 1:
                seg_weights[1:] = torch.cumprod(seg_alpha[:-1], dim=0)
            weights[start:end] = seg_weights
        if floor > 0.0:
            weights = weights.clamp_min(floor)
    return weights.detach(), alpha.detach()


def first_reject_truncation_mask(
    teacher_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    response_positions: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    """Per-block keep mask that zeros positions strictly after the first reject.

    Under greedy / DFlash acceptance a draft token ``d_i`` is accepted iff it
    equals the teacher argmax at that position. This returns ``keep_i`` where,
    within each draft block, ``keep_i = prod_{j<i} 1[d_j == teacher_argmax_j]``.
    The first reject position itself is kept (its target is conditioned only on
    accepted tokens, so it is clean and is the highest-value +1-accept signal);
    every position after it is zeroed because its teacher target is conditioned
    on an already-rejected prefix that the verifier never reaches at inference.

    This is the {0, 1} hard analogue of :func:`survival_accept_weights` and uses
    the same block-reset segmentation, so multiplying it into the survival
    weights truncates each block at its first reject.
    """
    with torch.no_grad():
        draft_token_ids = draft_token_ids.to(teacher_logits.device).long()
        teacher_argmax = teacher_logits.float().argmax(dim=-1)
        match = (draft_token_ids == teacher_argmax)

        n = int(match.shape[0])
        if n == 0:
            return match.float()
        if response_positions is None:
            response_positions = torch.arange(n, device=teacher_logits.device)
        else:
            response_positions = response_positions.to(teacher_logits.device)
        draft_span = max(int(block_size) - 1, 1)
        local_pos = torch.remainder(response_positions.long(), draft_span)
        reset = torch.ones(n, dtype=torch.bool, device=teacher_logits.device)
        if n > 1:
            reset[1:] = local_pos[1:] <= local_pos[:-1]
        starts = torch.nonzero(reset, as_tuple=False).flatten().tolist()
        starts.append(n)

        keep = torch.ones(n, dtype=torch.float32, device=teacher_logits.device)
        for i in range(len(starts) - 1):
            start, end = int(starts[i]), int(starts[i + 1])
            seg_match = match[start:end]
            seg_keep = torch.ones_like(seg_match, dtype=torch.float32)
            if seg_match.numel() > 1:
                seg_keep[1:] = torch.cumprod(seg_match[:-1].float(), dim=0)
            keep[start:end] = seg_keep
    return keep.detach()


def survival_weighted_forward_kl_components(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    draft_token_ids: torch.Tensor,
    response_positions: torch.Tensor | None,
    block_size: int,
    truncate_after_first_reject: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Unreduced survival-weighted forward-KL pieces.

    The caller can all-reduce ``weight_sum`` across blocks/ranks and then backprop
    ``weighted_sum / global_weight_sum``. This is required for the stage-2
    objective ``sum_i s_i KL_i / sum_i s_i`` when blocks are streamed one at a
    time to avoid retaining a full 4096-token graph.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must match in [N, V]; got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if draft_token_ids is None:
        raise ValueError("draft_token_ids are required for survival-weighted KL")
    n_tokens = int(student_logits.shape[0])
    if n_tokens == 0:
        zero = student_logits.reshape(-1)[:1].float().sum() * 0.0
        return zero, zero.detach(), {
            "num_tokens": 0,
            "raw_forward_kl": 0.0,
            "raw_survival_forward_kl": 0.0,
            "survival_weight_mean": 0.0,
            "survival_alpha_mean": 0.0,
            "top1_match": 0.0,
        }

    survival_weights, survival_alpha = survival_accept_weights(
        teacher_logits,
        draft_token_ids,
        student_logits=student_logits,
        response_positions=response_positions,
        block_size=block_size,
    )
    survival_weights = survival_weights.to(student_logits.device)
    if truncate_after_first_reject:
        keep = first_reject_truncation_mask(
            teacher_logits,
            draft_token_ids,
            response_positions=response_positions,
            block_size=block_size,
        ).to(device=student_logits.device, dtype=survival_weights.dtype)
        survival_weights = survival_weights * keep
    weight_sum = survival_weights.sum()
    active = survival_weights > 0
    if bool(active.any()):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        forward_kl = forward_kl_per_token(
            student_logits.index_select(0, active_idx),
            teacher_logits.index_select(0, active_idx),
        )
        active_weights = survival_weights.index_select(0, active_idx).to(forward_kl.device)
        weighted_sum = (forward_kl * active_weights).sum()
        raw_forward_kl = float(forward_kl.mean().detach())
        raw_survival_forward_kl = float(
            (weighted_sum / weight_sum.clamp_min(1e-8)).detach()
        )
    else:
        weighted_sum = student_logits.reshape(-1)[:1].float().sum() * 0.0
        raw_forward_kl = 0.0
        raw_survival_forward_kl = 0.0

    with torch.no_grad():
        teacher_argmax = teacher_logits.float().argmax(dim=-1)
        diagnostics = {
            "num_tokens": n_tokens,
            "raw_forward_kl": raw_forward_kl,
            "raw_survival_forward_kl": raw_survival_forward_kl,
            "survival_weight_mean": float(survival_weights.mean()),
            "survival_alpha_mean": float(survival_alpha.mean()),
            "top1_match": float((student_logits.argmax(dim=-1) == teacher_argmax).float().mean()),
        }
    return weighted_sum, weight_sum.detach(), diagnostics


def confidence_weights(
    c: torch.Tensor,
    lo: float = 0.3,
    hi: float = 0.9,
    shape: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise gate weights from teacher top-1 prob ``c`` ([N]).

    ``w_ce = clamp((c - lo) / (hi - lo), 0, 1)``, ``w_kl = 1 - w_ce``.
    Endpoints: ``c >= hi -> (1, 0)``, ``c <= lo -> (0, 1)``, midpoint -> (0.5, 0.5).
    ``shape`` is a hook for future ramps (e.g. smoothstep); only "linear" now.
    """
    if hi <= lo:
        raise ValueError(f"confidence gate requires hi > lo, got lo={lo}, hi={hi}")
    ramp = ((c - lo) / (hi - lo)).clamp(0.0, 1.0)
    if shape == "linear":
        w_ce = ramp
    elif shape == "smoothstep":
        w_ce = ramp * ramp * (3.0 - 2.0 * ramp)
    else:
        raise ValueError(f"Unknown confidence gate shape: {shape!r}")
    return w_ce, 1.0 - w_ce


def context_weights(deficit: torch.Tensor, scale: float) -> torch.Tensor:
    """Context gate from the student's left-context deficit ([N]).

    ``w_ctx = clamp(1 - deficit / scale, 0, 1)``. ``deficit`` is the number of
    left in-block neighbors the student is missing relative to the AR teacher at
    this token's commit round (``trajectory_state.context_deficit``); ``scale``
    (= ``D``, the TAD-style look-ahead window) sets how fast hard-CE weight
    decays with that deficit:

        deficit == 0       -> w_ctx = 1   (student saw the teacher's left context)
        deficit >= scale   -> w_ctx = 0   (teacher's label leans on unseen context)

    Multiplied into the confidence gate's ``w_ce`` so hard CE fires only where the
    teacher is confident AND the student has the context to reproduce it; the
    freed weight goes to forward KL (TAD's distant / soft-KL regime).
    """
    if scale <= 0:
        raise ValueError(f"context gate requires scale > 0, got {scale}")
    return (1.0 - deficit.float() / scale).clamp(0.0, 1.0)


def confidence_gated_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lo: float = 0.3,
    hi: float = 0.9,
    shape: str = "linear",
    context_deficit: torch.Tensor | None = None,
    context_scale: float = 16.0,
) -> tuple[torch.Tensor, dict]:
    """Per-token confidence-gated mix of hard CE and forward KL, mean over N tokens.

    Returns ``(loss_scalar, diagnostics)``. The teacher distribution `c`,
    `teacher_argmax`, and `log p_t` are all derived from the same teacher logits
    under no_grad; the gate carries no gradient.

    When ``context_deficit`` ([N], aligned to the same masked tokens as the
    logits) is given, the confidence gate's CE weight is additionally scaled by
    the context gate ``context_weights(context_deficit, context_scale)``:
    ``w_ce = w_ce_conf * w_ctx``, ``w_kl = 1 - w_ce``. Passing ``None`` (default)
    leaves the loss byte-identical to the confidence-only gate.

    Fused, low-peak-memory implementation: mathematically identical to
    ``w_ce * hard_ce_per_token + w_kl * forward_kl_per_token``, but computes the student
    log-softmax ONCE (shared by both terms) instead of once per fn. At V=151936
    each [N, V] f32 tensor is ~2.5 GB; calling both per-token fns transiently
    holds ~7 of them (two student log-softmaxes both kept for backward, plus
    several teacher copies) and OOMs at N~4096. This path holds at most ~3.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must match in [N, V]; got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )

    if student_logits.numel() == 0:
        # Sum-times-zero keeps a grad_fn (when student_logits.requires_grad) so the
        # caller's unconditional loss.backward() does not raise on an N==0 round; a
        # plain new_zeros() is a leaf with no grad_fn. Currently dormant (the collective
        # empty-rollout skip guarantees N>=1), but cheap insurance against future
        # refactors that loosen that invariant.
        zero = student_logits.float().sum() * 0.0
        return zero, {"num_tokens": 0}

    with torch.no_grad():
        teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_prob = teacher_log_prob.exp()
        c, teacher_argmax = teacher_prob.max(dim=-1)
        w_ce_conf, _ = confidence_weights(c, lo=lo, hi=hi, shape=shape)
        if context_deficit is not None:
            w_ctx = context_weights(context_deficit.to(c.device), context_scale)
            w_ce = w_ce_conf * w_ctx
        else:
            w_ctx = None
            w_ce = w_ce_conf
        w_kl = 1.0 - w_ce
        # The student-independent half of forward KL: sum_v p_t * log p_t.
        teacher_self_term = (teacher_prob * teacher_log_prob).sum(dim=-1)
        del teacher_log_prob  # free ~2.5 GB before the student forward

    # ONE student log-softmax, shared by CE and KL (this is the [N, V] tensor
    # autograd keeps for backward; computing it twice is what doubled live mem).
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    # hard CE vs teacher argmax == -log p_s at the argmax (== F.cross_entropy).
    ce = -student_log_prob.gather(1, teacher_argmax.unsqueeze(1)).squeeze(1)
    # forward KL = teacher_self_term - sum_v p_t * log p_s.
    kl = teacher_self_term - (teacher_prob * student_log_prob).sum(dim=-1)
    per_token = w_ce * ce + w_kl * kl
    loss = per_token.mean()

    with torch.no_grad():
        diagnostics = {
            "num_tokens": int(student_logits.shape[0]),
            "mean_c": float(c.mean()),
            "mean_w_ce": float(w_ce.mean()),
            "raw_ce": float(ce.mean()),
            "raw_kl": float(kl.mean()),
            "frac_ce_only": float((c >= hi).float().mean()),
            "frac_kl_only": float((c <= lo).float().mean()),
        }
        if w_ctx is not None:
            diagnostics["mean_w_ctx"] = float(w_ctx.mean())
            diagnostics["mean_deficit"] = float(context_deficit.float().mean())
    return loss, diagnostics
