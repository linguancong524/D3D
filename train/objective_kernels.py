"""Per-token CE/KL and top-k auxiliary losses for offline D3D training."""

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
