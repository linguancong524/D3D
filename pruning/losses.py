from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    weights = weights.to(values.dtype)
    numerator = (values * weights).sum()
    count = weights.sum()
    if not dist.is_available() or not dist.is_initialized():
        return numerator / count.clamp_min(1.0)
    global_count = count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    return numerator * dist.get_world_size() / global_count.clamp_min(1.0)


def _target_log_probability(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    logits = logits.float()
    target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
    return target_logits - torch.logsumexp(logits, dim=-1)


def target_log_probabilities(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    shape = targets.shape
    vocabulary = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary)
    flat_targets = targets.reshape(-1)
    chunks = []
    for start in range(0, len(flat_targets), chunk_size):
        stop = start + chunk_size
        chunks.append(
            checkpoint(
                _target_log_probability,
                flat_logits[start:stop],
                flat_targets[start:stop],
                use_reentrant=False,
            )
        )
    return torch.cat(chunks).reshape(shape)


def verifier_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return masked_mean(-target_log_probabilities(logits, targets), weights)


def _corrected_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    probability_margin: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    target_probability = teacher_probs.gather(-1, targets[:, None]).squeeze(-1)
    top_two = torch.topk(teacher_probs, k=2, dim=-1)
    best_other = torch.where(
        top_two.indices[:, 0].eq(targets),
        top_two.values[:, 1],
        top_two.values[:, 0],
    )
    deficit = best_other - target_probability
    correction = (deficit + probability_margin) / (1.0 + deficit).clamp_min(1e-8)
    correction = correction.clamp(0.0, 1.0)
    corrected = teacher_probs * (1.0 - correction[:, None])
    corrected = corrected.scatter_add(-1, targets[:, None], correction[:, None])
    corrected_log = corrected.clamp_min(1e-30).log()
    return (corrected * (corrected_log - student_log_probs)).sum(dim=-1)


def verifier_corrected_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    probability_margin: float = 0.01,
    chunk_size: int = 128,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    vocabulary = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocabulary)
    teacher = teacher_logits.reshape(-1, vocabulary)
    target = targets.reshape(-1)
    chunks = []
    for start in range(0, len(target), chunk_size):
        stop = start + chunk_size
        chunks.append(
            checkpoint(
                _corrected_kl,
                student[start:stop],
                teacher[start:stop],
                target[start:stop],
                temperature,
                probability_margin,
                use_reentrant=False,
            )
        )
    token_kl = torch.cat(chunks).reshape_as(targets)
    return masked_mean(token_kl, weights) * temperature**2


def _normalized_hidden_mse(
    student: torch.Tensor, teacher: torch.Tensor, epsilon: float
) -> torch.Tensor:
    student = student.float() * torch.rsqrt(
        student.float().square().mean(-1, keepdim=True) + epsilon
    )
    teacher = teacher.float() * torch.rsqrt(
        teacher.float().square().mean(-1, keepdim=True) + epsilon
    )
    return (student - teacher).square().mean(-1)


def boundary_hidden_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    position_weights: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    chunk_size: int = 128,
) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError(
            "Boundary states must have equal [anchor, layer, position, hidden] shapes"
        )
    expected = student.shape[:-1]
    if position_weights.shape == (expected[0], expected[2]):
        position_weights = position_weights[:, None, :].expand(expected)
    if position_weights.shape != expected:
        raise ValueError("Hidden-state weights have an incompatible shape")
    hidden_size = student.shape[-1]
    student_flat = student.reshape(-1, hidden_size)
    teacher_flat = teacher.detach().reshape(-1, hidden_size)
    chunks = []
    for start in range(0, len(student_flat), chunk_size):
        stop = start + chunk_size
        chunks.append(
            checkpoint(
                _normalized_hidden_mse,
                student_flat[start:stop],
                teacher_flat[start:stop],
                epsilon,
                use_reentrant=False,
            )
        )
    return masked_mean(torch.cat(chunks).reshape(expected), position_weights)


def hard_accept_length(
    predicted: torch.Tensor, targets: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    matches = predicted.eq(targets) & active.bool()
    accepted_tail = matches.to(torch.int64).cumprod(dim=-1)
    return (accepted_tail * active.to(torch.int64)).sum(dim=-1) + 1
