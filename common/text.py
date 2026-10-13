"""Prompt formatting, stop-token, and token-list helpers."""
from __future__ import annotations

from typing import Any, Iterable

import torch


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def eos_token_ids(tokenizer: Any, explicit: list[int] | None) -> list[int]:
    values: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        values.append(eos)
    elif isinstance(eos, (list, tuple)):
        values.extend(int(x) for x in eos)
    extra = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(extra, (list, tuple)):
        values.extend(int(x) for x in extra)
    # Keep legacy shell-provided stop IDs for Qwen3 while also honoring a
    # newer verifier tokenizer (Qwen3.6 uses 248046/248044).
    if explicit:
        values.extend(int(x) for x in explicit)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def contains_stop(tokens, stop_token_ids: Iterable[int] | None):
    if stop_token_ids is None:
        return False
    return any((tokens == int(stop_id)).any().item() for stop_id in stop_token_ids)


def trim_at_stop(output_ids, num_input_tokens: int, stop_token_ids: Iterable[int] | None):
    if stop_token_ids is None:
        return output_ids
    stops = torch.tensor(list(stop_token_ids), device=output_ids.device)
    hits = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(as_tuple=True)[0]
    if hits.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + hits[0].item() + 1]


def tokens_to_int_list(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if isinstance(tokens, torch.Tensor):
        return [int(t) for t in tokens.detach().cpu().tolist()]
    return [int(t) for t in tokens]
