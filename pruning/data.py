from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from .common import MASK_TOKEN_ID, causal_mask, stable_uint64


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass(frozen=True)
class AnchorBatch:
    clean_input_ids: torch.Tensor
    clean_attention_mask: torch.Tensor
    prompt_length: int
    draft_input_ids: torch.Tensor
    draft_attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logit_keep_indices: torch.Tensor
    active_token_mask: torch.Tensor
    target_ids: torch.Tensor


@dataclass(frozen=True)
class DataWindow:
    cursor_start: int
    cursor_end: int
    candidate_rows: int
    required_examples: int
    gradient_accumulation_steps: int
    world_size: int
    candidate_multiplier: float
    seed: int
    dataset_rows: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def make_data_window(
    *,
    cursor: int,
    steps: int,
    gradient_accumulation_steps: int,
    world_size: int,
    candidate_multiplier: float,
    seed: int,
    dataset_rows: int,
) -> DataWindow:
    if cursor < 0 or steps <= 0 or dataset_rows <= 0:
        raise ValueError("cursor, steps, and dataset_rows are invalid")
    if gradient_accumulation_steps <= 0 or world_size <= 0:
        raise ValueError("gradient accumulation and world size must be positive")
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be at least one")
    required = steps * gradient_accumulation_steps * world_size
    candidates = max(required, math.ceil(required * candidate_multiplier))
    return DataWindow(
        cursor_start=cursor,
        cursor_end=cursor + candidates,
        candidate_rows=candidates,
        required_examples=required,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        candidate_multiplier=float(candidate_multiplier),
        seed=seed,
        dataset_rows=dataset_rows,
    )


def _coprime_multiplier(size: int, seed: int, cycle: int) -> int:
    if size == 1:
        return 0
    candidate = stable_uint64(str(cycle), f"d3d-prune-affine-{seed}") % size or 1
    while math.gcd(candidate, size) != 1:
        candidate = (candidate + 1) % size or 1
    return int(candidate)


def permuted_row(position: int, rows: int, seed: int) -> int:
    cycle, offset = divmod(position, rows)
    if rows == 1:
        return 0
    multiplier = _coprime_multiplier(rows, seed, cycle)
    shift = stable_uint64(str(cycle), f"d3d-prune-shift-{seed}") % rows
    return int((multiplier * offset + shift) % rows)


def assigned_rows(window: DataWindow, rank: int) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for position in range(
        window.cursor_start + rank, window.cursor_end, window.world_size
    ):
        selected.setdefault(
            permuted_row(position, window.dataset_rows, window.seed), []
        ).append(position)
    return selected


def _chat_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode_example(
    tokenizer: Any,
    source: dict[str, Any],
    example_id: str,
    *,
    max_prompt_tokens: int,
    max_sequence_tokens: int,
) -> EncodedExample | None:
    if source.get("status") not in (None, "success"):
        return None
    raw_messages = source.get("conversations") or source.get("messages")
    if not isinstance(raw_messages, list):
        return None
    messages = []
    for item in raw_messages:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", item.get("from", ""))).lower()
        role = "user" if role == "human" else "assistant" if role == "gpt" else role
        message: dict[str, Any] = {
            "role": role,
            "content": item.get("content", item.get("value", "")) or "",
        }
        if role == "assistant" and item.get("reasoning_content"):
            message["reasoning_content"] = item["reasoning_content"]
        messages.append(message)
    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if not assistant_positions:
        return None
    messages = messages[: assistant_positions[-1] + 1]
    if len(messages) < 2:
        return None
    prompt_text = _chat_text(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = _chat_text(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if (
        not prompt_ids
        or len(prompt_ids) > max_prompt_tokens
        or len(full_ids) > max_sequence_tokens
        or full_ids[: len(prompt_ids)] != prompt_ids
    ):
        return None
    response_ids = full_ids[len(prompt_ids) :]
    if len(response_ids) < 2:
        return None
    return EncodedExample(
        example_id=example_id,
        prompt_ids=[int(token) for token in prompt_ids],
        response_ids=[int(token) for token in response_ids],
    )


class ProgressiveDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        window: DataWindow,
        rank: int,
        max_prompt_tokens: int,
        max_sequence_tokens: int,
        minimum_response_tokens: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.window = window
        self.rank = rank
        self.max_prompt_tokens = max_prompt_tokens
        self.max_sequence_tokens = max_sequence_tokens
        self.minimum_response_tokens = minimum_response_tokens

    def __iter__(self) -> Iterator[tuple[int, EncodedExample]]:
        positions = range(
            self.window.cursor_start + self.rank,
            self.window.cursor_end,
            self.window.world_size,
        )
        # Index only selected rows, not their decoded/tokenized contents. A
        # physical-order scan followed by immediate yields biases any fixed
        # training budget toward low row numbers, overriding the permutation.
        offsets = {
            permuted_row(position, self.window.dataset_rows, self.window.seed): 0
            for position in positions
        }
        observed = 0
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                if observed in offsets:
                    offsets[observed] = offset
                observed += 1
            if observed != self.window.dataset_rows:
                raise RuntimeError(
                    f"Dataset changed: expected {self.window.dataset_rows} rows, "
                    f"observed {observed}"
                )
            for virtual_position in positions:
                row = permuted_row(
                    virtual_position, self.window.dataset_rows, self.window.seed
                )
                stream.seek(offsets[row])
                source = json.loads(stream.readline())
                if not isinstance(source, dict):
                    raise TypeError(f"Training row {row} is not a JSON object")
                encoded = encode_example(
                    self.tokenizer,
                    source,
                    str(row),
                    max_prompt_tokens=self.max_prompt_tokens,
                    max_sequence_tokens=self.max_sequence_tokens,
                )
                if (
                    encoded is not None
                    and len(encoded.response_ids) >= self.minimum_response_tokens
                ):
                    yield virtual_position, encoded


def anchor_offsets(
    response_tokens: int,
    block_size: int,
    max_anchors: int,
    example_id: str,
    namespace: str,
) -> list[int]:
    offsets = list(range(max(response_tokens - block_size + 1, 0)))
    offsets.sort(
        key=lambda offset: (
            stable_uint64(f"{example_id}:{offset}", namespace),
            offset,
        )
    )
    return sorted(offsets[:max_anchors])


def build_anchor_batch(
    example: EncodedExample,
    *,
    block_size: int,
    max_anchors: int,
    namespace: str,
    device: torch.device,
    mask_token_id: int = MASK_TOKEN_ID,
) -> AnchorBatch | None:
    offsets = anchor_offsets(
        len(example.response_ids),
        block_size,
        max_anchors,
        example.example_id,
        namespace,
    )
    if not offsets:
        return None
    clean_ids = example.prompt_ids + example.response_ids
    clean_input_ids = torch.tensor(clean_ids, device=device).unsqueeze(0)
    clean_length = len(clean_ids)
    anchors = len(offsets)
    query_length = anchors * block_size
    draft_input_ids = torch.full(
        (1, query_length), mask_token_id, dtype=torch.long, device=device
    )
    position_ids = torch.empty((1, query_length), dtype=torch.long, device=device)
    active = torch.ones((anchors, block_size - 1), dtype=torch.bool, device=device)
    targets = torch.empty((anchors, block_size - 1), dtype=torch.long, device=device)
    anchor_positions = []
    for index, offset in enumerate(offsets):
        anchor_position = len(example.prompt_ids) + offset
        anchor_positions.append(anchor_position)
        start = index * block_size
        draft_input_ids[0, start] = example.response_ids[offset]
        position_ids[0, start : start + block_size] = torch.arange(
            anchor_position, anchor_position + block_size, device=device
        )
        targets[index] = torch.tensor(
            example.response_ids[offset + 1 : offset + block_size], device=device
        )

    attention = torch.zeros(
        (1, 1, query_length, clean_length + query_length),
        dtype=torch.bool,
        device=device,
    )
    clean_positions = torch.arange(clean_length, device=device)
    for index, anchor_position in enumerate(anchor_positions):
        start = index * block_size
        stop = start + block_size
        attention[0, :, start:stop, :clean_length] = clean_positions < anchor_position
        attention[0, :, start:stop, clean_length + start : clean_length + stop] = True
    keep = torch.cat(
        [
            torch.arange(
                index * block_size + 1,
                (index + 1) * block_size,
                device=device,
            )
            for index in range(anchors)
        ]
    )
    return AnchorBatch(
        clean_input_ids=clean_input_ids,
        clean_attention_mask=causal_mask(clean_length, clean_length, device),
        prompt_length=len(example.prompt_ids),
        draft_input_ids=draft_input_ids,
        draft_attention_mask=attention,
        position_ids=position_ids,
        logit_keep_indices=keep,
        active_token_mask=active,
        target_ids=targets,
    )
