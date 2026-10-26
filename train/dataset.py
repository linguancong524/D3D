from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .runtime import DistContext
from .types import OfflineExample, ParallelOfflineBatch, RowItem
from .blocks import build_packed_block_batch, pin_batch

@dataclass(frozen=True)
class PackedBatch:
    """A worker-prepared training batch plus lightweight bookkeeping."""

    parallel_batch: ParallelOfflineBatch | None
    errors: list[tuple[int, str, str]]
    prompt_count: int

    def pin_memory(self):
        return PackedBatch(
            parallel_batch=pin_batch(self.parallel_batch),
            errors=self.errors,
            prompt_count=self.prompt_count,
        )


def count_jsonl_rows(path: str, *, max_examples: int = 0) -> int:
    """Count non-empty JSONL rows, optionally capped."""

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if max_examples and count >= int(max_examples):
                    break
    return count


def _normalize_messages(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize common chat role/content field variants."""

    messages: list[dict[str, Any]] = []
    for message in conversations:
        role = message.get("role", message.get("from", ""))
        if role == "human":
            role = "user"
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        normalized = {"role": role, "content": message.get("content", message.get("value", "")) or ""}
        if role == "assistant" and message.get("reasoning_content"):
            normalized["reasoning_content"] = message["reasoning_content"]
        messages.append(normalized)
    return messages


def _last_assistant_index(messages: list[dict[str, Any]]) -> int | None:
    """Find the last assistant turn used as the supervised response."""

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def render_prompt_and_full_texts(
    tokenizer,
    row: dict[str, Any],
    *,
    enable_thinking: bool,
) -> tuple[str, str]:
    """Render prompt-only and prompt-plus-response texts with the chat template."""

    messages = _normalize_messages(row.get("conversations") or row.get("messages") or [])
    assistant_index = _last_assistant_index(messages)
    if assistant_index is None:
        raise ValueError("row has no assistant response")
    prompt_messages = messages[:assistant_index]
    full_messages = messages[: assistant_index + 1]
    if not prompt_messages:
        raise ValueError("row has no prompt messages before assistant response")
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return prompt_text, full_text


def split_prompt_response_token_ids(
    rendered_prompt_ids: list[int],
    rendered_full_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Split a templated conversation at its shared prompt token boundary."""

    boundary = 0
    for prompt_id, full_id in zip(rendered_prompt_ids, rendered_full_ids):
        if prompt_id != full_id:
            break
        boundary += 1
    return rendered_full_ids[:boundary], rendered_full_ids[boundary:]


def encode_offline_example(
    tokenizer,
    row: dict[str, Any],
    *,
    fallback_prompt_id: int,
    max_prompt_len: int,
    max_seq_len: int,
    enable_thinking: bool,
    device: torch.device,
) -> OfflineExample | None:
    """Tokenize one JSONL row and keep only valid prompt-response examples."""

    if row.get("status") not in (None, "success"):
        return None
    prompt_text, full_text = render_prompt_and_full_texts(
        tokenizer,
        row,
        enable_thinking=enable_thinking,
    )
    rendered_prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids_list = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_ids_list, response_ids_list = split_prompt_response_token_ids(
        rendered_prompt_ids,
        full_ids_list,
    )
    if not prompt_ids_list:
        return None
    if len(prompt_ids_list) > int(max_prompt_len):
        return None
    if len(full_ids_list) > int(max_seq_len):
        return None
    if len(response_ids_list) < 2:
        return None
    prompt_id = str(row.get("id", row.get("prompt_id", fallback_prompt_id)))
    return OfflineExample(
        prompt_id=prompt_id,
        prompt_ids=torch.tensor([prompt_ids_list], dtype=torch.long, device=device),
        response_ids=torch.tensor(response_ids_list, dtype=torch.long, device=device),
    )


class OfflineDataset(Dataset):
    """Rank-local JSONL rows indexed by byte offset for multi-worker loading."""

    def __init__(
        self,
        *,
        data_path: str,
        tokenizer,
        rank: int,
        world_size: int,
        max_examples: int,
        max_prompt_len: int,
        max_seq_len: int,
        enable_thinking: bool,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_examples = int(max_examples)
        self.max_prompt_len = int(max_prompt_len)
        self.max_seq_len = int(max_seq_len)
        self.enable_thinking = bool(enable_thinking)
        self.offsets: list[tuple[int, int]] = []
        self._file = None
        seen = 0
        with open(self.data_path, "rb") as f:
            while True:
                offset = int(f.tell())
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if self.max_examples and seen >= self.max_examples:
                    break
                row_index = seen
                seen += 1
                if row_index % self.world_size == self.rank:
                    self.offsets.append((row_index, offset))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _read_row(self, offset: int) -> dict[str, Any]:
        if self._file is None:
            self._file = open(self.data_path, "r", encoding="utf-8")
        self._file.seek(int(offset))
        return json.loads(self._file.readline())

    def __getitem__(self, index: int) -> RowItem:
        cpu_device = torch.device("cpu")
        row_index, offset = self.offsets[int(index)]
        try:
            example = encode_offline_example(
                self.tokenizer,
                self._read_row(offset),
                fallback_prompt_id=row_index,
                max_prompt_len=self.max_prompt_len,
                max_seq_len=self.max_seq_len,
                enable_thinking=self.enable_thinking,
                device=cpu_device,
            )
        except Exception as exc:
            return RowItem(
                example=None,
                error=(row_index, type(exc).__name__, str(exc)),
            )
        return RowItem(example=example)


def make_collate(
    *,
    block_size: int,
    mask_token_id: int,
    max_anchors: int,
):
    """Create a collate function that packs rows into a ParallelOfflineBatch."""

    def collate(rows: list[RowItem]) -> PackedBatch:
        examples = [
            row.example
            for row in rows
            if row.example is not None
            and int(row.example.response_ids.numel()) > 2 * int(block_size)
        ]
        errors = [row.error for row in rows if row.error is not None]
        parallel_batch = (
                build_packed_block_batch(
                examples,
                block_size=block_size,
                mask_token_id=mask_token_id,
                max_anchors=max_anchors,
                device=torch.device("cpu"),
            )
            if examples
            else None
        )
        return PackedBatch(
            parallel_batch=parallel_batch,
            errors=errors,
            prompt_count=len(examples),
        )

    return collate


def seed_worker(worker_id: int) -> None:
    """Seed Python RNG in workers for deterministic CPU-side utilities."""

    seed = torch.initial_seed() % (2**32)
    random.seed(int(seed) + int(worker_id))


def build_loader(
    *,
    args: argparse.Namespace,
    tokenizer,
    mask_token_id: int,
    ctx: DistContext,
) -> DataLoader:
    """Build a multi-worker loader for CPU-side tokenization and batch packing."""

    dataset = OfflineDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        rank=ctx.rank,
        world_size=ctx.world_size,
        max_examples=args.max_examples,
        max_prompt_len=args.max_prompt_len,
        max_seq_len=args.max_seq_len,
        enable_thinking=args.enable_thinking_template,
    )
    num_workers = int(args.workers)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "collate_fn": make_collate(
            block_size=args.block_size,
            mask_token_id=mask_token_id,
            max_anchors=args.parallel_anchors_per_example,
        ),
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(args.prefetch)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return DataLoader(**kwargs)
