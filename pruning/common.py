from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from models.token_ids import DEFAULT_MASK_TOKEN_ID

MASK_TOKEN_ID = DEFAULT_MASK_TOKEN_ID


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON atomically without leaving a partial artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def stable_uint64(value: str, namespace: str) -> int:
    return random.Random(f"{namespace}:{value}").getrandbits(64)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_selection_examples(path: Path, split: str | None = None) -> pd.DataFrame:
    """Load tokenized selection examples from Parquet or JSONL.

    Each row must contain ``prefix_ids`` and ``target_ids``.  ``analysis_id`` is
    optional and is generated from row order when absent.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError("Selection data must be .parquet, .jsonl, or .json")
    if split is not None and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == split].copy()
    missing = {"prefix_ids", "target_ids"} - set(frame.columns)
    if missing:
        raise ValueError(f"Selection data is missing columns: {sorted(missing)}")
    if "analysis_id" not in frame.columns:
        frame["analysis_id"] = [str(index) for index in range(len(frame))]
    frame["analysis_id"] = frame["analysis_id"].astype(str)
    if frame["analysis_id"].duplicated().any():
        raise ValueError("analysis_id must be unique")
    return frame.reset_index(drop=True)


def checkpoint_depth(checkpoint: Path) -> int:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return int(config["num_hidden_layers"])


def layer_provenance(checkpoint: Path) -> tuple[Path, int, list[int]]:
    """Return the original teacher, original depth, and retained layer mapping."""
    checkpoint = Path(checkpoint).resolve()
    depth = checkpoint_depth(checkpoint)
    metadata_path = checkpoint / "layer_prune.json"
    if not metadata_path.is_file():
        return checkpoint, depth, list(range(depth))
    metadata = json.loads(metadata_path.read_text())
    retained = [int(index) for index in metadata["retained_original_layers"]]
    if len(retained) != depth or retained != sorted(set(retained)):
        raise ValueError(f"Invalid retained-layer mapping in {metadata_path}")
    teacher = Path(metadata["teacher_checkpoint"]).resolve()
    original_depth = int(metadata["original_depth"])
    if not teacher.is_dir() or original_depth < depth:
        raise ValueError(f"Invalid teacher provenance in {metadata_path}")
    return teacher, original_depth, retained


def causal_mask(
    query_length: int,
    key_length: int,
    device: torch.device,
    *,
    query_start: int = 0,
) -> torch.Tensor:
    query_positions = torch.arange(
        query_start, query_start + query_length, device=device
    )
    key_positions = torch.arange(key_length, device=device)
    return (key_positions[None, :] <= query_positions[:, None])[None, None, :, :]


def draft_attention_mask(
    prefix_length: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Causal prefix followed by one bidirectional diffusion block."""
    total = prefix_length + block_size
    mask = torch.zeros((1, 1, total, total), dtype=torch.bool, device=device)
    if prefix_length:
        mask[:, :, :prefix_length, :prefix_length] = causal_mask(
            prefix_length, prefix_length, device
        )
    mask[:, :, prefix_length:, :] = True
    return mask


def hard_accept_length(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[int, float]:
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("predicted and target must be equal-length vectors")
    matches = predicted.eq(target)
    accepted_tail = int(matches.to(torch.int64).cumprod(dim=0).sum().item())
    emitted = accepted_tail + 1
    accuracy = float(matches.float().mean().item())
    return emitted, accuracy
