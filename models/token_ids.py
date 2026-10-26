"""Checkpoint-level token conventions, shared by pruning and evaluation."""
import json
from pathlib import Path


DEFAULT_MASK_TOKEN_ID = 151669


def checkpoint_mask_id(checkpoint, override: int | None = None) -> int:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    saved = config.get("mask_token_id")
    if saved is not None and override is not None and int(saved) != int(override):
        raise ValueError("--mask-token-id conflicts with the checkpoint's mask_token_id")
    value = override if override is not None else saved
    value = DEFAULT_MASK_TOKEN_ID if value is None else int(value)
    if not 0 <= value < int(config["vocab_size"]):
        raise ValueError(f"mask token ID {value} is outside checkpoint vocabulary {config['vocab_size']}")
    return value
