"""Small IO helpers: JSON writing and output-tag sanitising."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def safe_tag(path_or_name: str) -> str:
    tag = Path(path_or_name).name or str(path_or_name).strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) or "model"


def safe_custom_tag(tag: str | None, fallback: str) -> str:
    if tag:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag).strip()) or fallback
    return fallback


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp_path, out_path)
