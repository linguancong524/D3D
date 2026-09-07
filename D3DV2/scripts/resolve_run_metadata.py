#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def safe_tag(raw: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw) or fallback


def path_tag(path: str, fallback: str) -> str:
    return Path(path).name or path.strip().replace("/", "_") or fallback


def path_tail_tag(path: str, fallback: str, parts: int = 2) -> str:
    resolved = Path(path)
    names = [part for part in resolved.parts if part not in {"", resolved.anchor}]
    tail = names[-int(parts):] if names else []
    return "_".join(tail) or path_tag(path, fallback)


def model_type(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ""
    try:
        return str(json.loads(config_path.read_text()).get("model_type", ""))
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve sdar_t3d model type and filename-safe tags.")
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--drafter-tag", default="")
    parser.add_argument("--verifier", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = [
        model_type(args.drafter),
        safe_tag(args.drafter_tag or path_tail_tag(args.drafter, "drafter", parts=2), "drafter"),
        safe_tag(path_tag(args.verifier, "model"), "model"),
    ]
    print("|".join(values))


if __name__ == "__main__":
    main()
