#!/usr/bin/env python3
"""H20 probe: one anchor-conditioned draft block from a running drafter.

Opens N draft sessions against a live `sglang.launch_drafter --backend engine`
endpoint and asserts the service-level contract of the anchor-conditioned
block: ``block = [anchor] + [mask]*(block_size-1)``, i.e. every reply carries
exactly ``block_size`` tokens whose first token is the committed bonus (anchor)
that was NOT part of the draft prefix.

This is the GPU-side companion to the CPU regressions in
``test/registered/unit/spec/test_decoupled_anchor_block.py``: it exercises the
real variable-length extend (gap + block) through the native dLLM Engine.

Example:
  python scripts/decoupled/probe_anchor_block.py \\
    --endpoint ipc:///dev/shm/sglang-decoupled-32000.sock \\
    --tokenizer-path /workspace/models/Qwen3-8B \\
    --sessions 3 --output docs/perf/results/anchor-probe/probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sglang.srt.speculative.decoupled.client import DraftClient
from sglang.srt.speculative.decoupled.protocol import vocabulary_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="What is the value of 2 + 3 * 4? Show your reasoning briefly.",
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    encoded = tuple(tokenizer.encode(args.prompt))
    if len(encoded) < 2:
        parser.error("--prompt must encode to at least two tokens")
    # Split one real encoded sequence at the prompt/output boundary. Repeating
    # encoded[-1] in both fields would manufacture a duplicate token and let
    # the anchor assertion pass for a request shape the Target never emits.
    prompt = encoded[:-1]
    bonus = encoded[-1]
    outputs = (bonus,)

    client = DraftClient(
        endpoint=args.endpoint,
        rank=0,
        fingerprint=vocabulary_fingerprint(tokenizer.get_vocab()),
        block_size=args.block_size,
        timeout_seconds=args.timeout,
    )
    inputs = [(f"probe-{index}", prompt, outputs) for index in range(args.sessions)]
    sessions: list[dict[str, Any]] = []
    errors: list[str] = []
    record: dict[str, Any] = {
        "endpoint": args.endpoint,
        "tokenizer_path": args.tokenizer_path,
        "block_size": args.block_size,
        "prompt_len": len(prompt),
        "anchor": bonus,
        "sessions": sessions,
        "errors": errors,
    }
    try:
        replies = client.propose(inputs=inputs)
    finally:
        client.close()

    ok = True
    for rid, _, _ in inputs:
        reply = replies.get(rid)
        entry: dict[str, Any] = {
            "rid": rid,
            "error": reply.error if reply is not None else "no reply",
            "source": reply.source if reply is not None else "",
        }
        proposal = reply.proposal if reply is not None else None
        if proposal is None:
            entry["proposal_len"] = 0
            ok = False
        else:
            entry["proposal_len"] = len(proposal.tokens)
            entry["anchor"] = proposal.tokens[0]
            entry["anchor_matches"] = proposal.tokens[0] == bonus
            entry["top_tokens_rows"] = len(proposal.top_tokens)
            entry["tokens"] = list(proposal.tokens)
            if len(proposal.tokens) != args.block_size or proposal.tokens[0] != bonus:
                ok = False
        sessions.append(entry)
        if entry["error"]:
            errors.append(f"{rid}: {entry['error']}")
            ok = False

    record["ok"] = ok
    payload = json.dumps(record, indent=2)
    print(payload, flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
