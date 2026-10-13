#!/usr/bin/env python3
"""Measure serial request speed using the same inputs as decoupled_serving."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
import msgspec
import numpy as np
from transformers import AutoTokenizer

from sglang.benchmark.decoupled_serving import (
    compare_outputs,
    encode_dataset_prompts,
    generate,
)


async def stream_request(*, session, url, tokens, output_len, index):
    start = time.perf_counter()
    first = None
    count = 0
    payload = None
    async with session.post(
        url + "/generate",
        json={
            "input_ids": tokens,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        },
    ) as response:
        response.raise_for_status()
        async for raw_line in response.content:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            current = int(payload["meta_info"]["completion_tokens"])
            if current > count and first is None:
                first = (time.perf_counter(), current)
            count = current
    end = time.perf_counter()
    if count != output_len or first is None:
        raise RuntimeError(f"incomplete stream: index={index}, tokens={count}")
    meta = payload["meta_info"]
    decode_seconds = end - first[0]
    decode_tokens = count - first[1]
    return {
        "index": index,
        "worker": 0,
        "latency": end - start,
        "output_tokens": count,
        "error": "",
        "output_ids": payload["output_ids"],
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
        "spec_num_correct_drafts": int(meta.get("spec_num_correct_drafts", 0)),
        "ttft_seconds": first[0] - start,
        "first_chunk_tokens": first[1],
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "tpot_ms": decode_seconds * 1000 / decode_tokens,
    }


async def run(*, args, prompts):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        warmup = await generate(
            session=session,
            url=args.url,
            tokens=prompts[0],
            output_len=args.output_len,
            index=-1,
            worker=0,
            divergence_top_logprobs=0,
        )
        if warmup.error:
            raise RuntimeError(warmup.error)
        start = time.perf_counter()
        results = []
        for index, tokens in enumerate(prompts):
            results.append(
                await stream_request(
                    session=session,
                    url=args.url,
                    tokens=tokens,
                    output_len=args.output_len,
                    index=index,
                )
            )
        elapsed = time.perf_counter() - start
    return results, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", default="/workspace/models/Qwen3-8B")
    parser.add_argument("--dataset-jsonl", default="docs/scripts/test.jsonl")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rows = [
        json.loads(line) for line in Path(args.dataset_jsonl).read_text().splitlines()
    ]
    prompts = encode_dataset_prompts(
        rows=rows, tokenizer=tokenizer, limit=args.num_prompts
    )
    results, elapsed = asyncio.run(run(args=args, prompts=prompts))
    summary = {
        "valid": True,
        "completed_requests": len(results),
        "failed_requests": 0,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": len(results) * args.output_len / elapsed,
        "decode_tokens_per_second_p50": float(
            np.median([r["decode_tokens_per_second"] for r in results])
        ),
        "tpot_ms_p50": float(np.median([r["tpot_ms"] for r in results])),
        "ttft_ms_p50": float(np.median([r["ttft_seconds"] * 1000 for r in results])),
        "spec_verify_ct": sum(r["spec_verify_ct"] for r in results),
        "spec_num_correct_drafts": sum(r["spec_num_correct_drafts"] for r in results),
    }
    payload = {
        "configuration": {
            "model_path": args.model_path,
            "output_len": args.output_len,
            "dataset_jsonl": args.dataset_jsonl,
            "concurrency": 1,
        },
        "prompt_digest": hashlib.sha256(msgspec.json.encode(prompts)).hexdigest(),
        "summary": summary,
        "requests": results,
    }
    if args.reference_results:
        payload["correctness"] = compare_outputs(
            payload=payload,
            reference=json.loads(args.reference_results.read_text()),
        )
        summary["valid"] = payload["correctness"]["valid"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
