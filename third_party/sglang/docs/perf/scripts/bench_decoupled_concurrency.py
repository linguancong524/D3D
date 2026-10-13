#!/usr/bin/env python3
"""Measure per-request output speed with a bounded, closed-loop client pool."""

import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp
import msgspec

from bench_decoupled_serial import stream_request
from sglang.benchmark.decoupled_serving import generate


async def warmup(*, url, prompt, output_len):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        result = await generate(
            session=session, url=url, tokens=prompt, output_len=output_len,
            index=-1, worker=0, divergence_top_logprobs=0,
        )
        if result.error:
            raise RuntimeError(result.error)


async def measure(*, url, prompts, concurrency, output_len, metrics_snapshot=None):
    if metrics_snapshot is not None and concurrency != 1:
        raise ValueError("Per-request metric attribution requires C1")
    iterator = iter(enumerate(prompts))
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async def worker():
            for index, tokens in iterator:
                before = None if metrics_snapshot is None else metrics_snapshot()
                started_unix = time.time()
                try:
                    result = await stream_request(
                        session=session, url=url, tokens=tokens,
                        output_len=output_len, index=index,
                    )
                    result["output_tokens_per_second"] = output_len / result["latency"]
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, KeyError) as exc:
                    result = {"index": index, "error": str(exc), "output_tokens": 0}
                if before is not None:
                    result["request_started_unix"] = started_unix
                    after = metrics_snapshot()
                    result["client_delta"] = {
                        key: value - before.get(key, 0) for key, value in after.items()
                    }
                results.append(result)

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    return sorted(results, key=lambda result: result["index"]), elapsed


def percentile(*, values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compare_reference(*, requests, reference):
    expected = {item["index"]: item["output_ids"] for item in reference["requests"]}
    mismatches = [
        item["index"] for item in requests
        if item["error"] or item["output_ids"] != expected[item["index"]]
    ]
    return {"valid": not mismatches, "mismatches": mismatches}


def run_measurement(*, url, prompts, concurrency, output_len, model_path, reference,
                    metrics_snapshot=None):
    requests, elapsed = asyncio.run(measure(
        url=url, prompts=prompts, concurrency=concurrency, output_len=output_len,
        metrics_snapshot=metrics_snapshot,
    ))
    successful = [item for item in requests if not item["error"]]
    if not successful:
        raise RuntimeError("all requests failed")
    speed = [item["output_tokens_per_second"] for item in successful]
    decode = [item["decode_tokens_per_second"] for item in successful]
    latency = [item["latency"] for item in successful]
    summary = {
        "valid": len(successful) == len(requests),
        "completed_requests": len(successful), "failed_requests": len(requests) - len(successful),
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(item["output_tokens"] for item in successful) / elapsed,
        "per_request_output_tokens_per_second_p50": statistics.median(speed),
        "per_request_output_tokens_per_second_p10": percentile(values=speed, fraction=0.1),
        "per_request_output_tokens_per_second_p90": percentile(values=speed, fraction=0.9),
        "decode_tokens_per_second_p50": statistics.median(decode),
        "tpot_ms_p50": statistics.median(item["tpot_ms"] for item in successful),
        "ttft_ms_p50": statistics.median(item["ttft_seconds"] * 1000 for item in successful),
        "e2e_p95_seconds": percentile(values=latency, fraction=0.95),
        "spec_verify_ct": sum(item["spec_verify_ct"] for item in successful),
        "spec_num_correct_drafts": sum(item["spec_num_correct_drafts"] for item in successful),
    }
    payload = {
        "configuration": {"model_path": model_path, "concurrency": concurrency,
                          "output_len": output_len, "num_prompts": len(prompts),
                          "client_policy": "closed-loop pool; no client-side queue time in request latency"},
        "prompt_digest": hashlib.sha256(msgspec.json.encode(prompts)).hexdigest(),
        "summary": summary, "requests": requests,
    }
    if reference is not None:
        if payload["prompt_digest"] != reference["prompt_digest"]:
            raise ValueError("reference input tokens differ")
        payload["correctness"] = compare_reference(requests=requests, reference=reference)
        summary["valid"] &= payload["correctness"]["valid"]
    return payload


def write_json(*, path, payload):
    with Path(path).open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
