"""Measure one global workload across 3+1 or 4 target replicas.

Reports aggregate completed output tokens / common wall-clock interval, never
the sum of independently timed per-request speeds. Use --worker-urls from the
launcher READY line. Save --output under docs/perf/results/<run>/ and pipe
stdout with tee into docs/perf/logs/ for H20 experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path

import aiohttp
import msgspec
import numpy as np


class Result(msgspec.Struct):
    index: int
    worker: int
    latency: float
    output_tokens: int
    error: str = ""
    output_ids: tuple[int, ...] = ()
    output_top_logprobs: tuple = ()
    spec_verify_ct: int = 0
    spec_num_correct_drafts: int = 0


async def generate(
    *, session, url, tokens, output_len, index, worker, divergence_top_logprobs
) -> Result:
    start = time.perf_counter()
    try:
        async with session.post(
            url.rstrip("/") + "/generate",
            json={
                "input_ids": tokens,
                "stream": False,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                },
                **(
                    {
                        "return_logprob": True,
                        "top_logprobs_num": divergence_top_logprobs,
                    }
                    if divergence_top_logprobs > 0
                    else {}
                ),
            },
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        count = int(payload["meta_info"]["completion_tokens"])
        if count != output_len:
            raise ValueError(f"incomplete output: expected {output_len}, got {count}")
        return Result(
            index,
            worker,
            time.perf_counter() - start,
            count,
            output_ids=tuple(payload.get("output_ids", ())),
            output_top_logprobs=tuple(
                payload["meta_info"].get("output_top_logprobs", ())
            ),
            spec_verify_ct=int(payload["meta_info"].get("spec_verify_ct", 0)),
            spec_num_correct_drafts=int(
                payload["meta_info"].get("spec_num_correct_drafts", 0)
            ),
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as exc:
        return Result(index, worker, time.perf_counter() - start, 0, str(exc))


def summarize(*, results: list[Result], elapsed: float) -> dict:
    successful = [r for r in results if not r.error]
    latencies = [r.latency for r in successful]
    total_tokens = sum(r.output_tokens for r in successful)
    return {
        "elapsed_seconds": elapsed,
        "completed_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "output_tokens": total_tokens,
        "output_tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
        "requests_per_second": len(successful) / elapsed if elapsed > 0 else 0,
        "e2e_p50_seconds": float(np.percentile(latencies, 50)) if latencies else None,
        "e2e_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "e2e_p99_seconds": float(np.percentile(latencies, 99)) if latencies else None,
        "valid": bool(results) and len(successful) == len(results),
        "spec_verify_ct": sum(r.spec_verify_ct for r in successful),
        "spec_num_correct_drafts": sum(r.spec_num_correct_drafts for r in successful),
    }


def encode_dataset_prompts(*, rows: list[dict], tokenizer, limit: int) -> list:
    prompts = []
    for row in rows[:limit]:
        if "input_ids" in row:
            prompt = row["input_ids"]
        elif "text" in row:
            prompt = tokenizer.encode(row["text"])
        else:
            problem = (
                row["problem"]
                + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
            messages = [{"role": "user", "content": problem}]
            template_args = {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                prompt = tokenizer.apply_chat_template(messages, **template_args)
            except TypeError as exc:
                if "enable_thinking" not in str(exc):
                    raise
                template_args.pop("enable_thinking")
                prompt = tokenizer.apply_chat_template(messages, **template_args)
            if isinstance(prompt, Mapping):
                prompt = prompt["input_ids"]
        prompts.append(prompt)
    return prompts


def _top_logprobs_at(result: dict, index: int) -> list:
    rows = result.get("output_top_logprobs", ())
    return rows[index] if index < len(rows) else []


def compare_outputs(*, payload: dict, reference: dict) -> dict:
    if not reference["summary"]["valid"]:
        raise ValueError("reference run is invalid")
    if payload["prompt_digest"] != reference["prompt_digest"]:
        raise ValueError(
            "reference and current run have different input token sequences"
        )
    for name in ("model_path", "output_len"):
        if payload["configuration"][name] != reference["configuration"][name]:
            raise ValueError(f"reference configuration differs: {name}")
    expected = {r["index"]: r for r in reference["requests"]}
    if len(expected) != len(payload["requests"]):
        raise ValueError("reference request count differs")
    mismatches = []
    divergence_details = []
    for result in payload["requests"]:
        baseline = expected[result["index"]]
        if (
            len(result["output_ids"]) != result["output_tokens"]
            or len(baseline["output_ids"]) != baseline["output_tokens"]
        ):
            raise ValueError(
                "token-level comparison requires output_ids from both servers"
            )
        # ``Result.output_ids`` is a tuple in the live payload while a JSON
        # reference necessarily restores it as a list. Compare token values,
        # not the serialization container type.
        if result["error"] or tuple(result["output_ids"]) != tuple(
            baseline["output_ids"]
        ):
            mismatches.append(result["index"])
            current_ids = result["output_ids"]
            baseline_ids = baseline["output_ids"]
            first = next(
                (
                    index
                    for index, (current, expected_token) in enumerate(
                        zip(current_ids, baseline_ids)
                    )
                    if current != expected_token
                ),
                min(len(current_ids), len(baseline_ids)),
            )
            divergence_details.append(
                {
                    "index": result["index"],
                    "first_token": first,
                    "reference_tokens": baseline_ids[first : first + 4],
                    "current_tokens": current_ids[first : first + 4],
                    "reference_top_logprobs": _top_logprobs_at(baseline, first),
                    "current_top_logprobs": _top_logprobs_at(result, first),
                }
            )
    return {
        "matched": len(payload["requests"]) - len(mismatches),
        "mismatched_indices": mismatches,
        "divergence_details": divergence_details,
        "valid": not mismatches,
    }


async def run(args, prompts) -> dict:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Same warmup per target, excluded from the common measurement window.
        for index, url in enumerate(args.worker_urls):
            result = await generate(
                session=session,
                url=url,
                tokens=prompts[0],
                output_len=args.output_len,
                index=-1,
                worker=index,
                divergence_top_logprobs=args.divergence_top_logprobs,
            )
            if result.error:
                raise RuntimeError(f"warmup failed on {url}: {result.error}")
        rng = random.Random(args.seed)
        start = time.perf_counter()
        scheduled = start
        tasks = []
        for index, tokens in enumerate(prompts):
            if index and math.isfinite(args.request_rate):
                scheduled += rng.expovariate(args.request_rate)
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            worker = index % len(args.worker_urls)
            tasks.append(
                asyncio.create_task(
                    generate(
                        session=session,
                        url=args.worker_urls[worker],
                        tokens=tokens,
                        output_len=args.output_len,
                        index=index,
                        worker=worker,
                        divergence_top_logprobs=args.divergence_top_logprobs,
                    )
                )
            )
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
    return {
        "configuration": vars(args),
        "prompt_digest": hashlib.sha256(msgspec.json.encode(prompts)).hexdigest(),
        "summary": summarize(results=results, elapsed=elapsed),
        "requests": [msgspec.to_builtins(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument("--model-path", required=True, help="Target tokenizer path")
    parser.add_argument(
        "--dataset-jsonl", help="Optional rows containing input_ids, text, or problem"
    )
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=400)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--reference-results",
        help="Optional AR result JSON from the same workload; compare every output token",
    )
    parser.add_argument(
        "--divergence-top-logprobs",
        type=int,
        default=0,
        help="Save top-N candidates per token for first-divergence diagnosis",
    )
    parser.add_argument(
        "--output", required=True, help="New JSON artifact path (must not exist)"
    )
    args = parser.parse_args()
    if min(args.input_len, args.output_len, args.num_prompts, args.timeout) <= 0:
        parser.error("lengths, prompt count and timeout must be positive")
    if math.isnan(args.request_rate) or args.request_rate <= 0:
        parser.error("request rate must be positive or inf")
    if args.divergence_top_logprobs < 0:
        parser.error("divergence-top-logprobs must be nonnegative")
    output = Path(args.output)
    if output.exists():
        parser.error("output exists; choose a new file to preserve previous results")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.dataset_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.dataset_jsonl).read_text().splitlines()
            if line.strip()
        ]
        prompts = encode_dataset_prompts(
            rows=rows, tokenizer=tokenizer, limit=args.num_prompts
        )
        if len(prompts) != args.num_prompts or any(not p for p in prompts):
            parser.error("dataset must contain num-prompts nonempty prompts")
    else:
        rng = random.Random(args.seed)
        special = set(tokenizer.all_special_ids)
        vocab = sorted(set(tokenizer.get_vocab().values()) - special)
        prompts = [
            rng.choices(vocab, k=args.input_len) for _ in range(args.num_prompts)
        ]
    payload = asyncio.run(run(args, prompts))
    if args.reference_results:
        try:
            payload["correctness"] = compare_outputs(
                payload=payload,
                reference=json.loads(Path(args.reference_results).read_text()),
            )
        except (ValueError, KeyError, OSError) as exc:
            payload["correctness"] = {"valid": False, "error": str(exc)}
        payload["summary"]["valid"] &= payload["correctness"]["valid"]
    # JSON does not define Infinity; save the offered rate as an explicit string.
    if not math.isfinite(args.request_rate):
        payload["configuration"]["request_rate"] = "inf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps(payload["summary"], indent=2, allow_nan=False))
    if not payload["summary"]["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
