#!/usr/bin/env python3
"""Natural-EOS adapter for the archived H800 math C1 comparison.

Reuses the existing isolated server lifecycle, telemetry and metric snapshots,
but computes speed from actual output length rather than the requested cap.
No inference sources or fixed-length measurement helpers are modified.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp
import msgspec

import run_h800_hybrid_grid as GRID


async def stream(session, tokens, output_len, index, url):
    start = time.perf_counter()
    first = None
    payload = None
    async with session.post(url + "/generate", json={
        "input_ids": tokens, "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": output_len, "ignore_eos": False},
    }) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            count = int(payload["meta_info"]["completion_tokens"])
            if count > 0 and first is None:
                first = (time.perf_counter(), count)
    end = time.perf_counter()
    if payload is None or first is None:
        raise RuntimeError(f"Empty natural-EOS stream: {index}")
    meta = payload["meta_info"]
    count = int(meta["completion_tokens"])
    if not 0 < count <= output_len or (meta.get("finish_reason") or {}).get("type") not in {"stop", "length"}:
        raise RuntimeError(f"Invalid natural-EOS completion: {index}, {meta}")
    decode_tokens = count - first[1]
    decode_seconds = end - first[0]
    return {
        "index": index, "worker": 0, "error": "", "latency": end - start,
        "output_tokens": count, "output_ids": payload["output_ids"],
        "output_tokens_per_second": count / (end - start),
        "ttft_seconds": first[0] - start, "first_chunk_tokens": first[1],
        "decode_tokens_per_second": decode_tokens / decode_seconds if decode_tokens else None,
        "tpot_ms": 1000 * decode_seconds / decode_tokens if decode_tokens else None,
        "finish_reason": meta["finish_reason"],
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
        "spec_num_correct_drafts": int(meta.get("spec_num_correct_drafts", 0)),
    }


async def warmup(*, url, prompt, output_len):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        await stream(session, prompt, output_len, -1, url)


async def measure(*, url, prompts, output_len, metrics_snapshot):
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        started = time.perf_counter()
        for index, tokens in enumerate(prompts):
            before = metrics_snapshot() if metrics_snapshot else None
            started_unix = time.time()
            row = await stream(session, tokens, output_len, index, url)
            if before is not None:
                row["request_started_unix"] = started_unix
                after = metrics_snapshot()
                row["client_delta"] = GRID.delta(before=before, after=after)
            rows.append(row)
        elapsed = time.perf_counter() - started
    return rows, elapsed


def run_measurement(*, url, prompts, concurrency, output_len, model_path, reference, metrics_snapshot=None):
    if concurrency != 1:
        raise ValueError("This natural-EOS comparison requires C1")
    rows, elapsed = asyncio.run(measure(url=url, prompts=prompts, output_len=output_len,
                                       metrics_snapshot=metrics_snapshot))
    payload = {
        "configuration": {"model_path": model_path, "concurrency": concurrency,
                          "output_len": output_len, "num_prompts": len(prompts),
                          "ignore_eos": False, "output_len_semantics": "maximum, not fixed"},
        "prompt_digest": hashlib.sha256(msgspec.json.encode(prompts)).hexdigest(),
        "requests": rows,
        "summary": {
            "valid": True, "completed_requests": len(rows), "failed_requests": 0,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / elapsed,
            "per_request_output_tokens_per_second_p50": statistics.median(r["output_tokens_per_second"] for r in rows),
            "output_tokens_p50": statistics.median(r["output_tokens"] for r in rows),
            "total_output_tokens": sum(r["output_tokens"] for r in rows),
            "finish_reason_counts": {kind: sum(r["finish_reason"]["type"] == kind for r in rows)
                                     for kind in sorted({r["finish_reason"]["type"] for r in rows})},
            "ttft_ms_p50": statistics.median(r["ttft_seconds"] * 1000 for r in rows),
            "spec_verify_ct": sum(r["spec_verify_ct"] for r in rows),
            "spec_num_correct_drafts": sum(r["spec_num_correct_drafts"] for r in rows),
        },
    }
    if reference is not None:
        assert payload["prompt_digest"] == reference["prompt_digest"]
        expected = {r["index"]: r for r in reference["requests"]}
        mismatches = [r["index"] for r in rows if
                      (r["output_ids"], r["output_tokens"], r["finish_reason"]) !=
                      (expected[r["index"]]["output_ids"], expected[r["index"]]["output_tokens"], expected[r["index"]]["finish_reason"])]
        payload["correctness"] = {"valid": not mismatches, "mismatches": mismatches}
        payload["summary"]["valid"] = not mismatches
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.repetitions, args.concurrency, args.output_len, args.resume = 3, [1], 2048, False
    tokenizer = GRID.AutoTokenizer.from_pretrained(GRID.AB.TARGET)
    fingerprint = GRID.vocabulary_fingerprint(tokenizer.get_vocab())
    # Scoped adapter in this standalone process only; fixed-length helpers stay intact.
    GRID.run_measurement, GRID.warmup = run_measurement, warmup
    valid = True
    for dataset, order in (("math500", ("ar", "eager", "graph")),
                           ("gsm8k", ("ar", "graph", "eager"))):
        directory = args.output_dir / dataset
        directory.mkdir()
        source = args.fixed_archive / dataset / "sample.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        prompts = GRID.encode_dataset_prompts(rows=rows, tokenizer=tokenizer, limit=33)
        assert len(prompts) == 33
        GRID.write_json(path=directory / "dataset_manifest.json", payload={
            "sample_path": str(source), "sample_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fixed_source_manifest": str(args.fixed_archive / dataset / "dataset_manifest.json"),
            "measured_prompts": 32, "warmup_prompts": 1, "output_cap": 2048,
            "ignore_eos": False, "repetitions": 3, "order": order,
        })
        reference = None
        for mode in order:
            print(f"START natural {dataset}/{mode}", flush=True)
            endpoint = f"ipc:///dev/shm/h800-natural-{dataset}-{mode}.sock"
            launch = GRID.launch_args(block=16, topk=2, full_accept=True,
                                      endpoint=endpoint, partial_accept_counts=[1, 2, 3])
            overrides = json.loads((args.fixed_archive / f"{mode}.json").read_text())
            for key, value in overrides.items():
                if key not in vars(launch):
                    raise ValueError(key)
                setattr(launch, key, value)
            launch.draft_cuda_graph_max_tokens = GRID.LAUNCHER.remote_packed_graph_max_tokens(launch)
            specs = GRID.ar_specs() if mode == "ar" else GRID.LAUNCHER.commands(launch)
            config = {"mode": launch.mode, "block": 16, "topk": 2, "full_accept": mode != "ar",
                      "overrides": overrides, "performance_valid": True, "per_request_metrics": True,
                      "ignore_eos": False, "output_cap": 2048}
            source_paths = list(json.loads((args.fixed_archive / dataset / "ar_source_sha256.json").read_text()))
            source_paths += [str(Path(__file__).resolve().relative_to(GRID.ROOT)), "docs/perf/scripts/bench_decoupled_serial.py"]
            GRID.write_json(path=directory / f"{mode}_source_sha256.json", payload={
                filename: hashlib.sha256((GRID.ROOT / filename).read_bytes()).hexdigest()
                for filename in source_paths
            })
            measurements, generated = GRID.measure_config(
                specs=specs, directory=directory / mode, config=config,
                endpoint=None if mode == "ar" else endpoint, fingerprint=fingerprint,
                prompts=prompts[:-1], warmup_prompt=prompts[-1], reference=reference, args=args,
            )
            if mode == "ar":
                reference = generated
                GRID.write_json(path=directory / "ar_reference.json", payload=reference)
            valid &= all(m["valid"] for m in measurements)
            print(f"FINISH natural {dataset}/{mode}; valid={all(m['valid'] for m in measurements)}", flush=True)
    if not valid:
        raise RuntimeError("Natural-EOS output parity mismatch; evidence retained")


if __name__ == "__main__":
    main()
