import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from benchmark import call


def summarize(rows, elapsed_seconds):
    total_tokens = sum(row["output_tokens"] for row in rows)
    total_verifies = sum(row["meta_info"].get("spec_verify_ct", 0) for row in rows)
    decode_tokens = sum(max(0, row["output_tokens"] - 1) for row in rows)
    return {
        "requests": len(rows),
        "output_tokens": total_tokens,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": total_tokens / elapsed_seconds,
        "median_request_tokens_per_second": statistics.median(
            row["output_tokens"] / row["elapsed_seconds"] for row in rows
        ),
        "verify_calls": total_verifies,
        "acclen": decode_tokens / total_verifies if total_verifies else None,
        "accepted_draft_tokens_per_verify": (
            decode_tokens / total_verifies - 1 if total_verifies else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    if args.concurrency < 1 or args.samples < args.concurrency:
        parser.error("samples must be at least concurrency")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(args.dataset).read_bytes()
    dataset = [json.loads(line) for line in source.splitlines() if line.strip()]
    indices = random.Random(args.seed).sample(range(len(dataset)), args.samples + 1)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = []
    for index in indices:
        row = dataset[index]
        question = row["problem"] if "problem" in row else row["question"]
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": question +
             "\nPlease reason step by step and put your final answer in \\boxed{}."}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
            enable_thinking=False,
        ))
    sampling = {"temperature": 0, "max_new_tokens": args.max_new_tokens,
                "ignore_eos": False}
    call(args.url, "/generate", {"input_ids": prompts[-1],
                                   "sampling_params": sampling})

    rows = []
    started = time.perf_counter()
    for repetition in range(args.repetitions):
        for offset in range(0, args.samples, args.concurrency):
            batch = list(zip(indices[offset:offset + args.concurrency],
                             prompts[offset:offset + args.concurrency]))
            batch_started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as executor:
                futures = [executor.submit(call, args.url, "/generate", {
                    "input_ids": tokens, "sampling_params": sampling,
                }) for _, tokens in batch]
                results = [future.result() for future in futures]
            batch_elapsed = time.perf_counter() - batch_started
            for (index, tokens), result in zip(batch, results):
                rows.append({
                    "source_index": index, "repetition": repetition,
                    "input_ids": tokens, "text": result["text"],
                    "output_ids": result.get("output_ids"),
                    "output_tokens": result["meta_info"]["completion_tokens"],
                    "elapsed_seconds": batch_elapsed,
                    "meta_info": result["meta_info"],
                })
            print(f"batch {len(rows) // args.concurrency}: "
                  f"{len(batch)} requests / {batch_elapsed:.3f}s", flush=True)
    elapsed = time.perf_counter() - started
    report = {
        "configuration": vars(args),
        "dataset_sha256": hashlib.sha256(source).hexdigest(),
        "indices": indices[:-1], "warmup_index": indices[-1],
        "sampling_params": sampling, "rows": rows,
        "summary": summarize(rows, elapsed),
    }
    if args.baseline:
        reference = json.loads(Path(args.baseline).read_text())
        if (reference["indices"] != report["indices"] or
            reference["dataset_sha256"] != report["dataset_sha256"] or
            reference["sampling_params"] != sampling or
            len(reference["rows"]) != len(rows) or
            any(left["input_ids"] != right["input_ids"]
                for left, right in zip(reference["rows"], rows))):
            raise ValueError("baseline must use identical prompts and request count")
        report["summary"]["speedup"] = (
            report["summary"]["tokens_per_second"] /
            reference["summary"]["tokens_per_second"]
        )
        report["summary"]["exact_text_matches"] = sum(
            left["text"] == right["text"]
            for left, right in zip(reference["rows"], rows)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
