import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def call(base_url, endpoint, payload=None):
    request = urllib.request.Request(
        base_url + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def summarize(rows, elapsed_seconds=None):
    total_tokens = sum(row["output_tokens"] for row in rows)
    total_seconds = (
        elapsed_seconds if elapsed_seconds is not None
        else sum(row["elapsed_seconds"] for row in rows)
    )
    total_verifies = sum(row["meta_info"].get("spec_verify_ct", 0) for row in rows)
    decode_tokens = sum(max(0, row["output_tokens"] - 1) for row in rows)
    return {
        "requests": len(rows),
        "output_tokens": total_tokens,
        "elapsed_seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds,
        "median_request_tokens_per_second": statistics.median(
            row["output_tokens"] / row["elapsed_seconds"] for row in rows
        ),
        "verify_calls": total_verifies,
        "acclen": decode_tokens / total_verifies if total_verifies else None,
        "acclen_definition": "(output tokens minus one prefill token per request) / verify calls; includes target bonus",
        "accepted_draft_tokens_per_verify": (
            decode_tokens / total_verifies - 1 if total_verifies else None
        ),
    }


def generate_row(base_url, index, tokens, sampling, repetition):
    started = time.perf_counter()
    result = call(base_url, "/generate", {
        "input_ids": tokens, "sampling_params": sampling,
    })
    return {
        "source_index": index, "repetition": repetition,
        "input_ids": tokens, "text": result["text"],
        "output_ids": result.get("output_ids"),
        "output_tokens": result["meta_info"]["completion_tokens"],
        "elapsed_seconds": time.perf_counter() - started,
        "meta_info": result["meta_info"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--dataset", required=True,
        help="dataset name such as math500 (auto-download) or a local JSONL path",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    if args.samples < 1 or args.repetitions < 1 or args.concurrency < 1:
        parser.error("samples, repetitions and concurrency must be positive")
    from transformers import AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        source = dataset_path.read_bytes()
        dataset = [json.loads(line) for line in source.splitlines() if line.strip()]
        question_fn = lambda row: row["problem"] if "problem" in row else row["question"]
    else:
        from common.data import DATASETS, load_rows

        if args.dataset not in DATASETS:
            raise ValueError(
                f"unknown dataset {args.dataset!r}; use a registered name or JSONL path"
            )
        dataset, _, _ = load_rows(args.dataset, None)
        source = json.dumps(dataset, sort_keys=True, default=str).encode()
        question_fn = DATASETS[args.dataset]["question"]
    indices = random.Random(args.seed).sample(range(len(dataset)), args.samples + 1)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = []
    for index in indices:
        row = dataset[index]
        question = question_fn(row)
        text = question + "\nPlease reason step by step and put your final answer in \\boxed{}."
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
            enable_thinking=False,
        ))
    sampling = {
        "temperature": 0, "max_new_tokens": args.max_new_tokens,
        "ignore_eos": False,
    }
    call(args.url, "/generate", {"input_ids": prompts[-1], "sampling_params": sampling})
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        started = time.perf_counter()
        for repetition in range(args.repetitions):
            futures = [executor.submit(
                generate_row, args.url, index, tokens, sampling, repetition,
            ) for index, tokens in zip(indices[:-1], prompts[:-1])]
            for future in futures:
                row = future.result()
                rows.append(row)
                print(f"request {len(rows)}: {row['output_tokens']} tokens / "
                      f"{row['elapsed_seconds']:.3f}s", flush=True)
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
        if (
            reference["indices"] != report["indices"]
            or reference["dataset_sha256"] != report["dataset_sha256"]
            or reference["sampling_params"] != sampling
            or len(reference["rows"]) != len(rows)
            or reference["configuration"].get("concurrency", 1) != args.concurrency
            or any(left["input_ids"] != right["input_ids"]
                   for left, right in zip(reference["rows"], rows))
        ):
            raise ValueError("baseline must use identical prompts, sampling and request count")
        report["summary"]["speedup"] = (
            report["summary"]["tokens_per_second"]
            / reference["summary"]["tokens_per_second"]
        )
        report["summary"]["exact_text_matches"] = sum(
            left["text"] == right["text"]
            for left, right in zip(reference["rows"], rows)
        )
        report["summary"]["speedup_definition"] = "aggregate E2E output tokens/second divided by matched native SGLang AR"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
