#!/usr/bin/env python3
"""Compare complete Math500 answers from BF16 and FP8 Target AR, cache off."""

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from run_decoupled_serial_fp8_controls import BF16_REFERENCE, ROOT, SEARCH
from sglang.benchmark.decoupled_serving import encode_dataset_prompts
from transformers import AutoTokenizer


def last_boxed_answer(text):
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for position in range(match.end(), len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
            if depth == 0:
                answers.append(text[match.end() : position])
                break
    return answers[-1] if answers else None


def normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", answer)
    answer = answer.replace("^{\\circ}", "^\\circ")
    return re.sub(r"\s+", "", answer).strip("$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--fp8-reference-results", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    root = args.output_dir.resolve()
    root.mkdir()
    dataset = ROOT / "docs/scripts/test.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()][:10]
    indices = args.indices if args.indices is not None else list(range(10))
    if len(set(indices)) != len(indices) or any(
        index not in range(10) for index in indices
    ):
        parser.error("indices must be unique entries from 0 through 9")
    tokenizer = AutoTokenizer.from_pretrained(args.model_cache / "Qwen3-8B")
    prompts = encode_dataset_prompts(rows=rows, tokenizer=tokenizer, limit=10)
    SEARCH.write_json(
        path=root / "suite.json",
        value={
            "dataset": str(dataset),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "num_prompts": len(indices),
            "indices": indices,
            "max_output_tokens": args.max_output_tokens,
            "temperature": 0,
            "ignore_eos": False,
            "target_radix_cache_disabled": True,
            "note": (
                "Task-quality pilot, not performance measurement. Last complete "
                "boxed answer is graded by normalized exact match; unmatched "
                "answers require review for mathematical equivalence. Same "
                "first-ten inputs as speed grids; not full Math500 accuracy."
            ),
        },
    )
    for label, quantization, reference_path in (
        ("bf16_ar", None, BF16_REFERENCE),
        ("fp8_ar", "fp8", args.fp8_reference_results),
    ):
        directory = root / label
        directory.mkdir()
        config = directory / "dllm.yaml"
        config.write_text("block_size: 4\n")
        launch = SEARCH.launch_args(
            block=4,
            topk=4,
            config=config,
            mode="ar",
            model_cache=args.model_cache,
            target_quantization=quantization,
            disable_radix_cache=True,
        )
        workers = SEARCH.start_workers(args=launch, directory=directory)
        reference = json.loads(reference_path.read_text())
        answers = []
        try:
            for index, (row, tokens) in enumerate(zip(rows, prompts, strict=True)):
                if index not in indices:
                    continue
                request = urllib.request.Request(
                    "http://127.0.0.1:30000/generate",
                    data=json.dumps(
                        {
                            "input_ids": tokens,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": args.max_output_tokens,
                                "ignore_eos": False,
                            },
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=1800) as response:
                    payload = json.load(response)
                predicted = last_boxed_answer(payload["text"])
                prefix = min(256, len(payload["output_ids"]))
                result = {
                    "index": index,
                    "unique_id": row["unique_id"],
                    "expected_answer": row["answer"],
                    "boxed_answer": predicted,
                    "normalized_exact_match": normalize_answer(predicted)
                    == normalize_answer(row["answer"]),
                    "prefix_tokens": prefix,
                    "prefix_matches_performance_reference": payload["output_ids"][
                        :prefix
                    ]
                    == reference["requests"][index]["output_ids"][:prefix],
                    "response": payload,
                }
                answers.append(result)
                SEARCH.write_json(path=directory / "answers.json", value=answers)
                print(
                    f"QUALITY {label} index={index} "
                    f"tokens={payload['meta_info']['completion_tokens']} "
                    f"boxed={predicted!r} exact={result['normalized_exact_match']}",
                    flush=True,
                )
        finally:
            SEARCH.stop_workers(processes=workers)
    print("FP8 ANSWER QUALITY PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
