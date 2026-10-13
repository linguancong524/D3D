#!/usr/bin/env python3
"""Run same-day BF16 controls, FP8 references, and all-model FP8 searches."""

import argparse
import importlib.util
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "serial_search", ROOT / "docs/perf/scripts/search_decoupled_serial.py"
)
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)
BF16_REFERENCE = (
    ROOT / "docs/perf/results/decoupled-local-20261009-023239-671917/serial_n10.json"
)


def profile_control(*, directory, payload):
    relative = directory.resolve().relative_to(ROOT / "docs/perf/results")
    output = ROOT / "docs/perf/profiles" / relative
    output.mkdir(parents=True)

    def post(endpoint, data):
        request = urllib.request.Request(
            "http://127.0.0.1:30000/" + endpoint,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read().decode()

    post(
        "start_profile",
        {
            "output_dir": str(output),
            "activities": ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": True,
            "profile_id": directory.name,
        },
    )
    try:
        response = post(
            "generate",
            {
                "input_ids": payload["requests"][0]["output_ids"][:32],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
            },
        )
        (directory / "profile_request.json").write_text(response)
    finally:
        post("stop_profile", {})


def run_control(
    *,
    directory,
    block,
    topk,
    mode,
    quantization,
    reference,
    cache,
    disable_radix_cache=False,
):
    directory.mkdir(parents=True)
    config = directory / "dllm.yaml"
    config.write_text(f"block_size: {block}\n")
    args = SEARCH.launch_args(
        block=block,
        topk=topk,
        config=config,
        mode=mode,
        model_cache=cache,
        target_quantization=quantization,
        local_draft_quantization=quantization if mode == "local" else None,
        disable_radix_cache=disable_radix_cache,
    )
    workers = SEARCH.start_workers(args=args, directory=directory)
    results = []
    try:
        for attempt in (1, 2):
            bounds, payload = SEARCH.benchmark(
                directory=directory,
                attempt=attempt,
                reference=reference,
                diagnose_output_mismatch=True,
            )
            result = {
                "benchmark": payload["summary"],
                "correctness": payload.get("correctness"),
                "bounds": bounds,
            }
            results.append(result)
            SEARCH.write_json(path=directory / "analysis.json", value=results)
            print(
                f"CONTROL {directory.name} attempt={attempt} "
                f"decode={payload['summary']['decode_tokens_per_second_p50']:.2f} "
                f"parity={payload.get('correctness')}",
                flush=True,
            )
            if reference is None:
                reference = directory / "serial_n10.json"
        if quantization == "fp8" and (mode == "ar" or block == 4):
            profile_control(directory=directory, payload=payload)
    finally:
        SEARCH.stop_workers(processes=workers)


def run_bf16_controls(*, root, cache):
    controls = root / "bf16_controls"
    controls.mkdir()
    run_control(
        directory=controls / "ar",
        block=4,
        topk=4,
        mode="ar",
        quantization=None,
        reference=BF16_REFERENCE,
        cache=cache,
    )
    for block, topk in ((4, 4), (6, 4), (7, 4), (8, 4), (8, 2)):
        SEARCH.run_case(
            root=controls,
            block=block,
            topk=topk,
            reference=BF16_REFERENCE,
            minimum_ready=0.95,
            model_cache=cache,
        )
    for block in (4, 8):
        run_control(
            directory=controls / f"local_b{block}",
            block=block,
            topk=4,
            mode="local",
            quantization=None,
            reference=BF16_REFERENCE,
            cache=cache,
        )


def run_fp8_controls(*, root, cache):
    fp8_reference = root / "fp8_ar" / "serial_n10.json"
    run_control(
        directory=fp8_reference.parent,
        block=4,
        topk=4,
        mode="ar",
        quantization="fp8",
        reference=None,
        cache=cache,
    )
    for block in (4, 8):
        run_control(
            directory=root / f"fp8_local_b{block}",
            block=block,
            topk=4,
            mode="local",
            quantization="fp8",
            reference=fp8_reference,
            cache=cache,
        )
    return fp8_reference


def run_full_fp8_grid(*, root, cache, reference):
    command = [
        str(ROOT / ".venv/bin/python3"),
        str(ROOT / "docs/perf/scripts/search_decoupled_serial.py"),
        "--output-dir",
        str(root / "full_fp8_grid"),
        "--model-cache",
        str(cache),
        "--remote-draft-quantization",
        "fp8",
        "--target-quantization",
        "fp8",
        "--local-draft-quantization",
        "fp8",
        "--reference-results",
        str(reference),
        "--diagnose-output-mismatch",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    root = args.output_dir.resolve()
    root.mkdir()
    SEARCH.write_json(
        path=root / "suite.json",
        value={
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "bf16_reference": str(BF16_REFERENCE),
            "model_cache": str(args.model_cache),
            "minimum_ready": 0.95,
            "measurement_note": (
                "Quantization is standard online FP8 linear-layer quantization; "
                "attention, KV, norms and output head keep their supported dtype. "
                "Parity failures are retained for diagnosis and never count as "
                "passing fidelity. References exclude warmup; service counts "
                "include it, matching the original serial search."
            ),
        },
    )
    reference = run_fp8_controls(root=root, cache=args.model_cache)
    run_bf16_controls(root=root, cache=args.model_cache)
    run_full_fp8_grid(root=root, cache=args.model_cache, reference=reference)
    print("FP8 SUITE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
