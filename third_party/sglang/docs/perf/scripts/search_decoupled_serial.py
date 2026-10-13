#!/usr/bin/env python3
"""Search block widths and candidate top-k with a serial ready-rate guardrail."""

import argparse
import ast
import datetime
import importlib.util
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "decoupled_launcher", ROOT / "scripts/decoupled/launch.py"
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def write_json(*, path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def last_metrics(*, text, pattern):
    matches = re.findall(pattern, text)
    return ast.literal_eval(matches[-1]) if matches else {}


def offsets(*, directory):
    return {
        path.name: path.stat().st_size
        for path in directory.glob("*.log")
        if path.name in ("drafter.log", "target-0.log")
    }


def summarize(*, directory, bounds, benchmark):
    deltas = {}
    target_text = ""
    for name, end in bounds["end"].items():
        data = (directory / name).read_bytes()
        begin = bounds["begin"].get(name, 0)
        pattern = (
            r"Optional draft service metrics: (\{[^\n]+\})"
            if name == "drafter.log"
            else r"client_metrics=(\{[^\n]+\})"
        )
        old = last_metrics(text=data[:begin].decode(), pattern=pattern)
        new = last_metrics(text=data[:end].decode(), pattern=pattern)
        deltas[name] = {key: value - old.get(key, 0) for key, value in new.items()}
        if name == "target-0.log":
            target_text = data[begin:end].decode()
    service = deltas["drafter.log"]
    client = deltas["target-0.log"]
    ready = client["remote_optional_ready_lookup_ct"]
    hits = client["remote_optional_lookup_hit_ct"]
    lookups = ready + client["remote_optional_lookup_miss_ct"]
    forwards = service["packed_physical_forward_count"]
    verify = [
        float(value) * 1000
        for value in re.findall(
            r"target verify timing items=1 seconds=([0-9.]+)", target_text
        )
    ]
    return {
        "benchmark": benchmark["summary"],
        "correctness": benchmark["correctness"],
        "lookups": lookups,
        "ready": ready,
        "hits": hits,
        "ready_rate": ready / lookups,
        "hit_rate": hits / lookups,
        "hit_given_ready_rate": hits / ready if ready else None,
        "remote_gpu_ms": service["packed_model_forward_seconds"] * 1000 / forwards,
        "remote_service_ms": service["remote_optional_seconds"]
        * 1000
        / service["remote_optional_batch_ct"],
        "remote_items_per_batch": service["remote_optional_batch_item_ct"]
        / service["remote_optional_batch_ct"],
        "branches_per_item": service["remote_optional_branch_ct"]
        / service["remote_optional_batch_item_ct"],
        "verify_ms_p50": statistics.median(verify),
        "tokens_per_verify": benchmark["summary"]["completed_requests"]
        * 256
        / benchmark["summary"]["spec_verify_ct"],
        "client_delta": client,
        "service_delta": service,
    }


def launch_args(
    *,
    block,
    topk,
    config,
    mode,
    model_cache=None,
    remote_draft_quantization=None,
    target_quantization=None,
    local_draft_quantization=None,
    disable_radix_cache=False,
):
    template = (
        ROOT / "docs/perf/results/decoupled-hybrid-20261009-023916-785957/run.json"
    )
    args = json.loads(template.read_text())["args"]
    args.update(
        mode=mode,
        block_size=block,
        candidate_topk=topk,
        num_anchors=topk if mode == "hybrid" else 0,
        max_branches=(block - 1) * (topk - 1) + 1,
        accept_counts=list(range(1, block + 1)),
        dllm_algorithm_config=str(config),
        num_targets=1,
        gpus=[0, 3] if mode == "hybrid" else [0],
        max_running_requests=1,
        draft_batch_size=1,
        speculation_batch_size=1,
        draft_metrics_interval=1.0,
        draft_cuda_graph_max_tokens=1024,
        remote_draft_quantization=remote_draft_quantization,
        target_quantization=target_quantization,
        local_draft_quantization=local_draft_quantization,
        disable_radix_cache=disable_radix_cache,
    )
    if model_cache is not None:
        args["target_model"] = str(model_cache / "Qwen3-8B")
        args["draft_model"] = str(model_cache / "d3d-1.7b-b16_for_qwen3-8b")
        args["draft_tokenizer"] = args["target_model"]
    return argparse.Namespace(**args)


def worker_specs(*, args):
    specs = LAUNCHER.commands(args)
    # Every candidate uses the same BCG cap; all first-ten prompts fit in 512.
    for name, _, command in specs:
        if name.startswith("target-"):
            command += ["--cuda-graph-max-bs-prefill", "512"]
            if args.disable_radix_cache:
                command += ["--disable-radix-cache"]
            if args.target_quantization is not None:
                command += ["--quantization", args.target_quantization]
            if args.local_draft_quantization is not None:
                command += [
                    "--speculative-draft-model-quantization",
                    args.local_draft_quantization,
                ]
    return specs


def start_workers(*, args, directory):
    LAUNCHER.assert_idle(args.gpus)
    specs = worker_specs(args=args)
    write_json(
        path=directory / "run.json",
        value={
            "args": vars(args),
            "commands": specs,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    processes = []
    try:
        for name, gpu, command in specs:
            log = (directory / (name + ".log")).open("w")
            environment = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONUNBUFFERED": "1",
            }
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log.close()
            processes.append(process)
            if name == "drafter":
                LAUNCHER.wait_ready(
                    process=process,
                    timeout=600,
                    endpoint=LAUNCHER.resolve_draft_endpoint(args),
                )
        LAUNCHER.wait_ready(process=processes[-1], timeout=600, port=30000, http=True)
        return processes
    except BaseException:
        stop_workers(processes=processes)
        raise


def stop_workers(*, processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def benchmark(*, directory, attempt, reference, diagnose_output_mismatch=False):
    name = "serial_n10" if attempt == 1 else f"serial_n10_repeat{attempt}"
    bounds = {"begin": offsets(directory=directory), "begin_unix": time.time()}
    command = [
        str(ROOT / ".venv/bin/python3"),
        "docs/perf/scripts/bench_decoupled_serial.py",
        "--output",
        str(directory / (name + ".json")),
    ]
    if reference is not None:
        command += ["--reference-results", str(reference)]
    pipeline = (
        shlex.join(command)
        + " 2>&1 | tee "
        + shlex.quote(str(directory / (name + ".log")))
    )
    completed = subprocess.run(
        ["bash", "-o", "pipefail", "-c", pipeline], cwd=ROOT, check=False
    )
    time.sleep(2)
    bounds.update(end=offsets(directory=directory), end_unix=time.time())
    write_json(path=directory / (name + "_bounds.json"), value=bounds)
    payload = json.loads((directory / (name + ".json")).read_text())
    if completed.returncode != 0:
        parity_failure = (
            completed.returncode == 1
            and payload["summary"]["failed_requests"] == 0
            and payload["summary"]["completed_requests"] == 10
            and not payload["correctness"]["valid"]
        )
        if not (diagnose_output_mismatch and parity_failure):
            completed.check_returncode()
    return bounds, payload


def run_case(
    *,
    root,
    block,
    topk,
    reference,
    minimum_ready,
    model_cache,
    remote_draft_quantization=None,
    target_quantization=None,
    local_draft_quantization=None,
    diagnose_output_mismatch=False,
    disable_radix_cache=False,
):
    directory = root / f"b{block}_k{topk}"
    directory.mkdir()
    config = directory / "dllm.yaml"
    config.write_text(f"block_size: {block}\n")
    print(f"START block={block} topk={topk}", flush=True)
    processes = start_workers(
        args=launch_args(
            block=block,
            topk=topk,
            config=config,
            mode="hybrid",
            model_cache=model_cache,
            remote_draft_quantization=remote_draft_quantization,
            target_quantization=target_quantization,
            local_draft_quantization=local_draft_quantization,
            disable_radix_cache=disable_radix_cache,
        ),
        directory=directory,
    )
    measurements = []
    try:
        for attempt in (1, 2):
            bounds, payload = benchmark(
                directory=directory,
                attempt=attempt,
                reference=reference,
                diagnose_output_mismatch=diagnose_output_mismatch,
            )
            result = summarize(directory=directory, bounds=bounds, benchmark=payload)
            measurements.append(result)
            write_json(path=directory / "analysis.json", value=measurements)
            print(
                f"RESULT block={block} topk={topk} attempt={attempt} ready={result['ready_rate']:.4f} hit={result['hit_rate']:.4f} decode={result['benchmark']['decode_tokens_per_second_p50']:.2f}",
                flush=True,
            )
            if (
                attempt == 1
                and result["ready_rate"] >= minimum_ready
                and abs(result["ready_rate"] - minimum_ready) > 0.02
            ):
                break
    finally:
        stop_workers(processes=processes)
    ready_rate = sum(x["ready"] for x in measurements) / sum(
        x["lookups"] for x in measurements
    )
    return {
        "block": block,
        "topk": topk,
        "directory": str(directory),
        "ready_rate": ready_rate,
        "timely": ready_rate >= minimum_ready,
        "hit_rate": sum(x["hits"] for x in measurements)
        / sum(x["lookups"] for x in measurements),
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-block", type=int, default=4)
    parser.add_argument("--max-block", type=int, default=8)
    parser.add_argument("--topks", type=int, nargs="+", default=[4, 3, 2])
    parser.add_argument("--minimum-ready", type=float, default=0.95)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--remote-draft-quantization", choices=("fp8", "w8a8_int8"))
    parser.add_argument("--target-quantization", choices=("fp8",))
    parser.add_argument("--local-draft-quantization", choices=("fp8",))
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--diagnose-output-mismatch", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(exist_ok=args.resume)
    reference = args.reference_results or (
        ROOT
        / "docs/perf/results/decoupled-local-20261009-023239-671917/serial_n10.json"
    )
    write_json(
        path=args.output_dir / ("resume_search.json" if args.resume else "search.json"),
        value={
            "minimum_ready": args.minimum_ready,
            "blocks": list(range(args.min_block, args.max_block + 1)),
            "topks": args.topks,
            "remote_draft_quantization": args.remote_draft_quantization,
            "target_quantization": args.target_quantization,
            "local_draft_quantization": args.local_draft_quantization,
            "diagnose_output_mismatch": args.diagnose_output_mismatch,
            "disable_radix_cache": args.disable_radix_cache,
            "reference": str(reference),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "measurement_note": "Timing and lookup counts include one warmup; speed and parity exclude it. Single request; no remote lookup wait.",
        },
    )
    results = (
        json.loads((args.output_dir / "summary.json").read_text())
        if args.resume
        else []
    )
    complete = {(row["block"], row["topk"]) for row in results}

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    for block in range(args.min_block, args.max_block + 1):
        for topk in args.topks:
            if (block, topk) in complete:
                continue
            results.append(
                run_case(
                    root=args.output_dir,
                    block=block,
                    topk=topk,
                    reference=reference,
                    minimum_ready=args.minimum_ready,
                    model_cache=args.model_cache,
                    remote_draft_quantization=args.remote_draft_quantization,
                    target_quantization=args.target_quantization,
                    local_draft_quantization=args.local_draft_quantization,
                    diagnose_output_mismatch=args.diagnose_output_mismatch,
                    disable_radix_cache=args.disable_radix_cache,
                )
            )
            write_json(path=args.output_dir / "summary.json", value=results)
    print("SEARCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
