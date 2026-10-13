#!/usr/bin/env python3
"""Launch AR, local-required, or hybrid optional-packed comparisons.

Example:
  python scripts/decoupled/launch.py --mode hybrid --target-model /models/Qwen3-8B \\
    --draft-model /models/d3d-1.7b-b16_for_qwen3-8b \\
    --dllm-algorithm OneStep
Add --dry-run to inspect every command without touching GPUs or starting models.
"""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def resolve_draft_endpoint(args):
    endpoint = getattr(args, "draft_endpoint", None)
    if endpoint:
        return endpoint
    return f"ipc:///dev/shm/sglang-decoupled-{args.draft_port}.sock"


def remote_branch_group_count(args):
    if args.packed_single_row:
        return 1
    accept_counts = args.accept_counts or range(1, args.block_size + 1)
    num_positions = len(tuple(accept_counts))
    anchored_groups = (
        args.num_anchors + args.branch_group_size - 1
    ) // args.branch_group_size
    return max(0, num_positions - 1) * anchored_groups + 1


def remote_packed_tokens_per_item(args):
    return args.block_size * (
        args.max_branches + remote_branch_group_count(args)
    )


def remote_packed_graph_max_tokens(args):
    return args.speculation_batch_size * remote_packed_tokens_per_item(args)


def target_count(args):
    if args.mode == "ar4":
        return 4
    if args.mode == "ar3":
        return 3
    return args.num_targets


def commands(args):
    num_targets = target_count(args)
    remote_optional = args.mode == "hybrid"
    if args.remote_draft_quantization is not None and not remote_optional:
        raise ValueError("remote Draft quantization requires hybrid mode")
    if args.remote_draft_model is not None and not remote_optional:
        raise ValueError("remote Draft model requires hybrid mode")
    required_gpus = num_targets + int(remote_optional)
    if (
        len(args.gpus) < required_gpus
        or len(set(args.gpus)) != len(args.gpus)
        or (remote_optional and len(args.gpus) != required_gpus)
    ):
        raise ValueError(
            f"provide {required_gpus} distinct GPU indices"
            + ("" if remote_optional else " (extra idle GPUs are allowed)")
        )
    local_required = args.mode in ("local", "hybrid")
    draft_endpoint = resolve_draft_endpoint(args)
    if local_required and not args.draft_model:
        raise ValueError("local/hybrid require --draft-model")
    if remote_optional and args.dllm_algorithm != "OneStep":
        raise ValueError(
            "anchor-conditioned engine backend requires --dllm-algorithm OneStep"
        )
    result = []
    num_branch_groups = remote_branch_group_count(args)
    packed_tokens_per_item = remote_packed_tokens_per_item(args)
    if remote_optional:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_drafter",
            "--model-path",
            args.remote_draft_model or args.draft_model,
            "--backend",
            "engine",
            "--tokenizer-path",
            args.draft_tokenizer or args.target_model,
            "--endpoint",
            draft_endpoint,
            "--mode",
            "packed",
            "--block-size",
            str(args.block_size),
            "--num-anchors",
            str(args.num_anchors),
            "--max-branches",
            str(args.max_branches),
            "--candidate-topk",
            str(args.candidate_topk),
            "--branch-group-size",
            str(args.branch_group_size),
            "--max-context",
            str(args.context_length),
            "--max-sessions",
            str(num_targets * args.max_running_requests),
            "--max-sessions-per-peer",
            str(args.max_running_requests),
            "--batch-size",
            str(args.draft_batch_size),
            "--speculation-batch-size",
            str(args.speculation_batch_size),
            "--batching-window-ms",
            str(args.batching_window_ms),
            "--engine-packed-cuda-graph-max-requests",
            str(
                args.speculation_batch_size * num_branch_groups
            ),
            "--engine-packed-cuda-graph-max-tokens",
            str(args.draft_cuda_graph_max_tokens),
            "--engine-packed-cuda-graph-token-step",
            str(packed_tokens_per_item),
            "--metrics-interval",
            str(args.draft_metrics_interval),
            "--prefill-chunk-size",
            str(args.draft_prefill_chunk_size),
            "--dllm-algorithm",
            args.dllm_algorithm,
            "--dllm-algorithm-config",
            args.dllm_algorithm_config,
            "--engine-attention-backend",
            args.draft_attention_backend,
            "--engine-max-running-requests",
            str(
                num_targets * args.max_running_requests
                + max(
                    args.draft_batch_size,
                    args.speculation_batch_size * num_branch_groups,
                )
            ),
            "--dtype",
            args.draft_dtype,
        ]
        if args.draft_mem_fraction_static is not None:
            cmd += [
                "--engine-mem-fraction-static",
                str(args.draft_mem_fraction_static),
            ]
        if args.remote_draft_quantization is not None:
            cmd += ["--quantization", args.remote_draft_quantization]
        if args.draft_enable_cuda_graph:
            cmd += ["--engine-enable-cuda-graph"]
        if args.batched_branch_topk:
            cmd += ["--batched-branch-topk"]
        if args.packed_single_row:
            cmd += ["--packed-single-row"]
        if getattr(args, "packed_causal_pending", False):
            cmd += ["--packed-causal-pending"]
        if getattr(args, "engine_deterministic", False):
            cmd += ["--engine-deterministic"]
        if args.accept_counts:
            cmd += ["--accept-counts", *map(str, args.accept_counts)]
        if args.trust_remote_code:
            cmd += ["--trust-remote-code"]
        result.append(("drafter", args.gpus[num_targets], cmd))
    for rank in range(num_targets):
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.target_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.base_port + rank),
            "--mem-fraction-static",
            str(args.target_mem_fraction_static),
            "--tp-size",
            "1",
            "--context-length",
            str(args.context_length),
            "--max-running-requests",
            str(args.max_running_requests),
            "--attention-backend",
            "fa3",
            "--enable-deterministic-inference",
        ]
        if args.trust_remote_code:
            cmd += ["--trust-remote-code"]
        if args.disable_prefill_cuda_graph:
            cmd += ["--cuda-graph-backend-prefill=disabled"]
        if args.target_chunked_prefill_size is not None:
            cmd += [
                "--chunked-prefill-size",
                str(args.target_chunked_prefill_size),
            ]
        if local_required:
            cmd += [
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-algorithm",
                "DECOUPLED",
                "--speculative-num-draft-tokens",
                str(args.block_size),
                "--speculative-draft-attention-backend",
                args.draft_attention_backend,
                *(
                    ["--decoupled-local-prefix-reuse"]
                    if args.local_prefix_reuse
                    else []
                ),
                "--decoupled-draft-timeout",
                str(args.draft_timeout),
                "--decoupled-lookup-wait",
                str(args.lookup_wait_ms / 1000),
                "--decoupled-candidate-topk",
                str(args.candidate_topk),
                "--decoupled-num-anchors",
                str(args.num_anchors),
                "--decoupled-accept-counts",
                *map(str, args.accept_counts or range(1, args.block_size + 1)),
                "--decoupled-poll-interval",
                str(args.poll_interval_ms / 1000),
            ]
            cmd += [
                "--decoupled-local-cuda-graph-max-tokens",
                str(args.local_cuda_graph_max_tokens),
                "--decoupled-local-cuda-graph-max-requests",
                str(args.local_cuda_graph_max_requests),
            ]
            if args.timing_sync:
                cmd += ["--decoupled-timing-sync"]
            if args.local_cuda_graph_check:
                cmd += ["--decoupled-local-cuda-graph-check"]
            if args.wait_for_covered_result:
                cmd += ["--decoupled-wait-for-covered-result"]
            if args.shadow_local:
                cmd += ["--decoupled-shadow-local"]
            if remote_optional:
                cmd += ["--decoupled-draft-endpoint", draft_endpoint]
        result.append((f"target-{rank}", args.gpus[rank], cmd))
    return result


def assert_idle(gpus):
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True,
    )
    print(gpu_rows, flush=True)
    selected = {
        row.split(",")[1].strip()
        for row in gpu_rows.splitlines()
        if int(row.split(",")[0]) in gpus
    }
    if len(selected) != len(gpus):
        raise RuntimeError("selected GPUs are not all present")
    running = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(row.split(",")[0].strip() in selected for row in running.splitlines()):
        raise RuntimeError(
            "selected GPU has an existing compute process; refusing to start"
        )


def wait_ready(*, process, timeout, port=None, http=False, endpoint=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker exited with status {process.returncode}")
        try:
            if endpoint is not None:
                if endpoint.startswith("ipc://"):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.settimeout(1)
                        probe.connect(endpoint.removeprefix("ipc://"))
                        return
                if endpoint.startswith("tcp://"):
                    host, raw_port = endpoint.removeprefix("tcp://").rsplit(":", 1)
                    with socket.create_connection((host, int(raw_port)), timeout=1):
                        return
                raise ValueError("drafter endpoint must use ipc:// or tcp://")
            elif http:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return
            else:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
        except (OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"worker port {port} readiness deadline exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("ar4", "ar3", "local", "hybrid"), required=True
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument(
        "--remote-draft-model",
        help="Optional remote Draft checkpoint; local fallback keeps --draft-model.",
    )
    parser.add_argument("--dllm-algorithm", default="OneStep")
    parser.add_argument("--dllm-algorithm-config", default="/tmp/sdar_dllm.yaml")
    parser.add_argument("--draft-attention-backend", default="flashinfer")
    parser.add_argument("--draft-dtype", default="bfloat16")
    parser.add_argument(
        "--remote-draft-quantization",
        choices=("fp8", "w8a8_int8"),
        help="Quantize only the optional remote Draft; Target and local fallback stay unchanged.",
    )
    parser.add_argument("--draft-mem-fraction-static", type=float)
    parser.add_argument("--draft-enable-cuda-graph", action="store_true")
    parser.add_argument("--draft-cuda-graph-max-tokens", type=int)
    parser.add_argument("--local-prefix-reuse", action="store_true")
    parser.add_argument("--local-cuda-graph-max-tokens", type=int, default=0)
    parser.add_argument("--local-cuda-graph-max-requests", type=int, default=1)
    parser.add_argument("--local-cuda-graph-check", action="store_true")
    parser.add_argument("--timing-sync", action="store_true")
    parser.add_argument("--draft-prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--draft-batch-size", type=int, default=64)
    parser.add_argument("--speculation-batch-size", type=int, default=8)
    parser.add_argument("--batching-window-ms", type=float, default=1)
    parser.add_argument("--draft-metrics-interval", type=float, default=10)
    parser.add_argument("--draft-tokenizer")
    parser.add_argument("--num-targets", type=int, choices=(1, 3), default=3)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--draft-port", type=int, default=32000)
    parser.add_argument(
        "--draft-endpoint",
        help="Explicit drafter endpoint; defaults to ipc:///dev/shm using --draft-port",
    )
    parser.add_argument("--draft-timeout", type=float, default=30)
    parser.add_argument("--lookup-wait-ms", type=float, default=0)
    parser.add_argument("--wait-for-covered-result", action="store_true")
    parser.add_argument("--candidate-topk", type=int, default=4)
    parser.add_argument("--branch-group-size", type=int, default=3)
    parser.add_argument("--packed-single-row", action="store_true")
    parser.add_argument("--batched-branch-topk", action="store_true")
    parser.add_argument("--packed-causal-pending", action="store_true")
    parser.add_argument("--engine-deterministic", action="store_true")
    parser.add_argument("--poll-interval-ms", type=float, default=1)
    parser.add_argument("--shadow-local", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=0)
    parser.add_argument("--max-branches", type=int, default=16)
    parser.add_argument("--accept-counts", type=int, nargs="+")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=48)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--target-mem-fraction-static",
        type=float,
        default=0.85,
        help=(
            "KV/state pool fraction for every Target server. Lower it when the "
            "in-process local fallback Draft runs a dLLM logits path that needs "
            "several GiB of transient fp32 workspace on top of the pool "
            "(--mode local/hybrid on a card where the pool leaves <10 GiB free)."
        ),
    )
    parser.add_argument(
        "--target-chunked-prefill-size",
        type=int,
        help=(
            "Override --chunked-prefill-size on every Target server. The "
            "in-process local fallback Draft is a dLLM that materializes full "
            "fp32 logits for every position in a forward (SDAR hard-codes "
            "return_full_logits=True), so its transient workspace scales with "
            "the chunk size -- ~0.6 GiB per 1k tokens of chunk on a 152k vocab. "
            "Lower it when the Target shares a card with that Draft."
        ),
    )
    parser.add_argument(
        "--disable-prefill-cuda-graph",
        action="store_true",
        help=(
            "Pass --cuda-graph-backend-prefill=disabled to every Target server. "
            "Required when the installed driver predates the CUDA runtime that "
            "cuda-python targets (e.g. H800 on driver 5xx with cuda-python 13), "
            "where prefill graph capture fails to initialize."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.wait_for_covered_result and args.lookup_wait_ms <= 0:
        parser.error("--wait-for-covered-result requires --lookup-wait-ms > 0")
    if args.draft_cuda_graph_max_tokens is None:
        args.draft_cuda_graph_max_tokens = remote_packed_graph_max_tokens(args)
    specs = commands(args)
    num_targets = target_count(args)
    urls = [f"http://127.0.0.1:{args.base_port + i}" for i in range(num_targets)]
    print(
        json.dumps(
            {"mode": args.mode, "worker_urls": urls, "commands": specs}, indent=2
        ),
        flush=True,
    )
    if args.dry_run:
        return
    assert_idle(args.gpus)
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = root / "docs/perf/results" / f"decoupled-{args.mode}-{stamp}"
    log_dir.mkdir(parents=True)
    # Publish the archive location before starting workers so wrapper scripts
    # can preserve launcher output even when model startup fails.
    print(f"Logs: {log_dir}", flush=True)
    (log_dir / "run.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "commands": specs,
                "worker_urls": urls,
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
            },
            indent=2,
        )
    )
    processes, files = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, gpu, cmd in specs:
            log = (log_dir / f"{name}.log").open("w")
            files.append(log)
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONUNBUFFERED": "1",
            }
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            # The drafter binds only after its native Engine has loaded the
            # model, so readiness also gates Target startup.
            if name == "drafter":
                wait_ready(
                    process=process,
                    endpoint=resolve_draft_endpoint(args),
                    timeout=args.startup_timeout,
                )
        target_processes = processes[-len(urls) :]
        for rank, process in enumerate(target_processes):
            wait_ready(
                process=process,
                port=args.base_port + rank,
                http=True,
                timeout=args.startup_timeout,
            )
        print(f"READY: {' '.join(urls)}", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError("a worker exited; stopping the experiment")
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in files:
            log.close()


if __name__ == "__main__":
    main()
