#!/usr/bin/env python3
"""Sweep B4/8/16 x K2/3/4 x full-accept off/on at low client concurrency."""

import argparse
import asyncio
import datetime
import functools
import importlib.util
import itertools
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import msgspec
import zmq
from transformers import AutoTokenizer

import run_h800_hybrid_vs_ar as AB
from bench_decoupled_concurrency import run_measurement, warmup, write_json
from sglang.benchmark.decoupled_serving import encode_dataset_prompts
from sglang.srt.speculative.decoupled.protocol import Reply, Request, vocabulary_fingerprint

ROOT = AB.ROOT
SPEC = importlib.util.spec_from_file_location("grid_launcher", ROOT / "scripts/decoupled/launch.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def http_json(path):
    with urllib.request.urlopen(AB.URL + path, timeout=30) as response:
        return json.load(response)


def client_metrics():
    state = http_json("/get_server_info")["internal_states"][0]
    return state.get("decoupled_client_metrics", {})


def request_metrics(*, scheduler_pids=()):
    result = client_metrics()
    cpu_stat = Path("/sys/fs/cgroup/cpu.stat")
    if cpu_stat.exists():
        result.update({f"host_cpu_{key}": int(value)
                       for key, value in (line.split() for line in cpu_stat.read_text().splitlines())})
    pressure = Path("/sys/fs/cgroup/cpu.pressure")
    if pressure.exists():
        for line in pressure.read_text().splitlines():
            parts = line.split()
            values = dict(part.split("=", 1) for part in parts[1:])
            result[f"host_cpu_pressure_{parts[0]}_us"] = int(values["total"])
    for label, pid in scheduler_pids:
        run_ns, wait_ns, slices = map(int, Path(f"/proc/{pid}/schedstat").read_text().split())
        result[f"host_{label}_run_ns"] = run_ns
        result[f"host_{label}_runqueue_ns"] = wait_ns
        result[f"host_{label}_slices"] = slices
    return result


class RemoteStats:
    def __init__(self, *, endpoint, fingerprint, block):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)
        self.socket.connect(endpoint)
        self.decoder = msgspec.msgpack.Decoder(Reply)
        self._request(Request(rpc="grid-hello", op="hello", fingerprint=fingerprint, block_size=block))

    def _request(self, request):
        self.socket.send(msgspec.msgpack.encode(request))
        reply = self.decoder.decode(self.socket.recv())
        if reply.error:
            raise RuntimeError(reply.error)
        return reply

    def snapshot(self):
        deadline = time.monotonic() + 10
        while True:
            result = self._request(Request(rpc="grid-stats", op="stats")).metrics
            if result["remote_optional_queue"] == 0 and result["active_sessions"] == 0:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("remote work did not drain before metric snapshot")
            time.sleep(0.05)

    def close(self):
        self.socket.close()
        self.context.term()


def launch_args(*, block, topk, full_accept, endpoint, partial_accept_counts):
    template = Path(__file__).with_name("h800_hybrid_grid_defaults.json")
    args = json.loads(template.read_text())
    counts = list(partial_accept_counts) + ([block] if full_accept else [])
    branches = len(partial_accept_counts) * (topk - 1) + int(full_accept)
    positions = len(counts)
    args.update(
        block_size=block, candidate_topk=topk, num_anchors=topk - 1,
        max_branches=branches, accept_counts=counts,
        dllm_algorithm_config=f"scripts/decoupled/sdar_block{block}.yaml",
        draft_endpoint=endpoint, draft_metrics_interval=1.0,
        draft_cuda_graph_max_tokens=8 * block * (branches + positions),
        startup_timeout=900,
    )
    return argparse.Namespace(**args)


def start_workers(*, specs, directory, endpoint):
    LAUNCHER.assert_idle([0, 1])
    processes = []
    try:
        for name, gpu, command in specs:
            log_path = directory / f"{name}.log"
            if log_path.exists():
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S-%f")
                log_path = directory / f"{name}_resume-{stamp}.log"
            with log_path.open("x") as output:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=os.environ | {
                        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                    }, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                )
            processes.append(process)
            if name == "drafter":
                LAUNCHER.wait_ready(process=process, endpoint=endpoint, timeout=900)
        AB.wait_ready(processes[-1])
        return processes
    except BaseException:
        stop_workers(processes=processes)
        raise


def stop_workers(*, processes):
    for process in reversed(processes):
        AB.stop(process)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        occupied = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
        ], text=True).strip()
        if not occupied:
            return
        time.sleep(0.5)
    raise RuntimeError("GPU processes remain after experiment shutdown")


def delta(*, before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def measure_config(*, specs, directory, config, endpoint, fingerprint, prompts, warmup_prompt, reference, args):
    directory.mkdir(exist_ok=args.resume)
    manifest_path = directory / "run.json"
    run_manifest = {"configuration": config, "commands": specs}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(run_manifest)):
            raise ValueError(f"resumed commands differ: {directory.name}")
    else:
        write_json(path=manifest_path, payload=run_manifest)
    existing = {}
    for repetition in range(1, args.repetitions + 1):
        for concurrency in args.concurrency:
            artifact = directory / f"c{concurrency}_r{repetition}.json"
            if artifact.exists():
                payload = json.loads(artifact.read_text())
                if payload["experiment"] != config | {"repetition": repetition}:
                    raise ValueError(f"resumed configuration differs: {artifact}")
                existing[concurrency, repetition] = payload
                if reference is None and payload["summary"]["valid"]:
                    reference = payload
    if len(existing) == len(args.concurrency) * args.repetitions:
        print(f"REUSE: {directory.name}; all measurements archived", flush=True)
        return [config | {"concurrency": concurrency, "repetition": repetition,
                          "artifact": str(directory / f"c{concurrency}_r{repetition}.json"),
                          **payload["summary"]}
                for (concurrency, repetition), payload in existing.items()], reference
    processes = start_workers(specs=specs, directory=directory, endpoint=endpoint)
    stats = None
    measurements = []
    monitor = None
    monitor_log = None
    metrics_snapshot = None
    try:
        if config.get("per_request_metrics", False):
            groups = {process.pid: name for (name, _, _), process in zip(specs, processes)}
            process_rows = subprocess.check_output(["ps", "-eo", "pid=,pgid=,comm="], text=True)
            scheduler_pids = []
            for line in process_rows.splitlines():
                pid, group, comm = line.split(maxsplit=2)
                if int(group) in groups and "schedul" in comm:
                    scheduler_pids.append((groups[int(group)], int(pid)))
            write_json(path=directory / "scheduler_pids.json", payload=scheduler_pids)
            metrics_snapshot = functools.partial(request_metrics, scheduler_pids=scheduler_pids)
            monitor_log = (directory / "gpu_telemetry.csv").open("x")
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                "--format=csv", "-l", "1",
            ], stdout=monitor_log, stderr=subprocess.STDOUT)
        if endpoint is not None:
            stats = RemoteStats(endpoint=endpoint, fingerprint=fingerprint, block=config["block"])
        print(f"READY: {directory.name}", flush=True)
        for repetition in range(1, args.repetitions + 1):
            concurrencies = args.concurrency if repetition % 2 else list(reversed(args.concurrency))
            for concurrency in concurrencies:
                if (concurrency, repetition) in existing:
                    payload = existing[concurrency, repetition]
                    measurements.append(config | {"concurrency": concurrency, "repetition": repetition,
                                                  "artifact": str(directory / f"c{concurrency}_r{repetition}.json"),
                                                  **payload["summary"]})
                    continue
                with urllib.request.urlopen(AB.URL + "/flush_cache", timeout=30) as response:
                    if response.status != 200:
                        raise RuntimeError("cache flush failed")
                asyncio.run(warmup(url=AB.URL, prompt=warmup_prompt, output_len=args.output_len))
                remote_before = {} if stats is None else stats.snapshot()
                client_before = client_metrics()
                payload = run_measurement(
                    url=AB.URL, prompts=prompts, concurrency=concurrency,
                    output_len=args.output_len, model_path=AB.TARGET, reference=reference,
                    metrics_snapshot=metrics_snapshot,
                )
                payload["measurement_finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if reference is None and payload["summary"]["valid"]:
                    reference = payload
                remote_after = {} if stats is None else stats.snapshot()
                client_after = client_metrics()
                payload["experiment"] = config | {"repetition": repetition}
                payload["metrics"] = {
                    "client_before": client_before, "client_after": client_after,
                    "client_delta": delta(before=client_before, after=client_after),
                    "remote_before": remote_before, "remote_after": remote_after,
                    "remote_delta": delta(before=remote_before, after=remote_after),
                }
                artifact = directory / f"c{concurrency}_r{repetition}.json"
                write_json(path=artifact, payload=payload)
                summary = config | {"concurrency": concurrency, "repetition": repetition,
                                    "artifact": str(artifact), **payload["summary"]}
                measurements.append(summary)
                print("RESULT " + json.dumps(summary), flush=True)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if monitor_log is not None:
            monitor_log.close()
        if stats is not None:
            stats.close()
        stop_workers(processes=processes)
    return measurements, reference


def ar_specs():
    command = [
        AB.PYTHON, "-m", "sglang.launch_server", "--model-path", AB.TARGET,
        "--host", "127.0.0.1", "--port", "30000", "--mem-fraction-static", "0.68",
        "--tp-size", "1", "--context-length", "8192", "--max-running-requests", "16",
        "--attention-backend", "fa3", "--enable-deterministic-inference",
        "--trust-remote-code", "--cuda-graph-backend-prefill=disabled",
        "--chunked-prefill-size", "2048",
    ]
    return [("target-0", 0, command)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--topks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--partial-accept-counts", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--remote-draft-quantization", choices=["fp8"])
    args = parser.parse_args()
    if min(args.concurrency) < 1 or max(args.concurrency) > min(16, args.num_prompts):
        parser.error("concurrency must fit prompt count and the server cap of 16")
    if not args.partial_accept_counts or not all(1 <= count < min(args.blocks) for count in args.partial_accept_counts):
        parser.error("partial accept counts must be positive and smaller than every block")
    if args.resume and args.output_dir is None:
        parser.error("resume requires an explicit output directory")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive = args.output_dir or ROOT / "docs/perf/results" / f"h800-hybrid-grid-{stamp}"
    archive.mkdir(exist_ok=args.resume)
    manifest = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "working_diff": subprocess.check_output(["git", "diff"], text=True),
                "args": vars(args) | {"output_dir": str(archive)}, "dataset": AB.DATASET,
                "primary_metric": "median per-request output_tokens / E2E seconds; bounded closed-loop clients"}
    manifest_name = f"manifest_resume_{stamp}.json" if args.resume else "manifest.json"
    write_json(path=archive / manifest_name, payload=manifest)
    if args.resume:
        original_args = json.loads((archive / "manifest.json").read_text())["args"]
        for name in ("blocks", "topks", "concurrency", "repetitions", "num_prompts", "output_len"):
            if vars(args)[name] != original_args[name]:
                raise ValueError(f"resume changes experiment dimension: {name}")
        if args.remote_draft_quantization != original_args.get("remote_draft_quantization"):
            raise ValueError("resume changes remote draft quantization")
        stamp = archive.name.removeprefix("h800-hybrid-grid-")
    print(f"ARCHIVE: {archive}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(AB.TARGET)
    rows = [json.loads(line) for line in Path(AB.DATASET).read_text().splitlines() if line.strip()]
    all_prompts = encode_dataset_prompts(rows=rows, tokenizer=tokenizer, limit=args.num_prompts + 1)
    fingerprint = vocabulary_fingerprint(tokenizer.get_vocab())
    prompts, warmup_prompt = all_prompts[:-1], all_prompts[-1]
    measurements, reference = measure_config(
        specs=ar_specs(), directory=archive / "ar", config={"mode": "ar"}, endpoint=None,
        fingerprint=fingerprint, prompts=prompts, warmup_prompt=warmup_prompt, reference=None, args=args,
    )
    if reference is None:
        raise RuntimeError("no valid AR reference")
    reference_path = archive / "ar_reference.json"
    if reference_path.exists():
        if json.loads(reference_path.read_text()) != reference:
            raise ValueError("resumed AR reference differs")
    else:
        write_json(path=reference_path, payload=reference)
    for block, topk, full_accept in itertools.product(args.blocks, args.topks, (False, True)):
        name = f"b{block}_k{topk}_full{int(full_accept)}"
        config = {"mode": "hybrid", "block": block, "topk": topk, "full_accept": full_accept}
        if args.remote_draft_quantization is not None:
            config["remote_draft_quantization"] = args.remote_draft_quantization
        endpoint = f"ipc:///dev/shm/sglang-grid-{stamp}-{name}.sock"
        launch = launch_args(block=block, topk=topk, full_accept=full_accept, endpoint=endpoint,
                             partial_accept_counts=args.partial_accept_counts)
        launch.remote_draft_quantization = args.remote_draft_quantization
        try:
            result, _ = measure_config(
                specs=LAUNCHER.commands(launch), directory=archive / name, config=config,
                endpoint=endpoint, fingerprint=fingerprint, prompts=prompts,
                warmup_prompt=warmup_prompt, reference=reference, args=args,
            )
            measurements += result
        except Exception as exc:
            write_json(path=archive / name / "failure.json", payload={"error": repr(exc)})
            print(f"FAILED: {name}: {exc!r}", flush=True)
    write_json(path=archive / "measurements.json", payload=measurements)
    print(f"DONE: {archive}", flush=True)


if __name__ == "__main__":
    main()
