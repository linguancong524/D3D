#!/usr/bin/env python3
"""Run matched single-GPU AR and two-GPU hybrid workloads on H800."""

import datetime
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = str(ROOT / ".venv/bin/python3")
TARGET = "/inspire/ssd/project/sais-bio/public/Chinese_civilization/models/Qwen/Qwen3-8B"
DRAFT = "/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908"
DATASET = "/inspire/ssd/project/sais-bio/public/Chinese_civilization/BenchMark/MATH-500/test.jsonl"
URL = "http://127.0.0.1:30000"


def stop(process):
    if process.poll() is None:
        # Signal only the process group created by this experiment.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server startup exceeded 900 seconds")


def benchmark(command, log):
    with log.open("x") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    print(f"BENCHMARK: {log} status={result.returncode}", flush=True)
    if result.returncode:
        raise RuntimeError(f"benchmark failed; see {log}")


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = ROOT / "docs/perf/results" / f"h800-hybrid-vs-ar-{stamp}"
    directory.mkdir()
    print(f"ARCHIVE: {directory}", flush=True)
    common = [
        "--model-path", TARGET, "--host", "127.0.0.1", "--port", "30000",
        "--mem-fraction-static", "0.68", "--tp-size", "1",
        "--context-length", "8192", "--max-running-requests", "16",
        "--attention-backend", "fa3", "--enable-deterministic-inference",
        "--trust-remote-code", "--cuda-graph-backend-prefill=disabled",
        "--chunked-prefill-size", "2048",
    ]
    commands = {
        "ar": [PYTHON, "-m", "sglang.launch_server", *common],
        "hybrid": [
            PYTHON, "scripts/decoupled/launch.py", "--mode", "hybrid",
            "--num-targets", "1", "--gpus", "0", "1",
            "--target-model", TARGET, "--draft-model", DRAFT,
            "--dllm-algorithm-config", "scripts/decoupled/sdar_block16.yaml",
            "--block-size", "16", "--num-anchors", "2", "--max-branches", "6",
            "--accept-counts", "1", "2", "3", "--speculation-batch-size", "8",
            "--batching-window-ms", "1", "--max-running-requests", "16",
            "--target-mem-fraction-static", "0.68",
            "--target-chunked-prefill-size", "2048",
            "--draft-mem-fraction-static", "0.85",
            "--trust-remote-code", "--disable-prefill-cuda-graph",
            "--startup-timeout", "900",
            "--draft-endpoint", f"ipc:///dev/shm/sglang-h800-ab-{stamp}.sock",
        ],
    }
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_diff": subprocess.check_output(["git", "diff", "--", "scripts/decoupled/launch.py"], cwd=ROOT, text=True),
        "commands": commands,
        "dataset": DATASET,
        "repetitions": 2,
        "cache_policy": "flush_cache before each measurement; serial warmup remains enabled",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for mode, command in commands.items():
        gpu_processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
        ], text=True).strip()
        if gpu_processes:
            raise RuntimeError(f"GPUs are occupied: {gpu_processes}")
        with (directory / f"{mode}_gpu_before.txt").open("x") as output:
            subprocess.run(["nvidia-smi"], stdout=output, check=True)
        with (directory / f"{mode}_server.log").open("x") as output:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            try:
                wait_ready(process)
                print(f"READY: {mode}", flush=True)
                for repetition in (1, 2):
                    for workload in ("serial", "saturation"):
                        with urllib.request.urlopen(URL + "/flush_cache", timeout=30) as response:
                            if response.status != 200:
                                raise RuntimeError("cache flush failed")
                        artifact = directory / f"{mode}_{workload}_r{repetition}.json"
                        if workload == "serial":
                            bench = [PYTHON, "docs/perf/scripts/bench_decoupled_serial.py", "--url", URL, "--num-prompts", "10"]
                        else:
                            bench = [PYTHON, "-m", "sglang.benchmark.decoupled_serving", "--worker-urls", URL, "--num-prompts", "100", "--request-rate", "inf", "--seed", "42"]
                        bench += ["--model-path", TARGET, "--dataset-jsonl", DATASET, "--output-len", "256", "--output", str(artifact)]
                        benchmark(bench, directory / f"{mode}_{workload}_r{repetition}.log")
                        payload = json.loads(artifact.read_text())
                        print(json.dumps({"mode": mode, "workload": workload, "repetition": repetition, **payload["summary"]}), flush=True)
            finally:
                stop(process)
        time.sleep(5)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
