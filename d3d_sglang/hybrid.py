import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/sglang"


def commands(environment, extra_args):
    concurrency = int(environment.get("MAX_RUNNING_REQUESTS", "1"))
    block_size = int(environment.get("BLOCK_SIZE", "16"))
    anchors = int(environment.get("ANCHORS", "1"))
    if min(concurrency, anchors) < 1 or block_size < 2:
        raise ValueError("concurrency and anchors must be positive; block size must be at least two")
    if block_size not in (4, 8, 16):
        raise ValueError("Hybrid supports the upstream block sizes 4, 8 and 16")
    command = [
        sys.executable, "scripts/decoupled/launch.py", "--mode", "hybrid",
        "--num-targets", "1", "--gpus", environment.get("TARGET_GPU", "0"),
        environment.get("DRAFT_GPU", "1"),
        "--target-model", environment.get("TARGET_MODEL", "Qwen/Qwen3-8B"),
        "--draft-model", environment.get("DRAFT_MODEL", "lingc8/D3D-1.7B-to-8B"),
        "--dllm-algorithm-config", f"scripts/decoupled/sdar_block{block_size}.yaml",
        "--block-size", str(block_size), "--candidate-topk", str(anchors + 1),
        "--num-anchors", str(anchors), "--max-branches", str(block_size * anchors),
        "--accept-counts", *map(str, range(1, block_size + 1)),
        "--draft-dtype", "float16", "--draft-attention-backend", "flashinfer",
        "--draft-enable-cuda-graph", "--packed-single-row", "--batched-branch-topk",
        "--packed-causal-pending", "--batching-window-ms", "0",
        "--lookup-wait-ms", "12", "--wait-for-covered-result",
        "--local-cuda-graph-max-tokens", "1024",
        "--local-cuda-graph-max-requests", str(concurrency),
        "--context-length", "8192", "--max-running-requests", str(concurrency),
        "--target-mem-fraction-static", "0.68", "--target-chunked-prefill-size", "2048",
        "--draft-mem-fraction-static", "0.85", "--disable-prefill-cuda-graph",
        "--trust-remote-code", "--startup-timeout", "900",
        "--base-port", environment.get("TARGET_PORT", "30000"),
        "--draft-endpoint", environment.get(
            "HYBRID_DRAFT_ENDPOINT", f"ipc:///dev/shm/d3d-hybrid-{os.getpid()}.sock"
        ), *extra_args, "--dry-run",
    ]
    specs = json.loads(subprocess.check_output(
        command, cwd=UPSTREAM, env=environment, text=True,
    ))
    for name, gpu, worker_command in specs["commands"]:
        if name.startswith("target"):
            worker_command[worker_command.index("--max-running-requests") + 1] = str(concurrency + 1)
            if environment.get("DETERMINISTIC", "0") != "1":
                worker_command.remove("--enable-deterministic-inference")
            worker_command += [
                "--disable-radix-cache", "--cuda-graph-max-bs-decode",
                environment.get("CUDA_GRAPH_MAX_BS_DECODE", str(concurrency)),
                "--dtype", "bfloat16",
            ]
    return specs


def main():
    parser = argparse.ArgumentParser(description="Launch the upstream two-GPU Hybrid backend")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    environment = os.environ.copy()
    specs = commands(environment, extra_args)
    print(json.dumps(specs, indent=2), flush=True)
    if args.dry_run:
        return
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "d3d_upstream_launcher", UPSTREAM / "scripts/decoupled/launch.py"
    )
    launcher = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(launcher)
    launcher.assert_idle([int(gpu) for _, gpu, _ in specs["commands"]])
    log_dir = ROOT / "outputs/sglang" / f"hybrid-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    log_dir.mkdir(parents=True)
    (log_dir / "commands.json").write_text(json.dumps(specs, indent=2))
    print(f"Logs: {log_dir}", flush=True)
    processes = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, gpu, worker_command in specs["commands"]:
            with (log_dir / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    worker_command, cwd=UPSTREAM,
                    env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            processes.append(process)
            if name == "drafter":
                endpoint = worker_command[worker_command.index("--endpoint") + 1]
                launcher.wait_ready(process=process, endpoint=endpoint, timeout=900)
            else:
                port = int(worker_command[worker_command.index("--port") + 1])
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Target exited; inspect {log_dir}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Target startup timeout; inspect {log_dir}")
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=10,
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(2)
        print("READY: " + " ".join(specs["worker_urls"]), flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError(f"Hybrid worker exited; inspect {log_dir}")
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
