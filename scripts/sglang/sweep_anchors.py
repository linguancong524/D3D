import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def stop_tree(process):
    if process.poll() is not None:
        return
    parent = psutil.Process(process.pid)
    processes = parent.children(recursive=True) + [parent]
    for child in processes:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    if any(anchor < 1 for anchor in args.anchors):
        parser.error("anchor counts must be positive")
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(args.baseline).read_text())
    config = reference["configuration"]
    summaries = []
    for position, anchors in enumerate(args.anchors):
        name = f"anchor-{anchors}-run-{position + 1}"
        server_log = output / f"{name}-draft.log"
        metrics = output / f"{name}-metrics.json"
        report = output / f"{name}.json"
        if any(path.exists() for path in [server_log, metrics, report]):
            raise FileExistsError(name)
        environment = os.environ.copy()
        environment.update(
            ANCHORS=str(anchors), MAX_BRANCHES=str(16 * anchors), BLOCK_SIZE="16"
        )
        with server_log.open("w") as log:
            process = subprocess.Popen(
                ["bash", str(root / "scripts/sglang/launch.sh"), "draft",
                 "--metrics-file", str(metrics)],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 300
                while "READY " not in server_log.read_text():
                    if process.poll() is not None:
                        raise RuntimeError(f"Draft exited; inspect {server_log}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Draft not ready; inspect {server_log}")
                    time.sleep(2)
                print(f"Benchmarking anchors={anchors}, max_branches={16 * anchors}", flush=True)
                with (output / f"{name}-benchmark.log").open("w") as benchmark_log:
                    subprocess.run(
                        [sys.executable, str(root / "scripts/sglang/benchmark.py"),
                         "--dataset", args.dataset, "--tokenizer", args.tokenizer,
                         "--url", args.url, "--baseline", args.baseline,
                         "--samples", str(config["samples"]),
                         "--repetitions", str(config["repetitions"]),
                         "--max-new-tokens", str(config["max_new_tokens"]),
                         "--seed", str(config["seed"]), "--output", str(report)],
                        check=True, stdout=benchmark_log, stderr=subprocess.STDOUT,
                    )
                summary = json.loads(report.read_text())["summary"]
                summary.update(anchors=anchors, max_branches=16 * anchors, report=str(report))
                summaries.append(summary)
                (output / "summary.json").write_text(json.dumps(summaries, indent=2))
                print(json.dumps(summary, indent=2), flush=True)
            finally:
                stop_tree(process)
                time.sleep(3)


if __name__ == "__main__":
    main()
