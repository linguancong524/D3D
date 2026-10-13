#!/usr/bin/env python3
"""Reproducible C1 math test-split comparison: native AR vs hybrid eager/graph."""

import argparse
import hashlib
import json
import random
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MATH = Path("/inspire/ssd/project/sais-bio/public/Chinese_civilization/BenchMark/MATH-500/test.jsonl")
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SEED = 20261013


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def prepare(archive):
    archive.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(GSM, timeout=60) as response:
        gsm = response.read()
    sources = (("math500", MATH.read_bytes(), str(MATH), 500, "problem"),
               ("gsm8k", gsm, GSM, 1319, "question"))
    for name, raw, source, expected, field in sources:
        directory = archive / name
        directory.mkdir()
        with (directory / "test_source.jsonl").open("xb") as output:
            output.write(raw)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == expected, (name, len(rows))
        assert all(isinstance(row[field], str) and row[field].strip() for row in rows)
        assert len({row[field] for row in rows}) == len(rows)
        selected = random.Random(SEED).sample(range(len(rows)), 33)
        # Preserve the prior math prompt template; never include the answer.
        with (directory / "sample.jsonl").open("x") as output:
            for index in selected:
                output.write(json.dumps({"problem": rows[index][field], "source_index": index}, ensure_ascii=False) + "\n")
        write_json(directory / "dataset_manifest.json", {
            "source": source, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_rows": len(rows), "split": "test", "seed": SEED,
            "measured_source_indices": selected[:-1], "warmup_source_index": selected[-1],
            "sample_sha256": hashlib.sha256((directory / "sample.jsonl").read_bytes()).hexdigest(),
            "sampling": "32 uniformly sampled without replacement; one distinct holdout warmup",
            "prompt": "problem + step-by-step/boxed instruction; Qwen chat template; thinking disabled",
        })
    graph = json.loads((ROOT / "docs/perf/configs/h800_hybrid_local_graph_full_20261013.json").read_text())
    write_json(archive / "graph.json", graph)
    write_json(archive / "eager.json", graph | {"local_cuda_graph_max_tokens": 0})
    write_json(archive / "ar.json", {"mode": "ar"})
    write_json(archive / "plan.json", {
        "concurrency": 1, "num_prompts": 32, "repetitions": 3, "output_len": 512,
        "metric": "median of individual output_tokens / E2E including TTFT",
        "purpose": "fixed-output performance, not mathematical accuracy",
        "order": {"math500": ["ar", "eager", "graph"], "gsm8k": ["ar", "graph", "eager"]},
    })


def run(archive):
    for dataset, order in (("math500", ("ar", "eager", "graph")),
                           ("gsm8k", ("ar", "graph", "eager"))):
        directory = archive / dataset
        for mode in order:
            command = [str(ROOT / ".venv/bin/python3"),
                       "docs/perf/scripts/bench_h800_hybrid_optimization.py",
                       "--output-dir", str(directory), "--name", mode,
                       "--reference-dir", str(directory),
                       "--dataset-jsonl", str(directory / "sample.jsonl"),
                       "--overrides-file", str(archive / f"{mode}.json"),
                       "--concurrency", "1", "--repetitions", "3",
                       "--num-prompts", "32", "--output-len", "512",
                       "--per-request-metrics"]
            if mode == "ar":
                command.append("--create-reference")
            print(f"START {dataset}/{mode}", flush=True)
            with (directory / f"{mode}_runner.log").open("x") as output:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            print(f"FINISH {dataset}/{mode}: exit={result.returncode}", flush=True)
            if result.returncode:
                raise RuntimeError(f"See {directory}/{mode}_runner.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    prepare(args.archive)
    run(args.archive)


if __name__ == "__main__":
    main()
