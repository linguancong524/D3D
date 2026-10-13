#!/usr/bin/env python3
"""Run an isolated hybrid ablation against archived exact AR token sequences."""

import argparse
import hashlib
import json
from pathlib import Path

import run_h800_hybrid_grid as GRID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path(
        "docs/perf/results/h800-hybrid-grid-20261011-105356"))
    parser.add_argument("--name", required=True)
    overrides_group = parser.add_mutually_exclusive_group()
    overrides_group.add_argument("--overrides", default="{}")
    overrides_group.add_argument("--overrides-file", type=Path)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--dataset-jsonl", type=Path, default=Path(GRID.AB.DATASET))
    parser.add_argument("--create-reference", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--per-request-metrics", action="store_true")
    args = parser.parse_args()
    args.resume = False
    reference = None if args.create_reference else json.loads(
        (args.reference_dir / "ar_reference.json").read_text()
    )
    if reference is not None:
        args.output_len = reference["configuration"]["output_len"]
        args.num_prompts = reference["configuration"]["num_prompts"]
    overrides = json.loads(
        args.overrides_file.read_text() if args.overrides_file else args.overrides
    )
    endpoint = f"ipc:///dev/shm/h800-opt-{args.name}.sock"
    launch = GRID.launch_args(block=16, topk=3, full_accept=True,
                              endpoint=endpoint, partial_accept_counts=[1, 2, 3])
    for key, value in overrides.items():
        if key not in vars(launch):
            raise ValueError(f"Unknown launch option: {key}")
        setattr(launch, key, value)
    launch.draft_cuda_graph_max_tokens = GRID.LAUNCHER.remote_packed_graph_max_tokens(launch)
    if args.create_reference and launch.mode != "ar":
        parser.error("--create-reference requires native AR mode")
    specs = GRID.ar_specs() if launch.mode == "ar" else GRID.LAUNCHER.commands(launch)
    tokenizer = GRID.AutoTokenizer.from_pretrained(GRID.AB.TARGET)
    rows = [json.loads(line) for line in args.dataset_jsonl.read_text().splitlines()
            if line.strip()]
    rows = [
        {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )} if "messages" in row else row
        for row in rows
    ]
    prompts = GRID.encode_dataset_prompts(rows=rows, tokenizer=tokenizer, limit=args.num_prompts + 1)
    if len(prompts) != args.num_prompts + 1:
        raise ValueError("Dataset must contain the measured prompts plus a holdout warmup")
    fingerprint = GRID.vocabulary_fingerprint(tokenizer.get_vocab())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = args.output_dir / args.name
    GRID.write_json(path=args.output_dir / f"{args.name}_dataset.json", payload={
        "path": str(args.dataset_jsonl),
        "sha256": hashlib.sha256(args.dataset_jsonl.read_bytes()).hexdigest(),
        "num_prompts": args.num_prompts, "output_len": args.output_len,
    })
    source_paths = [
        "python/sglang/srt/speculative/decoupled/local_drafter.py",
        "python/sglang/srt/speculative/decoupled/client.py",
        "python/sglang/srt/speculative/decoupled/transient_state.py",
        "python/sglang/srt/speculative/decoupled/engine_backend.py",
        "python/sglang/srt/speculative/decoupled/packed_layout.py",
        "python/sglang/srt/speculative/decoupled/worker.py",
        "python/sglang/srt/layers/attention/flashinfer_backend.py",
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        "python/sglang/srt/layers/logits_processor.py",
        "python/sglang/srt/managers/tp_worker.py",
        "python/sglang/srt/managers/scheduler.py",
        "python/sglang/srt/model_executor/forward_batch_info.py",
        "python/sglang/srt/server_args.py",
        "python/sglang/srt/model_executor/graph_shared_output.py",
        "docs/perf/scripts/bench_h800_hybrid_optimization.py",
        "docs/perf/scripts/run_h800_hybrid_grid.py",
        "docs/perf/scripts/bench_decoupled_concurrency.py",
        "scripts/decoupled/launch.py",
    ]
    GRID.write_json(path=args.output_dir / f"{args.name}_source_sha256.json", payload={
        name: hashlib.sha256((GRID.ROOT / name).read_bytes()).hexdigest()
        for name in source_paths
    })
    config = {"mode": launch.mode, "block": launch.block_size,
              "topk": launch.candidate_topk, "full_accept": launch.block_size in launch.accept_counts,
              "overrides": overrides,
              "performance_valid": not (launch.shadow_local or launch.timing_sync or launch.local_cuda_graph_check)}
    if args.per_request_metrics:
        if args.concurrency != [1]:
            parser.error("--per-request-metrics requires --concurrency 1")
        config["per_request_metrics"] = True
    results, generated_reference = GRID.measure_config(
        specs=specs, directory=directory, config=config,
        endpoint=endpoint if launch.mode == "hybrid" else None,
        fingerprint=fingerprint, prompts=prompts[:-1], warmup_prompt=prompts[-1],
        reference=reference, args=args,
    )
    if not all(row["valid"] for row in results):
        raise RuntimeError("Output parity or completion validation failed")
    if args.create_reference:
        GRID.write_json(path=args.output_dir / "ar_reference.json", payload=generated_reference)
    print("DONE: native AR reference archived" if args.create_reference else
          "DONE: exact AR parity for every measured request", flush=True)


if __name__ == "__main__":
    main()
