#!/usr/bin/env python3
"""Profile fixed-shape single-row packed Draft forwards on one GPU."""

import argparse
from pathlib import Path

from sglang.srt.speculative.decoupled.engine_backend import DllmEngineBackend
from sglang.srt.speculative.decoupled.protocol import Proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[3, 6])
    parser.add_argument(
        "--accept-counts", type=int, nargs="+", default=list(range(1, 17))
    )
    parser.add_argument(
        "--model-path", default="/workspace/models/d3d-1.7b-b16_for_qwen3-8b"
    )
    parser.add_argument("--tokenizer-path", default="/workspace/models/Qwen3-8B")
    parser.add_argument("--quantization", choices=("fp8", "w8a8_int8"))
    parser.add_argument(
        "--algorithm-config", default="scripts/decoupled/sdar_block16.yaml"
    )
    args = parser.parse_args()
    if any(size < 1 or size > 8 for size in args.batch_sizes):
        raise ValueError("batch sizes must be between 1 and 8")
    if not args.accept_counts or any(
        count < 1 or count > 16 for count in args.accept_counts
    ):
        raise ValueError("accept counts must be between 1 and 16")
    if len(set(args.accept_counts)) != len(args.accept_counts):
        raise ValueError("accept counts must be unique")

    backend = DllmEngineBackend(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        block_size=16,
        num_anchors=1,
        accept_counts=tuple(args.accept_counts),
        max_branches=len(args.accept_counts),
        candidate_topk=4,
        branch_group_size=3,
        packed_single_row=True,
        packed_causal_pending=True,
        deterministic=True,
        dllm_algorithm="OneStep",
        dllm_algorithm_config=args.algorithm_config,
        context_length=8192,
        max_running_requests=73,
        chunked_prefill_size=1024,
        attention_backend="flashinfer",
        dtype="bfloat16",
        quantization=args.quantization,
        mem_fraction_static=0.65,
        enable_cuda_graph=True,
        packed_cuda_graph_max_requests=8,
        packed_cuda_graph_max_tokens=1024,
        packed_cuda_graph_token_step=16 * (len(args.accept_counts) + 1),
        trust_remote_code=True,
    )
    prefix = tuple(range(1000, 1511)) + (1511,)
    tokens = (1511,) + tuple(range(2001, 2016))
    proposal = Proposal(
        tokens=tokens,
        top_tokens=tuple((token, 3000 + index) for index, token in enumerate(tokens)),
    )
    try:
        for count in args.batch_sizes:
            sessions = [f"profile-{count}-{index}" for index in range(count)]
            backend.stage_many(items=[(session, prefix[:-1]) for session in sessions])
            print(f"profiling batch={count}", flush=True)
            backend.engine.start_profile(
                output_dir=str(args.output_dir / f"batch{count}"),
                activities=["CPU", "GPU"],
                with_stack=False,
                record_shapes=True,
                profile_id=f"b16-single-row-batch{count}",
            )
            try:
                outcomes = backend.speculate_many(
                    items=[(session, 0, prefix, proposal) for session in sessions]
                )
            finally:
                backend.engine.stop_profile()
            assert all(
                len(outcomes[session]) == len(args.accept_counts)
                for session in sessions
            )
            print(
                f"batch={count} cumulative_gpu_seconds="
                f"{backend.packed_model_forward_seconds:.6f}",
                flush=True,
            )
            for session in sessions:
                backend.close(session=session)
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
