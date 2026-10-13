"""Launch the optional-only native packed SDAR service."""

import argparse
import logging
import signal
import threading


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--backend",
        choices=("engine",),
        default="engine",
        help="native offline dLLM Engine with packed replay",
    )
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--quantization", choices=("fp8", "w8a8_int8"))
    parser.add_argument(
        "--dllm-algorithm",
        default="OneStep",
        help="native anchor-conditioned backend currently requires OneStep",
    )
    parser.add_argument("--dllm-algorithm-config")
    parser.add_argument("--engine-attention-backend", default="flashinfer")
    parser.add_argument("--engine-max-running-requests", type=int, default=192)
    parser.add_argument("--engine-mem-fraction-static", type=float)
    parser.add_argument(
        "--engine-enable-cuda-graph",
        action="store_true",
        help="enable token-bucket CUDA graphs for native packed replay",
    )
    parser.add_argument(
        "--engine-packed-cuda-graph-max-requests", type=int, default=8
    )
    parser.add_argument(
        "--engine-packed-cuda-graph-max-tokens", type=int, default=1024
    )
    parser.add_argument("--engine-packed-cuda-graph-token-step", type=int, default=0)
    parser.add_argument("--endpoint", default="ipc:///dev/shm/sglang-decoupled.sock")
    parser.add_argument("--mode", choices=("packed",), default="packed")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=0)
    parser.add_argument("--accept-counts", type=int, nargs="+", default=[])
    parser.add_argument("--max-branches", type=int, default=16)
    parser.add_argument("--candidate-topk", type=int, default=4)
    parser.add_argument("--branch-group-size", type=int, default=3)
    parser.add_argument("--packed-single-row", action="store_true")
    parser.add_argument("--batched-branch-topk", action="store_true")
    parser.add_argument("--packed-causal-pending", action="store_true")
    parser.add_argument("--engine-deterministic", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=512)
    parser.add_argument("--max-sessions-per-peer", type=int)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--lease-seconds", type=float, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--speculation-batch-size", type=int, default=8)
    parser.add_argument("--batching-window-ms", type=float, default=1)
    parser.add_argument("--metrics-interval", type=float, default=10)
    args = parser.parse_args()
    from sglang.srt.speculative.decoupled.service import DraftService, serve

    logging.basicConfig(level=logging.INFO)
    from sglang.srt.speculative.decoupled.engine_backend import DllmEngineBackend

    backend = DllmEngineBackend(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            block_size=args.block_size,
            num_anchors=args.num_anchors,
            accept_counts=tuple(args.accept_counts),
            max_branches=args.max_branches,
            candidate_topk=args.candidate_topk,
            branch_group_size=args.branch_group_size,
            packed_single_row=args.packed_single_row,
            batched_branch_topk=args.batched_branch_topk,
            packed_causal_pending=args.packed_causal_pending,
            deterministic=args.engine_deterministic,
            dllm_algorithm=args.dllm_algorithm,
            dllm_algorithm_config=args.dllm_algorithm_config,
            context_length=args.max_context,
            max_running_requests=args.engine_max_running_requests,
            chunked_prefill_size=args.prefill_chunk_size,
            attention_backend=args.engine_attention_backend,
            dtype=args.dtype,
            quantization=args.quantization,
            mem_fraction_static=args.engine_mem_fraction_static,
            enable_cuda_graph=args.engine_enable_cuda_graph,
            packed_cuda_graph_max_requests=args.engine_packed_cuda_graph_max_requests,
            packed_cuda_graph_max_tokens=args.engine_packed_cuda_graph_max_tokens,
            packed_cuda_graph_token_step=args.engine_packed_cuda_graph_token_step,
            trust_remote_code=args.trust_remote_code,
        )
    service = DraftService(
        backend=backend,
        max_sessions=args.max_sessions,
        max_sessions_per_peer=args.max_sessions_per_peer,
        max_context=args.max_context,
        lease_seconds=args.lease_seconds,
        batch_size=args.batch_size,
        speculation_batch_size=args.speculation_batch_size,
        batching_window_seconds=args.batching_window_ms / 1000,
    )
    # The dLLM Engine reconfigures logging via ServerArgs.prepare_server_args
    # (logging.basicConfig(level=<engine log_level> default "error", force=True)),
    # which silences the root logger and drops the DraftService metrics below.
    # Pin the decoupled service logger to INFO so metrics_interval stats are
    # always flushed to drafter.log, independent of the Engine's log level.
    logging.getLogger("sglang.srt.speculative.decoupled.service").setLevel(logging.INFO)
    logging.info(
        "Starting optional packed drafter at %s, block=%d fingerprint=%s",
        args.endpoint,
        args.block_size,
        backend.fingerprint,
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        serve(
            service=service,
            endpoint=args.endpoint,
            stop=stop,
            metrics_interval=args.metrics_interval,
        )
    finally:
        logging.info("Draft service stopped")


if __name__ == "__main__":
    main()
