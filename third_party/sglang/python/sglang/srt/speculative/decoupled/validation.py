"""Fail closed for configurations outside the initial greedy TP=1 contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


def configure_target(args: ServerArgs) -> None:
    if args.device != "cuda":
        raise ValueError("DECOUPLED v1 target verification requires CUDA")
    if args.skip_tokenizer_init:
        raise ValueError(
            "DECOUPLED needs tokenizer initialization for vocabulary negotiation"
        )
    if args.disaggregation_mode != "null":
        raise ValueError(
            "DECOUPLED v1 requires each target to own both prefill and decode"
        )
    if args.decoupled_draft_endpoint and not args.decoupled_draft_endpoint.startswith(
        ("tcp://", "ipc://")
    ):
        raise ValueError("optional draft endpoint must use tcp:// or ipc://")
    if args.decoupled_draft_timeout <= 0:
        raise ValueError("--decoupled-draft-timeout must be positive")
    if args.decoupled_wait_for_covered_result and args.decoupled_lookup_wait <= 0:
        raise ValueError("covered-result wait requires --decoupled-lookup-wait > 0")
    if args.tp_size != 1 or args.pp_size != 1 or args.dp_size != 1:
        raise ValueError(
            "DECOUPLED v1 uses independent TP=1 target servers; launch one per GPU"
        )
    if args.enable_dp_attention or args.enable_lora or args.enable_two_batch_overlap:
        raise ValueError("DECOUPLED v1 does not support DP attention, LoRA or TBO")
    if args.speculative_draft_model_path is None:
        raise ValueError("DECOUPLED requires a Target-local SDAR draft model")
    if args.decoupled_spec_role != "null":
        raise ValueError(
            "DECOUPLED uses request/reply transport, not the legacy PULL mesh role"
        )
    size = args.speculative_num_draft_tokens or 16
    if not 2 <= size <= 64:
        raise ValueError("DECOUPLED block size must be between 2 and 64")
    args.speculative_num_draft_tokens = size
    args.speculative_eagle_topk = 1
    args.speculative_num_steps = size - 1
    args.disable_overlap_schedule = True
    args.enable_mixed_chunk = False
    if args.max_running_requests is None:
        args.max_running_requests = 48


def validate_request(req: Req) -> str | None:
    # SamplingParams normalizes temperature=0 into top_k=1.
    if req.sampling_params.top_k != 1:
        return "DECOUPLED v1 supports greedy requests only (temperature=0)"
    if req.multimodal_inputs is not None:
        return "DECOUPLED v1 supports text-only requests"
    return None
