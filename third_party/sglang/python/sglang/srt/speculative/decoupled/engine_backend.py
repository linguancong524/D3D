"""Native SGLang dLLM backend without an HTTP serving hop.

The backend owns one long-lived :class:`sglang.Engine` on the draft GPU and
submits token-id batches directly through the offline API.  Compared with the
old HTTP adapter this removes TCP, JSON encoding, and the ``/generate`` web
server from the required draft path.  The Engine still owns a Scheduler
subprocess and uses local IPC internally; moving the Target/Draft data plane to
CUDA IPC or NCCL is a separate phase.

Committed prefixes are submitted as token ids.  SGLang's radix cache provides
prefix reuse, while ``draft_many`` lets commits from every Target replica enter
one native dLLM scheduling batch.
"""

from __future__ import annotations

import time

from typing import Any

from sglang.srt.dllm.config import resolve_dllm_block_size
from sglang.srt.managers.io_struct import (
    DllmKvCommit,
    DllmKvControlReqInput,
    DllmPackedBranch,
    DllmPackedItem,
    DllmPackedReqInput,
    DllmPrefixStageItem,
    DllmPrefixStageReqInput,
)

from .protocol import (
    Branch,
    Commit,
    Proposal,
    proposal_identity,
    select_branches,
    vocabulary_fingerprint,
)


def _load_algorithm_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Please install PyYAML to use YAML config files. `pip install pyyaml`"
        ) from exc
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file) or {}
    except OSError as exc:
        raise ValueError(f"Failed to read dLLM algorithm config: {path}") from exc
    if not isinstance(config, dict):
        raise TypeError("dLLM algorithm config must be a mapping")
    return config


class DllmEngineBackend:
    """Draft blocks through a persistent offline SGLang dLLM engine."""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str | None = None,
        block_size: int = 16,
        num_anchors: int = 0,
        accept_counts: tuple[int, ...] = (),
        max_branches: int = 16,
        candidate_topk: int = 4,
        branch_group_size: int = 3,
        packed_single_row: bool = False,
        batched_branch_topk: bool = False,
        packed_causal_pending: bool = False,
        deterministic: bool = False,
        dllm_algorithm: str = "OneStep",
        dllm_algorithm_config: str | None = None,
        context_length: int = 32768,
        max_running_requests: int = 192,
        chunked_prefill_size: int = 1024,
        attention_backend: str = "flashinfer",
        dtype: str = "bfloat16",
        quantization: str | None = None,
        mem_fraction_static: float | None = None,
        enable_cuda_graph: bool = False,
        packed_cuda_graph_max_requests: int = 8,
        packed_cuda_graph_max_tokens: int = 1024,
        packed_cuda_graph_token_step: int = 0,
        trust_remote_code: bool = False,
    ) -> None:
        if (
            block_size < 2
            or num_anchors < 0
            or max_branches < 1
            or candidate_topk < 1
            or branch_group_size < 1
        ):
            raise ValueError("invalid block size or packed budget")
        if min(context_length, max_running_requests, chunked_prefill_size) < 1:
            raise ValueError("engine resource limits must be positive")
        if dllm_algorithm != "OneStep":
            raise ValueError(
                "anchor-conditioned decoupled drafting currently requires "
                "dllm_algorithm='OneStep'"
            )
        if packed_cuda_graph_max_requests < 1 or packed_cuda_graph_max_tokens < block_size:
            raise ValueError("invalid packed CUDA graph capacity")
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.accept_counts = accept_counts or tuple(range(1, block_size + 1))
        if any(not 1 <= count <= block_size for count in self.accept_counts):
            raise ValueError("accept counts outside block")
        self.max_branches = max_branches
        self.candidate_topk = candidate_topk

        from transformers import AutoConfig, AutoTokenizer

        model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        architectures = model_config.architectures
        if not architectures:
            raise ValueError("dLLM model config has no architecture")
        engine_block_size = resolve_dllm_block_size(
            architecture=architectures[0],
            algorithm_config=_load_algorithm_config(dllm_algorithm_config),
        )
        if engine_block_size != block_size:
            raise ValueError(
                "decoupled block_size does not match the dLLM Engine: "
                f"backend={block_size}, engine={engine_block_size}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path, trust_remote_code=trust_remote_code
        )
        vocab = tokenizer.get_vocab()
        self.fingerprint = vocabulary_fingerprint(vocab)
        self.vocab_size = max(vocab.values()) + 1
        self.packed_physical_forward_count = 0
        self.packed_model_forward_seconds = 0.0
        self.packed_build_seconds = 0.0
        self.packed_prepare_seconds = 0.0
        self.packed_postprocess_seconds = 0.0
        self.packed_protocol_seconds = 0.0
        self.prefix_stage_physical_forward_count = 0
        self.prefix_stage_model_forward_seconds = 0.0

        from sglang.srt.entrypoints.engine import Engine

        engine_args: dict[str, Any] = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "attention_backend": attention_backend,
            "dllm_algorithm": dllm_algorithm,
            "dllm_algorithm_config": dllm_algorithm_config,
            "context_length": context_length,
            "max_running_requests": max_running_requests,
            "chunked_prefill_size": chunked_prefill_size,
            # OneStep writes generated token IDs over MASK positions after the
            # model forward, so the stored denoising KV does not belong to those
            # generated IDs. It is also block-dependent and cannot be shared by
            # a global token-prefix radix tree. Disable radix reuse until native
            # drafting has D3DV2-style session-owned pending-token replay/crop.
            "disable_radix_cache": True,
            # Prefix staging remains eager. The experimental decode graph has
            # a strict admission gate for fixed-width, all-denoising
            # [anchor]+masks batches and falls back to eager for every other
            # DLLM_EXTEND shape or phase.
            "disable_decode_cuda_graph": not enable_cuda_graph,
            "disable_prefill_cuda_graph": True,
            "dllm_packed_cuda_graph_max_requests": (
                packed_cuda_graph_max_requests if enable_cuda_graph else 0
            ),
            "dllm_packed_cuda_graph_max_tokens": packed_cuda_graph_max_tokens,
            "dllm_packed_cuda_graph_token_step": packed_cuda_graph_token_step,
            "dllm_branch_group_size": branch_group_size,
            "dllm_packed_single_row": packed_single_row,
            "dllm_batched_branch_topk": batched_branch_topk,
            "dllm_packed_causal_pending": packed_causal_pending,
            "enable_deterministic_inference": deterministic,
        }
        if quantization is not None:
            engine_args["quantization"] = quantization
        if enable_cuda_graph:
            # The shared Draft service batches at most 64 required sessions or
            # 8x6 optional branches. Capturing the generic dLLM default through
            # bs=256 wastes several model-sized graphs and can OOM at startup.
            graph_bs = [1, 2, 4, 8, 16, 24, 32, 48, 64]
            engine_args["cuda_graph_max_bs_decode"] = graph_bs[-1]
            engine_args["cuda_graph_bs_decode"] = graph_bs
        if mem_fraction_static is not None:
            engine_args["mem_fraction_static"] = mem_fraction_static
        self.engine = Engine(**engine_args)

    def stage_many(self, *, items: list[tuple[str, tuple[int, ...]]]) -> None:
        if not items:
            return
        response = self.engine.dllm_prefix_stage(
            DllmPrefixStageReqInput(
                rid="dllm-prefix-stage",
                items=[
                    DllmPrefixStageItem(session=session, tokens=list(tokens))
                    for session, tokens in items
                ],
            )
        )
        self.prefix_stage_physical_forward_count += response.physical_forward_count
        self.prefix_stage_model_forward_seconds += response.model_forward_seconds

    def commit_many(self, *, items: list[Commit]) -> None:
        if not items or not hasattr(self.engine, "dllm_kv_control"):
            return
        self.engine.dllm_kv_control(
            DllmKvControlReqInput(
                rid="dllm-kv-commit",
                commits=[
                    DllmKvCommit(
                        session=item.session,
                        revision=item.revision,
                        proposal_id=item.proposal_id,
                        committed_prefix=list(item.committed_prefix),
                    )
                    for item in items
                ],
            )
        )

    def speculate(
        self,
        *,
        session: str,
        revision: int,
        prefix: tuple[int, ...],
        proposal: Proposal,
    ) -> dict[Branch, Proposal]:
        return self.speculate_many(
            items=[(session, revision, prefix, proposal)]
        )[session]

    def speculate_many(
        self,
        *,
        items: list[tuple[str, int, tuple[int, ...], Proposal]],
    ) -> dict[str, dict[Branch, Proposal]]:
        """Batch optimistic outcomes in one native explicit-mask forward."""
        outcomes: dict[str, dict[Branch, Proposal]] = {
            session: {} for session, _, _, _ in items
        }
        selected = {
            session: tuple(
                select_branches(
                    proposal=proposal,
                    num_anchors=self.num_anchors,
                    accept_counts=self.accept_counts,
                    max_branches=self.max_branches,
                )
            )
            for session, _, _, proposal in items
        }
        if not hasattr(self.engine, "dllm_packed_replay"):
            raise RuntimeError("optional drafter requires native packed replay")
        packed_items = [
            DllmPackedItem(
                session=session,
                revision=revision,
                proposal_id=proposal_identity(proposal),
                committed_tokens=list(prefix[:-1]),
                proposal_tokens=list(proposal.tokens),
                branches=[
                    DllmPackedBranch(
                        num_accept_tokens=branch.num_accept_tokens,
                        bonus_token=branch.bonus_token,
                    )
                    for branch in selected[session]
                ],
            )
            for session, revision, prefix, proposal in items
        ]
        round_trip_start = time.monotonic()
        response = self.engine.dllm_packed_replay(
            DllmPackedReqInput(
                rid="dllm-packed-replay",
                items=packed_items,
                branch_token_count=self.block_size,
                topk=max(2, self.candidate_topk),
            )
        )
        self.packed_physical_forward_count += response.physical_forward_count
        self.packed_model_forward_seconds += response.model_forward_seconds
        self.packed_build_seconds += response.build_seconds
        self.packed_prepare_seconds += response.prepare_seconds
        self.packed_postprocess_seconds += response.postprocess_seconds
        # Everything the round trip spends outside the Scheduler phases: request
        # encode, ZMQ, and reply decode.
        self.packed_protocol_seconds += max(
            0.0,
            time.monotonic()
            - round_trip_start
            - response.model_forward_seconds
            - response.build_seconds
            - response.prepare_seconds
            - response.postprocess_seconds,
        )
        for item_output in response.items:
            session_outcomes = outcomes[item_output.session]
            for outcome in item_output.outcomes:
                branch = Branch(
                    outcome.branch.num_accept_tokens,
                    outcome.branch.bonus_token,
                )
                session_outcomes[branch] = Proposal(
                    tuple(outcome.tokens),
                    tuple(tuple(row) for row in outcome.top_tokens),
                )
        return outcomes

    def close(self, *, session: str) -> None:
        if hasattr(self.engine, "dllm_kv_control"):
            self.engine.dllm_kv_control(
                DllmKvControlReqInput(
                    rid="dllm-kv-close", close_sessions=[session]
                )
            )

    def shutdown(self) -> None:
        self.engine.shutdown()
