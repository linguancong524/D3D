"""Stateless full-replay path for Target-local required SDAR drafting.

The GPU runner consumes this layout in one full-replay forward. Keeping the planner
pure makes the safety-critical mask independently testable. Required drafting
deliberately does not retain SDAR KV: cached-prefix execution changes proposal
quality for this checkpoint, while causal full replay preserves the validated
acceptance baseline. Remote optional drafting owns its transient KV separately.
"""

from __future__ import annotations

import logging
import time

import msgspec
import numpy as np
import torch

from .protocol import vocabulary_fingerprint

logger = logging.getLogger(__name__)


def validate_vocabulary_compatibility(
    *,
    target_tokenizer,
    target_vocab_size: int,
    draft_tokenizer,
    draft_vocab_size: int,
    mask_token: int,
) -> str:
    """Require identical Target token IDs, allowing the Draft-only MASK.

    SDAR checkpoints for Qwen add ``<|MASK|>`` at the first unused tokenizer
    ID while keeping the model's padded embedding vocabulary the same size as
    the Target. Requiring byte-for-byte tokenizer equality would therefore
    reject the intended checkpoint. Every Target token must still map to the
    exact same ID, and MASK is the only permitted Draft-only entry.
    """
    if target_vocab_size != draft_vocab_size:
        raise ValueError(
            "DECOUPLED requires equal Target/local draft model vocabulary sizes: "
            f"target={target_vocab_size}, draft={draft_vocab_size}"
        )
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    mismatches = {
        token
        for token, token_id in target_vocab.items()
        if draft_vocab.get(token) != token_id
    }
    draft_only = set(draft_vocab) - set(target_vocab)
    if mismatches or len(draft_only) != 1 or {
        draft_vocab[token] for token in draft_only
    } != {mask_token}:
        raise ValueError(
            "DECOUPLED requires identical Target token IDs and exactly one "
            "Draft-only MASK token"
        )
    return vocabulary_fingerprint(target_vocab)


class RequiredDraftLayout(msgspec.Struct, frozen=True):
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    attention_mask: torch.Tensor
    replay_len: int


class LocalContinuousPrefixState(msgspec.Struct):
    tokens: tuple[int, ...]
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]]


def plan_required_draft_layout(
    *,
    committed_prefix: tuple[int, ...],
    block_size: int,
    mask_token: int,
) -> RequiredDraftLayout:
    """Replay the causal prefix and append one anchor-conditioned SDAR block.

    Replay rows are causal. Block rows see the replayed prefix and the complete
    denoising block; replay rows cannot see the anchor/MASK scratch.
    """
    if block_size < 2:
        raise ValueError("required draft block must contain anchor and MASK")
    if not committed_prefix:
        raise ValueError("committed prefix must contain a Target bonus")
    safe_prefix = committed_prefix[:-1]
    block = (committed_prefix[-1],) + (mask_token,) * (block_size - 1)
    tokens = safe_prefix + block
    positions = tuple(range(len(tokens)))
    query_len = len(tokens)
    replay_len = len(safe_prefix)
    # Construct on the host without one PyTorch dispatch per prefix token.
    # Block rows see every key, while replay rows remain strictly causal.
    offsets = np.arange(query_len)
    mask = offsets[None, :] <= offsets[:, None]
    mask[replay_len:, :] = True

    return RequiredDraftLayout(
        input_tokens=tokens,
        positions=positions,
        attention_mask=torch.from_numpy(mask),
        replay_len=replay_len,
    )


class LocalSDARDrafter:
    """In-process stateless drafter with an independent non-radix KV pool."""

    def __init__(self, *, server_args, gpu_id, ps, nccl_port, target_worker) -> None:
        from sglang.srt.dllm.config import DLLM_PARAMS
        from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker

        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            target_model_config=target_worker.model_runner.model_config,
            algo_label="DECOUPLED local required",
        )
        self.worker = bundle.draft_worker
        self.model_runner = bundle.draft_model_runner
        architectures = self.model_runner.model_config.hf_config.architectures or ()
        if not architectures or architectures[0] not in DLLM_PARAMS:
            raise ValueError("DECOUPLED local draft must be an SDAR diffusion model")
        self.mask_token = int(DLLM_PARAMS[architectures[0]]["mask_id"])

        from sglang.srt.utils.hf_transformers import get_tokenizer

        draft_tokenizer = get_tokenizer(
            server_args.speculative_draft_model_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            tokenizer_revision=server_args.speculative_draft_model_revision,
            tokenizer_backend=server_args.tokenizer_backend,
        )
        self.fingerprint = validate_vocabulary_compatibility(
            target_tokenizer=target_worker.tokenizer,
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            draft_tokenizer=draft_tokenizer,
            draft_vocab_size=self.model_runner.model_config.vocab_size,
            mask_token=self.mask_token,
        )
        self.block_size = int(server_args.speculative_num_draft_tokens)
        # Target verification is a topk=1 chain. Anchor alternatives belong
        # only to the SDAR proposal and must not change the Target tree shape.
        self.topk = int(server_args.decoupled_candidate_topk)
        if self.topk < 1:
            raise ValueError("DECOUPLED candidate top-k must be positive")
        self._proposal_log_budget = 3
        self.req_to_token_pool = None
        self.token_allocator = None
        self.tree_cache = None
        self.prefix_reuse = server_args.decoupled_local_prefix_reuse
        from sglang.srt.runtime_context import get_spec

        self.graph_max_tokens = get_spec().decoupled_local_cuda_graph_max_tokens
        self.graph_max_requests = get_spec().decoupled_local_cuda_graph_max_requests
        self.graph_check = get_spec().decoupled_local_cuda_graph_check
        if self.graph_max_tokens < 0 or self.graph_max_requests < 1:
            raise ValueError("invalid local SDAR CUDA graph capacity")
        if self.graph_max_tokens and self.prefix_reuse:
            raise ValueError("local SDAR CUDA graph requires stateless full replay")
        if self.graph_check and not self.graph_max_tokens:
            raise ValueError("local SDAR graph comparison requires graph capture")
        if self.graph_check:
            logger.warning("Local SDAR graph comparison enabled; timings are diagnostic only")
        self.continuous_prefix_states: dict[str, LocalContinuousPrefixState] = {}

    def allocate_with_target(self, target_worker) -> None:
        """Resolve one data-driven token cap for two independent KV pools."""
        target = target_worker.model_runner
        local = self.model_runner
        target.init_kv_cache_configurator()
        local.init_kv_cache_configurator()
        budget = target.kv_cache_configurator._profile_available_bytes(
            target.pre_model_load_memory
        )
        target_full = target.kv_cache_configurator.config_from_budget(budget)
        local_full = local.kv_cache_configurator.config_from_budget(budget)
        if target_full.max_total_num_tokens <= 0 or local_full.max_total_num_tokens <= 0:
            raise RuntimeError("DECOUPLED cannot allocate Target/local SDAR KV pools")
        target_cost = budget / target_full.max_total_num_tokens
        local_cost = budget / local_full.max_total_num_tokens
        shared_cap = int(budget / (target_cost + local_cost))
        if shared_cap < self.block_size:
            raise RuntimeError("DECOUPLED combined KV capacity is below one draft block")
        target_config = target.kv_cache_configurator.config_from_budget(
            int(target_cost * shared_cap), cap_tokens=shared_cap
        )
        local_config = local.kv_cache_configurator.config_from_budget(
            int(local_cost * shared_cap), cap_tokens=shared_cap
        )
        shared_cap = min(
            target_config.max_total_num_tokens, local_config.max_total_num_tokens
        )
        target_config = target.kv_cache_configurator.config_from_budget(
            int(target_cost * shared_cap), cap_tokens=shared_cap
        )
        local_config = local.kv_cache_configurator.config_from_budget(
            int(local_cost * shared_cap), cap_tokens=shared_cap
        )

        # config_from_budget intentionally leaves request capacity unresolved;
        # ordinary ModelRunner allocation fills it in via
        # _resolve_memory_pool_config(). The paired DECOUPLED allocator bypasses
        # that entry point, so finalize both independent pools explicitly.
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
        from sglang.srt.runtime_context import get_schedule

        for runner, config in ((target, target_config), (local, local_config)):
            configurator = runner.kv_cache_configurator
            config.max_running_requests = configurator.resolve_max_num_reqs(
                config.max_total_num_tokens
            )
            config = create_memory_pool_configurator(
                configurator
            ).finalize_with_max_running_requests(config)
            config.mem_fraction_static = get_schedule().mem_fraction_static
            if runner is target:
                target_config = config
            else:
                local_config = config

        target_worker.alloc_memory_pool(memory_pool_config=target_config)
        self.worker.alloc_memory_pool(memory_pool_config=local_config)
        self.req_to_token_pool, self.token_allocator = self.worker.get_memory_pool()
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        self.tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_allocator,
                page_size=self.token_allocator.page_size,
            )
        )

    def init_attention_backends(self) -> None:
        self.worker.init_attention_backends()

    def init_cuda_graphs(self) -> None:
        # Initialize eager fallback without accidentally capturing Target verify
        # geometry inherited through the shared ServerArgs.
        self.worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        if not self.graph_max_tokens:
            return
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )
        from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
            PackedDllmGraphConfig,
        )
        from sglang.srt.utils import get_available_gpu_memory

        runner = self.model_runner
        if runner.device != "cuda" or check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        ):
            raise ValueError("local SDAR packed graph requires enabled CUDA decode graphs")
        if self.graph_max_tokens > self.req_to_token_pool.max_context_len:
            raise ValueError("local SDAR graph token capacity exceeds KV context capacity")
        if self.graph_max_requests + 1 > self.req_to_token_pool.size:
            raise ValueError("local SDAR graph needs an extra padding request slot")
        config = PackedDllmGraphConfig(
            block_size=self.block_size,
            max_requests=self.graph_max_requests,
            max_tokens=self.graph_max_tokens,
            token_step=max(128, self.block_size),
        )
        # Do not resize/replace the Target's shared output storage: its graphs
        # already hold those addresses. The local graph owns its larger buffer.
        runner.graph_shared_output = GraphSharedOutput(
            device=runner.device,
            max_rows=self.graph_max_tokens,
        )
        before_mem = get_available_gpu_memory(runner.device, runner.gpu_id)
        start = time.perf_counter()
        runner.decode_cuda_graph_runner = DecodeCudaGraphRunner(
            runner,
            packed_dllm_config=config,
        )
        elapsed = time.perf_counter() - start
        memory = before_mem - get_available_gpu_memory(runner.device, runner.gpu_id)
        runner.graph_memory_usage = runner.graph_memory_usage | {"draft_decode": memory}
        runner.graph_time_usage = runner.graph_time_usage | {"draft_decode": elapsed}
        logger.info(
            "Local SDAR packed CUDA graph captured: requests<=%d tokens<=%d, %.2f s, %.2f GiB",
            self.graph_max_requests,
            self.graph_max_tokens,
            elapsed,
            memory,
        )

    def _free_tail(self, *, req_pool_idx: int, target: int, end: int) -> None:
        if end <= target:
            return
        from sglang.srt.utils.common import ceil_align

        start = (
            target
            if self.token_allocator.page_size == 1
            else ceil_align(target, self.token_allocator.page_size)
        )
        if start < end:
            self.token_allocator.free(
                self.req_to_token_pool.req_to_token[req_pool_idx, start:end]
            )

    def _build_requests(self, sources):
        from array import array

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        synthetic = []
        layouts = []
        for source in sources:
            committed = tuple(source.origin_input_ids) + tuple(source.output_ids)
            layout = plan_required_draft_layout(
                committed_prefix=committed,
                block_size=self.block_size,
                mask_token=self.mask_token,
            )
            req = Req(
                f"local-required-{source.rid}:{id(source)}",
                "",
                array("q", layout.input_tokens),
                SamplingParams(max_new_tokens=1),
                dllm_config=None,
                eos_token_ids=self.model_runner.model_config.hf_eos_token_id,
                vocab_size=self.model_runner.model_config.vocab_size,
            )
            req.init_next_round_input(None)
            req.set_extend_range(0, len(layout.input_tokens))
            synthetic.append(req)
            layouts.append(layout)
        return synthetic, layouts

    def _forward(
        self, *, synthetic, layouts, phase_metrics=None, check_graph=False
    ) -> torch.Tensor:
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if not synthetic:
            return None
        batch = ScheduleBatch.init_new(
            reqs=synthetic,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.DECOUPLED,
            dllm_config=None,
        )
        batch.prepare_for_extend()
        if batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        batch.forward_mode = ForwardMode.DLLM_EXTEND
        positions = torch.tensor(
            [position for layout in layouts for position in layout.positions],
            dtype=torch.int64,
            device=batch.device,
        )
        attention_mask = torch.cat(
            [layout.attention_mask.flatten() for layout in layouts]
        ).to(device=batch.device, dtype=torch.uint8)
        graph_ct_before = (
            phase_metrics.get("local_required_cuda_graph_ct", 0)
            if phase_metrics is not None else 0
        )
        output = self.worker.forward_dllm_packed(
            batch,
            positions=positions,
            attention_mask=attention_mask,
            metrics=phase_metrics,
        )
        logits = output.full_logits
        expected_rows = sum(len(layout.input_tokens) for layout in layouts)
        if logits is None or logits.shape[0] != expected_rows:
            raise RuntimeError("local SDAR did not return aligned per-token logits")
        if (
            check_graph
            and phase_metrics is not None
            and phase_metrics.get("local_required_cuda_graph_ct", 0) > graph_ct_before
        ):
            # Diagnostic only; keep graph storage alive before eager can reuse
            # output buffers. Never include this extra forward in speed results.
            logits = logits.clone()
            eager = self.worker.forward_dllm_packed(
                batch,
                positions=positions,
                attention_mask=attention_mask,
                allow_cuda_graph=False,
            ).full_logits
            graph_proposals = self._postprocess(
                sources=synthetic, layouts=layouts, logits=logits
            )
            eager_proposals = self._postprocess(
                sources=synthetic, layouts=layouts, logits=eager
            )
            row = 0
            for layout in layouts:
                start = row + layout.replay_len + 1
                end = row + len(layout.input_tokens)
                graph_block = logits[start:end].float()
                eager_block = eager[start:end].float()
                max_error = (graph_block - eager_block).abs().max().item()
                graph_best = graph_block.argmax(dim=-1, keepdim=True)
                regret = (
                    eager_block.max(dim=-1).values
                    - eager_block.gather(-1, graph_best).squeeze(-1)
                ).max().item()
                phase_metrics["local_graph_check_logit_max_abs"] = max(
                    phase_metrics["local_graph_check_logit_max_abs"], max_error
                )
                phase_metrics["local_graph_check_flip_regret_max"] = max(
                    phase_metrics["local_graph_check_flip_regret_max"], regret
                )
                phase_metrics["local_graph_check_nonfinite_ct"] += int(
                    not torch.isfinite(graph_block).all().item()
                    or not torch.isfinite(eager_block).all().item()
                )
                row = end
            for rid, proposal in graph_proposals.items():
                other = eager_proposals[rid]
                phase_metrics["local_graph_check_ct"] += 1
                phase_metrics["local_graph_check_tokens_exact_ct"] += int(
                    proposal.tokens == other.tokens
                )
                phase_metrics["local_graph_check_topk_exact_ct"] += int(
                    proposal.top_tokens == other.top_tokens
                )
        return logits

    def _postprocess(self, *, sources, layouts, logits) -> dict[str, object]:
        proposals = {}
        row = 0
        for source, layout in zip(sources, layouts):
            width = len(layout.input_tokens)
            proposal = proposal_from_logits(
                anchor=layout.input_tokens[layout.replay_len],
                full_logits=logits[row : row + width],
                replay_len=layout.replay_len,
                block_size=self.block_size,
                topk=self.topk,
            )
            row += width
            proposals[source.rid] = proposal
            if self._proposal_log_budget > 0:
                logger.info(
                    "DECOUPLED local required proposal rid=%s anchor=%d tokens=%s replay_len=%d",
                    source.rid,
                    proposal.tokens[0],
                    proposal.tokens,
                    layout.replay_len,
                )
                self._proposal_log_budget -= 1
        return proposals

    def release_request(self, *, rid: str) -> None:
        self.continuous_prefix_states.pop(rid, None)

    def clear_prefix_states(self) -> None:
        self.continuous_prefix_states.clear()

    def _release_requests(self, *, synthetic) -> None:
        for req in synthetic:
            if req.req_pool_idx is None:
                continue
            allocated_len = int(req.kv.kv_allocated_len) if req.kv is not None else 0
            self._free_tail(req_pool_idx=req.req_pool_idx, target=0, end=allocated_len)
            self.req_to_token_pool.free_slots.append(req.req_pool_idx)
            req.req_pool_idx = None

    def _continuous_attention_many(
        self,
        *,
        attention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query_lens: list[int],
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None],
        store_kv: bool,
        ragged_wrapper,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        qkv, _ = attention.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )
        from sglang.srt.models.utils import apply_qk_norm

        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=attention.q_norm,
            k_norm=attention.k_norm,
            head_dim=attention.head_dim,
            allow_inplace=False,
        )
        q, k = attention.rotary_emb(positions, q, k)
        q_parts = q.split(query_lens)
        k_parts = k.split(query_lens)
        v_parts = v.split(query_lens)
        queries = []
        keys = []
        values = []
        retained = []
        key_lens = []
        for q_part, k_part, v_part, cached in zip(
            q_parts, k_parts, v_parts, past_kv
        ):
            query = q_part.view(-1, attention.num_heads, attention.head_dim)
            key = k_part.view(-1, attention.num_kv_heads, attention.head_dim)
            value = v_part.view(-1, attention.num_kv_heads, attention.head_dim)
            if cached is not None:
                key = torch.cat([cached[0], key], dim=0)
                value = torch.cat([cached[1], value], dim=0)
            queries.append(query)
            keys.append(key)
            values.append(value)
            key_lens.append(key.shape[0])
            retained.append(
                (key.detach(), value.detach())
                if store_kv
                else (cached if cached is not None else (key, value))
            )

        output = ragged_wrapper.run(
            torch.cat(queries), torch.cat(keys), torch.cat(values)
        )
        output, _ = attention.o_proj(output.reshape(-1, attention.q_size))
        return output, retained

    def _continuous_forward_many(
        self,
        *,
        input_tokens: list[tuple[int, ...]],
        positions: list[tuple[int, ...]],
        layer_kv: list[list[tuple[torch.Tensor, torch.Tensor]] | None],
        store_kv: bool,
        causal: bool,
    ) -> tuple[list[torch.Tensor], list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        model = self.model_runner.model
        core = model.model
        device = self.model_runner.device
        query_lens = [len(tokens) for tokens in input_tokens]
        flat_tokens = tuple(token for row in input_tokens for token in row)
        flat_positions = tuple(position for row in positions for position in row)
        input_ids = torch.tensor(flat_tokens, dtype=torch.int64, device=device)
        position_tensor = torch.tensor(flat_positions, dtype=torch.int64, device=device)
        hidden_states = core.embed_tokens(input_ids)
        ragged_wrapper = self.model_runner.attn_backend.prefill_wrapper_ragged
        cu_query_lens = torch.tensor(
            [0, *torch.tensor(query_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        key_lens = [
            query_len
            + (
                0
                if cache is None
                else cache[0][0].shape[0]
            )
            for query_len, cache in zip(query_lens, layer_kv)
        ]
        cu_key_lens = torch.tensor(
            [0, *torch.tensor(key_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        ragged_wrapper.plan(
            cu_query_lens,
            cu_key_lens,
            self.model_runner.model_config.num_attention_heads,
            self.model_runner.model_config.get_num_kv_heads(1, 1),
            self.model_runner.model_config.head_dim,
            causal=causal,
            q_data_type=hidden_states.dtype,
            kv_data_type=hidden_states.dtype,
            sm_scale=self.model_runner.model.model.layers[0].self_attn.scale,
        )
        retained_by_req = [[] for _ in input_tokens]
        for layer_index, layer in enumerate(core.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states, retained = self._continuous_attention_many(
                attention=layer.self_attn,
                hidden_states=hidden_states,
                positions=position_tensor,
                query_lens=query_lens,
                past_kv=[
                    None if cache is None else cache[layer_index]
                    for cache in layer_kv
                ],
                store_kv=store_kv,
                ragged_wrapper=ragged_wrapper,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            for req_index, current_kv in enumerate(retained):
                retained_by_req[req_index].append(current_kv)
        hidden_states = core.norm(hidden_states)
        logits = torch.nn.functional.linear(
            hidden_states,
            model.lm_head.weight[: self.model_runner.model_config.vocab_size],
        )
        return list(logits.split(query_lens)), retained_by_req

    def _propose_many_continuous(self, *, reqs) -> dict[str, object]:
        records = []
        for source in reqs:
            committed = tuple(source.origin_input_ids) + tuple(source.output_ids)
            safe_prefix = committed[:-1]
            state = self.continuous_prefix_states.get(source.rid)
            if state is not None and safe_prefix[: len(state.tokens)] != state.tokens:
                state = None
            cached_tokens = () if state is None else state.tokens
            records.append((source, committed, safe_prefix, state, cached_tokens))

        active = [record for record in records if len(record[2]) > len(record[4])]
        if active:
            _, retained = self._continuous_forward_many(
                input_tokens=[safe[len(cached) :] for _, _, safe, _, cached in active],
                positions=[
                    tuple(range(len(cached), len(safe)))
                    for _, _, safe, _, cached in active
                ],
                layer_kv=[None if state is None else state.layer_kv for _, _, _, state, _ in active],
                store_kv=True,
                causal=True,
            )
            for record, layer_kv in zip(active, retained):
                source, _, safe_prefix, _, _ = record
                self.continuous_prefix_states[source.rid] = LocalContinuousPrefixState(
                    tokens=safe_prefix, layer_kv=layer_kv
                )

        blocks = [
            (committed[-1],) + (self.mask_token,) * (self.block_size - 1)
            for _, committed, _, _, _ in records
        ]
        block_logits, _ = self._continuous_forward_many(
            input_tokens=blocks,
            positions=[
                tuple(range(len(safe), len(safe) + self.block_size))
                for _, _, safe, _, _ in records
            ],
            layer_kv=[
                self.continuous_prefix_states[source.rid].layer_kv
                for source, _, _, _, _ in records
            ],
            store_kv=False,
            causal=False,
        )
        return {
            source.rid: proposal_from_logits(
                anchor=committed[-1],
                full_logits=logits,
                replay_len=0,
                block_size=self.block_size,
                topk=self.topk,
            )
            for (source, committed, _, _, _), logits in zip(records, block_logits)
        }

    def propose_many(self, *, reqs, phase_metrics=None) -> dict[str, object]:
        """Run local misses together, using a packed graph when admitted."""
        if self.req_to_token_pool is None:
            raise RuntimeError("local SDAR memory pool is not initialized")
        if self.prefix_reuse:
            return self._propose_many_continuous(reqs=reqs)
        start = time.perf_counter()
        cpu_start = time.thread_time()
        synthetic, layouts = self._build_requests(reqs)
        build_end = time.perf_counter()
        if phase_metrics is not None:
            phase_metrics["local_required_build_seconds"] += build_end - start
            phase_metrics["local_required_input_tokens"] += sum(
                len(layout.input_tokens) for layout in layouts
            )
        try:
            logits = self._forward(
                synthetic=synthetic,
                layouts=layouts,
                phase_metrics=phase_metrics,
                check_graph=self.graph_check,
            )
            forward_end = time.perf_counter()
            if phase_metrics is not None:
                phase_metrics["local_required_forward_dispatch_seconds"] += forward_end - build_end
            proposals = self._postprocess(sources=reqs, layouts=layouts, logits=logits)
            if phase_metrics is not None:
                phase_metrics["local_required_postprocess_seconds"] += time.perf_counter() - forward_end
            return proposals
        finally:
            release_start = time.perf_counter()
            self._release_requests(synthetic=synthetic)
            if phase_metrics is not None:
                phase_metrics["local_required_release_seconds"] += time.perf_counter() - release_start
                phase_metrics["local_required_thread_cpu_seconds"] += time.thread_time() - cpu_start


def proposal_from_logits(
    *,
    anchor: int,
    full_logits: torch.Tensor,
    replay_len: int,
    block_size: int,
    topk: int,
):
    """Create aligned proposal/top-token rows from one required forward."""
    from .protocol import Proposal

    if full_logits.ndim != 2:
        raise ValueError("required draft logits must be [tokens, vocab]")
    block_logits = full_logits[replay_len : replay_len + block_size]
    if block_logits.shape[0] != block_size:
        raise ValueError("required draft logits do not cover the denoising block")
    draft_tokens = torch.argmax(block_logits[1:], dim=-1).tolist()
    tokens = (int(anchor),) + tuple(int(token) for token in draft_tokens)
    if topk <= 0:
        return Proposal(tokens)
    _, top_tokens = torch.topk(
        block_logits, k=min(topk, block_logits.shape[-1]), dim=-1
    )
    rows = [(int(anchor),)]
    rows.extend(
        tuple(int(token) for token in row) for row in top_tokens[1:].tolist()
    )
    return Proposal(tokens, tuple(rows))
