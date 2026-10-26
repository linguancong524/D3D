from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .common import draft_attention_mask, hard_accept_length


def load_drafter(
    checkpoint: str,
    dtype: torch.dtype,
    *,
    attention_implementation: str = "sdpa",
) -> Any:
    """Load the diffusion drafter used by the layer-pruning experiments."""
    from models.d3d_training.configuration_d3d import D3DConfig
    from models.d3d_training.modeling_d3d import D3DForCausalLM

    config = D3DConfig.from_pretrained(checkpoint)
    # Recovery computes AR/CE/KL from explicit logits. Historical checkpoints
    # may enable a fused-CE training mode that suppresses logits entirely.
    config.fuse_cross_entropy = False
    return D3DForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=dtype,
        attn_implementation=attention_implementation,
    )


def disable_cache(model: Any) -> None:
    model.config.use_cache = False


def decoder_layer(
    model: Any,
    layer_index: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return model.model.layers[layer_index](
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        store_kv=False,
        cache_position=position_ids[0],
        position_embeddings=position_embeddings,
    )[0]


@torch.inference_mode()
def evaluate_single_layer_removals(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_tail: torch.Tensor,
    block_size: int,
    candidate_batch_size: int,
) -> tuple[tuple[int, float], dict[int, tuple[int, float]]]:
    """Evaluate all one-layer deletions while sharing the unchanged prefixes."""
    layer_count = len(model.model.layers)
    hidden_states = model.model.embed_tokens(input_ids)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    chunk_starts = set(range(0, layer_count, candidate_batch_size))
    prefix_states: dict[int, torch.Tensor] = {}

    for layer_index in range(layer_count):
        if layer_index in chunk_starts:
            prefix_states[layer_index] = hidden_states
        hidden_states = decoder_layer(
            model,
            layer_index,
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
        )

    current = model.model.norm(hidden_states)[:, -block_size:, :]
    current_logits = model.lm_head(current)[0, 1:]
    current_metrics = hard_accept_length(current_logits.argmax(-1), target_tail)

    candidates: dict[int, tuple[int, float]] = {}
    for start in range(0, layer_count, candidate_batch_size):
        removed_layers = list(
            range(start, min(start + candidate_batch_size, layer_count))
        )
        batch = len(removed_layers)
        changed = prefix_states[start].expand(batch, -1, -1)
        removed = torch.tensor(removed_layers, dtype=torch.long, device=changed.device)
        for layer_index in range(start, layer_count):
            before = changed
            after = decoder_layer(
                model,
                layer_index,
                before,
                attention_mask,
                position_ids,
                position_embeddings,
            )
            changed = torch.where(
                removed.eq(layer_index).view(batch, 1, 1), before, after
            )
        changed = model.model.norm(changed)[:, -block_size:, :]
        logits = model.lm_head(changed)[:, 1:]
        for row, layer_index in enumerate(removed_layers):
            candidates[layer_index] = hard_accept_length(
                logits[row].argmax(-1), target_tail
            )
    return current_metrics, candidates


@torch.inference_mode()
def score_tokenized_examples(
    model: Any,
    examples: Any,
    *,
    block_size: int,
    mask_token_id: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    """Measure offline emitted tokens on a fixed tokenized example set."""
    rows: list[dict[str, float | int | str]] = []
    model.eval()
    disable_cache(model)
    for example in examples.itertuples(index=False):
        prefix = torch.tensor(
            [int(token) for token in example.prefix_ids], device=device
        ).unsqueeze(0)
        target = torch.tensor(
            [int(token) for token in example.target_ids[:block_size]], device=device
        )
        if target.numel() != block_size:
            raise ValueError(f"{example.analysis_id} has an incomplete target block")
        block = target.clone()
        block[1:] = mask_token_id
        inputs = torch.cat([prefix, block[None, :]], dim=1)
        positions = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
        output = model(
            inputs,
            attention_mask=draft_attention_mask(
                len(example.prefix_ids), block_size, device
            ),
            position_ids=positions,
            use_cache=False,
            store_kv=False,
        )
        logits = output.logits[0, -block_size + 1 :, :]
        emitted, accuracy = hard_accept_length(logits.argmax(-1), target[1:])
        rows.append(
            {
                "analysis_id": str(example.analysis_id),
                "emitted_tokens": emitted,
                "draft_token_accuracy": accuracy,
            }
        )
    return rows


@torch.no_grad()
def prefill_cache(model: Any, batch: Any, expected_layers: int) -> DynamicCache:
    was_training = model.training
    model.eval()
    sequence_length = int(batch.clean_input_ids.shape[1])
    output = model(
        batch.clean_input_ids,
        attention_mask=batch.clean_attention_mask,
        position_ids=torch.arange(
            sequence_length, device=batch.clean_input_ids.device
        ).unsqueeze(0),
        past_key_values=DynamicCache(),
        use_cache=True,
        store_kv=True,
        logits_to_keep=1,
    )
    if was_training:
        model.train()
    cache = output.past_key_values
    if len(cache) != expected_layers:
        raise RuntimeError(f"Cache has {len(cache)} layers, expected {expected_layers}")
    return cache


@torch.no_grad()
def teacher_outputs(
    teacher: Any, batch: Any, retained_original_layers: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = prefill_cache(teacher, batch, len(teacher.model.layers))
    output, hidden_states = _draft_outputs_with_raw_boundaries(teacher, batch, cache)
    logits = _reshape_logits(output.logits, batch)
    boundaries = _reshape_boundaries(hidden_states, batch, retained_original_layers)
    return logits.detach(), boundaries.detach()


def student_outputs(
    student: Any, batch: Any, layer_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = prefill_cache(student, batch, layer_count)
    output, hidden_states = _draft_outputs_with_raw_boundaries(student, batch, cache)
    logits = _reshape_logits(output.logits, batch)
    boundaries = _reshape_boundaries(hidden_states, batch, list(range(layer_count)))
    return logits, boundaries


def _draft_outputs_with_raw_boundaries(
    model: Any, batch: Any, cache: DynamicCache
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    # D3D returns raw intermediate boundaries but replaces the last one with
    # the final-norm output. Capture its input so a pruned student's final
    # boundary has the same semantics as the teacher's intermediate boundary.
    # The norm and logits themselves remain unchanged. Keeping the tensor (not
    # a detached copy) preserves recovery gradients through the final layer.
    before_norm: list[torch.Tensor] = []

    def capture(module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        before_norm.append(inputs[0])

    handle = model.model.norm.register_forward_pre_hook(capture)
    try:
        output = model(
            batch.draft_input_ids,
            attention_mask=batch.draft_attention_mask,
            position_ids=batch.position_ids,
            past_key_values=cache,
            use_cache=False,
            store_kv=False,
            output_hidden_states=True,
            logits_to_keep=batch.logit_keep_indices,
        )
    finally:
        handle.remove()
    if output.hidden_states is None or len(before_norm) != 1:
        raise RuntimeError("The model did not expose one pre-final-norm boundary")
    return output, (*output.hidden_states[:-1], before_norm[0])


def _reshape_logits(logits: torch.Tensor, batch: Any) -> torch.Tensor:
    anchors, tail = batch.target_ids.shape
    active = torch.nonzero(
        batch.active_token_mask.reshape(-1), as_tuple=False
    ).flatten()
    flat = logits.new_zeros((anchors * tail, logits.shape[-1]))
    flat[active] = logits.squeeze(0)
    return flat.reshape(anchors, tail, -1)


def _reshape_boundaries(
    hidden_states: tuple[torch.Tensor, ...],
    batch: Any,
    layers: list[int],
) -> torch.Tensor:
    if hidden_states is None:
        raise RuntimeError("The model did not return hidden states")
    anchors, tail = batch.target_ids.shape
    query_indices = torch.arange(
        batch.draft_input_ids.shape[1], device=batch.draft_input_ids.device
    )
    query_indices = query_indices[query_indices.remainder(tail + 1).ne(0)]
    by_layer = torch.stack(
        [hidden_states[layer + 1][:, query_indices, :] for layer in layers], dim=1
    ).squeeze(0)
    return (
        by_layer.permute(1, 0, 2)
        .reshape(anchors, tail, len(layers), -1)
        .permute(0, 2, 1, 3)
    )
