from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from .runtime import load_model, patch_transformers_losskwargs_for_d3d, d3d_forward, stage_checkpoint_with_code_model
from .types import ParallelOfflineBatch

def _unwrap_module(module):
    """Unwrap common DDP/FSDP containers without requiring Accelerate."""
    seen: set[int] = set()
    while hasattr(module, "module") and id(module) not in seen:
        seen.add(id(module))
        module = module.module
    return module


@contextmanager
def capture_final_norm_hidden(module):
    """Capture the full final-norm output before any logit position slicing."""
    unwrapped = _unwrap_module(module)
    backbone = getattr(unwrapped, "model", None)
    final_norm = getattr(backbone, "norm", None)
    if not isinstance(final_norm, torch.nn.Module):
        raise RuntimeError(
            "trajectory distillation requires the D3D final norm at model.norm"
        )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError("D3D final norm must return one tensor")
        if "hidden" in captured:
            raise RuntimeError("D3D final norm ran more than once in one capture")
        captured["hidden"] = output

    handle = final_norm.register_forward_hook(hook)
    try:
        yield captured
        if "hidden" not in captured:
            raise RuntimeError("D3D final norm did not run during captured forward")
    finally:
        handle.remove()


def dummy_zero_forward_loss(
    draft_ddp,
    *,
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the empty-rank block forward while deferring its backward."""
    output = d3d_forward(
        draft_ddp,
        torch.tensor([[int(mask_token_id)]], device=device),
        prefix_len=0,
        past_key_values=DynamicCache(),
        store_kv=False,
    )
    return output.logits.float().sum() * 0.0


def _checkpointed_d3d_layer_forward(self, *args, **kwargs):
    original_forward = self._d3d_checkpoint_original_forward
    if self.training and torch.is_grad_enabled():
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            **kwargs,
        )
    return original_forward(*args, **kwargs)


def enable_gradient_checkpointing(model) -> int:
    """Checkpoint D3D layers without dropping the clean block KV cache."""

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    base_model = getattr(model, "model", None)
    if hasattr(base_model, "gradient_checkpointing"):
        # Some D3D revisions checkpoint here, while others delegate to
        # GradientCheckpointingLayer.__call__. The latter silently removes
        # past_key_value, which would break block training against clean KV.
        base_model.gradient_checkpointing = False
    layers = getattr(base_model, "layers", ())
    for layer in layers:
        layer.gradient_checkpointing = False
        if not hasattr(layer, "_d3d_checkpoint_original_forward"):
            layer._d3d_checkpoint_original_forward = layer.forward
            layer.forward = MethodType(_checkpointed_d3d_layer_forward, layer)
    return len(layers)


def load_random_model(
    path: str,
    *,
    dtype: torch.dtype,
    attn_implementation: str,
    seed: int,
    draft_code_model: str = "",
):
    """Load the draft architecture with deterministic random initialization."""

    patch_transformers_losskwargs_for_d3d()
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    code_model = draft_code_model or str(Path(__file__).resolve().parents[1] / "models/d3d_training")
    with stage_checkpoint_with_code_model(path, code_model) as staged_path:
        config = AutoConfig.from_pretrained(staged_path, trust_remote_code=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return AutoModelForCausalLM.from_config(config, **kwargs)


def _causal_compatible_d3d_attention_forward(self, *args, **kwargs):
    """Honor draft_causal even when legacy D3D hard-codes causal=False."""

    original_forward = self._d3d_causal_original_forward
    if not bool(kwargs.get("draft_causal", False)):
        return original_forward(*args, **kwargs)

    forward_function = getattr(original_forward, "__func__", original_forward)
    function_globals = forward_function.__globals__
    original_flash_attention = function_globals["flash_attn_func"]

    def causal_flash_attention(*flash_args, **flash_kwargs):
        flash_kwargs["causal"] = True
        return original_flash_attention(*flash_args, **flash_kwargs)

    function_globals["flash_attn_func"] = causal_flash_attention
    try:
        return original_forward(*args, **kwargs)
    finally:
        if function_globals.get("flash_attn_func") is causal_flash_attention:
            function_globals["flash_attn_func"] = original_flash_attention


def enable_causal_flash_attention_compatibility(model) -> int:
    """Patch D3D attention calls so draft_causal survives checkpoint recompute."""

    patched_modules = 0
    for module in model.modules():
        if getattr(module, "supports_draft_causal", False):
            # The bundled implementation handles both SDPA and FlashAttention.
            # Do not monkey-patch its kernel globals.
            continue
        if hasattr(module, "_d3d_causal_original_forward"):
            patched_modules += 1
            continue
        forward = getattr(module, "forward", None)
        forward_function = getattr(forward, "__func__", forward)
        forward_code = getattr(forward_function, "__code__", None)
        function_globals = getattr(forward_function, "__globals__", None)
        if (
            forward_code is None
            or "flash_attn_func" not in forward_code.co_names
            or not isinstance(function_globals, dict)
            or not callable(function_globals.get("flash_attn_func"))
        ):
            continue
        module._d3d_causal_original_forward = module.forward
        module.forward = MethodType(
            _causal_compatible_d3d_attention_forward,
            module,
        )
        patched_modules += 1
    return patched_modules


@torch.no_grad()
def prefill_clean_kv_cache(
    draft_module,
    clean_input_ids: torch.LongTensor,
    clean_lens: torch.LongTensor,
) -> DynamicCache:
    """Build a causal KV cache over the clean prompt+response sequence."""

    clean_lens = clean_lens.to(device=clean_input_ids.device, dtype=torch.long)
    batch_size = int(clean_input_ids.shape[0])
    max_clean_len = int(clean_lens.max().item())
    if max_clean_len <= 0:
        raise ValueError("clean_lens must contain at least one positive length")
    if bool((clean_lens <= 0).any().item()):
        raise ValueError("clean_lens must be positive for every sample")

    clean_input_ids = clean_input_ids[:, :max_clean_len]
    device = clean_input_ids.device
    query_indices = torch.arange(max_clean_len, device=device).view(1, max_clean_len, 1)
    key_indices = torch.arange(max_clean_len, device=device).view(1, 1, max_clean_len)
    lens = clean_lens.view(batch_size, 1, 1)
    # Clean prefill is standard causal; block isolation is applied later.
    valid_causal = (query_indices < lens) & (key_indices < lens) & (key_indices <= query_indices)
    padding_self = (query_indices >= lens) & (query_indices == key_indices)
    attention_mask = (valid_causal | padding_self).view(
        batch_size,
        1,
        max_clean_len,
        max_clean_len,
    )
    use_flash_causal = (
        bool(getattr(draft_module, "supports_draft_causal", False))
        and bool((clean_lens == max_clean_len).all().item())
    )
    if use_flash_causal:
        # None + draft_causal=True is causal in BOTH supported backends.
        # Unknown checkpoint implementations keep the explicit triangle.
        attention_mask = None
    position_ids = torch.arange(max_clean_len, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
    cache = DynamicCache()
    was_training = bool(getattr(draft_module, "training", False))
    if was_training:
        draft_module.eval()
    try:
        draft_module(
            clean_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
            logits_to_keep=1,
            draft_causal=use_flash_causal,
        )
    finally:
        if was_training:
            draft_module.train()
    if len(cache.layers) == 0:
        raise RuntimeError("draft clean prefill did not populate the KV cache")
    # Zero padded KV slots so later packed block attention cannot read garbage.
    padding_positions = torch.arange(max_clean_len, device=device).view(1, 1, max_clean_len, 1) >= clean_lens.view(
        batch_size,
        1,
        1,
        1,
    )
    for layer in cache.layers:
        keys, values = layer.keys, layer.values
        keys.masked_fill_(padding_positions, 0.0)
        values.masked_fill_(padding_positions, 0.0)
    return cache


def dummy_clean_cache_forward(
    draft_module,
    *,
    mask_token_id: int,
    device: torch.device,
) -> None:
    """Match a valid rank's eval/no-grad clean-cache FSDP forward."""

    prefill_clean_kv_cache(
        draft_module,
        torch.tensor([[int(mask_token_id)]], dtype=torch.long, device=device),
        torch.tensor([1], dtype=torch.long, device=device),
    )


def detach_draft_cache(
    cache: DynamicCache,
    *,
    clean_lens: torch.LongTensor,
) -> DynamicCache:
    """Detach cached clean KV before the block loss backward pass."""

    detached = DynamicCache()
    max_clean_len = int(clean_lens.max().item())
    padding_positions = torch.arange(
        max_clean_len,
        device=clean_lens.device,
    ).view(1, 1, max_clean_len, 1) >= clean_lens.view(-1, 1, 1, 1)
    for layer_idx, layer in enumerate(cache.layers):
        keys, values = layer.keys, layer.values
        detached.update(
            keys.detach().masked_fill(padding_positions, 0.0),
            values.detach().masked_fill(padding_positions, 0.0),
            layer_idx,
        )
    return detached


@torch.no_grad()
def teacher_logits_at_positions(
    verifier,
    clean_input_ids: torch.LongTensor,
    clean_lens: torch.LongTensor,
    positions: torch.LongTensor,
    active_token_mask: torch.BoolTensor,
):
    """Fetch teacher logits only at selected clean-sequence positions."""

    positions = positions.to(device=clean_input_ids.device, dtype=torch.long)
    active_token_mask = active_token_mask.to(device=clean_input_ids.device, dtype=torch.bool)
    if positions.shape != active_token_mask.shape:
        raise ValueError(
            f"positions and active_token_mask must match; got "
            f"{tuple(positions.shape)} vs {tuple(active_token_mask.shape)}"
        )
    active_positions = positions[active_token_mask]
    if int(active_positions.numel()) == 0:
        raise ValueError("active positions must not be empty")
    keep_positions, inverse = torch.unique(
        active_positions,
        sorted=True,
        return_inverse=True,
    )
    batch_size, clean_len = clean_input_ids.shape
    clean_lens = clean_lens.to(device=clean_input_ids.device, dtype=torch.long)
    attention_mask = (
        torch.arange(clean_len, device=clean_input_ids.device).view(1, -1)
        < clean_lens.view(-1, 1)
    ).long()
    batch_indices = torch.arange(batch_size, device=clean_input_ids.device).view(-1, 1)
    batch_indices = batch_indices.expand_as(active_token_mask)[active_token_mask]
    kwargs = {
        "input_ids": clean_input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "logits_to_keep": keep_positions,
    }
    try:
        logits = verifier(**kwargs).logits
        return logits[batch_indices, inverse, :]
    except TypeError:
        kwargs.pop("logits_to_keep", None)
        logits = verifier(**kwargs).logits
        return logits[batch_indices, active_positions, :]


def packed_draft_forward(
    draft_ddp,
    batch: ParallelOfflineBatch,
    *,
    clean_cache,
    gradient_checkpointing: bool,
    block_forward_context=nullcontext,
):
    """Run one packed draft forward inside an optional architecture context."""

    with block_forward_context():
        return draft_ddp(
            batch.draft_input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            past_key_values=clean_cache,
            use_cache=not bool(gradient_checkpointing),
            store_kv=False,
            logits_to_keep=batch.logit_keep_indices,
        )


def tokenizer_load_kwargs(*, fix_mistral_regex: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if fix_mistral_regex:
        kwargs["fix_mistral_regex"] = True
    return kwargs


def load_draft_checkpoint(
    draft_model: str,
    *,
    draft_code_model: str = "",
    dtype: torch.dtype,
    attn_implementation: str,
):
    """Load original Qwen3 or trained D3D tensors into the bundled drafter."""
    return load_model(
        draft_model,
        code_model=draft_code_model or str(Path(__file__).resolve().parents[1] / "models/d3d_training"),
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
