#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from common.runtime import check_hardware, load_dataset_jsonl, resolve_model_path, runtime_metadata
LOCAL_CACHE_DIR = Path(os.environ.get("D3D_DATA_CACHE", REPO_ROOT / "data_cache"))
LOCAL_ANSWER_DIR = Path(os.environ.get("D3D_ANSWER_CACHE", REPO_ROOT / "answer_cache"))
MASK_ID = 151669
STOP_TOKEN_IDS = [151643, 151645]


@dataclass
class AcceptanceStep:
    accepted_draft_tokens: int
    accept_length: int
    emitted_count: int
    emitted_tokens: list[int]
    draft_tokens: list[int]
    target_tokens: list[int]
    stopped: bool
    mode: str = "draft"


def _safe_tag(path_or_name: str) -> str:
    tag = Path(path_or_name).name or str(path_or_name).strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) or "model"


def _safe_custom_tag(tag: str | None, fallback: str) -> str:
    if tag:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag).strip()) or fallback
    return fallback


def _checkpoint_model_type(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=model_path, filename='config.json'))
    try:
        model_type = json.loads(config_path.read_text()).get("model_type")
        return "d3d" if model_type == "sdar" else model_type
    except json.JSONDecodeError:
        return None


def _is_d3d_checkpoint(model_path: str) -> bool:
    return _checkpoint_model_type(model_path) == "d3d"


def _resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def _ensure_loss_kwargs_compat() -> None:
    from typing import TypedDict
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs


def _from_pretrained(cls: Any, model_path: str, dtype: Any, *, attn_implementation: str | None, config: Any = None):
    kwargs: dict[str, Any] = {"trust_remote_code": False, "dtype": dtype}
    if config is not None:
        kwargs['config'] = config
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return cls.from_pretrained(model_path, **kwargs)
    except TypeError as error:
        if 'dtype' not in str(error):
            raise
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        return cls.from_pretrained(model_path, **kwargs)


def _load_model(model_path: str, dtype: Any, *, attn_implementation: str | None):
    if _is_d3d_checkpoint(model_path):
        _ensure_loss_kwargs_compat()
        from models.d3d.configuration_d3d import D3DConfig
        from models.d3d.modeling_d3d import D3DForCausalLM

        config = D3DConfig.from_pretrained(model_path)
        return _from_pretrained(D3DForCausalLM, model_path, dtype,
                                config=config, attn_implementation=attn_implementation)

    import transformers

    # Qwen3.5/3.6 is registered as a multimodal conditional-generation model
    # even when it is used in language-only mode. Prefer that loader so we do
    # not first construct/fail several incompatible AutoModel classes.
    checkpoint_type = _checkpoint_model_type(model_path)
    if checkpoint_type == "qwen3_5":
        candidates = []
        image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if image_text_cls is not None:
            candidates.append(image_text_cls)
        candidates.append(AutoModelForCausalLM)
    else:
        candidates = [AutoModelForCausalLM]

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        candidate = getattr(transformers, name, None)
        if candidate is not None:
            candidates.append(candidate)
    candidates.append(AutoModel)

    last_error: Exception | None = None
    for cls in candidates:
        try:
            return _from_pretrained(cls, model_path, dtype, attn_implementation=attn_implementation)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def _disable_default_cache(model: Any) -> None:
    configs = [getattr(model, "config", None)]
    for attr in ("text_config", "language_config"):
        nested = getattr(configs[0], attr, None) if configs[0] is not None else None
        if nested is not None:
            configs.append(nested)
    for config in configs:
        if hasattr(config, "use_cache"):
            config.use_cache = False


def _sync(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)


def _event_pair(device: torch.device | str):
    device_t = torch.device(device)
    if device_t.type != "cuda" or not torch.cuda.is_available():
        return None, None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_elapsed_s(start, end) -> float:
    if start is None or end is None:
        return 0.0
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _synced_time() -> float:
    _sync()
    return time.perf_counter()


def _pos(start: int, end: int, device):
    return torch.arange(start, end, device=device).unsqueeze(0)


def _full_mask(q_len: int, kv_len: int, device):
    return torch.ones((1, 1, q_len, kv_len), dtype=torch.bool, device=device)


def _causal_mask(q_len: int, kv_len: int, device, *, query_start: int = 0):
    q_pos = torch.arange(int(query_start), int(query_start) + int(q_len), device=device)
    k_pos = torch.arange(int(kv_len), device=device)
    return (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _block_causal_mask(seq_len: int, block_size: int, device):
    block_size = max(1, int(block_size))
    block_ids = torch.arange(int(seq_len), device=device) // block_size
    return (block_ids[:, None] >= block_ids[None, :]).unsqueeze(0).unsqueeze(0)


def _draft_attention_mask(
    prefix_len: int,
    block_len: int,
    device,
    attention_mask_mode: str,
    block_size: int,
):
    total_len = int(prefix_len) + int(block_len)
    if attention_mask_mode == "block_causal":
        block_size = max(1, int(block_size))
        block_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
        num_prefix_blocks = (int(prefix_len) + block_size - 1) // block_size
        for block_idx in range(num_prefix_blocks):
            start = block_idx * block_size
            end = min((block_idx + 1) * block_size, int(prefix_len))
            block_ids[start:end] = block_idx
        block_ids[int(prefix_len) :] = num_prefix_blocks
        return (block_ids[:, None] >= block_ids[None, :]).unsqueeze(0).unsqueeze(0)
    if attention_mask_mode != "causal":
        raise ValueError(f"unsupported attention_mask_mode={attention_mask_mode!r}")

    mask = torch.zeros((1, 1, total_len, total_len), dtype=torch.bool, device=device)
    if prefix_len > 0:
        mask[:, :, : int(prefix_len), : int(prefix_len)] = _causal_mask(
            int(prefix_len),
            int(prefix_len),
            device,
        )
    if block_len > 0:
        mask[:, :, int(prefix_len) :, :] = True
    return mask


def _prefix_attention_mask(seq_len: int, device, attention_mask_mode: str, block_size: int):
    if attention_mask_mode == "block_causal":
        return _block_causal_mask(seq_len, block_size, device)
    if attention_mask_mode == "causal":
        return _causal_mask(seq_len, seq_len, device)
    raise ValueError(f"unsupported attention_mask_mode={attention_mask_mode!r}")


def _fused_draft_mask(
    pending_len: int,
    block_len: int,
    past_len: int,
    device,
    attention_mask_mode: str = "block_causal",
):
    q_len = int(pending_len) + int(block_len)
    kv_len = int(past_len) + q_len
    mask = torch.ones((1, 1, q_len, kv_len), dtype=torch.bool, device=device)
    if attention_mask_mode == "causal":
        if pending_len > 0:
            mask[:, :, : int(pending_len), :] = _causal_mask(
                int(pending_len),
                kv_len,
                device,
                query_start=int(past_len),
            )
        return mask
    if attention_mask_mode != "block_causal":
        raise ValueError(f"unsupported attention_mask_mode={attention_mask_mode!r}")
    if pending_len > 0 and block_len > 0:
        block_start = int(past_len) + int(pending_len)
        mask[:, :, : int(pending_len), block_start:] = False
    return mask


def _cache_len(cache: Any | None) -> int:
    return int(cache.get_seq_length()) if cache is not None else 0


def _crop_cache(cache: Any | None, length: int) -> None:
    if cache is not None:
        cache.crop(int(length))


def _sampling_probs(logits, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
    if temperature is None or temperature <= 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    logits = logits.float() / float(temperature)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits.scatter_(-1, sorted_idx, sorted_logits)
    return torch.softmax(logits, dim=-1)


def _sample_tokens(logits, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
    probs = _sampling_probs(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if temperature is None or temperature <= 1e-5:
        return probs.argmax(dim=-1)
    return torch.multinomial(
        probs.reshape(-1, probs.shape[-1]),
        1,
    ).reshape(logits.shape[:-1])


def _sample_with_prob(logits, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
    probs = _sampling_probs(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if temperature is None or temperature <= 1e-5:
        tokens = probs.argmax(dim=-1)
    else:
        tokens = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]),
            1,
        ).reshape(logits.shape[:-1])
    token_probs = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, token_probs


def _sample_with_sparse_probs(
    logits,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if temperature is None or temperature <= 1e-5:
        tokens = logits.argmax(dim=-1)
        return tokens, tokens.unsqueeze(-1), torch.ones_like(tokens.unsqueeze(-1), dtype=torch.float32)

    vocab_size = int(logits.shape[-1])
    support_size = min(int(top_k), vocab_size) if int(top_k) > 0 else vocab_size
    scaled = logits.float() / float(temperature)
    support_logits, support_ids = torch.topk(scaled, k=support_size, dim=-1)
    if top_p and top_p < 1.0:
        support_probs = torch.softmax(support_logits, dim=-1)
        remove = support_probs.cumsum(dim=-1) > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        support_logits = support_logits.masked_fill(remove, torch.finfo(support_logits.dtype).min)
    support_probs = torch.softmax(support_logits, dim=-1)
    local_tokens = torch.multinomial(
        support_probs.reshape(-1, support_size),
        1,
    ).reshape(logits.shape[:-1])
    tokens = support_ids.gather(-1, local_tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, support_ids, support_probs


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _eos_token_ids(tokenizer: Any, explicit: list[int] | None) -> list[int]:
    values: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        values.append(eos)
    elif isinstance(eos, (list, tuple)):
        values.extend(int(x) for x in eos)
    extra = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(extra, (list, tuple)):
        values.extend(int(x) for x in extra)
    # Keep legacy shell-provided stop IDs for Qwen3 while also honoring a
    # newer verifier tokenizer (Qwen3.6 uses 248046/248044).
    if explicit:
        values.extend(int(x) for x in explicit)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _contains_stop(tokens, stop_token_ids: Iterable[int] | None):
    if stop_token_ids is None:
        return False
    return any((tokens == int(stop_id)).any().item() for stop_id in stop_token_ids)


def _trim_at_stop(output_ids, num_input_tokens: int, stop_token_ids: Iterable[int] | None):
    if stop_token_ids is None:
        return output_ids
    stops = torch.tensor(list(stop_token_ids), device=output_ids.device)
    hits = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(as_tuple=True)[0]
    if hits.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + hits[0].item() + 1]


def _tokens_to_int_list(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if isinstance(tokens, torch.Tensor):
        return [int(t) for t in tokens.detach().cpu().tolist()]
    return [int(t) for t in tokens]


def summarize_acceptance(steps: list[AcceptanceStep], draft_tokens: int) -> dict[str, Any]:
    if not steps:
        return {
            "num_verify_steps": 0,
            "avg_accepted_draft_tokens": 0.0,
            "avg_accept_length": 0.0,
            "avg_emitted_tokens": 0.0,
            "draft_accept_rate": 0.0,
            "reject_at_0_rate": 0.0,
            "all_draft_accepted_rate": 0.0,
            "no_draft_num_steps": 0,
            "avg_no_draft_verify_tokens": 0.0,
            "accept_length_histogram": {str(i): 0 for i in range(1, int(draft_tokens) + 2)},
        }
    no_draft_steps = [s for s in steps if s.mode == "no-draft"]
    draft_steps = [s for s in steps if len(s.draft_tokens) > 0]
    accepted = [s.accepted_draft_tokens for s in steps]
    accept_lengths = [s.accept_length for s in steps]
    emitted = [s.emitted_count for s in steps]
    total_draft_tokens = sum(len(s.draft_tokens) for s in steps)
    return {
        "num_verify_steps": len(steps),
        "avg_accepted_draft_tokens": statistics.fmean(accepted),
        "avg_accept_length": statistics.fmean(accept_lengths),
        "avg_emitted_tokens": statistics.fmean(emitted),
        "draft_accept_rate": sum(accepted) / total_draft_tokens if total_draft_tokens else 0.0,
        "reject_at_0_rate": (
            sum(1 for s in draft_steps if s.accepted_draft_tokens == 0) / len(draft_steps)
            if draft_steps
            else 0.0
        ),
        "all_draft_accepted_rate": (
            sum(1 for s in draft_steps if s.accepted_draft_tokens == len(s.draft_tokens)) / len(draft_steps)
            if draft_steps
            else 0.0
        ),
        "no_draft_num_steps": len(no_draft_steps),
        "avg_no_draft_verify_tokens": (
            statistics.fmean(s.emitted_count for s in no_draft_steps) if no_draft_steps else 0.0
        ),
        "accept_length_histogram": {
            str(i): accept_lengths.count(i) for i in range(1, int(draft_tokens) + 2)
        },
    }


def summarize_timing_metrics(per_prompt: list[dict[str, Any]], *, elapsed_s: float) -> dict[str, float]:
    total_output_tokens = sum(int(x.get("output_tokens", 0)) for x in per_prompt)
    total_decode_tokens = sum(int(x.get("decode_tokens", x.get("output_tokens", 0))) for x in per_prompt)
    decode_times = [float(x.get("decode_elapsed_s", 0.0)) for x in per_prompt]
    prompt_tps = [float(x.get("tokens_per_s", 0.0)) for x in per_prompt]
    time_per_output_token = [
        float(x.get("time_per_output_token", 0.0))
        for x in per_prompt
        if int(x.get("decode_tokens", x.get("output_tokens", 0))) > 0
    ]
    total_decode_elapsed_s = sum(decode_times)
    aggregate_decode_tps = total_decode_tokens / total_decode_elapsed_s if total_decode_elapsed_s > 0 else 0.0
    mean_tokens_per_s = statistics.fmean(prompt_tps) if prompt_tps else 0.0
    return {
        "total_decode_elapsed_s": total_decode_elapsed_s,
        "mean_tokens_per_s": mean_tokens_per_s,
        "mean_prompt_decode_tps": mean_tokens_per_s,
        "mean_time_per_output_token": (
            statistics.fmean(time_per_output_token) if time_per_output_token else 0.0
        ),
        "aggregate_decode_tokens_per_s": aggregate_decode_tps,
        "aggregate_decode_tps": aggregate_decode_tps,
        "elapsed_tokens_per_s": total_output_tokens / float(elapsed_s) if elapsed_s > 0 else 0.0,
    }


def headline_metrics(
    d3d_tps: float,
    baseline_tps: float,
    accept_length: float,
    *,
    metric: str = "decode",
) -> dict[str, Any]:
    """Report the two headline numbers: speedup over the single-card AR baseline and accept length."""
    d3d_tps = float(d3d_tps or 0.0)
    baseline_tps = float(baseline_tps or 0.0)
    return {
        "metric": metric,
        "d3d_tps": d3d_tps,
        "baseline_tps": baseline_tps,
        "speedup": d3d_tps / baseline_tps if baseline_tps > 0 else None,
        "accept_length": float(accept_length or 0.0),
    }


def _effective_draft_tokens(draft_tokens: int | None, block_size: int) -> int:
    if draft_tokens is not None:
        if draft_tokens <= 0:
            raise ValueError("--draft-tokens must be positive")
        return int(draft_tokens)
    if block_size <= 1:
        raise ValueError("--block-size must be > 1 when --draft-tokens is omitted")
    return int(block_size) - 1


def _get_num_transfer_tokens(num_masks: int, steps: int):
    steps = max(1, int(steps))
    base = num_masks // steps
    extra = num_masks % steps
    return [base + (1 if i < extra else 0) for i in range(steps)]


def _select_transfer(block, mask_index, x0, x0_p, k, remasking, threshold):
    if k <= 0 or not mask_index.any():
        return
    scores = x0_p.masked_fill(~mask_index, -1.0)
    masked_count = int(mask_index.sum().item())
    k = min(int(k), masked_count)
    if "dynamic" in (remasking or ""):
        confident = (scores >= threshold) & mask_index
        if confident.any():
            idx = confident.nonzero(as_tuple=True)[1]
            if idx.numel() > k:
                _, local = torch.topk(scores[:, idx], k, dim=-1)
                idx = idx[local[0]]
        else:
            idx = torch.topk(scores, k, dim=-1).indices[0]
    else:
        idx = torch.topk(scores, k, dim=-1).indices[0]
    block[:, idx] = x0[:, idx]
    return idx


def _can_skip_transfer_probs(step: int, schedule: list[int], mask_index, remasking: str, temperature: float) -> bool:
    if step != len(schedule) - 1:
        return False
    if "dynamic" in (remasking or ""):
        return False
    if temperature is not None and temperature > 1e-5:
        return False
    return int(mask_index.sum().item()) == int(schedule[step])


def _argmax_with_dummy_probs(logits):
    return logits.argmax(dim=-1), torch.zeros(logits.shape[:-1], dtype=torch.float32, device=logits.device)


@torch.inference_mode()
def d3d_draft_block(
    drafter,
    block,
    prefix_len: int,
    denoising_steps: int,
    remasking: str,
    threshold: float,
    temperature: float,
    top_k: int,
    top_p: float,
    mask_id: int,
    prefix_ids=None,
    past_key_values=None,
    attention_mask_mode: str = "block_causal",
    block_size: int = 32,
    return_proposal_probs: bool = False,
    return_sparse_proposal: bool = False,
):
    device = block.device
    cur = block.clone()
    num_masks = int((cur == mask_id).sum().item())
    if num_masks == 0:
        if return_proposal_probs or return_sparse_proposal:
            return cur, None
        return cur
    schedule = _get_num_transfer_tokens(num_masks, denoising_steps)
    proposal_probs = None
    proposal_ids = None
    proposal_sparse_probs = None
    for step, k in enumerate(schedule):
        mask_index = cur == mask_id
        if not mask_index.any():
            break
        if prefix_ids is not None:
            draft_input = torch.cat([prefix_ids, cur], dim=1)
            logits = drafter(
                draft_input,
                attention_mask=_draft_attention_mask(
                    prefix_ids.shape[1],
                    cur.shape[1],
                    device,
                    attention_mask_mode,
                    block_size,
                ),
                position_ids=_pos(0, draft_input.shape[1], device),
                use_cache=False,
                logits_to_keep=cur.shape[1],
            ).logits
        elif past_key_values is not None:
            logits = drafter(
                cur,
                attention_mask=_full_mask(cur.shape[1], prefix_len + cur.shape[1], device),
                position_ids=_pos(prefix_len, prefix_len + cur.shape[1], device),
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
                logits_to_keep=cur.shape[1],
            ).logits
        else:
            logits = drafter(
                cur,
                attention_mask=_full_mask(cur.shape[1], cur.shape[1], device),
                position_ids=_pos(0, cur.shape[1], device),
                use_cache=False,
                logits_to_keep=cur.shape[1],
            ).logits
        if step == len(schedule) - 1:
            k = int(mask_index.sum().item())
        if return_proposal_probs and temperature > 1e-5 and proposal_probs is None:
            proposal_probs = torch.zeros(
                (*cur.shape, int(logits.shape[-1])),
                dtype=torch.float32,
                device=device,
            )
        if return_sparse_proposal and temperature > 1e-5 and proposal_ids is None:
            support_size = min(int(top_k), int(logits.shape[-1])) if int(top_k) > 0 else int(logits.shape[-1])
            proposal_ids = torch.zeros((*cur.shape, support_size), dtype=torch.long, device=device)
            proposal_sparse_probs = torch.zeros((*cur.shape, support_size), dtype=torch.float32, device=device)
        if _can_skip_transfer_probs(step, schedule, mask_index, remasking, temperature):
            x0, x0_p = _argmax_with_dummy_probs(logits)
        elif return_sparse_proposal and temperature > 1e-5:
            x0, step_ids, step_sparse_probs = _sample_with_sparse_probs(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            x0_p = step_sparse_probs.gather(
                -1,
                (step_ids == x0.unsqueeze(-1)).long().argmax(dim=-1, keepdim=True),
            ).squeeze(-1)
        else:
            x0, x0_p = _sample_with_prob(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        selected_idx = _select_transfer(cur, mask_index, x0, x0_p, k, remasking, threshold)
        if proposal_probs is not None:
            step_probs = _sampling_probs(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            proposal_probs[:, selected_idx, :] = step_probs[:, selected_idx, :]
        if proposal_ids is not None:
            proposal_ids[:, selected_idx, :] = step_ids[:, selected_idx, :]
            proposal_sparse_probs[:, selected_idx, :] = step_sparse_probs[:, selected_idx, :]
    if return_sparse_proposal:
        return cur, (proposal_ids, proposal_sparse_probs)
    if return_proposal_probs:
        return cur, proposal_probs
    return cur


@torch.inference_mode()
def d3d_draft_block_fused_commit(
    drafter,
    block,
    *,
    draft_kv,
    draft_kv_len: int,
    pending_commit,
    prefix_len: int,
    denoising_steps: int,
    remasking: str,
    threshold: float,
    temperature: float,
    top_k: int,
    top_p: float,
    mask_id: int,
    attention_mask_mode: str = "block_causal",
    return_metrics: bool = False,
    return_topk: int = 0,
    return_proposal_probs: bool = False,
    return_sparse_proposal: bool = False,
):
    device = block.device
    cur = block.clone()
    pending_len = int(pending_commit.shape[1]) if pending_commit is not None else 0
    if pending_len != int(prefix_len) - int(draft_kv_len):
        raise RuntimeError(
            f"draft pending length mismatch: pending={pending_len}, "
            f"prefix_len={prefix_len}, draft_kv_len={draft_kv_len}"
        )

    num_masks = int((cur == mask_id).sum().item())
    if num_masks == 0:
        if pending_len > 0:
            draft_input = torch.cat([pending_commit, cur], dim=1)
            drafter(
                draft_input,
                attention_mask=_fused_draft_mask(
                    pending_len,
                    cur.shape[1],
                    draft_kv_len,
                    device,
                    attention_mask_mode,
                ),
                position_ids=_pos(draft_kv_len, draft_kv_len + draft_input.shape[1], device),
                past_key_values=draft_kv,
                use_cache=True,
                store_kv=True,
                logits_to_keep=1,
            )
            _crop_cache(draft_kv, prefix_len)
        if return_metrics or int(return_topk) > 0:
            metrics = {
                "top_tokens": None,
                "top_logprobs": None,
            }
            if return_proposal_probs or return_sparse_proposal:
                return cur, metrics, None
            return cur, metrics
        return cur

    schedule = _get_num_transfer_tokens(num_masks, denoising_steps)
    pending_written = False
    proposal_probs = None
    proposal_ids = None
    proposal_sparse_probs = None
    final_logits = None
    for step, k in enumerate(schedule):
        mask_index = cur == mask_id
        if not mask_index.any():
            break

        if pending_len > 0 and not pending_written:
            draft_input = torch.cat([pending_commit, cur], dim=1)
            output = drafter(
                draft_input,
                attention_mask=_fused_draft_mask(
                    pending_len,
                    cur.shape[1],
                    draft_kv_len,
                    device,
                    attention_mask_mode,
                ),
                position_ids=_pos(draft_kv_len, draft_kv_len + draft_input.shape[1], device),
                past_key_values=draft_kv,
                use_cache=True,
                store_kv=True,
                logits_to_keep=cur.shape[1],
            )
            logits = output.logits
            _crop_cache(draft_kv, prefix_len)
            pending_written = True
        else:
            logits = drafter(
                cur,
                attention_mask=_full_mask(cur.shape[1], prefix_len + cur.shape[1], device),
                position_ids=_pos(prefix_len, prefix_len + cur.shape[1], device),
                past_key_values=draft_kv,
                use_cache=True,
                store_kv=False,
                logits_to_keep=cur.shape[1],
            ).logits

        if step == len(schedule) - 1:
            k = int(mask_index.sum().item())
            final_logits = logits
        if return_proposal_probs and temperature > 1e-5 and proposal_probs is None:
            proposal_probs = torch.zeros(
                (*cur.shape, int(logits.shape[-1])),
                dtype=torch.float32,
                device=device,
            )
        if return_sparse_proposal and temperature > 1e-5 and proposal_ids is None:
            support_size = min(int(top_k), int(logits.shape[-1])) if int(top_k) > 0 else int(logits.shape[-1])
            proposal_ids = torch.zeros((*cur.shape, support_size), dtype=torch.long, device=device)
            proposal_sparse_probs = torch.zeros((*cur.shape, support_size), dtype=torch.float32, device=device)
        need_metrics = (return_metrics or int(return_topk) > 0) and step == len(schedule) - 1
        if need_metrics:
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            if temperature is None or temperature <= 1e-5:
                x0 = logits.argmax(dim=-1)
                x0_p = probs.max(dim=-1).values
            elif return_sparse_proposal and temperature > 1e-5:
                x0, step_ids, step_sparse_probs = _sample_with_sparse_probs(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                x0_p = step_sparse_probs.gather(
                    -1,
                    (step_ids == x0.unsqueeze(-1)).long().argmax(dim=-1, keepdim=True),
                ).squeeze(-1)
            else:
                x0, x0_p = _sample_with_prob(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        elif _can_skip_transfer_probs(step, schedule, mask_index, remasking, temperature):
            x0, x0_p = _argmax_with_dummy_probs(logits)
        elif return_sparse_proposal and temperature > 1e-5:
            x0, step_ids, step_sparse_probs = _sample_with_sparse_probs(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            x0_p = step_sparse_probs.gather(
                -1,
                (step_ids == x0.unsqueeze(-1)).long().argmax(dim=-1, keepdim=True),
            ).squeeze(-1)
        else:
            x0, x0_p = _sample_with_prob(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        selected_idx = _select_transfer(cur, mask_index, x0, x0_p, k, remasking, threshold)
        if proposal_probs is not None:
            step_probs = _sampling_probs(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            proposal_probs[:, selected_idx, :] = step_probs[:, selected_idx, :]
        if proposal_ids is not None:
            proposal_ids[:, selected_idx, :] = step_ids[:, selected_idx, :]
            proposal_sparse_probs[:, selected_idx, :] = step_sparse_probs[:, selected_idx, :]

    metrics = {
        "top_tokens": None,
        "top_logprobs": None,
    }
    if int(return_topk) > 0 and final_logits is not None:
        log_probs = torch.log_softmax(final_logits.float(), dim=-1)
        k = min(int(return_topk), int(log_probs.shape[-1]))
        top_logprobs, top_tokens = torch.topk(log_probs, k=k, dim=-1)
        metrics["top_tokens"] = top_tokens.detach()
        metrics["top_logprobs"] = top_logprobs.detach()
    if return_metrics or int(return_topk) > 0 or return_proposal_probs or return_sparse_proposal:
        if return_sparse_proposal:
            return cur, metrics, (proposal_ids, proposal_sparse_probs)
        if return_proposal_probs:
            return cur, metrics, proposal_probs
        return cur, metrics
    return cur


@torch.inference_mode()
def run_d3d_generate(
    drafter,
    verifier,
    input_ids,
    max_new_tokens: int,
    stop_token_ids=None,
    temperature: float = 0.0,
    block_size: int = 4,
    denoising_steps: int = 1,
    threshold: float = 0.9,
    remasking: str = "low_confidence",
    top_k: int = 0,
    top_p: float = 1.0,
    target_top_k: int = 0,
    target_top_p: float = 1.0,
    return_stats: bool = False,
    attention_mask_mode: str = "block_causal",
    collect_stage_timings: bool = False,
    draft_device: str | None = None,
):
    device = next(verifier.parameters()).device
    draft_device = draft_device or str(next(drafter.parameters()).device)
    input_ids = input_ids.to(device)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.full(
        (1, max_length + block_size + 1),
        MASK_ID,
        dtype=input_ids.dtype,
        device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    _sync(device)
    start_time = time.perf_counter()
    _sync(device)
    verifier_prefill_start = time.perf_counter()
    first_logits = verifier(input_ids, use_cache=False).logits[:, -1:, :]
    _sync(device)
    verifier_prefill_time = time.perf_counter() - verifier_prefill_start
    output_ids[:, num_input_tokens] = _sample_tokens(
        first_logits,
        temperature=temperature,
        top_k=target_top_k,
        top_p=target_top_p,
    )[:, 0]

    acceptance_lengths = []
    draft_token_blocks: list[Any] = []
    target_token_blocks: list[Any] = []
    emitted_token_blocks: list[Any] = []
    stopped_blocks: list[bool] = []
    stage_timing_records: list[dict[str, Any]] = []
    draft_time = 0.0
    verify_time = 0.0
    start = num_input_tokens
    while start < max_length:
        step_index = len(acceptance_lengths)
        prefix_len_before = int(start)
        draft_wall_s = 0.0
        draft_gpu_s = 0.0
        verify_wall_s = 0.0
        verify_gpu_s = 0.0
        cur_block_size = min(block_size, max_length - start)
        if cur_block_size <= 0:
            break
        block = output_ids[:, start : start + cur_block_size].clone()
        if cur_block_size > 1:
            prefix_ids = output_ids[:, :start]
            block[:, 1:] = MASK_ID
            _sync(draft_device)
            draft_start = time.perf_counter()
            draft_start_event, draft_end_event = (
                _event_pair(draft_device) if collect_stage_timings else (None, None)
            )
            if draft_start_event is not None:
                draft_start_event.record(torch.cuda.current_stream(draft_device))
            draft_result = d3d_draft_block(
                drafter,
                block.to(draft_device),
                prefix_len=start,
                denoising_steps=denoising_steps,
                remasking=remasking,
                threshold=threshold,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_id=MASK_ID,
                prefix_ids=prefix_ids.to(draft_device),
                attention_mask_mode=attention_mask_mode,
                block_size=block_size,
                return_proposal_probs=temperature > 1e-5,
            )
            if temperature > 1e-5:
                block, draft_proposal_probs = draft_result
                draft_proposal_probs = draft_proposal_probs.to(device)
            else:
                block = draft_result
                draft_proposal_probs = None
            block = block.to(device)
            if draft_end_event is not None:
                draft_end_event.record(torch.cuda.current_stream(draft_device))
            _sync(draft_device)
            draft_wall_s = time.perf_counter() - draft_start
            draft_gpu_s = _event_elapsed_s(draft_start_event, draft_end_event)
            draft_time += draft_wall_s

        verify_input = torch.cat([output_ids[:, :start], block], dim=1)
        _sync(device)
        verify_start = time.perf_counter()
        verify_start_event, verify_end_event = (
            _event_pair(device) if collect_stage_timings else (None, None)
        )
        if verify_start_event is not None:
            verify_start_event.record(torch.cuda.current_stream(device))
        logits = verifier(verify_input, use_cache=False).logits[:, start : start + cur_block_size, :]
        if verify_end_event is not None:
            verify_end_event.record(torch.cuda.current_stream(device))
        _sync(device)
        verify_wall_s = time.perf_counter() - verify_start
        verify_gpu_s = _event_elapsed_s(verify_start_event, verify_end_event)
        verify_time += verify_wall_s
        if temperature > 1e-5 and cur_block_size > 1:
            from single.speculative_sampling import verify_speculative_block

            target_probs = _sampling_probs(
                logits,
                temperature=temperature,
                top_k=target_top_k,
                top_p=target_top_p,
            )
            draft_probs = draft_proposal_probs[:, 1:cur_block_size, :]
            verify_result = verify_speculative_block(
                block[:, 1:cur_block_size],
                target_probs,
                draft_probs,
            )
            accepted_draft = verify_result.accepted_draft_tokens
            correction_token = verify_result.correction_token.to(device)
            posterior = target_probs.argmax(dim=-1)
        else:
            posterior = _sample_tokens(
                logits,
                temperature=temperature,
                top_k=target_top_k,
                top_p=target_top_p,
            )
            if cur_block_size > 1:
                matches = block[:, 1:] == posterior[:, :-1]
                accepted_draft = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
            else:
                accepted_draft = 0
            correction_token = posterior[:, accepted_draft]

        commit_len = accepted_draft + 1
        commit_tokens = block[:, :commit_len]
        output_ids[:, start : start + commit_len] = commit_tokens
        output_ids[:, start + accepted_draft + 1] = correction_token
        start += commit_len
        acceptance_lengths.append(commit_len)
        draft_token_blocks.append(block[0, 1:].detach())
        target_token_blocks.append(posterior[0].detach())
        emitted_token_blocks.append(commit_tokens[0].detach())
        if collect_stage_timings:
            stage_timing_records.append(
                {
                    "step_index": step_index,
                    "prefix_len_before": prefix_len_before,
                    "generated_len_before": prefix_len_before - num_input_tokens,
                    "block_size": int(cur_block_size),
                    "accepted_draft_tokens": int(accepted_draft),
                    "accept_length": int(commit_len),
                    "prefix_len_after": int(start),
                    "generated_len_after": int(start) - num_input_tokens,
                    "draft_wall_s": float(draft_wall_s),
                    "draft_gpu_s": float(draft_gpu_s),
                    "verify_wall_s": float(verify_wall_s),
                    "verify_gpu_s": float(verify_gpu_s),
                    "round_wall_s": float(draft_wall_s + verify_wall_s),
                    "round_gpu_s": float(draft_gpu_s + verify_gpu_s),
                }
            )

        generated = output_ids[:, num_input_tokens : min(start + 1, max_length)]
        stopped = _contains_stop(generated, stop_token_ids)
        stopped_blocks.append(bool(stopped))
        if stopped:
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = _trim_at_stop(output_ids, num_input_tokens, stop_token_ids)
    _sync(device)
    total_time = time.perf_counter() - start_time

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_per_output_token=total_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        draft_token_blocks=draft_token_blocks,
        target_token_blocks=target_token_blocks,
        emitted_token_blocks=emitted_token_blocks,
        stopped_blocks=stopped_blocks,
        stage_timing_records=stage_timing_records,
        block_size=block_size,
        draft_time=draft_time,
        verify_time=verify_time,
        verifier_prefill_time=verifier_prefill_time,
        num_verify_steps=len(acceptance_lengths),
        mean_draft_time_per_step=draft_time / max(len(acceptance_lengths), 1),
        mean_verify_time_per_step=verify_time / max(len(acceptance_lengths), 1),
        decode_elapsed_s=total_time,
    )


@torch.inference_mode()
def run_d3d_generate_cached(
    drafter,
    verifier,
    input_ids,
    max_new_tokens: int,
    stop_token_ids=None,
    temperature: float = 0.0,
    block_size: int = 4,
    denoising_steps: int = 1,
    threshold: float = 0.9,
    remasking: str = "low_confidence",
    top_k: int = 0,
    top_p: float = 1.0,
    target_top_k: int = 0,
    target_top_p: float = 1.0,
    return_stats: bool = False,
    attention_mask_mode: str = "block_causal",
    collect_stage_timings: bool = False,
    draft_device: str | None = None,
):
    device = next(verifier.parameters()).device
    draft_device = draft_device or str(next(drafter.parameters()).device)
    input_ids = input_ids.to(device)
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)
    if max_new_tokens <= 0:
        return input_ids if not return_stats else SimpleNamespace(
            output_ids=input_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=0,
            time_per_output_token=0.0,
            acceptance_lengths=[],
            draft_token_blocks=[],
            target_token_blocks=[],
            emitted_token_blocks=[],
            stopped_blocks=[],
            stage_timing_records=[],
            block_size=block_size,
            draft_time=0.0,
            draft_transient_time=0.0,
            draft_commit_time=0.0,
            draft_prefill_time=0.0,
            verify_time=0.0,
            verifier_prefill_time=0.0,
            num_verify_steps=0,
            mean_draft_time_per_step=0.0,
            mean_verify_time_per_step=0.0,
            decode_elapsed_s=0.0,
        )

    output_ids = torch.full(
        (1, max_length + 1),
        MASK_ID,
        dtype=input_ids.dtype,
        device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    target_kv = DynamicCache()
    draft_kv = DynamicCache()
    kv_len = num_input_tokens
    draft_kv_len = num_input_tokens
    pending_commit = input_ids.to(draft_device).new_empty((1, 0))

    _sync(device)
    start_time = time.perf_counter()

    _sync(device)
    verifier_prefill_start = time.perf_counter()
    target_output = verifier(
        input_ids,
        position_ids=_pos(0, num_input_tokens, device),
        past_key_values=target_kv,
        use_cache=True,
    )
    target_kv = target_output.past_key_values
    anchor = _sample_tokens(
        target_output.logits[:, -1:, :],
        temperature=temperature,
        top_k=target_top_k,
        top_p=target_top_p,
    )[:, 0]
    output_ids[:, kv_len] = anchor
    _sync(device)
    verifier_prefill_time = time.perf_counter() - verifier_prefill_start

    _sync(draft_device)
    draft_prefill_start = time.perf_counter()
    draft_output = drafter(
        input_ids.to(draft_device),
        attention_mask=_prefix_attention_mask(num_input_tokens, draft_device, attention_mask_mode, block_size),
        position_ids=_pos(0, num_input_tokens, draft_device),
        past_key_values=draft_kv,
        use_cache=True,
        store_kv=True,
    )
    draft_kv = draft_output.past_key_values
    _sync(draft_device)
    draft_prefill_time = time.perf_counter() - draft_prefill_start

    if _cache_len(target_kv) != kv_len:
        raise RuntimeError(f"target_kv length mismatch after prefill: {_cache_len(target_kv)} != {kv_len}")
    if _cache_len(draft_kv) != kv_len:
        raise RuntimeError(f"draft_kv length mismatch after prefill: {_cache_len(draft_kv)} != {kv_len}")

    acceptance_lengths: list[int] = []
    draft_token_blocks: list[Any] = []
    target_token_blocks: list[Any] = []
    emitted_token_blocks: list[Any] = []
    stopped_blocks: list[bool] = []
    stage_timing_records: list[dict[str, Any]] = []
    draft_transient_time = 0.0
    draft_gpu_time = 0.0
    draft_commit_time = 0.0
    verify_time = 0.0
    verify_gpu_time = 0.0
    decoding_start=time.perf_counter()
    while kv_len < max_length:
        step_index = len(acceptance_lengths)
        prefix_len_before = int(kv_len)
        draft_wall_s = 0.0
        draft_gpu_s = 0.0
        verify_wall_s = 0.0
        verify_gpu_s = 0.0
        cur_block_size = min(int(block_size), max_length - kv_len)
        if cur_block_size <= 0:
            break

        block = torch.full(
            (1, cur_block_size),
            MASK_ID,
            dtype=input_ids.dtype,
            device=device,
        )
        block[:, 0] = anchor

        if cur_block_size > 1:
            _crop_cache(draft_kv, draft_kv_len)
            _sync(draft_device)
            draft_start = time.perf_counter()
            draft_start_event, draft_end_event = (
                _event_pair(draft_device) if collect_stage_timings else (None, None)
            )
            if draft_start_event is not None:
                draft_start_event.record(torch.cuda.current_stream(draft_device))
            draft_result = d3d_draft_block_fused_commit(
                drafter,
                block.to(draft_device),
                draft_kv=draft_kv,
                draft_kv_len=draft_kv_len,
                pending_commit=pending_commit,
                prefix_len=kv_len,
                denoising_steps=denoising_steps,
                remasking=remasking,
                threshold=threshold,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_id=MASK_ID,
                attention_mask_mode=attention_mask_mode,
                return_proposal_probs=temperature > 1e-5,
            )
            if temperature > 1e-5:
                block, _, draft_proposal_probs = draft_result
                draft_proposal_probs = draft_proposal_probs.to(device)
            else:
                block = draft_result
                draft_proposal_probs = None
            block = block.to(device)
            if pending_commit.shape[1] > 0:
                draft_kv_len = kv_len
                pending_commit = input_ids.to(draft_device).new_empty((1, 0))
                if _cache_len(draft_kv) != draft_kv_len:
                    raise RuntimeError(
                        f"draft_kv length mismatch after fused commit: "
                        f"{_cache_len(draft_kv)} != {draft_kv_len}"
                    )
            if draft_end_event is not None:
                draft_end_event.record(torch.cuda.current_stream(draft_device))
            _sync(draft_device)
            draft_wall_s = time.perf_counter() - draft_start
            draft_gpu_s = _event_elapsed_s(draft_start_event, draft_end_event)
            draft_transient_time += draft_wall_s
            draft_gpu_time += draft_gpu_s

        _crop_cache(target_kv, kv_len)
        _sync(device)
        verify_start = time.perf_counter()
        verify_start_event, verify_end_event = (
            _event_pair(device) if collect_stage_timings else (None, None)
        )
        if verify_start_event is not None:
            verify_start_event.record(torch.cuda.current_stream(device))
        target_output = verifier(
            block,
            position_ids=_pos(kv_len, kv_len + cur_block_size, device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv = target_output.past_key_values
        if verify_end_event is not None:
            verify_end_event.record(torch.cuda.current_stream(device))
        _sync(device)
        verify_wall_s = time.perf_counter() - verify_start
        verify_gpu_s = _event_elapsed_s(verify_start_event, verify_end_event)
        verify_time += verify_wall_s
        verify_gpu_time += verify_gpu_s
        if temperature > 1e-5 and cur_block_size > 1:
            from single.speculative_sampling import verify_speculative_block

            target_probs = _sampling_probs(
                target_output.logits,
                temperature=temperature,
                top_k=target_top_k,
                top_p=target_top_p,
            )
            draft_probs = draft_proposal_probs[:, 1:cur_block_size, :]
            verify_result = verify_speculative_block(
                block[:, 1:cur_block_size],
                target_probs,
                draft_probs,
            )
            accepted_tail = verify_result.accepted_draft_tokens
            correction_token = verify_result.correction_token.to(device)
            posterior = target_probs.argmax(dim=-1)
        else:
            posterior = _sample_tokens(
                target_output.logits,
                temperature=temperature,
                top_k=target_top_k,
                top_p=target_top_p,
            )
            if cur_block_size > 1:
                matches = block[:, 1:] == posterior[:, :-1]
                accepted_tail = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
            else:
                accepted_tail = 0
            correction_token = posterior[:, accepted_tail]

        commit_len = accepted_tail + 1
        commit_tokens = block[:, :commit_len]
        next_anchor = correction_token
        output_ids[:, kv_len : kv_len + commit_len] = commit_tokens

        _crop_cache(target_kv, kv_len + commit_len)
        pending_commit = commit_tokens.to(draft_device)

        kv_len += commit_len
        acceptance_lengths.append(commit_len)
        draft_token_blocks.append(block[0, 1:].detach())
        target_token_blocks.append(posterior[0].detach())
        emitted_token_blocks.append(commit_tokens[0].detach())
        if collect_stage_timings:
            stage_timing_records.append(
                {
                    "step_index": step_index,
                    "prefix_len_before": prefix_len_before,
                    "generated_len_before": prefix_len_before - num_input_tokens,
                    "block_size": int(cur_block_size),
                    "accepted_draft_tokens": int(accepted_tail),
                    "accept_length": int(commit_len),
                    "prefix_len_after": int(kv_len),
                    "generated_len_after": int(kv_len) - num_input_tokens,
                    "draft_wall_s": float(draft_wall_s),
                    "draft_gpu_s": float(draft_gpu_s),
                    "verify_wall_s": float(verify_wall_s),
                    "verify_gpu_s": float(verify_gpu_s),
                    "round_wall_s": float(draft_wall_s + verify_wall_s),
                    "round_gpu_s": float(draft_gpu_s + verify_gpu_s),
                }
            )
        anchor = next_anchor
        if kv_len < max_length:
            output_ids[:, kv_len] = anchor

        generated = output_ids[:, num_input_tokens : min(kv_len + 1, max_length)]
        stopped = _contains_stop(generated, stop_token_ids)
        stopped_blocks.append(bool(stopped))
        if stopped:
            break

    output_ids = output_ids[:, : min(kv_len + 1, max_length)]
    output_ids = _trim_at_stop(output_ids, num_input_tokens, stop_token_ids)
    _sync(device)
    decoding_time = time.perf_counter() - decoding_start

    if not return_stats:
        return output_ids

    num_output_tokens = int(output_ids.shape[1] - num_input_tokens)
    draft_time = draft_transient_time + draft_commit_time
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_per_output_token=decoding_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        draft_token_blocks=draft_token_blocks,
        target_token_blocks=target_token_blocks,
        emitted_token_blocks=emitted_token_blocks,
        stopped_blocks=stopped_blocks,
        stage_timing_records=stage_timing_records,
        block_size=block_size,
        draft_time=draft_time,
        draft_gpu_time=draft_gpu_time,
        draft_transient_time=draft_transient_time,
        draft_commit_time=draft_commit_time,
        draft_prefill_time=draft_prefill_time,
        verify_time=verify_time,
        verify_gpu_time=verify_gpu_time,
        verifier_prefill_time=verifier_prefill_time,
        num_verify_steps=len(acceptance_lengths),
        mean_draft_time_per_step=draft_time / max(len(acceptance_lengths), 1),
        mean_draft_gpu_time_per_step=draft_gpu_time / max(len(acceptance_lengths), 1),
        mean_verify_time_per_step=verify_time / max(len(acceptance_lengths), 1),
        mean_verify_gpu_time_per_step=verify_gpu_time / max(len(acceptance_lengths), 1),
        decode_elapsed_s=decoding_time,
    )


def _gsm8k_gold(row):
    return row["answer"].split("####")[-1].strip().replace(",", "")


def _boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None


def _humaneval_question(row):
    prompt = row.get("prompt", "")
    if prompt.startswith("Write a solution to the following problem"):
        return prompt
    return (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{prompt}\n```"
    )


def _livecodebench_question(row):
    if row.get('prompt'):
        return row['prompt']
    parts = ['Write a solution to the following problem and make sure that it passes the tests:']
    for field in ('question_title', 'question_content'):
        value = str(row.get(field) or '').replace('\r\n', '\n').strip()
        if value:
            parts.append(value)
    if row.get('starter_code'):
        parts.append(f"Starter code:\n```python\n{row['starter_code']}\n```")
    return '\n\n'.join(parts)


DATASETS = {
    "gsm8k": {
        "load": ("openai/gsm8k", "main", "test"),
        "question": lambda r: (
            r.get("prompt")
            or r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        "gold": _gsm8k_gold,
        "score": "math",
    },
    "math500": {
        "load": ("HuggingFaceH4/MATH-500", None, "test"),
        "question": lambda r: (
            r.get("prompt")
            or r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        "gold": lambda r: r["answer"].strip(),
        "score": "math",
    },
    "aime25": {
        "load": ("MathArena/aime_2025", None, "train"),
        "question": lambda r: (
            r.get("prompt")
            or r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        # Some local prompt caches intentionally contain no answer file.
        # Keep generation/throughput evaluation available and score accuracy only
        # when a caller supplies an answer-enriched row.
        "gold": lambda r: (
            str(r["answer"]).strip() if r.get("answer") is not None else None
        ),
        "score": "math",
    },
    "humaneval": {
        "load": ("openai/openai_humaneval", None, "test"),
        "question": _humaneval_question,
        "gold": None,
        "score": None,
    },
    "mbpp": {
        "load": ("google-research-datasets/mbpp", "sanitized", "test"),
        "question": lambda r: r.get("prompt") or r["text"],
        "gold": None,
        "score": None,
    },
    "mt-bench": {
        "load": ("HuggingFaceH4/mt_bench_prompts", None, "train"),
        "question": lambda r: (
            r.get("prompt")
            if isinstance(r.get("prompt"), str)
            else (r.get("prompt") or [""])[0]
        ),
        "gold": None,
        "score": None,
    },
    "livecodebench-v6": {
        "load": ("livecodebench/code_generation_lite", None, "test"),
        "filename": "test6.jsonl",
        "question": _livecodebench_question,
        "gold": None,
        "score": None,
    },
}


def _load_jsonl(path: Path):
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_local_dataset(dataset: str):
    cache_path = LOCAL_CACHE_DIR / f"{dataset}.jsonl"
    answer_path = LOCAL_ANSWER_DIR / f"{dataset}.jsonl"
    if not cache_path.exists():
        return None
    prompts = _load_jsonl(cache_path)
    answers = _load_jsonl(answer_path) if answer_path.exists() else [None] * len(prompts)
    if len(prompts) != len(answers):
        raise ValueError(f"local cache length mismatch: {cache_path}={len(prompts)} {answer_path}={len(answers)}")
    rows = []
    for idx, (prompt_row, answer_row) in enumerate(zip(prompts, answers)):
        turns = prompt_row.get("turns") or []
        if not turns:
            raise ValueError(f"missing turns in {cache_path}")
        row = dict(answer_row) if answer_row is not None else {}
        row.setdefault("dataset", dataset)
        row.setdefault("dataset_index", idx)
        row["prompt"] = turns[0]
        if dataset == "gsm8k" and answer_row is not None:
            row["gold"] = _gsm8k_gold(answer_row)
        elif dataset == "math500" and answer_row is not None:
            row["gold"] = answer_row["answer"].strip()
        rows.append(row)
    return rows


def load_rows(dataset: str, max_samples: int | None):
    cfg = dict(DATASETS[dataset])
    rows = _load_local_dataset(dataset)
    dataset_source = "local_cache"
    if rows is None:
        if cfg.get("load") is None:
            raise FileNotFoundError(f"missing local cache for {dataset}: {LOCAL_CACHE_DIR / f'{dataset}.jsonl'}")
        from datasets import load_dataset

        name, subset, split = cfg["load"]
        options = {'revision': os.environ.get('D3D_DATASET_REVISION', 'main')}
        if dataset == 'livecodebench-v6':
            rows, metadata = load_dataset_jsonl(name, cfg['filename'], options['revision'])
            cfg['fingerprint'] = metadata['sha256']
        else:
            loaded = load_dataset(name, subset, split=split, **options)
            cfg['fingerprint'] = loaded._fingerprint
            rows = list(loaded)
        cfg['revision'] = options['revision']
        dataset_source = "huggingface"
        for idx, row in enumerate(rows):
            row.setdefault("dataset", dataset)
            row.setdefault("dataset_index", idx)
    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]
    cfg['source'] = dataset_source
    cfg['samples'] = len(rows)
    cfg['sample_sha256'] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()
    ).hexdigest()
    return rows, cfg, dataset_source


def _extract_pred(text):
    boxed = _boxed(text)
    if boxed is not None:
        return boxed.replace("$", "").replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _norm(x):
    if x is None:
        return None
    x = x.strip().rstrip(".").replace("\\!", "").replace(" ", "")
    try:
        f = float(x)
        if not math.isfinite(f):
            return x
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return x


def _score_text(cfg: dict[str, Any], text: str, gold: str | None):
    if cfg.get("score") != "math" or gold is None:
        return None, None
    pred = _extract_pred(text)
    return pred, pred is not None and _norm(pred) == _norm(gold)


def _gold_for_row(cfg: dict[str, Any], row):
    if "gold" in row:
        return row["gold"]
    gold_fn = cfg.get("gold")
    if not gold_fn:
        return None
    try:
        return gold_fn(row)
    except KeyError:
        return None


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[max(0, min(idx, len(xs) - 1))])


def _compact_acceptance(per_sample_accept: list[float], flat_accept: list[int], block_size: int) -> dict[str, Any]:
    hist = Counter(int(x) for x in flat_accept)
    return {
        "num_samples": len(per_sample_accept),
        "num_verify_steps": len(flat_accept),
        "block_size": int(block_size),
        "mean_accept_per_sample": statistics.fmean(per_sample_accept) if per_sample_accept else 0.0,
        "mean_accept_per_verify": statistics.fmean(flat_accept) if flat_accept else 0.0,
        "median_accept_per_verify": statistics.median(flat_accept) if flat_accept else 0.0,
        "p90_accept_per_verify": _percentile(flat_accept, 0.90),
        "p95_accept_per_verify": _percentile(flat_accept, 0.95),
        "min_accept_per_verify": min(flat_accept) if flat_accept else 0,
        "max_accept_per_verify": max(flat_accept) if flat_accept else 0,
        "histogram": {str(k): hist.get(k, 0) for k in range(int(block_size) + 1)},
    }


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp_path, out_path)


def _summary_from_payload(args, payload: dict[str, Any], summary_json: str | None) -> dict[str, Any]:
    prompts = payload["prompts"]
    flat = [int(step["accept_length"]) for prompt in prompts for step in prompt.get("steps", [])]
    per_sample = [
        float(prompt.get("summary", {}).get("avg_accept_length", 0.0))
        for prompt in prompts
    ]
    summary = payload["summary"]
    compact = {
        "dataset": args.dataset,
        "dataset_source": payload.get("dataset_source"),
        "drafter": args.drafter,
        "drafter_tag": args.drafter_tag,
        "verifier": args.verifier,
        "num_samples": len(prompts),
        "block_size": args.block_size,
        "denoising_steps": args.denoising_steps,
        "confidence_threshold": args.confidence_threshold,
        "attention_mask_mode": args.attention_mask_mode,
        "target_attn_implementation": args.target_attn_implementation,
        "draft_attn_implementation": args.draft_attn_implementation,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "use_kvcache": bool(args.use_kvcache),
        "acceptance": {
            **_compact_acceptance(per_sample, flat, args.block_size),
            "mean_accepted_draft_tokens_per_verify": float(summary.get("avg_accepted_draft_tokens", 0.0)),
            "draft_accept_rate": float(summary.get("draft_accept_rate", 0.0)),
            "reject_at_0_rate": float(summary.get("reject_at_0_rate", 0.0)),
            "all_draft_accepted_rate": float(summary.get("all_draft_accepted_rate", 0.0)),
            "no_draft_num_steps": int(summary.get("no_draft_num_steps", 0)),
            "avg_no_draft_verify_tokens": float(summary.get("avg_no_draft_verify_tokens", 0.0)),
        },
        "timing": {
            "total_decode_elapsed_s": float(summary.get("total_decode_elapsed_s", 0.0)),
            "mean_time_per_output_token": float(summary.get("mean_time_per_output_token", 0.0)),
            "aggregate_decode_tps": float(summary.get("aggregate_decode_tps", 0.0)),
            "elapsed_tokens_per_s": float(summary.get("elapsed_tokens_per_s", 0.0)),
        },
        "accuracy": payload.get("accuracy"),
        "summary_path": str(Path(summary_json).resolve()) if summary_json else None,
        "results_path": str(Path(args.output_json).resolve()) if args.output_json else None,
    }
    baseline_timing = (payload.get("baseline") or {}).get("decode_timing") or {}
    compact["headline"] = headline_metrics(
        summary.get("aggregate_decode_tps", 0.0),
        baseline_timing.get("aggregate_decode_tps", 0.0),
        summary.get("avg_accept_length", 0.0),
        metric="decode_tps_excluding_prefill",
    )
    if summary_json:
        _write_json(summary_json, compact)
    return compact


def _build_prompt(question: str, enable_thinking: bool) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": question}],
        "enable_thinking": enable_thinking,
    }


def _d3d_steps(
    acceptance_lengths: list[int],
    block_size: int,
    draft_token_blocks: list[Any] | None = None,
    target_token_blocks: list[Any] | None = None,
    emitted_token_blocks: list[Any] | None = None,
    stopped_blocks: list[bool] | None = None,
    stage_timing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps = []
    for i, length in enumerate(acceptance_lengths):
        accepted = max(int(length) - 1, 0)
        if draft_token_blocks is not None and i < len(draft_token_blocks):
            draft_tokens = _tokens_to_int_list(draft_token_blocks[i])
        else:
            draft_tokens = [0] * max(int(block_size) - 1, 0)
        target_tokens = (
            _tokens_to_int_list(target_token_blocks[i])
            if target_token_blocks is not None and i < len(target_token_blocks)
            else []
        )
        emitted_tokens = (
            _tokens_to_int_list(emitted_token_blocks[i])
            if emitted_token_blocks is not None and i < len(emitted_token_blocks)
            else []
        )
        stopped = bool(stopped_blocks[i]) if stopped_blocks is not None and i < len(stopped_blocks) else False
        step = asdict(
            AcceptanceStep(
                accepted_draft_tokens=accepted,
                accept_length=int(length),
                emitted_count=len(emitted_tokens) if emitted_token_blocks is not None else int(length),
                emitted_tokens=emitted_tokens,
                draft_tokens=draft_tokens,
                target_tokens=target_tokens,
                stopped=stopped,
            )
        )
        if stage_timing_records is not None and i < len(stage_timing_records):
            record = stage_timing_records[i]
            step["sequence"] = {
                "prefix_len_before": int(record["prefix_len_before"]),
                "generated_len_before": int(record["generated_len_before"]),
                "block_size": int(record["block_size"]),
                "prefix_len_after": int(record["prefix_len_after"]),
                "generated_len_after": int(record["generated_len_after"]),
            }
            step["timing"] = {
                "draft_wall_s": float(record["draft_wall_s"]),
                "draft_gpu_s": float(record["draft_gpu_s"]),
                "verify_wall_s": float(record["verify_wall_s"]),
                "verify_gpu_s": float(record["verify_gpu_s"]),
                "round_wall_s": float(record["round_wall_s"]),
                "round_gpu_s": float(record["round_gpu_s"]),
            }
        steps.append(step)
    return steps


def _summary_step(step: dict[str, Any]) -> AcceptanceStep:
    return AcceptanceStep(
        accepted_draft_tokens=int(step["accepted_draft_tokens"]),
        accept_length=int(step["accept_length"]),
        emitted_count=int(step["emitted_count"]),
        emitted_tokens=list(step.get("emitted_tokens", [])),
        draft_tokens=list(step.get("draft_tokens", [])),
        target_tokens=list(step.get("target_tokens", [])),
        stopped=bool(step.get("stopped", False)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {args.dataset}; supported: {sorted(DATASETS)}")
    check_hardware(args.verifier_device, args.draft_device)
    target_path = resolve_model_path(args.verifier)
    draft_path = resolve_model_path(args.drafter)
    tokenizer_path = resolve_model_path(args.tokenizer) if args.tokenizer else target_path
    args.runtime = runtime_metadata({'target': target_path, 'draft': draft_path,
                                     'tokenizer': tokenizer_path})
    args.drafter_tag = _safe_custom_tag(args.drafter_tag, _safe_tag(args.drafter))
    args.verifier_tag = _safe_tag(args.verifier)
    args.verifier, args.drafter, args.tokenizer = target_path, draft_path, tokenizer_path
    model_type = args.model_type or _checkpoint_model_type(args.drafter)
    if args.use_kvcache and model_type not in {"d3d", "qwen3"}:
        raise ValueError("--use-kvcache is currently supported only for D3D/Qwen3 single-chain paths")

    dtype = _resolve_dtype(args.dtype)
    target_attn = os.environ.get("TARGET_ATTN_IMPLEMENTATION", args.target_attn_implementation)
    draft_attn = os.environ.get("DRAFT_ATTN_IMPLEMENTATION", args.draft_attn_implementation)
    args.target_attn_implementation = target_attn
    args.draft_attn_implementation = draft_attn
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    eos_token_ids = _eos_token_ids(tokenizer, args.eos_token_id)
    if not eos_token_ids:
        eos_token_ids = STOP_TOKEN_IDS

    verifier_device = args.verifier_device or "cuda:0"
    draft_device = args.draft_device or verifier_device
    target = _load_model(args.verifier, dtype, attn_implementation=target_attn).to(verifier_device).eval()
    draft = _load_model(args.drafter, dtype, attn_implementation=draft_attn).to(draft_device).eval()
    _disable_default_cache(target)
    _disable_default_cache(draft)

    load_limit = None if args.sample_index is not None else args.max_samples
    rows, cfg, dataset_source = load_rows(args.dataset, load_limit)
    if args.sample_index is not None:
        sample_index = int(args.sample_index)
        rows = [
            row for i, row in enumerate(rows)
            if int(row.get("dataset_index", i)) == sample_index or i == sample_index
        ]
        if not rows:
            raise ValueError(f"sample_index={sample_index} not found in dataset={args.dataset}")
        rows = rows[:1]
    print(f"Loaded {len(rows)} {args.dataset} samples from {dataset_source}", flush=True)

    start_time = _synced_time()
    per_prompt: list[dict[str, Any]] = []
    spec_ok = spec_eval = 0
    draft_tokens = _effective_draft_tokens(args.draft_tokens, args.block_size)

    for i, row in enumerate(rows):
        question = cfg["question"](row)
        gold = _gold_for_row(cfg, row)
        prompt = _build_prompt(question, args.enable_thinking)

        if model_type == "qwen3":
            text = _apply_chat_template(tokenizer, prompt["messages"], args.enable_thinking)
            ids = tokenizer.encode(text, return_tensors="pt").to(verifier_device)
            st = run_causal_generate(
                draft,
                target,
                ids,
                args.max_new_tokens,
                eos_token_ids,
                args.temperature,
                block_size=args.block_size,
                return_stats=True,
            )
            out_text = tokenizer.decode(st.output_ids[0, st.num_input_tokens:], skip_special_tokens=False)
            steps = _d3d_steps(
                [int(x) for x in st.acceptance_lengths],
                args.block_size,
                getattr(st, "draft_token_blocks", None),
                getattr(st, "target_token_blocks", None),
                getattr(st, "emitted_token_blocks", None),
                getattr(st, "stopped_blocks", None),
            )
            result = {
                "input_tokens": int(st.num_input_tokens),
                "output_tokens": int(st.num_output_tokens),
                "text": out_text,
                "summary": summarize_acceptance(
                    [_summary_step(step) for step in steps],
                    draft_tokens,
                ),
                "steps": steps,
                "target_first_token": None,
                "decode_elapsed_s": float(st.decode_elapsed_s),
                "tokens_per_s": (
                    int(st.num_output_tokens) / float(st.decode_elapsed_s)
                    if float(st.decode_elapsed_s) > 0
                    else 0.0
                ),
                "time_per_output_token": float(st.time_per_output_token),
                "timing": {
                    "draft_time": float(st.draft_time),
                    "draft_prefill_time": float(getattr(st, "draft_prefill_time", 0.0)),
                    "verify_time": float(st.verify_time),
                    "verifier_prefill_time": float(st.verifier_prefill_time),
                    "mean_draft_time_per_step": float(st.mean_draft_time_per_step),
                    "mean_verify_time_per_step": float(st.mean_verify_time_per_step),
                    "time_per_output_token": float(st.time_per_output_token),
                    "causal_drafter": True,
                },
            }
        elif model_type == "d3d":
            text = _apply_chat_template(tokenizer, prompt["messages"], args.enable_thinking)
            ids = tokenizer.encode(text, return_tensors="pt").to(verifier_device)
            d3d_generate = run_d3d_generate_cached if args.use_kvcache else run_d3d_generate
            st = d3d_generate(
                draft,
                target,
                ids,
                args.max_new_tokens,
                eos_token_ids,
                args.temperature,
                block_size=args.block_size,
                denoising_steps=args.denoising_steps,
                threshold=args.confidence_threshold,
                remasking=args.remasking_strategy,
                top_k=args.proposal_top_k,
                top_p=args.proposal_top_p,
                target_top_k=args.target_top_k,
                target_top_p=args.target_top_p,
                return_stats=True,
                attention_mask_mode=args.attention_mask_mode,
                collect_stage_timings=args.collect_stage_timings,
                draft_device=draft_device,
            )
            out_text = tokenizer.decode(st.output_ids[0, st.num_input_tokens:], skip_special_tokens=False)
            steps = _d3d_steps(
                [int(x) for x in st.acceptance_lengths],
                args.block_size,
                getattr(st, "draft_token_blocks", None),
                getattr(st, "target_token_blocks", None),
                getattr(st, "emitted_token_blocks", None),
                getattr(st, "stopped_blocks", None),
                getattr(st, "stage_timing_records", None),
            )
            result = {
                "input_tokens": int(st.num_input_tokens),
                "output_tokens": int(st.num_output_tokens),
                "text": out_text,
                "summary": summarize_acceptance(
                    [_summary_step(step) for step in steps],
                    draft_tokens,
                ),
                "steps": steps,
                "target_first_token": None,
                "decode_elapsed_s": float(st.decode_elapsed_s),
                "tokens_per_s": (
                    int(st.num_output_tokens) / float(st.decode_elapsed_s)
                    if float(st.decode_elapsed_s) > 0
                    else 0.0
                ),
                "time_per_output_token": float(st.time_per_output_token),
                "timing": {
                    "draft_time": float(st.draft_time),
                    "draft_gpu_time": float(getattr(st, "draft_gpu_time", 0.0)),
                    "draft_prefill_time": float(getattr(st, "draft_prefill_time", 0.0)),
                    "draft_transient_time": float(getattr(st, "draft_transient_time", st.draft_time)),
                    "draft_commit_time": float(getattr(st, "draft_commit_time", 0.0)),
                    "verify_time": float(st.verify_time),
                    "verify_gpu_time": float(getattr(st, "verify_gpu_time", 0.0)),
                    "verifier_prefill_time": float(st.verifier_prefill_time),
                    "mean_draft_time_per_step": float(st.mean_draft_time_per_step),
                    "mean_draft_gpu_time_per_step": float(
                        getattr(st, "mean_draft_gpu_time_per_step", 0.0)
                    ),
                    "mean_verify_time_per_step": float(st.mean_verify_time_per_step),
                    "mean_verify_gpu_time_per_step": float(
                        getattr(st, "mean_verify_gpu_time_per_step", 0.0)
                    ),
                    "time_per_output_token": float(st.time_per_output_token),
                    "collect_stage_timings": bool(args.collect_stage_timings),
                },
            }
        else:
            raise ValueError(f"unsupported drafter model_type={model_type!r}")

        pred, hit = _score_text(cfg, result["text"], gold)
        if hit is not None:
            spec_eval += 1
            spec_ok += int(hit)
        result.update(
            {
                "index": i,
                "dataset_index": int(row.get("dataset_index", i)),
                "prompt": question,
                "gold": gold,
                "spec_pred": pred,
                "spec_correct": hit,
            }
        )
        per_prompt.append(result)

        payload = _build_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type)
        payload["accuracy"] = {
            "spec_evaluated": spec_eval,
            "spec_accuracy": spec_ok / spec_eval if spec_eval else None,
        }
        _write_json(args.output_json, payload)
        if args.verbose:
            print(
                json.dumps(
                    {
                        "index": i,
                        "output_tokens": result["output_tokens"],
                        "decode_elapsed_s": result["decode_elapsed_s"],
                        **result["summary"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = _build_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type)
    payload["accuracy"] = {
        "spec_evaluated": spec_eval,
        "spec_accuracy": spec_ok / spec_eval if spec_eval else None,
    }
    _write_json(args.output_json, payload)
    compact = _summary_from_payload(args, payload, args.summary_json)
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return payload


def _build_payload(
    args: argparse.Namespace,
    per_prompt: list[dict[str, Any]],
    draft_tokens: int,
    eos_token_ids: list[int],
    start_time: float,
    dataset_source: str,
    model_type: str | None,
) -> dict[str, Any]:
    all_steps = [
        _summary_step(step)
        for result in per_prompt
        for step in result.get("steps", [])
    ]
    elapsed_s = _synced_time() - start_time
    summary = summarize_acceptance(all_steps, draft_tokens)
    summary.update(
        {
            "dataset": args.dataset,
            "dataset_source": dataset_source,
            "model_type": model_type,
            "num_prompts": len(per_prompt),
            "total_output_tokens": sum(int(x["output_tokens"]) for x in per_prompt),
            "elapsed_s": elapsed_s,
            **summarize_timing_metrics(per_prompt, elapsed_s=elapsed_s),
            "target_model": args.verifier,
            "draft_model": args.drafter,
            "draft_tokens": draft_tokens,
            "block_size": args.block_size,
            "denoising_steps": args.denoising_steps,
            "remasking_strategy": args.remasking_strategy,
            "confidence_threshold": args.confidence_threshold,
            "attention_mask_mode": args.attention_mask_mode,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_ids": eos_token_ids,
            "temperature": args.temperature,
            "proposal_top_k": args.proposal_top_k,
            "proposal_top_p": args.proposal_top_p,
            "target_top_k": args.target_top_k,
            "target_top_p": args.target_top_p,
            "use_kvcache": bool(args.use_kvcache),
            "collect_stage_timings": bool(args.collect_stage_timings),
        }
    )
    return {"summary": summary, "dataset_source": dataset_source, "prompts": per_prompt,
            "runtime": getattr(args, 'runtime', {}), "config": vars(args)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D3D synchronous draft verification and diagnostics.")
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--drafter-tag", default=None)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--draft-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--proposal-top-k", type=int, default=0)
    parser.add_argument("--proposal-top-p", type=float, default=1.0)
    parser.add_argument("--target-top-k", type=int, default=0)
    parser.add_argument("--target-top-p", type=float, default=1.0)
    parser.add_argument("--verifier-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--eos-token-id", type=int, action="append", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-kvcache", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--remasking-strategy", default="low_confidence")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--attention-mask-mode", choices=["block_causal", "causal"], default="block_causal")
    parser.add_argument("--collect-stage-timings", action="store_true")
    parser.add_argument("--target-attn-implementation", default="flash_attention_2")
    parser.add_argument("--draft-attn-implementation", default="flash_attention_2")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
