#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
import traceback
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

if __package__:
    from . import eval_sync as sync
else:
    import eval_sync as sync

def _device(name: str | torch.device) -> torch.device:
    return torch.device(name)


def _sync_device(device: str | torch.device) -> None:
    if torch.cuda.is_available() and _device(device).type == "cuda":
        torch.cuda.synchronize(_device(device))


def _event_pair(device: torch.device):
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _event_elapsed_s(start, end) -> float:
    if start is None or end is None:
        return 0.0
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _verifier_process_main(
    connection,
    model_path: str,
    device_name: str,
    dtype_name: str,
    attn_implementation: str | None,
    cpu_affinity: tuple[int, ...],
) -> None:
    try:
        torch.set_grad_enabled(False)
        if cpu_affinity:
            os.sched_setaffinity(0, set(cpu_affinity))
        device = _device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = sync._resolve_dtype(dtype_name)
        verifier = sync._load_model(
            model_path,
            dtype,
            attn_implementation=attn_implementation,
        ).to(device).eval()
        sync._disable_default_cache(verifier)
        target_kv = None
        connection.send({"status": "ready"})

        while True:
            message = connection.recv()
            command = message["command"]
            if command == "close":
                break
            if command == "prefill":
                input_ids = torch.tensor(message["input_ids"], dtype=torch.long, device=device)
                target_kv = DynamicCache()
                _sync_device(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    target_output = verifier(
                        input_ids,
                        position_ids=sync._pos(0, input_ids.shape[1], device),
                        past_key_values=target_kv,
                        use_cache=True,
                    )
                    target_kv = target_output.past_key_values
                    anchor = sync._sample_tokens(
                        target_output.logits[:, -1:, :],
                        temperature=float(message["temperature"]),
                        top_k=int(message.get("target_top_k", 0)),
                        top_p=float(message.get("target_top_p", 1.0)),
                    )[:, 0]
                _sync_device(device)
                connection.send(
                    {
                        "status": "ok",
                        "anchor": anchor.detach().cpu().tolist(),
                        "elapsed_s": time.perf_counter() - start,
                        "cache_len": sync._cache_len(target_kv),
                    }
                )
                continue
            if command == "verify":
                worker_wall_start = time.perf_counter()
                if target_kv is None:
                    raise RuntimeError("verifier process received verify before prefill")
                kv_len = int(message["kv_len"])
                block = torch.tensor(message["block"], dtype=torch.long, device=device)
                sync._crop_cache(target_kv, kv_len)
                start_event, end_event = _event_pair(device)
                if start_event is not None:
                    start_event.record(torch.cuda.current_stream(device))
                else:
                    wall_start = time.perf_counter()
                with torch.inference_mode():
                    target_output = verifier(
                        block,
                        position_ids=sync._pos(kv_len, kv_len + block.shape[1], device),
                        past_key_values=target_kv,
                        use_cache=True,
                    )
                    target_kv = target_output.past_key_values
                    temperature = float(message["temperature"])
                    target_probs = sync._sampling_probs(
                        target_output.logits,
                        temperature=temperature,
                        top_k=int(message.get("target_top_k", 0)),
                        top_p=float(message.get("target_top_p", 1.0)),
                    )
                    posterior = target_probs.argmax(dim=-1)
                    tail_tokens = block[:, 1:]
                    if tail_tokens.shape[1] == 0:
                        accepted_tail = 0
                        correction_token = sync._sample_tokens(
                            target_output.logits[:, -1:, :],
                            temperature=temperature,
                            top_k=int(message.get("target_top_k", 0)),
                            top_p=float(message.get("target_top_p", 1.0)),
                        )[:, 0]
                    elif temperature > 1e-5:
                        from single.speculative_sampling import verify_speculative_block_sparse

                        draft_token_ids = message.get("draft_token_ids")
                        draft_token_probs = message.get("draft_token_probs")
                        if draft_token_ids is None or draft_token_probs is None:
                            raise RuntimeError("temperature > 0 verification requires sparse draft q")
                        draft_token_ids = draft_token_ids.to(device=device, dtype=torch.long)
                        draft_token_probs = draft_token_probs.to(device=device, dtype=torch.float32)
                        verify_result = verify_speculative_block_sparse(
                            tail_tokens,
                            target_probs,
                            draft_token_ids,
                            draft_token_probs,
                        )
                        accepted_tail = verify_result.accepted_draft_tokens
                        correction_token = verify_result.correction_token.to(device)
                    else:
                        if tail_tokens.shape[1] > 0:
                            matches = tail_tokens == posterior[:, :-1]
                            accepted_tail = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
                        else:
                            accepted_tail = 0
                        correction_token = posterior[:, accepted_tail]
                if end_event is not None:
                    end_event.record(torch.cuda.current_stream(device))
                    elapsed_s = _event_elapsed_s(start_event, end_event)
                else:
                    _sync_device(device)
                    elapsed_s = time.perf_counter() - wall_start
                posterior_values = posterior.detach().cpu().tolist()
                response_ready_at = time.perf_counter()
                connection.send(
                    {
                        "status": "ok",
                        "posterior": posterior_values,
                        "accepted_tail": int(accepted_tail),
                        "correction_token": correction_token.detach().cpu().tolist(),
                        "elapsed_s": elapsed_s,
                        "worker_wall_s": response_ready_at - worker_wall_start,
                        "dispatch_delay_s": max(
                            worker_wall_start - float(message.get("submitted_at", worker_wall_start)),
                            0.0,
                        ),
                        "response_ready_at": response_ready_at,
                    }
                )
                continue
            raise ValueError(f"unknown verifier process command: {command}")
    except BaseException as exc:
        try:
            connection.send(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class VerifierProcessClient:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        attn_implementation: str | None,
        cpu_affinity: tuple[int, ...] = (),
        visible_device: str | None = None,
    ) -> None:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._process = context.Process(
            target=_verifier_process_main,
            args=(child_connection, model_path, device, dtype, attn_implementation, cpu_affinity),
            daemon=True,
        )
        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        try:
            self._process.start()
        finally:
            if original_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
        child_connection.close()
        self._verify_submitted_at: float | None = None

    def _recv(self) -> dict[str, Any]:
        response = self._connection.recv()
        if response.get("status") == "error":
            raise RuntimeError(
                f"verifier process failed: {response.get('error')}\n{response.get('traceback', '')}"
            )
        return response

    def wait_ready(self) -> None:
        response = self._recv()
        if response.get("status") != "ready":
            raise RuntimeError(f"unexpected verifier startup response: {response}")

    def prefill(
        self,
        input_ids,
        temperature: float,
        target_top_k: int,
        target_top_p: float,
    ) -> tuple[list[list[int]], float, int]:
        self._connection.send(
            {
                "command": "prefill",
                "input_ids": input_ids.detach().cpu().tolist(),
                "temperature": float(temperature),
                "target_top_k": int(target_top_k),
                "target_top_p": float(target_top_p),
            }
        )
        response = self._recv()
        return response["anchor"], float(response["elapsed_s"]), int(response["cache_len"])

    def submit_verify(
        self,
        block,
        draft_proposal,
        kv_len: int,
        temperature: float,
        target_top_k: int,
        target_top_p: float,
    ) -> float:
        submit_start = time.perf_counter()
        block_values = block.detach().cpu().tolist()
        submitted_at = time.perf_counter()
        self._connection.send(
            {
                "command": "verify",
                "block": block_values,
                "kv_len": int(kv_len),
                "temperature": float(temperature),
                "target_top_k": int(target_top_k),
                "target_top_p": float(target_top_p),
                "draft_token_ids": (
                    draft_proposal[0].detach().to(device="cpu", dtype=torch.int32)
                    if draft_proposal is not None
                    else None
                ),
                "draft_token_probs": (
                    draft_proposal[1].detach().to(device="cpu", dtype=torch.float32)
                    if draft_proposal is not None
                    else None
                ),
                "submitted_at": submitted_at,
            }
        )
        self._verify_submitted_at = submitted_at
        return time.perf_counter() - submit_start

    def receive_verify(self) -> tuple[list[list[int]], int, list[int], dict[str, float]]:
        response = self._recv()
        received_at = time.perf_counter()
        submitted_at = self._verify_submitted_at or received_at
        self._verify_submitted_at = None
        return (
            response["posterior"],
            int(response["accepted_tail"]),
            response["correction_token"],
            {
            "gpu_s": float(response["elapsed_s"]),
            "worker_wall_s": float(response.get("worker_wall_s", response["elapsed_s"])),
            "dispatch_delay_s": float(response.get("dispatch_delay_s", 0.0)),
            "response_pipe_s": max(
                received_at - float(response.get("response_ready_at", received_at)),
                0.0,
            ),
            "round_trip_s": max(received_at - submitted_at, 0.0),
            },
        )

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._connection.send({"command": "close"})
            except (BrokenPipeError, EOFError):
                pass
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()


def _parse_cpu_affinity(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    cpus: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            cpus.extend(range(int(start_text), int(end_text) + 1))
        else:
            cpus.append(int(item))
    return tuple(dict.fromkeys(cpus))


def _parse_packed_accepted_counts(value: str) -> tuple[int, ...] | None:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    counts = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not counts:
        raise argparse.ArgumentTypeError("packed accepted counts must be 'all' or a comma-separated list")
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("packed accepted counts must be positive integers")
    return counts


def _isolated_cuda_device(device_name: str) -> tuple[str | None, str]:
    device = _device(device_name)
    if device.type != "cuda":
        return None, device_name
    logical_index = int(device.index or 0)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
        if logical_index >= len(entries):
            raise ValueError(
                f"CUDA device {device_name} is outside CUDA_VISIBLE_DEVICES={visible_devices}"
            )
        return entries[logical_index], "cuda:0"
    return str(logical_index), "cuda:0"


@torch.inference_mode()
def d3d_draft_from_anchor(
    drafter,
    *,
    anchor,
    block_size: int,
    prefix_len: int,
    draft_kv,
    dtype,
    denoising_steps: int,
    threshold: float,
    remasking: str,
    temperature: float,
    top_k: int,
    top_p: float,
    mask_id: int,
    attention_mask_mode: str = "block_causal",
    return_topk: int = 0,
):
    device = anchor.device
    block = torch.full((1, int(block_size)), int(mask_id), dtype=dtype, device=device)
    block[:, 0] = anchor.reshape(1)
    if int(block_size) > 1:
        if int(return_topk) > 0:
            empty_pending = block[:, :0]
            draft_result = sync.d3d_draft_block_fused_commit(
                drafter,
                block,
                draft_kv=draft_kv,
                draft_kv_len=int(prefix_len),
                pending_commit=empty_pending,
                prefix_len=int(prefix_len),
                denoising_steps=int(denoising_steps),
                remasking=remasking,
                threshold=float(threshold),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                mask_id=int(mask_id),
                attention_mask_mode=attention_mask_mode,
                return_topk=int(return_topk),
                return_sparse_proposal=temperature > 1e-5,
            )
            if temperature > 1e-5:
                block, metrics, proposal_probs = draft_result
            else:
                block, metrics = draft_result
                proposal_probs = None
            return block, metrics.get("top_tokens"), proposal_probs
        draft_result = sync.d3d_draft_block(
            drafter,
            block,
            prefix_len=int(prefix_len),
            denoising_steps=int(denoising_steps),
            remasking=remasking,
            threshold=float(threshold),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            mask_id=int(mask_id),
            past_key_values=draft_kv,
            attention_mask_mode=attention_mask_mode,
            return_sparse_proposal=temperature > 1e-5,
        )
        if temperature > 1e-5:
            block, proposal_probs = draft_result
        else:
            block = draft_result
            proposal_probs = None
    else:
        proposal_probs = None
    return block, None, proposal_probs


@torch.inference_mode()
def d3d_draft_all_outcomes_packed(
    drafter,
    *,
    pending_block,
    branch_token_count: int,
    draft_kv,
    draft_kv_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    mask_id: int,
    attention_mask_mode: str,
    pending_top_tokens=None,
    anchor_num: int = 0,
    accepted_counts: tuple[int, ...] | None = None,
) -> tuple[
    dict[tuple[int, int | None], Any],
    dict[tuple[int, int | None], Any],
    dict[tuple[int, int | None], Any],
]:
    device = pending_block.device
    pending_len = int(pending_block.shape[1])
    branch_token_count = int(branch_token_count)
    if pending_len <= 0 or branch_token_count <= 0:
        return {}, {}, {}

    selected_counts = (
        tuple(range(1, pending_len + 1))
        if accepted_counts is None
        else tuple(count for count in accepted_counts if count <= pending_len)
    )
    branch_specs: list[tuple[int, int | None, Any | None]] = []
    for accepted_len in selected_counts:
        if int(anchor_num) == 0 or accepted_len == pending_len:
            branch_specs.append((accepted_len, None, None))
            continue
        if pending_top_tokens is None:
            raise RuntimeError("packed anchor candidates require pending top-token logits")
        if int(pending_top_tokens.shape[1]) <= accepted_len:
            raise RuntimeError(
                f"pending top-token positions do not cover accepted_count={accepted_len}"
            )
        required_topk = int(anchor_num) + 1
        if int(pending_top_tokens.shape[2]) < required_topk:
            raise RuntimeError(
                f"pending top-token width {pending_top_tokens.shape[2]} is smaller than "
                f"required candidate width {required_topk} for anchor_num={anchor_num}"
            )
        returned_token = pending_block[:, accepted_len]
        selected_anchors = 0
        for rank in range(required_topk):
            anchor = pending_top_tokens[:, accepted_len, rank].to(device=device)
            if torch.equal(anchor.reshape(-1), returned_token.reshape(-1)):
                continue
            anchor_token = int(anchor.reshape(-1)[0].item())
            branch_specs.append((accepted_len, anchor_token, anchor))
            selected_anchors += 1
            if selected_anchors == int(anchor_num):
                break
        if selected_anchors != int(anchor_num):
            raise RuntimeError(
                f"only selected {selected_anchors} anchors for anchor_num={anchor_num}"
            )

    if not branch_specs:
        return {}, {}, {}

    candidate_token_count = len(branch_specs) * branch_token_count
    query_len = pending_len + candidate_token_count
    total_kv_len = int(draft_kv_len) + query_len

    draft_input = torch.full(
        (1, query_len),
        int(mask_id),
        dtype=pending_block.dtype,
        device=device,
    )
    draft_input[:, :pending_len] = pending_block
    position_ids = torch.empty((1, query_len), dtype=torch.long, device=device)
    position_ids[:, :pending_len] = torch.arange(
        int(draft_kv_len),
        int(draft_kv_len) + pending_len,
        device=device,
    )
    attention_mask = torch.zeros(
        (1, 1, query_len, total_kv_len),
        dtype=torch.bool,
        device=device,
    )

    if attention_mask_mode == "causal":
        for pending_pos in range(pending_len):
            attention_mask[
                0,
                0,
                pending_pos,
                : int(draft_kv_len) + pending_pos + 1,
            ] = True
    elif attention_mask_mode == "block_causal":
        attention_mask[
            0,
            0,
            :pending_len,
            : int(draft_kv_len) + pending_len,
        ] = True
    else:
        raise ValueError(f"unsupported attention_mask_mode={attention_mask_mode!r}")

    segments: list[tuple[tuple[int, int | None], int, int, Any | None]] = []
    segment_start = pending_len
    for accepted_len, anchor_token, anchor in branch_specs:
        segment_stop = segment_start + branch_token_count
        if anchor is not None:
            draft_input[:, segment_start] = anchor
        position_ids[:, segment_start:segment_stop] = torch.arange(
            int(draft_kv_len) + accepted_len,
            int(draft_kv_len) + accepted_len + branch_token_count,
            device=device,
        )
        rows = slice(segment_start, segment_stop)
        attention_mask[0, 0, rows, : int(draft_kv_len)] = True
        attention_mask[
            0,
            0,
            rows,
            int(draft_kv_len) : int(draft_kv_len) + accepted_len,
        ] = True
        attention_mask[
            0,
            0,
            rows,
            int(draft_kv_len) + segment_start : int(draft_kv_len) + segment_stop,
        ] = True
        segments.append(((accepted_len, anchor_token), segment_start, segment_stop, anchor))
        segment_start = segment_stop

    try:
        output = drafter(
            draft_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=draft_kv,
            use_cache=True,
            store_kv=True,
            logits_to_keep=candidate_token_count,
        )
        if temperature > 1e-5:
            candidate_tokens, candidate_token_ids, candidate_token_probs = sync._sample_with_sparse_probs(
                output.logits,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
        else:
            candidate_tokens = sync._sample_tokens(
                output.logits,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
            candidate_token_ids = None
            candidate_token_probs = None
        candidate_top_tokens = None
        if int(anchor_num) > 0:
            topk = min(int(anchor_num) + 1, int(output.logits.shape[-1]))
            candidate_top_tokens = torch.topk(output.logits, k=topk, dim=-1).indices
        sync._crop_cache(draft_kv, int(draft_kv_len) + pending_len)
    except BaseException:
        try:
            sync._crop_cache(draft_kv, int(draft_kv_len))
        except BaseException:
            pass
        raise

    candidate_map = {}
    candidate_top_tokens_map = {}
    candidate_probs_map = {}
    for key, segment_start, segment_stop, anchor in segments:
        output_start = segment_start - pending_len
        output_stop = segment_stop - pending_len
        candidate = candidate_tokens[:, output_start:output_stop].clone()
        if anchor is not None:
            candidate[:, 0] = anchor
        candidate_map[key] = candidate
        if candidate_token_ids is not None:
            candidate_probs_map[key] = (
                candidate_token_ids[:, output_start:output_stop].clone(),
                candidate_token_probs[:, output_start:output_stop].clone(),
            )
        if candidate_top_tokens is not None:
            candidate_top_tokens_map[key] = candidate_top_tokens[
                :, output_start:output_stop
            ].clone()
    return candidate_map, candidate_top_tokens_map, candidate_probs_map


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(x) for x in values).items())}


def _redraft_span_accept_lengths(sources: list[str], acceptance_lengths: list[int]) -> list[int]:
    """Sum accepted tokens from each redraft through the step before the next redraft."""
    spans: list[int] = []
    current_sum: int | None = None
    for source, accept_len in zip(sources, acceptance_lengths):
        if source == "redraft":
            if current_sum is not None:
                spans.append(current_sum)
            current_sum = int(accept_len)
        elif current_sum is not None:
            current_sum += int(accept_len)
    if current_sum is not None:
        spans.append(current_sum)
    return spans


def _high_speed_continuous_stats(
    block_sizes: list[int],
    acceptance_lengths: list[int],
    min_consecutive_all_pass_blocks: int = 3,
) -> dict[str, int | float]:
    """Count every block in all-pass runs meeting the minimum run length."""
    total_blocks = len(acceptance_lengths)
    high_speed_blocks = 0
    high_speed_regions = 0
    run_blocks = 0

    def commit_run() -> None:
        nonlocal high_speed_blocks, high_speed_regions
        if run_blocks >= int(min_consecutive_all_pass_blocks):
            high_speed_regions += 1
            high_speed_blocks += run_blocks

    for block_size, accept_len in zip(block_sizes, acceptance_lengths):
        if int(accept_len) == int(block_size):
            run_blocks += 1
            continue
        commit_run()
        run_blocks = 0
    commit_run()

    return {
        "high_speed_min_consecutive_all_pass_blocks": int(min_consecutive_all_pass_blocks),
        "high_speed_continuous_blocks": int(high_speed_blocks),
        "high_speed_total_blocks": int(total_blocks),
        "high_speed_continuous_block_rate": (
            high_speed_blocks / total_blocks if total_blocks else 0.0
        ),
        "high_speed_continuous_regions": int(high_speed_regions),
    }


def _verify_source_stats(
    sources: list[str],
    block_sizes: list[int],
    acceptance_lengths: list[int],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for source, block_size, accept_len in zip(sources, block_sizes, acceptance_lengths):
        grouped.setdefault(source, []).append((int(block_size), int(accept_len)))

    out: dict[str, dict[str, float | int]] = {}
    for source, rows in sorted(grouped.items()):
        all_pass = sum(1 for block_size, accept_len in rows if accept_len == block_size)
        out[source] = {
            "count": len(rows),
            "all_pass_count": all_pass,
            "all_pass_rate": all_pass / len(rows) if rows else 0.0,
            "mean_block_size": statistics.fmean(block_size for block_size, _ in rows),
            "mean_accept_len": statistics.fmean(accept_len for _, accept_len in rows),
            "accept_len_histogram": _histogram(
                [accept_len for _, accept_len in rows]
            ),
            "reject_at_0_rate": (
                sum(1 for _, accept_len in rows if accept_len <= 1) / len(rows)
                if rows
                else 0.0
            ),
        }
    return out


def _d3d_steps_dynamic(
    acceptance_lengths: list[int],
    block_sizes: list[int],
    verify_sources: list[str] | None = None,
    draft_token_blocks: list[Any] | None = None,
    target_token_blocks: list[Any] | None = None,
    emitted_token_blocks: list[Any] | None = None,
    stopped_blocks: list[bool] | None = None,
) -> list[dict[str, Any]]:
    steps = []
    for i, (length, block_size) in enumerate(zip(acceptance_lengths, block_sizes)):
        source = verify_sources[i] if verify_sources is not None and i < len(verify_sources) else "redraft"
        accepted = int(length) if source.startswith("async_") else max(int(length) - 1, 0)
        if draft_token_blocks is not None and i < len(draft_token_blocks):
            draft_tokens = sync._tokens_to_int_list(draft_token_blocks[i])
        else:
            draft_len = max(int(block_size) - 1, 0)
            draft_tokens = [0] * draft_len
        target_tokens = (
            sync._tokens_to_int_list(target_token_blocks[i])
            if target_token_blocks is not None and i < len(target_token_blocks)
            else []
        )
        emitted_tokens = (
            sync._tokens_to_int_list(emitted_token_blocks[i])
            if emitted_token_blocks is not None and i < len(emitted_token_blocks)
            else []
        )
        stopped = bool(stopped_blocks[i]) if stopped_blocks is not None and i < len(stopped_blocks) else False
        steps.append(
            asdict(
                sync.AcceptanceStep(
                    accepted_draft_tokens=accepted,
                    accept_length=int(length),
                    emitted_count=len(emitted_tokens) if emitted_token_blocks is not None else int(length),
                    emitted_tokens=emitted_tokens,
                    draft_tokens=draft_tokens,
                    target_tokens=target_tokens,
                    stopped=stopped,
                    mode="no-draft" if len(draft_tokens) == 0 else "draft",
                )
            )
        )
    return steps


@torch.inference_mode()
def run_d3d_generate_async(
    drafter,
    verifier_client: VerifierProcessClient,
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
    verifier_device: str = "cuda:0",
    draft_device: str = "cuda:1",
    async_blocks: int = 1,
    attention_mask_mode: str = "block_causal",
    anchor_num: int = 0,
    packed_accepted_counts: tuple[int, ...] | None = None,
):
    if int(denoising_steps) != 1:
        raise ValueError("eval_async.py currently requires --denoising-steps 1")
    if int(block_size) <= 1:
        raise ValueError("eval_async.py requires --block-size > 1")
    if int(async_blocks) < 1:
        raise ValueError("--async-blocks must be >= 1")
    if int(anchor_num) < 0:
        raise ValueError("--anchor-num must be non-negative")
    if temperature > 1e-5 and int(top_k) <= 0:
        raise ValueError("temperature > 0 async sampling requires --proposal-top-k > 0")
    if packed_accepted_counts is not None and any(
        int(count) < 1 for count in packed_accepted_counts
    ):
        raise ValueError("--packed-accepted-counts must contain positive integers")

    draft_device_t = _device(draft_device)

    input_ids_draft = input_ids.to(draft_device_t)
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)
    if max_new_tokens <= 0:
        return input_ids_draft if not return_stats else SimpleNamespace(
            output_ids=input_ids_draft,
            num_input_tokens=num_input_tokens,
            num_output_tokens=0,
            time_per_output_token=0.0,
            acceptance_lengths=[],
            draft_token_blocks=[],
            target_token_blocks=[],
            emitted_token_blocks=[],
            stopped_blocks=[],
            block_size=block_size,
            draft_time=0.0,
            draft_prefill_time=0.0,
            draft_transient_time=0.0,
            draft_commit_time=0.0,
            verify_time=0.0,
            verifier_prefill_time=0.0,
            num_verify_steps=0,
            mean_draft_time_per_step=0.0,
            mean_verify_time_per_step=0.0,
            decode_elapsed_s=0.0,
            verify_block_sizes=[],
            async_stats={},
        )

    output_ids = torch.full(
        (1, max_length + 1),
        sync.MASK_ID,
        dtype=input_ids_draft.dtype,
        device=draft_device_t,
    )
    output_ids[:, :num_input_tokens] = input_ids_draft

    draft_kv = DynamicCache()
    kv_len = num_input_tokens

    _sync_device(draft_device_t)
    start_time = time.perf_counter()

    anchor_values, verifier_prefill_time, target_cache_len = verifier_client.prefill(
        input_ids,
        temperature,
        target_top_k,
        target_top_p,
    )
    anchor_target = torch.tensor(anchor_values, dtype=input_ids_draft.dtype, device=draft_device_t)
    output_ids[:, kv_len] = anchor_target

    draft_prefill_start = time.perf_counter()
    draft_output = drafter(
        input_ids_draft,
        attention_mask=sync._prefix_attention_mask(
            num_input_tokens,
            draft_device_t,
            attention_mask_mode,
            block_size,
        ),
        position_ids=sync._pos(0, num_input_tokens, draft_device_t),
        past_key_values=draft_kv,
        use_cache=True,
        store_kv=True,
    )
    draft_kv = draft_output.past_key_values
    _sync_device(draft_device_t)
    draft_prefill_time = time.perf_counter() - draft_prefill_start

    if target_cache_len != kv_len:
        raise RuntimeError(f"target_kv length mismatch after prefill: {target_cache_len} != {kv_len}")
    if sync._cache_len(draft_kv) != kv_len:
        raise RuntimeError(f"draft_kv length mismatch after prefill: {sync._cache_len(draft_kv)} != {kv_len}")

    acceptance_lengths: list[int] = []
    draft_token_blocks: list[Any] = []
    target_token_blocks: list[Any] = []
    emitted_token_blocks: list[Any] = []
    stopped_blocks: list[bool] = []
    verify_block_sizes: list[int] = []
    verify_sources: list[str] = []
    draft_sync_time = 0.0
    redraft_gpu_time = 0.0
    async_draft_time = 0.0
    async_draft_gpu_time = 0.0
    verify_time = 0.0
    state_rollback_time = 0.0
    state_rollback_gpu_time = 0.0
    verifier_worker_wall_time = 0.0
    verifier_dispatch_delay_time = 0.0
    verifier_response_pipe_time = 0.0
    verifier_round_trip_time = 0.0
    verify_submit_wall_time = 0.0
    overlap_wall_time = 0.0
    wait_for_verify_time = 0.0
    sync_redraft_steps = 0
    async_ready_hits = 0
    all_pass_steps = 0
    rollback_steps = 0
    async_blocks_per_step: list[int] = []
    async_useful_blocks = 0
    async_discarded_blocks = 0
    async_prefix_candidate_lens: list[int] = []
    packed_outcome_hits = 0
    packed_partial_hits = 0
    packed_full_hits = 0
    packed_candidate_tokens = 0
    packed_anchor_candidates_generated = 0
    packed_anchor_lookup_attempts = 0
    packed_anchor_matches = 0
    packed_unanchored_candidates_generated = 0
    packed_unanchored_hits = 0

    def draft_recovery(anchor, prefix_len: int):
        nonlocal draft_sync_time, redraft_gpu_time, sync_redraft_steps
        sync._crop_cache(draft_kv, prefix_len)
        recovery_size = min(int(block_size), max_length - int(prefix_len))
        if recovery_size <= 0:
            return None, None, None
        _sync_device(draft_device_t)
        draft_start = time.perf_counter()
        start_event, end_event = _event_pair(draft_device_t)
        if start_event is not None:
            start_event.record(torch.cuda.current_stream(draft_device_t))
        recovery, recovery_top_tokens, recovery_probs = d3d_draft_from_anchor(
            drafter,
            anchor=anchor,
            block_size=recovery_size,
            prefix_len=prefix_len,
            draft_kv=draft_kv,
            dtype=input_ids_draft.dtype,
            denoising_steps=denoising_steps,
            threshold=threshold,
            remasking=remasking,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            mask_id=sync.MASK_ID,
            attention_mask_mode=attention_mask_mode,
            return_topk=int(anchor_num) + 1 if int(anchor_num) > 0 else 0,
        )
        if end_event is not None:
            end_event.record(torch.cuda.current_stream(draft_device_t))
        _sync_device(draft_device_t)
        draft_sync_time += time.perf_counter() - draft_start
        redraft_gpu_time += _event_elapsed_s(start_event, end_event)
        sync_redraft_steps += 1
        return recovery, recovery_top_tokens, recovery_probs

    def draft_async_outcomes(current, current_top_tokens, prefix_len: int):
        nonlocal packed_candidate_tokens, packed_anchor_candidates_generated
        nonlocal packed_unanchored_candidates_generated
        remaining_after_current = max_length - (int(prefix_len) + int(current.shape[1]))
        branch_token_count = min(
            int(async_blocks) * int(block_size),
            remaining_after_current,
        )
        if branch_token_count <= 0:
            return {}, {}, {}, 0
        sync._crop_cache(draft_kv, int(prefix_len))
        candidates, candidate_top_tokens, candidate_probs = d3d_draft_all_outcomes_packed(
            drafter,
            pending_block=current,
            branch_token_count=branch_token_count,
            draft_kv=draft_kv,
            draft_kv_len=int(prefix_len),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            mask_id=sync.MASK_ID,
            attention_mask_mode=attention_mask_mode,
            pending_top_tokens=current_top_tokens,
            anchor_num=int(anchor_num),
            accepted_counts=packed_accepted_counts,
        )
        packed_candidate_tokens += sum(int(x.shape[1]) for x in candidates.values())
        async_prefix_candidate_lens.extend(int(x.shape[1]) for x in candidates.values())
        packed_anchor_candidates_generated += sum(1 for _, anchor in candidates if anchor is not None)
        packed_unanchored_candidates_generated += sum(1 for _, anchor in candidates if anchor is None)
        return candidates, candidate_top_tokens, candidate_probs, len(candidates)

    decoding_start = time.perf_counter()
    current_tokens, current_top_tokens, current_probs = draft_recovery(anchor_target, kv_len)
    current_source = "redraft"
    _sync_device(draft_device_t)
    while kv_len < max_length and current_tokens is not None:
        current_len = min(int(current_tokens.shape[1]), max_length - kv_len)
        current_tokens = current_tokens[:, :current_len]
        if current_probs is not None:
            current_probs = (
                current_probs[0][:, :current_len],
                current_probs[1][:, :current_len],
            )
        if current_len <= 0:
            break
        if temperature > 1e-5 and current_len > 1:
            if current_probs is None:
                raise RuntimeError("temperature > 0 draft block is missing proposal probabilities")
            if current_probs[0].shape[:2] != current_tokens.shape:
                raise RuntimeError(
                    "draft proposal shape mismatch: "
                    f"tokens={tuple(current_tokens.shape)}, ids={tuple(current_probs[0].shape)}, "
                    f"probs={tuple(current_probs[1].shape)}"
                )
        if current_source.startswith("packed_"):
            async_ready_hits += 1

        overlap_start = time.perf_counter()
        verify_submit_wall_time += verifier_client.submit_verify(
            current_tokens,
            (
                current_probs[0][:, 1:current_len],
                current_probs[1][:, 1:current_len],
            ) if current_probs is not None else None,
            kv_len,
            temperature,
            target_top_k,
            target_top_p,
        )

        _sync_device(draft_device_t)
        async_start = time.perf_counter()
        async_start_event, async_end_event = _event_pair(draft_device_t)
        if async_start_event is not None:
            async_start_event.record(torch.cuda.current_stream(draft_device_t))
        (
            outcome_candidates,
            outcome_top_tokens,
            outcome_probs,
            async_count,
        ) = draft_async_outcomes(current_tokens, current_top_tokens, kv_len)
        if async_end_event is not None:
            async_end_event.record(torch.cuda.current_stream(draft_device_t))
        _sync_device(draft_device_t)
        async_draft_time += time.perf_counter() - async_start
        async_draft_gpu_time += _event_elapsed_s(async_start_event, async_end_event)
        async_blocks_per_step.append(async_count)

        before_wait = time.perf_counter()
        (
            posterior_values,
            accepted_tail,
            correction_values,
            verify_metrics,
        ) = verifier_client.receive_verify()
        posterior = torch.tensor(
            posterior_values,
            dtype=current_tokens.dtype,
            device=draft_device_t,
        )
        wait_for_verify_time += time.perf_counter() - before_wait
        verify_time += verify_metrics["gpu_s"]
        state_rollback_time += verify_metrics.get("state_rollback_s", 0.0)
        state_rollback_gpu_time += verify_metrics.get("state_rollback_gpu_s", 0.0)
        verifier_worker_wall_time += verify_metrics["worker_wall_s"]
        verifier_dispatch_delay_time += verify_metrics["dispatch_delay_s"]
        verifier_response_pipe_time += verify_metrics["response_pipe_s"]
        verifier_round_trip_time += verify_metrics["round_trip_s"]
        overlap_wall_time += time.perf_counter() - overlap_start

        commit_len = int(accepted_tail) + 1
        all_pass = commit_len == current_len
        commit_tokens = current_tokens[:, :commit_len]
        output_ids[:, kv_len : kv_len + commit_len] = commit_tokens
        new_kv_len = kv_len + commit_len
        next_anchor_target = torch.tensor(
            correction_values,
            dtype=current_tokens.dtype,
            device=draft_device_t,
        )

        next_anchor_token = int(next_anchor_target.reshape(-1)[0].item())
        if int(anchor_num) == 0 or all_pass:
            candidate_key = (commit_len, None)
        else:
            candidate_key = (commit_len, next_anchor_token)
            if any(key[0] == commit_len for key in outcome_candidates):
                packed_anchor_lookup_attempts += 1
        selected_outcome = outcome_candidates.get(candidate_key)
        selected_top_tokens = outcome_top_tokens.get(candidate_key)
        selected_probs = outcome_probs.get(candidate_key)
        if selected_outcome is not None and temperature > 1e-5 and selected_probs is None:
            raise RuntimeError(f"packed outcome {candidate_key} is missing proposal probabilities")
        if all_pass:
            all_pass_steps += 1
        else:
            rollback_steps += 1
            sync._crop_cache(draft_kv, new_kv_len)

        next_group = None
        next_group_top_tokens = None
        next_group_probs = None
        next_group_source = None
        if selected_outcome is not None:
            if candidate_key[1] is not None:
                packed_anchor_matches += 1
            else:
                packed_unanchored_hits += 1
            selected_outcome[:, 0] = next_anchor_target.reshape(1)
            next_group = selected_outcome
            next_group_top_tokens = selected_top_tokens
            next_group_probs = selected_probs
            next_group_source = "packed_full" if all_pass else f"packed_outcome_{commit_len}"
            packed_outcome_hits += 1
            if all_pass:
                packed_full_hits += 1
            else:
                packed_partial_hits += 1
            async_useful_blocks += 1
            async_discarded_blocks += max(async_count - 1, 0)
        else:
            async_discarded_blocks += async_count

        acceptance_lengths.append(commit_len)
        draft_token_blocks.append(current_tokens[0, 1:].detach())
        target_token_blocks.append(posterior[0].detach())
        emitted_token_blocks.append(commit_tokens[0].detach())
        verify_block_sizes.append(current_len)
        verify_sources.append(current_source)
        kv_len = new_kv_len
        anchor_target = next_anchor_target.to(draft_device_t)
        if kv_len < max_length:
            output_ids[:, kv_len] = anchor_target

        generated = output_ids[:, num_input_tokens : min(kv_len + 1, max_length)]
        stopped = sync._contains_stop(generated, stop_token_ids)
        stopped_blocks.append(bool(stopped))
        if stopped or kv_len >= max_length:
            break

        if next_group is not None:
            current_tokens = next_group
            current_top_tokens = next_group_top_tokens
            current_probs = next_group_probs
            current_source = str(next_group_source or "packed_outcome")
        else:
            current_tokens, current_top_tokens, current_probs = draft_recovery(anchor_target, kv_len)
            current_source = "redraft"

    output_ids = output_ids[:, : min(kv_len + 1, max_length)]
    output_ids = sync._trim_at_stop(output_ids, num_input_tokens, stop_token_ids)
    _sync_device(draft_device_t)
    decoding_time = time.perf_counter() - decoding_start

    if not return_stats:
        return output_ids

    num_output_tokens = int(output_ids.shape[1] - num_input_tokens)
    decode_tokens = max(num_output_tokens - 1, 0)
    num_steps = len(acceptance_lengths)
    draft_time = draft_sync_time + async_draft_time
    verify_source_stats = _verify_source_stats(verify_sources, verify_block_sizes, acceptance_lengths)
    redraft_span_accept_lengths = _redraft_span_accept_lengths(verify_sources, acceptance_lengths)
    high_speed_stats = _high_speed_continuous_stats(
        verify_block_sizes,
        acceptance_lengths,
    )
    async_stats = {
        "mode": "process_two_gpu_all_positions_packed",
        "qwen35_rollback_mode": os.environ.get(
            "QWEN35_ROLLBACK_MODE", "replay"
        ),
        "verifier_device": str(verifier_device),
        "draft_device": str(draft_device_t),
        "async_blocks": int(async_blocks),
        "packed_all_positions": True,
        "packed_branch_token_count": int(async_blocks) * int(block_size),
        "packed_outcome_hits": int(packed_outcome_hits),
        "packed_partial_hits": int(packed_partial_hits),
        "packed_full_hits": int(packed_full_hits),
        "packed_candidate_tokens": int(packed_candidate_tokens),
        "anchor_num": int(anchor_num),
        "packed_accepted_counts": (
            "all" if packed_accepted_counts is None else [int(x) for x in packed_accepted_counts]
        ),
        "packed_anchor_candidates_generated": int(packed_anchor_candidates_generated),
        "packed_anchor_lookup_attempts": int(packed_anchor_lookup_attempts),
        "packed_anchor_matches": int(packed_anchor_matches),
        "packed_anchor_match_rate": (
            packed_anchor_matches / packed_anchor_lookup_attempts
            if packed_anchor_lookup_attempts
            else 0.0
        ),
        "packed_unanchored_candidates_generated": int(packed_unanchored_candidates_generated),
        "packed_unanchored_hits": int(packed_unanchored_hits),
        "sync_redraft_steps": int(sync_redraft_steps),
        "async_ready_hits": int(async_ready_hits),
        "all_pass_steps": int(all_pass_steps),
        "rollback_steps": int(rollback_steps),
        "async_useful_blocks": int(async_useful_blocks),
        "async_discarded_blocks": int(async_discarded_blocks),
        "async_blocks_generated": int(sum(async_blocks_per_step)),
        "async_prefix_candidate_lens": async_prefix_candidate_lens,
        "candidate_len_histogram": _histogram(async_prefix_candidate_lens),
        "verify_sources": verify_sources,
        "verify_source_stats": verify_source_stats,
        "redraft_span_accept_lengths": redraft_span_accept_lengths,
        "redraft_span_accept_len_histogram": _histogram(redraft_span_accept_lengths),
        **high_speed_stats,
        "packed_outcome_hit_rate": packed_outcome_hits / max(num_steps, 1),
        "packed_partial_hit_rate": packed_partial_hits / max(rollback_steps, 1),
        "mean_async_blocks_per_verify": (
            statistics.fmean(async_blocks_per_step) if async_blocks_per_step else 0.0
        ),
        "max_async_blocks_per_verify": max(async_blocks_per_step) if async_blocks_per_step else 0,
        "async_blocks_per_verify": async_blocks_per_step,
        "draft_sync_time": float(draft_sync_time),
        "redraft_gpu_time": float(redraft_gpu_time),
        "async_draft_time": float(async_draft_time),
        "async_draft_gpu_time": float(async_draft_gpu_time),
        "async_draft_cpu_overhead": float(max(async_draft_time - async_draft_gpu_time, 0.0)),
        "verifier_gpu_time": float(verify_time),
        "state_rollback_time": float(state_rollback_time),
        "state_rollback_gpu_time": float(state_rollback_gpu_time),
        "verifier_worker_wall_time": float(verifier_worker_wall_time),
        "verifier_dispatch_delay_time": float(verifier_dispatch_delay_time),
        "verifier_response_pipe_time": float(verifier_response_pipe_time),
        "verifier_round_trip_time": float(verifier_round_trip_time),
        "verify_submit_wall_time": float(verify_submit_wall_time),
        "overlap_wall_time": float(overlap_wall_time),
        "wait_for_verify_time": float(wait_for_verify_time),
        "mean_redraft_wall_time_per_redraft": draft_sync_time / max(sync_redraft_steps, 1),
        "mean_redraft_gpu_time_per_redraft": redraft_gpu_time / max(sync_redraft_steps, 1),
        "mean_verifier_gpu_time_per_step": verify_time / max(num_steps, 1),
        "mean_verifier_worker_wall_time_per_step": verifier_worker_wall_time / max(num_steps, 1),
        "mean_verifier_dispatch_delay_per_step": verifier_dispatch_delay_time / max(num_steps, 1),
        "mean_verifier_response_pipe_time_per_step": verifier_response_pipe_time / max(num_steps, 1),
        "mean_verifier_round_trip_time_per_step": verifier_round_trip_time / max(num_steps, 1),
        "mean_verify_submit_wall_time_per_step": verify_submit_wall_time / max(num_steps, 1),
        "mean_overlap_wall_time_per_step": overlap_wall_time / max(num_steps, 1),
        "mean_async_draft_time_per_step": async_draft_time / max(num_steps, 1),
        "mean_async_draft_gpu_time_per_step": async_draft_gpu_time / max(num_steps, 1),
        "mean_async_draft_cpu_overhead_per_step": (
            max(async_draft_time - async_draft_gpu_time, 0.0) / max(num_steps, 1)
        ),
    }
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        decode_tokens=decode_tokens,
        time_per_output_token=decoding_time / max(decode_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        draft_token_blocks=draft_token_blocks,
        target_token_blocks=target_token_blocks,
        emitted_token_blocks=emitted_token_blocks,
        stopped_blocks=stopped_blocks,
        block_size=block_size,
        draft_time=draft_time,
        draft_transient_time=async_draft_time,
        draft_commit_time=0.0,
        draft_prefill_time=draft_prefill_time,
        verify_time=verify_time,
        verifier_prefill_time=verifier_prefill_time,
        num_verify_steps=num_steps,
        mean_draft_time_per_step=draft_time / max(num_steps, 1),
        mean_verify_time_per_step=verify_time / max(num_steps, 1),
        decode_elapsed_s=decoding_time,
        verify_block_sizes=verify_block_sizes,
        async_stats=async_stats,
    )


def _summarize_async(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [p.get("timing", {}).get("async", {}) for p in per_prompt]
    stats = [s for s in stats if s]
    if not stats:
        return {}
    total_steps = sum(int(s.get("sync_redraft_steps", 0)) + int(s.get("async_ready_hits", 0)) for s in stats)
    total_generated = sum(int(s.get("async_blocks_generated", 0)) for s in stats)
    total_useful = sum(int(s.get("async_useful_blocks", 0)) for s in stats)
    total_discarded = sum(int(s.get("async_discarded_blocks", 0)) for s in stats)
    candidate_lens = [int(x) for s in stats for x in s.get("async_prefix_candidate_lens", [])]
    redraft_span_accept_lens = [
        int(x) for s in stats for x in s.get("redraft_span_accept_lengths", [])
    ]
    high_speed_blocks = sum(int(s.get("high_speed_continuous_blocks", 0)) for s in stats)
    high_speed_total_blocks = sum(int(s.get("high_speed_total_blocks", 0)) for s in stats)
    total_redraft_steps = sum(int(s.get("sync_redraft_steps", 0)) for s in stats)
    redraft_wall_time = sum(float(s.get("draft_sync_time", 0.0)) for s in stats)
    redraft_gpu_time = sum(float(s.get("redraft_gpu_time", 0.0)) for s in stats)
    verifier_gpu_time = sum(float(s.get("verifier_gpu_time", 0.0)) for s in stats)
    state_rollback_time = sum(
        float(s.get("state_rollback_time", 0.0)) for s in stats
    )
    state_rollback_gpu_time = sum(
        float(s.get("state_rollback_gpu_time", 0.0)) for s in stats
    )
    verifier_worker_wall_time = sum(float(s.get("verifier_worker_wall_time", 0.0)) for s in stats)
    verifier_dispatch_delay_time = sum(float(s.get("verifier_dispatch_delay_time", 0.0)) for s in stats)
    verifier_response_pipe_time = sum(float(s.get("verifier_response_pipe_time", 0.0)) for s in stats)
    verifier_round_trip_time = sum(float(s.get("verifier_round_trip_time", 0.0)) for s in stats)
    verify_submit_wall_time = sum(float(s.get("verify_submit_wall_time", 0.0)) for s in stats)
    overlap_wall_time = sum(float(s.get("overlap_wall_time", 0.0)) for s in stats)
    async_draft_time = sum(float(s.get("async_draft_time", 0.0)) for s in stats)
    async_draft_gpu_time = sum(float(s.get("async_draft_gpu_time", 0.0)) for s in stats)
    source_accum: dict[str, dict[str, Any]] = {}
    for stat in stats:
        for source, row in stat.get("verify_source_stats", {}).items():
            count = float(row.get("count", 0))
            if count <= 0:
                continue
            bucket = source_accum.setdefault(
                source,
                {
                    "count": 0.0,
                    "all_pass_count": 0.0,
                    "block_size_sum": 0.0,
                    "accept_len_sum": 0.0,
                    "reject_count": 0.0,
                    "accept_len_histogram": {},
                },
            )
            bucket["count"] += count
            bucket["all_pass_count"] += float(row.get("all_pass_count", 0))
            bucket["block_size_sum"] += float(row.get("mean_block_size", 0.0)) * count
            bucket["accept_len_sum"] += float(row.get("mean_accept_len", 0.0)) * count
            bucket["reject_count"] += float(row.get("reject_at_0_rate", 0.0)) * count
            histogram = bucket["accept_len_histogram"]
            for accept_len, histogram_count in row.get("accept_len_histogram", {}).items():
                histogram[str(accept_len)] = histogram.get(str(accept_len), 0) + int(histogram_count)
    verify_source_stats = {
        source: {
            "count": int(bucket["count"]),
            "all_pass_count": int(bucket["all_pass_count"]),
            "all_pass_rate": bucket["all_pass_count"] / bucket["count"] if bucket["count"] else 0.0,
            "mean_block_size": bucket["block_size_sum"] / bucket["count"] if bucket["count"] else 0.0,
            "mean_accept_len": bucket["accept_len_sum"] / bucket["count"] if bucket["count"] else 0.0,
            "accept_len_histogram": {
                key: int(value)
                for key, value in sorted(
                    bucket["accept_len_histogram"].items(),
                    key=lambda item: int(item[0]),
                )
            },
            "reject_at_0_rate": bucket["reject_count"] / bucket["count"] if bucket["count"] else 0.0,
        }
        for source, bucket in sorted(source_accum.items())
    }
    return {
        "mode": stats[0].get("mode"),
        "qwen35_rollback_mode": stats[0].get(
            "qwen35_rollback_mode", "replay"
        ),
        "verifier_device": stats[0].get("verifier_device"),
        "draft_device": stats[0].get("draft_device"),
        "async_blocks": int(stats[0].get("async_blocks", 1)),
        "packed_all_positions": True,
        "packed_branch_token_count": int(stats[0].get("packed_branch_token_count", 0)),
        "anchor_num": int(stats[0].get("anchor_num", 0)),
        "packed_accepted_counts": stats[0].get("packed_accepted_counts", "all"),
        "packed_anchor_candidates_generated": sum(
            int(s.get("packed_anchor_candidates_generated", 0)) for s in stats
        ),
        "packed_anchor_lookup_attempts": sum(
            int(s.get("packed_anchor_lookup_attempts", 0)) for s in stats
        ),
        "packed_anchor_matches": sum(
            int(s.get("packed_anchor_matches", 0)) for s in stats
        ),
        "packed_anchor_match_rate": (
            sum(int(s.get("packed_anchor_matches", 0)) for s in stats)
            / max(sum(int(s.get("packed_anchor_lookup_attempts", 0)) for s in stats), 1)
        ),
        "packed_unanchored_candidates_generated": sum(
            int(s.get("packed_unanchored_candidates_generated", 0)) for s in stats
        ),
        "packed_unanchored_hits": sum(
            int(s.get("packed_unanchored_hits", 0)) for s in stats
        ),
        "packed_outcome_hits": sum(int(s.get("packed_outcome_hits", 0)) for s in stats),
        "packed_partial_hits": sum(int(s.get("packed_partial_hits", 0)) for s in stats),
        "packed_full_hits": sum(int(s.get("packed_full_hits", 0)) for s in stats),
        "packed_candidate_tokens": sum(int(s.get("packed_candidate_tokens", 0)) for s in stats),
        "total_steps": int(total_steps),
        "sync_redraft_steps": int(total_redraft_steps),
        "async_ready_hits": sum(int(s.get("async_ready_hits", 0)) for s in stats),
        "all_pass_steps": sum(int(s.get("all_pass_steps", 0)) for s in stats),
        "rollback_steps": sum(int(s.get("rollback_steps", 0)) for s in stats),
        "async_blocks_generated": int(total_generated),
        "async_useful_blocks": int(total_useful),
        "async_discarded_blocks": int(total_discarded),
        "useful_async_block_rate": total_useful / total_generated if total_generated else 0.0,
        "mean_async_candidate_len": statistics.fmean(candidate_lens) if candidate_lens else 0.0,
        "candidate_len_histogram": _histogram(candidate_lens),
        "verify_source_stats": verify_source_stats,
        "redraft_span_accept_len_histogram": _histogram(redraft_span_accept_lens),
        "high_speed_min_consecutive_all_pass_blocks": int(
            stats[0].get("high_speed_min_consecutive_all_pass_blocks", 3)
        ),
        "high_speed_continuous_blocks": int(high_speed_blocks),
        "high_speed_total_blocks": int(high_speed_total_blocks),
        "high_speed_continuous_block_rate": (
            high_speed_blocks / high_speed_total_blocks if high_speed_total_blocks else 0.0
        ),
        "high_speed_continuous_regions": sum(
            int(s.get("high_speed_continuous_regions", 0)) for s in stats
        ),
        "packed_outcome_hit_rate": (
            sum(int(s.get("packed_outcome_hits", 0)) for s in stats) / max(total_steps, 1)
        ),
        "packed_partial_hit_rate": (
            sum(int(s.get("packed_partial_hits", 0)) for s in stats)
            / max(sum(int(s.get("rollback_steps", 0)) for s in stats), 1)
        ),
        "mean_async_blocks_per_verify": statistics.fmean(
            float(s.get("mean_async_blocks_per_verify", 0.0)) for s in stats
        ),
        "max_async_blocks_per_verify": max(int(s.get("max_async_blocks_per_verify", 0)) for s in stats),
        "mean_redraft_wall_time_per_redraft": redraft_wall_time / max(total_redraft_steps, 1),
        "mean_redraft_gpu_time_per_redraft": redraft_gpu_time / max(total_redraft_steps, 1),
        "mean_verifier_gpu_time_per_step": verifier_gpu_time / max(total_steps, 1),
        "state_rollback_time": state_rollback_time,
        "state_rollback_gpu_time": state_rollback_gpu_time,
        "mean_state_rollback_gpu_time_per_step": (
            state_rollback_gpu_time / max(total_steps, 1)
        ),
        "mean_verifier_worker_wall_time_per_step": verifier_worker_wall_time / max(total_steps, 1),
        "mean_verifier_dispatch_delay_per_step": verifier_dispatch_delay_time / max(total_steps, 1),
        "mean_verifier_response_pipe_time_per_step": verifier_response_pipe_time / max(total_steps, 1),
        "mean_verifier_round_trip_time_per_step": verifier_round_trip_time / max(total_steps, 1),
        "mean_verify_submit_wall_time_per_step": verify_submit_wall_time / max(total_steps, 1),
        "mean_overlap_wall_time_per_step": overlap_wall_time / max(total_steps, 1),
        "mean_async_draft_time_per_step": async_draft_time / max(total_steps, 1),
        "mean_async_draft_gpu_time_per_step": async_draft_gpu_time / max(total_steps, 1),
        "mean_async_draft_cpu_overhead_per_step": (
            max(async_draft_time - async_draft_gpu_time, 0.0) / max(total_steps, 1)
        ),
        "wait_for_verify_time": sum(float(s.get("wait_for_verify_time", 0.0)) for s in stats),
    }


def _build_async_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type):
    payload = sync._build_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type)
    payload["summary"]["async"] = _summarize_async(per_prompt)
    return payload


def _write_async_summary(args, payload: dict[str, Any]) -> dict[str, Any]:
    compact = sync._summary_from_payload(args, payload, args.summary_json)
    compact["async"] = payload.get("summary", {}).get("async", {})
    if args.summary_json:
        sync._write_json(args.summary_json, compact)
    return compact


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset not in sync.DATASETS:
        raise ValueError(f"unsupported dataset {args.dataset}; supported: {sorted(sync.DATASETS)}")
    args.verifier_device = args.verifier_device or "cuda:0"
    args.draft_device = args.draft_device or "cuda:1"
    sync.check_hardware(args.verifier_device, args.draft_device)
    torch.manual_seed(args.seed)
    target_path = sync.resolve_model_path(args.verifier)
    draft_path = sync.resolve_model_path(args.drafter)
    tokenizer_path = sync.resolve_model_path(args.tokenizer) if args.tokenizer else target_path
    args.runtime = sync.runtime_metadata({'target': target_path, 'draft': draft_path,
                                          'tokenizer': tokenizer_path})
    args.use_kvcache = True
    args.drafter_tag = sync._safe_custom_tag(args.drafter_tag, sync._safe_tag(args.drafter))
    args.verifier_tag = sync._safe_tag(args.verifier)
    args.verifier, args.drafter, args.tokenizer = target_path, draft_path, tokenizer_path

    model_type = args.model_type or sync._checkpoint_model_type(draft_path)
    if model_type != "d3d":
        raise ValueError(f"eval_async.py currently supports only D3D drafter checkpoints, got {model_type!r}")
    if int(args.denoising_steps) != 1:
        raise ValueError("eval_async_packed.py currently requires --denoising-steps 1")
    if int(args.async_blocks) < 1:
        raise ValueError("--async-blocks must be >= 1")

    dtype = sync._resolve_dtype(args.dtype)
    target_attn = os.environ.get("TARGET_ATTN_IMPLEMENTATION", args.target_attn_implementation)
    draft_attn = os.environ.get("DRAFT_ATTN_IMPLEMENTATION", args.draft_attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    eos_token_ids = sync._eos_token_ids(tokenizer, args.eos_token_id) or sync.STOP_TOKEN_IDS

    verifier_cpu_affinity = _parse_cpu_affinity(args.verifier_cpu_affinity)
    draft_cpu_affinity = _parse_cpu_affinity(args.draft_cpu_affinity)
    verifier_visible_device, verifier_worker_device = _isolated_cuda_device(
        args.verifier_device
    )
    verifier_client = VerifierProcessClient(
        target_path,
        verifier_worker_device,
        args.dtype,
        target_attn,
        verifier_cpu_affinity,
        verifier_visible_device,
    )
    if draft_cpu_affinity:
        os.sched_setaffinity(0, set(draft_cpu_affinity))
    draft = sync._load_model(
        draft_path,
        dtype,
        attn_implementation=draft_attn,
    ).to(args.draft_device).eval()
    sync._disable_default_cache(draft)
    verifier_client.wait_ready()

    rows, cfg, dataset_source = sync.load_rows(args.dataset, args.max_samples)
    print(f"Loaded {len(rows)} {args.dataset} samples from {dataset_source}", flush=True)

    args.dataset_metadata = {key: cfg.get(key) for key in ('load', 'filename', 'source', 'revision', 'fingerprint', 'samples', 'sample_sha256')}
    prepared = []
    for row in rows:
        question = cfg['question'](row)
        prompt = sync._build_prompt(question, args.enable_thinking)
        text = sync._apply_chat_template(tokenizer, prompt['messages'], args.enable_thinking)
        prepared.append(tokenizer.encode(text, return_tensors='pt'))
    if not prepared:
        verifier_client.close()
        raise ValueError('Dataset contains no prompts.')
    generation_options = dict(
        block_size=args.block_size, denoising_steps=args.denoising_steps,
        threshold=args.confidence_threshold, remasking=args.remasking_strategy,
        top_k=args.proposal_top_k, top_p=args.proposal_top_p,
        target_top_k=args.target_top_k, target_top_p=args.target_top_p, return_stats=True,
        verifier_device=args.verifier_device, draft_device=args.draft_device,
        async_blocks=args.async_blocks, attention_mask_mode=args.attention_mask_mode,
        anchor_num=args.anchor_num,
        packed_accepted_counts=args.packed_accepted_counts,
    )
    if args.warmup_new_tokens:
        print('WARMUP (excluded from timing)', flush=True)
        run_d3d_generate_async(draft, verifier_client, prepared[0], args.warmup_new_tokens,
                                eos_token_ids, args.temperature, **generation_options)
    start_time = sync._synced_time()
    per_prompt: list[dict[str, Any]] = []
    spec_ok = spec_eval = 0
    draft_tokens = max(
        sync._effective_draft_tokens(args.draft_tokens, args.block_size),
        int(args.async_blocks) * int(args.block_size),
    )

    for i, row in enumerate(rows):
        question = cfg["question"](row)
        gold = sync._gold_for_row(cfg, row)
        ids = prepared[i]
        st = run_d3d_generate_async(
            draft,
            verifier_client,
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
            verifier_device=args.verifier_device,
            draft_device=args.draft_device,
            async_blocks=args.async_blocks,
            attention_mask_mode=args.attention_mask_mode,
            anchor_num=args.anchor_num,
            packed_accepted_counts=args.packed_accepted_counts,
        )
        out_text = tokenizer.decode(st.output_ids[0, st.num_input_tokens:], skip_special_tokens=False)
        step_block_sizes = [int(x) for x in getattr(st, "verify_block_sizes", [])]
        if len(step_block_sizes) != len(st.acceptance_lengths):
            step_block_sizes = [int(args.block_size)] * len(st.acceptance_lengths)
        steps = _d3d_steps_dynamic(
            [int(x) for x in st.acceptance_lengths],
            step_block_sizes,
            st.async_stats.get("verify_sources"),
            getattr(st, "draft_token_blocks", None),
            getattr(st, "target_token_blocks", None),
            getattr(st, "emitted_token_blocks", None),
            getattr(st, "stopped_blocks", None),
        )
        result = {
            "input_tokens": int(st.num_input_tokens),
            "output_tokens": int(st.num_output_tokens),
            "decode_tokens": int(st.decode_tokens),
            "text": out_text,
            "summary": sync.summarize_acceptance(
                [sync.AcceptanceStep(**step) for step in steps],
                draft_tokens,
            ),
            "steps": steps,
            "target_first_token": None,
            "decode_elapsed_s": float(st.decode_elapsed_s),
            "tokens_per_s": (
                int(st.decode_tokens) / float(st.decode_elapsed_s)
                if float(st.decode_elapsed_s) > 0
                else 0.0
            ),
            "time_per_output_token": float(st.time_per_output_token),
            "timing": {
                "draft_time": float(st.draft_time),
                "draft_prefill_time": float(getattr(st, "draft_prefill_time", 0.0)),
                "draft_transient_time": float(getattr(st, "draft_transient_time", st.draft_time)),
                "draft_commit_time": float(getattr(st, "draft_commit_time", 0.0)),
                "verify_time": float(st.verify_time),
                "verifier_prefill_time": float(st.verifier_prefill_time),
                "mean_draft_time_per_step": float(st.mean_draft_time_per_step),
                "mean_verify_time_per_step": float(st.mean_verify_time_per_step),
                "time_per_output_token": float(st.time_per_output_token),
                "async": st.async_stats,
            },
        }

        pred, hit = sync._score_text(cfg, result["text"], gold)
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

        payload = _build_async_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type)
        payload["accuracy"] = {
            "spec_evaluated": spec_eval,
            "spec_accuracy": spec_ok / spec_eval if spec_eval else None,
        }
        sync._write_json(args.output_json, payload)
        if args.verbose:
            print(
                json.dumps(
                    {
                        "index": i,
                        "output_tokens": result["output_tokens"],
                        "decode_elapsed_s": result["decode_elapsed_s"],
                        **result["summary"],
                        "async": result["timing"]["async"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = _build_async_payload(args, per_prompt, draft_tokens, eos_token_ids, start_time, dataset_source, model_type)
    payload["accuracy"] = {
        "spec_evaluated": spec_eval,
        "spec_accuracy": spec_ok / spec_eval if spec_eval else None,
    }
    sync._write_json(args.output_json, payload)
    compact = _write_async_summary(args, payload)
    verifier_client.close()
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = sync.build_arg_parser()
    parser.description = "Two-GPU all-acceptance-position packed D3D runner."
    parser.set_defaults(verifier_device="cuda:0", draft_device="cuda:1")
    parser.add_argument('--warmup-new-tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--verifier-cpu-affinity", default=None)
    parser.add_argument("--draft-cpu-affinity", default=None)
    parser.add_argument(
        "--async-blocks",
        type=int,
        default=1,
        help="Number of block-size chunks generated for every packed acceptance outcome.",
    )
    parser.add_argument(
        "--anchor-num",
        type=int,
        default=0,
        help=(
            "Number of non-primary anchor candidates at each rejection position; 0 "
            "generates unanchored outcomes and replaces the first token with the "
            "verifier correction token on reuse."
        ),
    )
    parser.add_argument(
        "--packed-accepted-counts",
        type=_parse_packed_accepted_counts,
        default=None,
        help="Accepted counts to pack, as 'all' or a comma-separated positive integer list.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
