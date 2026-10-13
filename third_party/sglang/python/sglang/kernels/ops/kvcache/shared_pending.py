"""Gather shared-pending KV rows from the persistent pool or current forward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_shared_pending_kv_kernel(
    pool_k,
    pool_v,
    current_k,
    current_v,
    source_is_current,
    source_indices,
    output_k,
    output_v,
    pool_k_stride,
    pool_v_stride,
    current_k_stride,
    current_v_stride,
    output_k_stride,
    output_v_stride,
    row_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_elements
    source_index = tl.load(source_indices + row)
    use_current = tl.load(source_is_current + row)
    k_ptr = tl.where(
        use_current,
        current_k + source_index * current_k_stride + offsets,
        pool_k + source_index * pool_k_stride + offsets,
    )
    v_ptr = tl.where(
        use_current,
        current_v + source_index * current_v_stride + offsets,
        pool_v + source_index * pool_v_stride + offsets,
    )
    tl.store(
        output_k + row * output_k_stride + offsets,
        tl.load(k_ptr, mask=mask),
        mask=mask,
    )
    tl.store(
        output_v + row * output_v_stride + offsets,
        tl.load(v_ptr, mask=mask),
        mask=mask,
    )


def gather_shared_pending_kv(
    *,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    source_is_current: torch.Tensor,
    source_indices: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
) -> None:
    """Materialize one packed ragged KV buffer without Python-side concatenation."""
    num_rows = source_indices.numel()
    if num_rows == 0:
        return
    row_elements = current_k[0].numel()
    block = min(1024, triton.next_power_of_2(row_elements))
    grid = (num_rows, triton.cdiv(row_elements, block))
    _gather_shared_pending_kv_kernel[grid](
        pool_k,
        pool_v,
        current_k,
        current_v,
        source_is_current,
        source_indices,
        output_k,
        output_v,
        pool_k.stride(0),
        pool_v.stride(0),
        current_k.stride(0),
        current_v.stride(0),
        output_k.stride(0),
        output_v.stride(0),
        row_elements=row_elements,
        BLOCK=block,
    )
