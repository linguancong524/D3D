"""One-step denoising dLLM algorithm (for speculative drafting).

Unlike LowConfidence / JointThreshold (multi-step denoising for final
quality, up to block_size+1 forwards per block), OneStep fills every masked
position in a block during the *first* forward, so each block costs exactly
one forward pass. The per-position argmax predictions are a low-cost draft:
the downstream target model verifies and corrects them.

Usage:
    --dllm-algorithm OneStep
    --dllm-algorithm-config <yaml with block_size: 16>
"""

from typing import Any

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class OneStep(DllmAlgorithm):
    """Each step fills all masked positions with their argmax prediction."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)

    def max_steps(self, block_size: int) -> int:
        return 1

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: list[Any],
    ) -> list[bool]:
        batch_size = forward_batch.batch_size
        input_ids = forward_batch.input_ids
        block_mask_index = input_ids == self.mask_id

        x = torch.argmax(full_logits, dim=-1)
        new_input_ids = torch.where(block_mask_index, x, input_ids)
        # In-place to preserve the input_ids tensor identity (CUDA graph safe).
        forward_batch.input_ids.copy_(new_input_ids)

        # After the single step every masked position is filled, so the block
        # is done regardless of the pre-step state. Request segments may carry
        # a variable number of tokens (the block is extended from the anchor).
        return [True] * batch_size


Algorithm = OneStep
