"""Smoke-test DFlash on Target features plus an unverified SDAR block.

This intentionally tests an out-of-training-distribution input: the first SDAR
proposal is included as DFlash noise/context tokens without running Target
verification on it. It is not an acceptance-rate or serving benchmark.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import ModuleType
from typing import TypedDict
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from sglang.srt.speculative.decoupled.local_drafter import (
    plan_required_draft_layout,
    proposal_from_logits,
)


def build_prespec_tokens(
    *, first_tokens: torch.Tensor, mask_token: int, block_size: int
) -> torch.Tensor:
    if first_tokens.ndim != 2 or first_tokens.shape[1] != block_size:
        raise ValueError("first_tokens must have shape [batch, block_size]")
    second_masks = torch.full_like(first_tokens, mask_token)
    return torch.cat((first_tokens, second_masks), dim=1)


def load_first_draft_for_smoke(model_path: str, device: torch.device):
    """Load the old SDAR HF export with process-local compatibility shims.

    Production DECOUPLED uses SGLang's native SDAR loader. This shim only lets
    the offline experiment run its equivalent required-draft attention mask.
    """
    import transformers.utils as transformers_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    flash_ops = ModuleType("flash_attn.ops")
    flash_ops.__path__ = []
    flash_triton = ModuleType("flash_attn.ops.triton")
    flash_triton.__path__ = []
    flash_norm = ModuleType("flash_attn.ops.triton.layer_norm")

    def rms_norm_fn(x, weight, bias=None, eps=1e-6):
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + eps
        )
        result = normalized.to(x.dtype) * weight
        return result if bias is None else result + bias

    flash_norm.rms_norm_fn = rms_norm_fn
    modules = {
        "flash_attn.ops": flash_ops,
        "flash_attn.ops.triton": flash_triton,
        "flash_attn.ops.triton.layer_norm": flash_norm,
    }
    loss_kwargs = TypedDict("LossKwargs", {}, total=False)

    def default_rope_init(config, device=None):
        head_dim = int(config.head_dim)
        inv_freq = 1.0 / (
            float(config.rope_theta)
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        return inv_freq, 1.0

    with (
        patch.object(transformers_utils, "LossKwargs", loss_kwargs, create=True),
        patch.dict("sys.modules", modules),
        patch.dict(ROPE_INIT_FUNCTIONS, {"default": default_rope_init}),
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # The old export expects this PretrainedConfig default, removed by
        # Transformers 5. The pad row is not used by this smoke-test prompt.
        config.pad_token_id = config.eos_token_id
        model, load_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    missing_keys = load_info.get("missing_keys", ())
    print(f"first Draft missing weight keys: {len(missing_keys)}", flush=True)
    if missing_keys:
        raise RuntimeError(f"first Draft weight load is incomplete: {missing_keys[:5]}")
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="/workspace/models/Qwen3-8B")
    parser.add_argument(
        "--first-draft-model",
        default="/workspace/models/d3d-1.7b-b16_for_qwen3-8b",
    )
    parser.add_argument("--dflash-model", required=True)
    parser.add_argument(
        "--dataset-jsonl", type=Path, default=Path("docs/scripts/test.jsonl")
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--dflash-device", default="cuda:3")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--first-tokens",
        type=int,
        nargs="+",
        help="Use a previously generated first block for isolated DFlash timing",
    )
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.repetitions < 1 or args.sample_index < 0:
        parser.error("invalid prompt-tokens, repetitions, or sample-index")

    target_config = json.loads((Path(args.target_model) / "config.json").read_text())
    dflash_config = json.loads((Path(args.dflash_model) / "config.json").read_text())
    block_size = int(dflash_config["block_size"])
    mask_token = int(dflash_config["dflash_config"]["mask_token_id"])
    if args.first_tokens is not None and len(args.first_tokens) != block_size:
        parser.error(f"--first-tokens requires exactly {block_size} token IDs")
    target_layer_ids = tuple(dflash_config["dflash_config"]["target_layer_ids"])
    if max(target_layer_ids) >= int(target_config["num_hidden_layers"]):
        raise ValueError("DFlash target_layer_ids exceed Target layer count")
    if int(dflash_config["hidden_size"]) != int(target_config["hidden_size"]):
        raise ValueError("DFlash and Target hidden widths differ")

    with args.dataset_jsonl.open() as dataset:
        for index, line in enumerate(dataset):
            if index == args.sample_index:
                prompt = json.loads(line)["problem"]
                break
        else:
            raise ValueError(f"Math-500 sample index {args.sample_index} is missing")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[
        :, : args.prompt_tokens
    ]
    prompt_len = int(prompt_ids.shape[1])
    target_device = torch.device(args.target_device)
    dflash_device = torch.device(args.dflash_device)
    if target_device == dflash_device:
        raise ValueError(
            "use separate Target and DFlash GPUs for this transfer smoke test"
        )

    with torch.inference_mode():
        torch.manual_seed(42)
        print("loading Target", flush=True)
        first_draft = None
        if args.first_tokens is None:
            print("loading SDAR first Draft", flush=True)
            first_draft = load_first_draft_for_smoke(
                args.first_draft_model, target_device
            )
        else:
            print("using supplied first Draft tokens", flush=True)
        target = (
            AutoModelForCausalLM.from_pretrained(
                args.target_model, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to(target_device)
            .eval()
        )
        print("loading remote DFlash", flush=True)
        dflash, dflash_load_info = AutoModel.from_pretrained(
            args.dflash_model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        dflash_missing_keys = dflash_load_info.get("missing_keys", ())
        print(
            f"remote DFlash missing weight keys: {len(dflash_missing_keys)}", flush=True
        )
        if dflash_missing_keys:
            raise RuntimeError(
                f"remote DFlash weight load is incomplete: {dflash_missing_keys[:5]}"
            )
        dflash = dflash.to(dflash_device).eval()

        prompt_ids = prompt_ids.to(target_device)
        target_output = target(
            prompt_ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
        )
        target_hidden = torch.cat(
            [
                target_output.hidden_states[layer_id + 1]
                for layer_id in target_layer_ids
            ],
            dim=-1,
        )
        if args.first_tokens is None:
            bonus_tokens = target_output.logits[:, -1:].argmax(dim=-1)
            layout = plan_required_draft_layout(
                committed_prefix=tuple(prompt_ids[0].tolist())
                + (int(bonus_tokens.item()),),
                block_size=block_size,
                mask_token=mask_token,
            )
            with sdpa_kernel(SDPBackend.MATH):
                first_output = first_draft(
                    input_ids=torch.tensor(
                        [layout.input_tokens], dtype=torch.long, device=target_device
                    ),
                    attention_mask=layout.attention_mask.to(target_device)[None, None],
                    position_ids=torch.tensor(
                        [layout.positions], dtype=torch.long, device=target_device
                    ),
                    use_cache=False,
                )
            block_logits = first_output.logits[
                0, layout.replay_len : layout.replay_len + block_size
            ]
            finite_logits = torch.isfinite(block_logits)
            print(
                "first Draft finite logits: "
                f"{int(finite_logits.sum())}/{block_logits.numel()}",
                flush=True,
            )
            if not bool(finite_logits.all()):
                raise RuntimeError("first Draft logits are non-finite in HF smoke path")
            first_proposal = proposal_from_logits(
                anchor=int(bonus_tokens.item()),
                full_logits=first_output.logits[0],
                replay_len=layout.replay_len,
                block_size=block_size,
                topk=1,
            )
            first_values = first_proposal.tokens
        else:
            first_values = args.first_tokens
        first_tokens = torch.tensor(
            [first_values], dtype=torch.long, device=target_device
        )
        if int(first_tokens.min()) < 0 or int(first_tokens.max()) >= int(
            target_config["vocab_size"]
        ):
            raise ValueError(
                "first Draft proposed a token outside the Target vocabulary"
            )

        # The remote DFlash worker needs the Target's feature taps, embeddings,
        # and vocabulary head. Copy the small feature buffer each call, but keep
        # the embedding and head resident on the remote GPU as serving would.
        remote_embed = target.get_input_embeddings().weight.detach().to(dflash_device)
        remote_head = target.get_output_embeddings().weight.detach().to(dflash_device)
        noise_tokens = build_prespec_tokens(
            first_tokens=first_tokens.to(dflash_device),
            mask_token=mask_token,
            block_size=block_size,
        )
        position_ids = torch.arange(
            prompt_len + 2 * block_size, device=dflash_device
        ).unsqueeze(0)

        def prespec_forward(
            *, input_tokens: torch.Tensor, context_hidden: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            remote_hidden = context_hidden.to(dflash_device)
            noise_embedding = F.embedding(input_tokens, remote_embed)
            draft_hidden = dflash(
                target_hidden=remote_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
            logits = F.linear(draft_hidden[:, -block_size:], remote_head)
            return logits.argmax(dim=-1), logits

        for _ in range(2):
            second_tokens, second_logits = prespec_forward(
                input_tokens=noise_tokens, context_hidden=target_hidden
            )
        torch.cuda.synchronize(dflash_device)
        elapsed_ms = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            second_tokens, second_logits = prespec_forward(
                input_tokens=noise_tokens, context_hidden=target_hidden
            )
            torch.cuda.synchronize(dflash_device)
            elapsed_ms.append(1000 * (time.perf_counter() - start))
        blank_first_tokens = noise_tokens.clone()
        blank_first_tokens[:, :block_size] = mask_token
        without_first_tokens, without_first_logits = prespec_forward(
            input_tokens=blank_first_tokens, context_hidden=target_hidden
        )
        without_target_tokens, without_target_logits = prespec_forward(
            input_tokens=noise_tokens, context_hidden=torch.zeros_like(target_hidden)
        )
        result = {
            "mode": "unverified_first_block_as_dflash_noise_context",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "sample_index": args.sample_index,
            "prompt_len": prompt_len,
            "block_size": block_size,
            "target_layer_ids": target_layer_ids,
            "target_hidden_shape": list(target_hidden.shape),
            "dflash_noise_shape": list(noise_tokens.shape),
            "first_tokens": first_tokens.tolist()[0],
            "first_tokens_source": (
                "offline_d3d" if args.first_tokens is None else "supplied"
            ),
            "second_tokens": second_tokens.tolist()[0],
            "num_tokens_changed_without_first_block": int(
                (second_tokens != without_first_tokens).sum()
            ),
            "num_tokens_changed_without_target_hidden": int(
                (second_tokens != without_target_tokens).sum()
            ),
            "max_logit_change_without_first_block": float(
                (second_logits - without_first_logits).float().abs().max()
            ),
            "max_logit_change_without_target_hidden": float(
                (second_logits - without_target_logits).float().abs().max()
            ),
            "second_tokens_valid": bool(
                torch.all(second_tokens >= 0)
                and torch.all(second_tokens < int(target_config["vocab_size"]))
            ),
            "elapsed_ms": elapsed_ms,
            "note": "Forward-only smoke test; first block is unverified and the 32-token DFlash noise input is out of training distribution.",
        }
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
