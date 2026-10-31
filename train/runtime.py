from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from transformers.cache_utils import DynamicCache

@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def resolve_mask_token_id(
    tokenizer, mask_token: str = "<MASK>", *,
    mask_token_id: int | None = None, vocab_size: int | None = None,
) -> int:
    """Use an existing embedding row, without changing the Qwen tokenizer."""
    mask_id = mask_token_id
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if mask_id is None or (unk_id is not None and int(mask_id) == int(unk_id)):
            raise ValueError(f"tokenizer has no {mask_token!r}; provide --mask-token-id")
    mask_id = int(mask_id)
    if mask_id < 0 or (vocab_size is not None and mask_id >= int(vocab_size)):
        raise ValueError(f"mask token ID {mask_id} is outside model vocabulary {vocab_size}")
    return mask_id


def distributed_context_from_accelerator(accelerator) -> DistContext:
    return DistContext(
        rank=int(accelerator.process_index),
        local_rank=int(accelerator.local_process_index),
        world_size=int(accelerator.num_processes),
    )


def mixed_precision_from_dtype(dtype_name: str) -> str:
    if dtype_name == "bf16":
        return "bf16"
    if dtype_name == "fp16":
        return "fp16"
    return "no"


def add_d3d_fsdp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--no-fsdp", dest="fsdp", action="store_false")
    parser.set_defaults(fsdp=True)
    parser.add_argument("--fsdp-cpu-offload", action="store_true")
    parser.add_argument("--fsdp-reshard-after-forward", action="store_true")
    parser.add_argument(
        "--no-fsdp-reshard-after-forward",
        dest="fsdp_reshard_after_forward",
        action="store_false",
    )
    parser.set_defaults(fsdp_reshard_after_forward=True)
    parser.add_argument(
        "--fsdp-transformer-layer-cls",
        action="append",
        default=None,
        help="Transformer layer class to wrap. Defaults to D3DDecoderLayer.",
    )


def build_d3d_fsdp_plugin_from_args(args: argparse.Namespace):
    if not bool(getattr(args, "fsdp", True)):
        return None
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None
    if str(getattr(args, "dtype", "bf16")) != "bf16":
        raise ValueError("D3D FSDP requires --dtype bf16")
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp import MixedPrecisionPolicy

    layer_cls = getattr(args, "fsdp_transformer_layer_cls", None) or ["D3DDecoderLayer"]
    return FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=list(layer_cls),
        reshard_after_forward=bool(getattr(args, "fsdp_reshard_after_forward", True)),
        mixed_precision_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        cpu_offload=bool(getattr(args, "fsdp_cpu_offload", False)),
        state_dict_type="FULL_STATE_DICT",
    )


def all_reduce_sum_tensor(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def full_attention_mask(q_len: int, kv_len: int, device: torch.device) -> torch.BoolTensor:
    return torch.ones((int(q_len), int(kv_len)), dtype=torch.bool, device=device)


def position_ids(start: int, length: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


def d3d_forward(
    model,
    input_ids: torch.LongTensor,
    *,
    prefix_len: int,
    past_key_values: DynamicCache,
    store_kv: bool,
    logits_to_keep: int = 0,
    causal: bool = False,
):
    length = int(input_ids.shape[1])
    return model(
        input_ids,
        attention_mask=full_attention_mask(length, prefix_len + length, input_ids.device),
        position_ids=position_ids(prefix_len, length, input_ids.device),
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=store_kv,
        logits_to_keep=logits_to_keep,
        draft_causal=bool(causal),
    )


def dummy_zero_backward(draft_ddp, *, mask_token_id: int, device: torch.device) -> None:
    cache = DynamicCache()
    dummy = torch.tensor([[int(mask_token_id)]], dtype=torch.long, device=device)
    out = d3d_forward(
        draft_ddp,
        dummy,
        prefix_len=0,
        past_key_values=cache,
        store_kv=False,
    )
    loss = out.logits.float().sum() * 0.0
    loss.backward()


def lr_for_step(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    min_lr_ratio = float(min_lr_ratio)
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * float(step) / float(max(warmup_steps, 1))
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def accumulation_window_size(
    *,
    global_step: int,
    total_steps: int,
    accumulation_steps: int,
) -> int:
    accumulation_steps = max(int(accumulation_steps), 1)
    global_step = int(global_step)
    total_steps = int(total_steps)
    window_start = ((global_step - 1) // accumulation_steps) * accumulation_steps + 1
    window_end = min(window_start + accumulation_steps - 1, total_steps)
    return max(window_end - window_start + 1, 1)


def should_step_optimizer(
    *,
    global_step: int,
    total_steps: int,
    accumulation_steps: int,
) -> bool:
    accumulation_steps = max(int(accumulation_steps), 1)
    global_step = int(global_step)
    return global_step % accumulation_steps == 0 or global_step >= int(total_steps)


def should_save_final_checkpoint(*, global_step: int, save_every: int) -> bool:
    return int(save_every) > 0 and int(global_step) > 0 and int(global_step) % int(save_every) != 0


def save_checkpoint(
    *,
    draft_module,
    tokenizer,
    output_dir: Path,
    step: int,
    ctx: DistContext,
    accelerator=None,
    code_model: str | Path | None = None,
) -> None:
    if accelerator is not None:
        state_dict = accelerator.get_state_dict(draft_module)
        if accelerator.is_main_process:
            ckpt_dir = output_dir / f"step-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(draft_module)
            unwrapped.save_pretrained(str(ckpt_dir), state_dict=state_dict, safe_serialization=True)
            tokenizer.save_pretrained(str(ckpt_dir))
            if code_model is not None:
                restore_checkpoint_remote_code(ckpt_dir, Path(code_model))
            print(f"[save] step={step} path={ckpt_dir}", flush=True)
        accelerator.wait_for_everyone()
        return

    if ctx.is_distributed:
        dist.barrier()
    if ctx.is_main:
        ckpt_dir = output_dir / f"step-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        draft_module.save_pretrained(str(ckpt_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(ckpt_dir))
        if code_model is not None:
            restore_checkpoint_remote_code(ckpt_dir, Path(code_model))
        print(f"[save] step={step} path={ckpt_dir}", flush=True)
    if ctx.is_distributed:
        dist.barrier()


class WandbRun:
    def __init__(self, enabled: bool, *, project: str, name: str, config: dict[str, Any]):
        self.enabled = False
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] wandb not installed; logging disabled", flush=True)
            return
        self._wandb = wandb
        self._wandb.init(
            project=project, name=name, config=config,
            settings=wandb.Settings(finish_timeout=60, finish_timeout_raises=False),
        )
        self.enabled = True

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if self.enabled:
            self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()


def dtype_from_arg(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def patch_transformers_losskwargs_for_d3d(transformers_utils=None) -> bool:
    """Patch older Transformers so D3D remote code can import ``LossKwargs``.

    The D3D checkpoint's remote ``modeling_d3d.py`` imports ``LossKwargs`` only
    for ``Unpack[...]`` type annotations. Some local environments have all other
    needed APIs but do not re-export this empty TypedDict from
    ``transformers.utils``. Adding it before dynamic module import is behaviorally
    inert and avoids pinning the launch script to one specific env.
    """
    if transformers_utils is None:
        import transformers.utils as transformers_utils
    if hasattr(transformers_utils, "LossKwargs"):
        return False

    class LossKwargs(TypedDict, total=False):
        pass

    transformers_utils.LossKwargs = LossKwargs
    return True


@contextmanager
def stage_checkpoint_with_code_model(path: str, code_model: str):
    checkpoint_path = Path(path).expanduser().resolve()
    code_path = Path(code_model).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="d3d_code_model_") as tmp_dir:
        staged_path = Path(tmp_dir)
        for item in checkpoint_path.iterdir():
            target = staged_path / item.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(item, target_is_directory=item.is_dir())

        for filename in ("modeling_d3d.py", "configuration_d3d.py"):
            source = code_path / filename
            if not source.exists():
                continue
            target = staged_path / filename
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(source, target)

        checkpoint_config_path = checkpoint_path / "config.json"
        code_config_path = code_path / "config.json"
        if checkpoint_config_path.exists() and code_config_path.exists():
            target_config_path = staged_path / "config.json"
            if target_config_path.exists() or target_config_path.is_symlink():
                target_config_path.unlink()
            config = json.loads(checkpoint_config_path.read_text())
            code_config = json.loads(code_config_path.read_text())
            for key in ("auto_map", "model_type", "architectures"):
                if key in code_config:
                    config[key] = code_config[key]
            config["fuse_cross_entropy"] = False
            target_config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n"
            )

        yield str(staged_path)


def restore_checkpoint_remote_code(checkpoint_path: Path, code_model_path: Path) -> None:
    """Replace FSDP wrapper metadata with the original D3D remote code."""
    checkpoint_path = Path(checkpoint_path)
    code_model_path = Path(code_model_path)
    for filename in ("modeling_d3d.py", "configuration_d3d.py"):
        source = code_model_path / filename
        if source.exists():
            shutil.copy2(source, checkpoint_path / filename)

    checkpoint_config_path = checkpoint_path / "config.json"
    code_config_path = code_model_path / "config.json"
    if not checkpoint_config_path.exists() or not code_config_path.exists():
        return
    config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
    code_config = json.loads(code_config_path.read_text(encoding="utf-8"))
    for key in ("auto_map", "model_type", "architectures"):
        if key in code_config:
            config[key] = code_config[key]
    checkpoint_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model(
    path: str,
    *,
    dtype: torch.dtype,
    attn_implementation: str,
    code_model: str | None = None,
):
    patch_transformers_losskwargs_for_d3d()
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if code_model:
        with stage_checkpoint_with_code_model(path, code_model) as staged_path:
            model, info = AutoModelForCausalLM.from_pretrained(
                staged_path, output_loading_info=True, **kwargs,
            )
            failures = {key: info.get(key) for key in (
                "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
            ) if info.get(key)}
            if failures:
                raise ValueError(f"Checkpoint does not exactly match the D3D architecture: {failures}")
            return model
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)
