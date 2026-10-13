"""Build a per-channel W8A8 INT8 SDAR checkpoint for remote Draft trials.

Only transformer linear weights are quantized. Embeddings, norms, and the
vocabulary head remain in their source dtype, matching SGLang's unquantized
VocabParallelEmbedding / ParallelLMHead methods.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LINEAR_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("linear weight must be a floating-point matrix")
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    if not torch.isfinite(scale).all():
        raise ValueError("linear weight contains non-finite values")
    quantized = torch.round(weight.float() / scale).clamp_(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def convert_checkpoint(*, source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir() or not (source / "model.safetensors.index.json").is_file():
        raise ValueError("source must be a sharded safetensors model directory")
    if output == source or source in output.parents:
        raise ValueError("output must be separate from the source checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config") is not None:
        raise ValueError("source checkpoint is already quantized")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    if not shard_names or not all((source / name).is_file() for name in shard_names):
        raise ValueError("checkpoint index references a missing shard")
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in {
            "config.json",
            "model.safetensors.index.json",
            *shard_names,
        }:
            shutil.copy2(path, output / path.name)

    num_quantized = 0
    total_size = 0
    for shard_name in shard_names:
        converted = {}
        with safe_open(source / shard_name, framework="pt", device="cpu") as shard:
            for name in sorted(shard.keys()):
                tensor = shard.get_tensor(name)
                if name.endswith(LINEAR_WEIGHT_SUFFIXES):
                    quantized, scale = quantize_per_channel(tensor)
                    converted[name] = quantized
                    scale_name = name.removesuffix(".weight") + ".weight_scale"
                    converted[scale_name] = scale
                    weight_map[scale_name] = shard_name
                    num_quantized += 1
                else:
                    converted[name] = tensor
                del tensor
        save_file(converted, output / shard_name)
        total_size += sum(t.numel() * t.element_size() for t in converted.values())
        print(f"converted {shard_name}: {len(converted)} tensors", flush=True)
        del converted

    if num_quantized == 0:
        raise ValueError("source checkpoint has no SDAR transformer linear weights")
    index["metadata"]["total_size"] = total_size
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    config["quantization_config"] = {"quant_method": "w8a8_int8"}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        "source": str(source),
        "output": str(output),
        "num_quantized_linear_weights": num_quantized,
        "total_weight_bytes": total_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(source=args.source, output=args.output)))


if __name__ == "__main__":
    main()
