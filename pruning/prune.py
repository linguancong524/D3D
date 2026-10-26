from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from .common import atomic_write_json, checkpoint_depth

LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physically remove the layer recorded in selection.json."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def renamed_key(name: str, old_to_new: dict[int, int]) -> str | None:
    match = LAYER_KEY.match(name)
    if match is None:
        return name
    old_layer = int(match.group(1))
    if old_layer not in old_to_new:
        return None
    return f"model.layers.{old_to_new[old_layer]}.{match.group(2)}"


def source_weight_map(checkpoint: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        return dict(index["weight_map"]), dict(index.get("metadata", {}))
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"No safetensors weights found in {checkpoint}")
    with safe_open(model_path, framework="pt", device="cpu") as stream:
        names = list(stream.keys())
    return {name: model_path.name for name in names}, {}


def copy_support_files(source: Path, destination: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "layer_prune.json",
        "recovery_info.json",
        "layer_schedule.json",
        "checkpoint_validation.json",
    }
    for path in source.iterdir():
        if (
            path.is_file()
            and path.name not in excluded
            and not path.name.startswith(
                ("compact_model_", "progressive_checkpoint_", "recovery_checkpoint_")
            )
            and path.suffix != ".safetensors"
        ):
            shutil.copy2(path, destination / path.name)


def write_pruned_weights(
    source: Path,
    destination: Path,
    old_to_new: dict[int, int],
) -> dict[str, Any]:
    weight_map, source_metadata = source_weight_map(source)
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for old_name, shard in weight_map.items():
        new_name = renamed_key(old_name, old_to_new)
        if new_name is not None:
            by_shard.setdefault(shard, []).append((old_name, new_name))

    output_map: dict[str, str] = {}
    total_parameters = 0
    total_size = 0
    nonempty_shards = [shard for shard, entries in sorted(by_shard.items()) if entries]
    for number, source_shard in enumerate(nonempty_shards, start=1):
        output_shard = f"model-{number:05d}-of-{len(nonempty_shards):05d}.safetensors"
        tensors = {}
        with safe_open(source / source_shard, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
            for old_name, new_name in by_shard[source_shard]:
                tensor = stream.get_tensor(old_name)
                tensors[new_name] = tensor
                total_parameters += tensor.numel()
                total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, destination / output_shard, metadata=metadata)
        output_map.update({name: output_shard for name in tensors})

    metadata = {
        **source_metadata,
        "total_parameters": total_parameters,
        "total_size": total_size,
    }
    index = {"metadata": metadata, "weight_map": dict(sorted(output_map.items()))}
    atomic_write_json(destination / "model.safetensors.index.json", index)
    return metadata


def prune(selection_path: Path, output: Path) -> None:
    selection_path = selection_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    selection = json.loads(selection_path.read_text())
    parent = Path(selection["parent_checkpoint"]).resolve()
    parent_depth = int(selection["parent_depth"])
    child_depth = int(selection["child_depth"])
    removed = int(selection["removed_compact_layer"])
    if checkpoint_depth(parent) != parent_depth:
        raise ValueError("The selected parent no longer has the recorded depth")
    if child_depth != parent_depth - 1 or not 0 <= removed < parent_depth:
        raise ValueError("selection.json does not describe one valid layer removal")

    retained_parent_layers = [
        index for index in range(parent_depth) if index != removed
    ]
    old_to_new = {
        old_layer: new_layer
        for new_layer, old_layer in enumerate(retained_parent_layers)
    }
    retained_original_layers = [
        int(layer) for layer in selection["child_retained_original_layers"]
    ]
    if len(retained_original_layers) != child_depth:
        raise ValueError("The child original-layer mapping has the wrong length")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building.", dir=output.parent)
    )
    try:
        copy_support_files(parent, temporary)
        config = json.loads((parent / "config.json").read_text())
        config["num_hidden_layers"] = child_depth
        if selection.get("mask_token_id") is not None:
            from models.token_ids import checkpoint_mask_id
            config["mask_token_id"] = checkpoint_mask_id(parent, selection["mask_token_id"])
        if "max_window_layers" in config:
            config["max_window_layers"] = min(
                int(config["max_window_layers"]), child_depth
            )
        atomic_write_json(temporary / "config.json", config)
        weight_metadata = write_pruned_weights(parent, temporary, old_to_new)
        provenance = {
            "format_version": 1,
            "method": "progressive_single_layer_pruning",
            "parent_checkpoint": str(parent),
            "teacher_checkpoint": str(Path(selection["teacher_checkpoint"]).resolve()),
            "original_depth": int(selection["original_depth"]),
            "parent_depth": parent_depth,
            "child_depth": child_depth,
            "removed_parent_layer": removed,
            "removed_original_layer": int(selection["removed_original_layer"]),
            "retained_parent_layers": retained_parent_layers,
            "retained_original_layers": retained_original_layers,
            "block_size": int(selection["block_size"]),
            "selection_file": str(selection_path),
            "parameters": int(weight_metadata["total_parameters"]),
        }
        atomic_write_json(temporary / "layer_prune.json", provenance)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    prune(args.selection, args.output)
    print(f"Pruned checkpoint written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
