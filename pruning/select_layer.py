from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .common import (
    atomic_write_json,
    checkpoint_depth,
    draft_attention_mask,
    layer_provenance,
    load_selection_examples,
    torch_dtype,
)
from .modeling import disable_cache, evaluate_single_layer_removals, load_drafter
from models.token_ids import checkpoint_mask_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the least damaging decoder layer on a fixed tokenized split."
        )
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="layer_selection")
    parser.add_argument("--expected-samples", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


@torch.inference_mode()
def _evaluate_shard(
    *,
    parent: str,
    examples: pd.DataFrame,
    device_name: str,
    dtype: str,
    block_size: int,
    mask_token_id: int,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    model = load_drafter(parent, torch_dtype(dtype)).to(device).eval()
    disable_cache(model)
    rows = []
    for example in examples.itertuples(index=False):
        prefix_ids = [int(token) for token in example.prefix_ids]
        target_ids = [int(token) for token in example.target_ids[:block_size]]
        if len(target_ids) != block_size:
            raise ValueError(f"{example.analysis_id} has an incomplete target block")
        prefix = torch.tensor(prefix_ids, device=device).unsqueeze(0)
        target = torch.tensor(target_ids, device=device)
        block = target.clone()
        block[1:] = mask_token_id
        input_ids = torch.cat([prefix, block[None, :]], dim=1)
        current, candidates = evaluate_single_layer_removals(
            model,
            input_ids,
            draft_attention_mask(len(prefix_ids), block_size, device),
            torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            target[1:],
            block_size,
            candidate_batch_size,
        )
        for layer, candidate in candidates.items():
            rows.append(
                {
                    "analysis_id": str(example.analysis_id),
                    "removed_compact_layer": layer,
                    "parent_emitted_tokens": current[0],
                    "candidate_emitted_tokens": candidate[0],
                    "parent_draft_token_accuracy": current[1],
                    "candidate_draft_token_accuracy": candidate[1],
                }
            )
    return rows


def summarize(
    metrics: pd.DataFrame,
    layer_count: int,
    retained_original_layers: list[int],
    prompt_order: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, int]:
    parent: np.ndarray | None = None
    emitted = []
    accuracies = []
    for layer in range(layer_count):
        values = (
            metrics.loc[metrics["removed_compact_layer"] == layer]
            .set_index("analysis_id")
            .loc[prompt_order]
        )
        observed_parent = values["parent_emitted_tokens"].to_numpy(np.float64)
        if parent is None:
            parent = observed_parent
        elif not np.array_equal(parent, observed_parent):
            raise RuntimeError("Parent scores differ across deletion candidates")
        emitted.append(values["candidate_emitted_tokens"].to_numpy(np.float64))
        accuracies.append(values["candidate_draft_token_accuracy"].to_numpy(np.float64))
    assert parent is not None
    emitted_matrix = np.stack(emitted, axis=1)
    accuracy_matrix = np.stack(accuracies, axis=1)
    means = emitted_matrix.mean(axis=0)
    accuracy_means = accuracy_matrix.mean(axis=0)
    parent_mean = float(parent.mean())
    damage = (parent_mean - means) / parent_mean

    rng = np.random.default_rng(bootstrap_seed)
    counts = rng.multinomial(
        len(prompt_order),
        np.full(len(prompt_order), 1.0 / len(prompt_order)),
        size=bootstrap_samples,
    )
    weights = counts / len(prompt_order)
    parent_bootstrap = weights @ parent
    candidate_bootstrap = weights @ emitted_matrix
    damage_bootstrap = (
        parent_bootstrap[:, None] - candidate_bootstrap
    ) / parent_bootstrap[:, None]
    low, high = np.quantile(damage_bootstrap, [0.025, 0.975], axis=0)

    ranking = sorted(
        range(layer_count),
        key=lambda layer: (
            -means[layer],
            -accuracy_means[layer],
            retained_original_layers[layer],
        ),
    )
    rank = {layer: index + 1 for index, layer in enumerate(ranking)}
    rows = []
    for layer in range(layer_count):
        rows.append(
            {
                "rank": rank[layer],
                "removed_compact_layer": layer,
                "removed_original_layer": retained_original_layers[layer],
                "selected": layer == ranking[0],
                "parent_mean_accept_length": parent_mean,
                "candidate_mean_accept_length": float(means[layer]),
                "candidate_token_accuracy": float(accuracy_means[layer]),
                "relative_acceptance_damage": float(damage[layer]),
                "damage_ci95_low": float(low[layer]),
                "damage_ci95_high": float(high[layer]),
            }
        )
    return pd.DataFrame(rows).sort_values("rank"), ranking[0]


def main() -> None:
    args = build_parser().parse_args()
    parent = args.parent.resolve()
    args.mask_token_id = checkpoint_mask_id(parent, args.mask_token_id)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.block_size < 2 or args.candidate_batch_size <= 0:
        raise ValueError("Invalid block or candidate batch size")
    examples = load_selection_examples(args.examples, args.split)
    if args.expected_samples and len(examples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} examples, found {len(examples)}"
        )
    layer_count = checkpoint_depth(parent)
    teacher, original_depth, retained_original_layers = layer_provenance(parent)
    if len(retained_original_layers) != layer_count:
        raise ValueError("Parent layer provenance does not match its config")
    if len(args.devices) != len(set(args.devices)):
        raise ValueError("--devices must be unique")

    started = time.time()
    shards = [
        examples.iloc[index :: len(args.devices)].copy()
        for index in range(len(args.devices))
    ]
    jobs = [
        {
            "parent": str(parent),
            "examples": shard,
            "device_name": device,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "mask_token_id": args.mask_token_id,
            "candidate_batch_size": args.candidate_batch_size,
        }
        for device, shard in zip(args.devices, shards, strict=True)
    ]
    if len(jobs) == 1:
        rows = _evaluate_shard(**jobs[0])
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_evaluate_shard, **job) for job in jobs]
            rows = [row for future in futures for row in future.result()]
    metrics = pd.DataFrame(rows)
    summary, selected = summarize(
        metrics,
        layer_count,
        retained_original_layers,
        examples["analysis_id"].tolist(),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    selected_row = summary.loc[summary["selected"]].iloc[0]
    child_mapping = [
        original
        for compact, original in enumerate(retained_original_layers)
        if compact != selected
    ]
    selection = {
        "format_version": 1,
        "method": "progressive_single_layer_pruning",
        "parent_checkpoint": str(parent),
        "teacher_checkpoint": str(teacher),
        "original_depth": original_depth,
        "parent_depth": layer_count,
        "child_depth": layer_count - 1,
        "parent_retained_original_layers": retained_original_layers,
        "removed_compact_layer": selected,
        "removed_original_layer": retained_original_layers[selected],
        "child_retained_original_layers": child_mapping,
        "block_size": args.block_size,
        "selection_samples": len(examples),
        "mask_token_id": args.mask_token_id,
        "selection_data": str(args.examples.resolve()),
        "parent_mean_accept_length": float(selected_row["parent_mean_accept_length"]),
        "child_step0_mean_accept_length": float(
            selected_row["candidate_mean_accept_length"]
        ),
        "relative_acceptance_damage": float(selected_row["relative_acceptance_damage"]),
        "elapsed_seconds": time.time() - started,
    }
    output.mkdir(parents=True)
    metrics.sort_values(["removed_compact_layer", "analysis_id"]).to_parquet(
        output / "candidate_metrics.parquet", index=False
    )
    summary.to_csv(output / "candidate_summary.csv", index=False)
    atomic_write_json(output / "selection.json", selection)
    print(
        f"Selected compact layer {selected} (original layer "
        f"{retained_original_layers[selected]}) for D{layer_count} -> D{layer_count - 1}."
    )


if __name__ == "__main__":
    main()
