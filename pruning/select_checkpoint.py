from __future__ import annotations

import argparse
import concurrent.futures
import gc
import multiprocessing
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .common import atomic_write_json, load_selection_examples
from .modeling import load_drafter, score_tokenized_examples
from models.token_ids import checkpoint_mask_id

STEP_PATTERN = re.compile(r"step-(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Choose the recovery checkpoint with the highest validation acceptance."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="layer_validation")
    parser.add_argument("--expected-samples", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    return parser


def checkpoint_step(path: Path) -> int:
    match = STEP_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        checkpoints = [path.resolve() for path in args.checkpoints]
    else:
        checkpoints = sorted(
            (path.resolve() for path in args.run_dir.glob("step-*")),
            key=checkpoint_step,
        )
    checkpoints = [path for path in checkpoints if (path / "config.json").is_file()]
    if not checkpoints:
        raise ValueError("No complete checkpoints were found")
    return checkpoints


@torch.inference_mode()
def _score_worker(
    checkpoints: list[str],
    examples: pd.DataFrame,
    device_name: str,
    block_size: int,
    mask_token_id: int | None,
) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    rows = []
    for checkpoint in checkpoints:
        model = load_drafter(checkpoint, torch.bfloat16).to(device).eval()
        scores = score_tokenized_examples(
            model,
            examples,
            block_size=block_size,
            mask_token_id=checkpoint_mask_id(checkpoint, mask_token_id),
            device=device,
        )
        for score in scores:
            rows.append({"checkpoint": checkpoint, **score})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoints = discover_checkpoints(args)
    examples = load_selection_examples(args.examples, args.split)
    if args.expected_samples and len(examples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} examples, found {len(examples)}"
        )
    if len(args.devices) != len(set(args.devices)):
        raise ValueError("--devices must be unique")

    assignments = [
        checkpoints[index :: len(args.devices)] for index in range(len(args.devices))
    ]
    jobs = [
        (
            [str(path) for path in assigned],
            examples,
            device,
            args.block_size,
            args.mask_token_id,
        )
        for device, assigned in zip(args.devices, assignments, strict=True)
        if assigned
    ]
    if len(jobs) == 1:
        rows = _score_worker(*jobs[0])
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_score_worker, *job) for job in jobs]
            rows = [row for future in futures for row in future.result()]

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("checkpoint", as_index=False).agg(
        mean_accept_length=("emitted_tokens", "mean"),
        mean_draft_token_accuracy=("draft_token_accuracy", "mean"),
        samples=("analysis_id", "count"),
    )
    summary["step"] = summary["checkpoint"].map(
        lambda value: checkpoint_step(Path(value))
    )
    summary = summary.sort_values(
        ["mean_accept_length", "mean_draft_token_accuracy", "step"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    best = summary.iloc[0]

    output.mkdir(parents=True)
    metrics.sort_values(["checkpoint", "analysis_id"]).to_parquet(
        output / "checkpoint_metrics.parquet", index=False
    )
    summary.to_csv(output / "checkpoint_summary.csv", index=False)
    atomic_write_json(
        output / "selected_checkpoint.json",
        {
            "format_version": 1,
            "selection_metric": "mean_accept_length",
            "selected_checkpoint": str(best["checkpoint"]),
            "selected_step": int(best["step"]),
            "mean_accept_length": float(best["mean_accept_length"]),
            "mean_draft_token_accuracy": float(best["mean_draft_token_accuracy"]),
            "samples": int(best["samples"]),
        },
    )
    print(
        f"Selected {best['checkpoint']} with mean accept length "
        f"{best['mean_accept_length']:.6f}."
    )


if __name__ == "__main__":
    main()
