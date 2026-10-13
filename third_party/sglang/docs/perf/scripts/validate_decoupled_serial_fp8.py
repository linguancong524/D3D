#!/usr/bin/env python3
"""Independently restart the fastest and highest-hit timely FP8 candidates."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "serial_search", ROOT / "docs/perf/scripts/search_decoupled_serial.py"
)
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)
CONTROL_SPEC = importlib.util.spec_from_file_location(
    "serial_controls", ROOT / "docs/perf/scripts/run_decoupled_serial_fp8_controls.py"
)
CONTROLS = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(CONTROLS)


def run_matched_controls(*, directory, row, reference, cache, control_root):
    block, topk = row["block"], row["topk"]
    case = f"b{block}_k{topk}"
    local = directory / ("local_" + case)
    CONTROLS.run_control(
        directory=local,
        block=block,
        topk=topk,
        mode="local",
        quantization=None,
        reference=reference,
        cache=cache,
    )
    existing = (
        control_root / "bf16_controls" / case,
        control_root / "bf16_coverage_extension" / case,
    )
    if not any((path / "analysis.json").exists() for path in existing):
        hybrid = directory / "matched_bf16_hybrid"
        hybrid.mkdir(exist_ok=True)
        SEARCH.run_case(
            root=hybrid,
            block=block,
            topk=topk,
            reference=reference,
            minimum_ready=0.95,
            model_cache=cache,
        )
    return str(local)


def select_candidates(rows):
    qualified = [
        row
        for row in rows
        if row["timely"]
        and all(sample["correctness"]["valid"] for sample in row["measurements"])
    ]
    if not qualified:
        return []
    highest_hit = max(qualified, key=lambda row: row["hit_rate"])
    fastest = max(
        qualified,
        key=lambda row: row["measurements"][0]["benchmark"][
            "decode_tokens_per_second_p50"
        ],
    )
    unique = {(row["block"], row["topk"]): row for row in (highest_hit, fastest)}
    return list(unique.values())


def complete_extension_controls(*, remote_root, control_root, cache):
    extension = remote_root / "coverage_extension/summary.json"
    if not extension.exists():
        return
    directory = control_root / "bf16_coverage_extension"
    directory.mkdir(exist_ok=True)
    for row in json.loads(extension.read_text()):
        case = directory / f"b{row['block']}_k{row['topk']}"
        if (case / "analysis.json").exists():
            continue
        result = SEARCH.run_case(
            root=directory,
            block=row["block"],
            topk=row["topk"],
            reference=CONTROLS.BF16_REFERENCE,
            minimum_ready=0.95,
            model_cache=cache,
        )
        SEARCH.write_json(path=case / "summary.json", value=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    args.output_dir.mkdir()
    complete_extension_controls(
        remote_root=args.remote_dir,
        control_root=args.controls_dir,
        cache=args.model_cache,
    )
    selections = []
    for name, grid, reference, target_quantization in (
        (
            "remote_fp8",
            args.remote_dir / "grid",
            ROOT
            / "docs/perf/results/decoupled-local-20261009-023239-671917/serial_n10.json",
            None,
        ),
        (
            "full_fp8",
            args.controls_dir / "full_fp8_grid",
            args.controls_dir / "fp8_ar/serial_n10.json",
            "fp8",
        ),
    ):
        rows = json.loads((grid / "summary.json").read_text())
        if name == "remote_fp8":
            extension = args.remote_dir / "coverage_extension/summary.json"
            if extension.exists():
                rows += json.loads(extension.read_text())
        candidates = select_candidates(rows)
        reference_repeat_valid = True
        if name == "full_fp8":
            repeats = json.loads(
                (args.controls_dir / "fp8_ar/analysis.json").read_text()
            )
            reference_repeat_valid = all(
                sample["correctness"] is None or sample["correctness"]["valid"]
                for sample in repeats
            )
            if not reference_repeat_valid:
                candidates = []
        directory = args.output_dir / name
        directory.mkdir()
        results = []
        local_controls = []
        for row in candidates:
            results.append(
                SEARCH.run_case(
                    root=directory,
                    block=row["block"],
                    topk=row["topk"],
                    reference=reference,
                    minimum_ready=0.95,
                    model_cache=args.model_cache,
                    remote_draft_quantization="fp8",
                    target_quantization=target_quantization,
                    local_draft_quantization=target_quantization,
                    diagnose_output_mismatch=target_quantization is not None,
                )
            )
            if name == "remote_fp8":
                local_controls.append(
                    run_matched_controls(
                        directory=directory,
                        row=row,
                        reference=reference,
                        cache=args.model_cache,
                        control_root=args.controls_dir,
                    )
                )
        selections.append(
            {
                "mode": name,
                "initial_candidates": candidates,
                "reference_repeat_valid": reference_repeat_valid,
                "rechecks": results,
                "matched_local_controls": local_controls,
            }
        )
        SEARCH.write_json(path=args.output_dir / "selection.json", value=selections)
    print("FP8 CANDIDATE VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
