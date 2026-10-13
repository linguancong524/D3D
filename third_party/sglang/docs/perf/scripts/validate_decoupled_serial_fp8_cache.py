#!/usr/bin/env python3
"""Validate timely FP8 hybrid candidates against a repeatable cache-off AR."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "serial_controls", ROOT / "docs/perf/scripts/run_decoupled_serial_fp8_controls.py"
)
CONTROLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLS)
SEARCH = CONTROLS.SEARCH


def repeat_valid(directory):
    rows = json.loads((directory / "analysis.json").read_text())
    return all(
        row["correctness"] is None or row["correctness"]["valid"] for row in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    directory = args.probe_dir / "hybrid_followup"
    directory.mkdir()
    ar = args.probe_dir / "fp8_ar_cacheoff"
    state = {"initial_ar_repeat_valid": repeat_valid(ar), "cases": []}
    if state["initial_ar_repeat_valid"]:
        reference = ar / "serial_n10.json"
        restart = directory / "fp8_ar_restart"
        CONTROLS.run_control(
            directory=restart,
            block=4,
            topk=4,
            mode="ar",
            quantization="fp8",
            reference=reference,
            cache=args.model_cache,
            disable_radix_cache=True,
        )
        state["restarted_ar_valid"] = repeat_valid(restart)
        if state["restarted_ar_valid"]:
            rows = json.loads(
                (args.controls_dir / "full_fp8_grid/summary.json").read_text()
            )
            timely = [row for row in rows if row["timely"]]
            if timely:
                highest_hit = max(timely, key=lambda row: row["hit_rate"])
                fastest = max(
                    timely,
                    key=lambda row: row["measurements"][0]["benchmark"][
                        "decode_tokens_per_second_p50"
                    ],
                )
                candidates = {
                    (row["block"], row["topk"]): row for row in (highest_hit, fastest)
                }
                for block, topk in candidates:
                    result = SEARCH.run_case(
                        root=directory,
                        block=block,
                        topk=topk,
                        reference=reference,
                        minimum_ready=0.95,
                        model_cache=args.model_cache,
                        remote_draft_quantization="fp8",
                        target_quantization="fp8",
                        local_draft_quantization="fp8",
                        disable_radix_cache=True,
                        diagnose_output_mismatch=True,
                    )
                    local = directory / f"local_b{block}_k{topk}"
                    CONTROLS.run_control(
                        directory=local,
                        block=block,
                        topk=topk,
                        mode="local",
                        quantization="fp8",
                        reference=reference,
                        cache=args.model_cache,
                        disable_radix_cache=True,
                    )
                    state["cases"].append(
                        {"hybrid": result, "matched_local": str(local)}
                    )
                    SEARCH.write_json(path=directory / "summary.json", value=state)
    SEARCH.write_json(path=directory / "summary.json", value=state)
    print("FP8 CACHE-OFF HYBRID VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
