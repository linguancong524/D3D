#!/usr/bin/env python3
"""Repeat AR and local FP8 controls with only Target radix cache disabled."""

import argparse
import os
from pathlib import Path

from run_decoupled_serial_fp8_controls import BF16_REFERENCE, ROOT, SEARCH, run_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    directory = args.output_dir.resolve()
    directory.mkdir()
    reference = directory / "fp8_ar_cacheoff/serial_n10.json"
    for name, mode, block, quantization, comparison in (
        ("fp8_ar_cacheoff", "ar", 4, "fp8", None),
        ("fp8_local_b4_cacheoff", "local", 4, "fp8", reference),
        ("fp8_local_b8_cacheoff", "local", 8, "fp8", reference),
        ("bf16_ar_cacheoff", "ar", 4, None, BF16_REFERENCE),
    ):
        run_control(
            directory=directory / name,
            block=block,
            topk=4,
            mode=mode,
            quantization=quantization,
            reference=comparison,
            cache=args.model_cache,
            disable_radix_cache=True,
        )
    SEARCH.write_json(
        path=directory / "complete.json",
        value={
            "reference": str(reference),
            "target_radix_cache_disabled": True,
            "local_prefix_reuse": "unchanged from original configuration",
            "measured_requests": 80,
            "unique_inputs": 10,
        },
    )


if __name__ == "__main__":
    main()
