#!/usr/bin/env python3
"""Audit FP8 serial experiments and compare their archived raw token outputs."""

import argparse
import ast
import json
import math
import re
import statistics
from pathlib import Path

from sglang.benchmark.decoupled_serving import compare_outputs


def read_json(path):
    return json.loads(path.read_text())


def checked_parity(*, payload, reference):
    result = compare_outputs(payload=payload, reference=reference)
    direct_matches = sum(
        row["output_ids"] == other["output_ids"]
        for row, other in zip(payload["requests"], reference["requests"], strict=True)
    )
    if result["matched"] != direct_matches:
        raise ValueError("parity helper disagrees with direct token-list comparison")
    return result


def checked_lookup_counters(*, directory, result_path, measurement):
    if "lookups" not in measurement:
        return None
    bounds = read_json(result_path.with_name(result_path.stem + "_bounds.json"))
    data = (directory / "target-0.log").read_bytes()
    pattern = r"DECOUPLED request finish client_metrics=(\{[^\n]+\})"
    previous = re.findall(pattern, data[: bounds["begin"]["target-0.log"]].decode())
    current = re.findall(pattern, data[: bounds["end"]["target-0.log"]].decode())
    before = ast.literal_eval(previous[-1]) if previous else {}
    after = ast.literal_eval(current[-1])
    ready_key = "remote_optional_ready_lookup_ct"
    miss_key = "remote_optional_lookup_miss_ct"
    hit_key = "remote_optional_lookup_hit_ct"
    delta = {
        key: after[key] - before.get(key, 0) for key in (ready_key, miss_key, hit_key)
    }
    ready, misses, hits = (delta[key] for key in (ready_key, miss_key, hit_key))
    lookups = ready + misses
    if (lookups, ready, hits) != (
        measurement["lookups"],
        measurement["ready"],
        measurement["hits"],
    ):
        raise ValueError(
            f"lookup summary disagrees with request-finish log: {result_path}"
        )
    if not 0 <= hits <= ready <= lookups:
        raise ValueError(f"invalid lookup membership: {result_path}")
    return {"lookups": lookups, "ready": ready, "hits": hits}


def checked_sse_summary(payload):
    requests = payload["requests"]
    summary = payload["summary"]
    expected = {
        "decode_tokens_per_second_p50": statistics.median(
            row["decode_tokens_per_second"] for row in requests
        ),
        "tpot_ms_p50": statistics.median(row["tpot_ms"] for row in requests),
        "ttft_ms_p50": statistics.median(row["ttft_seconds"] for row in requests)
        * 1000,
        "output_tokens_per_second": sum(row["output_tokens"] for row in requests)
        / summary["elapsed_seconds"],
        "spec_verify_ct": sum(row["spec_verify_ct"] for row in requests),
    }
    for key, value in expected.items():
        if not math.isclose(summary[key], value, abs_tol=1e-10):
            raise ValueError(f"SSE summary does not reconcile: {key}")
    return expected


def main_path_costs(*, directory, result_path):
    bounds = read_json(result_path.with_name(result_path.stem + "_bounds.json"))
    data = (directory / "target-0.log").read_bytes()
    text = data[
        bounds["begin"]["target-0.log"] : bounds["end"]["target-0.log"]
    ].decode()
    finishes = list(
        re.finditer(r"DECOUPLED request finish client_metrics=(\{[^\n]+\})", text)
    )
    if not finishes:
        return None
    if len(finishes) != 11:
        raise ValueError(f"expected warmup + 10 request finishes in {result_path}")
    measured = text[finishes[0].end() :]
    warmup = ast.literal_eval(finishes[0].group(1))
    final = ast.literal_eval(finishes[-1].group(1))
    client = {key: value - warmup.get(key, 0) for key, value in final.items()}
    phases = {}
    for name, label in (("local_draft", "local fallback"), ("verify", "target verify")):
        values = [
            float(value)
            for value in re.findall(
                label + r" timing items=1 seconds=([0-9.]+)", measured
            )
        ]
        phases[name] = {
            "count": len(values),
            "seconds": sum(values),
            "mean_ms": statistics.mean(values) * 1000 if values else None,
        }
    payload = read_json(result_path)
    verify = payload["summary"]["spec_verify_ct"]
    if phases["verify"]["count"] != verify:
        raise ValueError(f"verify count mismatch in {result_path}")
    if (
        phases["local_draft"]["count"] + client["remote_optional_lookup_hit_ct"]
        != verify
    ):
        raise ValueError(f"hit / fallback accounting mismatch in {result_path}")
    return {
        "warmup_excluded": True,
        "phases": phases,
        "client": client,
        "remainder_seconds": payload["summary"]["elapsed_seconds"]
        - sum(phase["seconds"] for phase in phases.values()),
        "tokens_per_verify": sum(row["output_tokens"] for row in payload["requests"])
        / verify,
    }


def audit_case(*, directory, bf16_reference, fp8_reference, fp8_cacheoff_reference):
    result_paths = sorted(
        path
        for path in directory.glob("serial_n10*.json")
        if not path.name.endswith("_bounds.json")
    )
    metrics = read_json(directory / "analysis.json")
    if len(result_paths) != len(metrics):
        raise ValueError(f"result / measurement count mismatch in {directory}")
    benchmarks = []
    for path, measurement in zip(result_paths, metrics, strict=True):
        payload = read_json(path)
        successful = all(
            not row["error"] and row["output_tokens"] == len(row["output_ids"]) == 256
            for row in payload["requests"]
        )
        if (
            not successful
            or len(payload["requests"]) != 10
            or payload["summary"]["completed_requests"] != 10
            or payload["summary"]["failed_requests"] != 0
        ):
            raise ValueError(f"incomplete benchmark: {path}")
        benchmarks.append(
            {
                "path": str(path),
                "summary": payload["summary"],
                "sse_summary_check": checked_sse_summary(payload),
                "parity_bf16": checked_parity(
                    payload=payload, reference=bf16_reference
                ),
                "parity_fp8_ar": (
                    checked_parity(payload=payload, reference=fp8_reference)
                    if fp8_reference is not None
                    else None
                ),
                "parity_fp8_ar_cacheoff": (
                    checked_parity(payload=payload, reference=fp8_cacheoff_reference)
                    if fp8_cacheoff_reference is not None
                    else None
                ),
                "main_path_costs": main_path_costs(
                    directory=directory, result_path=path
                ),
                "raw_counter_check": checked_lookup_counters(
                    directory=directory,
                    result_path=path,
                    measurement=measurement,
                ),
            }
        )
    config = read_json(directory / "run.json")
    actual_precision = {}
    for name, _, command in config["commands"]:
        actual_precision[name] = {
            flag: command[command.index(flag) + 1] if flag in command else None
            for flag in ("--quantization", "--speculative-draft-model-quantization")
        }
    for sample in metrics:
        if "service_delta" in sample and sample["service_delta"]["error_ct"] != 0:
            raise ValueError(f"remote service errors in {directory}")
    return {
        "directory": str(directory),
        "args": config["args"],
        "actual_precision": actual_precision,
        "analysis": metrics,
        "benchmarks": benchmarks,
    }


def checked_grid_summaries(*, roots, cases):
    by_directory = {str(Path(case["directory"]).resolve()): case for case in cases}
    checked = []
    for root in roots:
        for path in sorted(root.rglob("summary.json")):
            rows = read_json(path)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if "timely" not in row or "measurements" not in row:
                    continue
                case = by_directory[str(Path(row["directory"]).resolve())]
                counters = [
                    sample["raw_counter_check"] for sample in case["benchmarks"]
                ]
                lookups = sum(sample["lookups"] for sample in counters)
                ready = sum(sample["ready"] for sample in counters) / lookups
                hit = sum(sample["hits"] for sample in counters) / lookups
                if (
                    not math.isclose(ready, row["ready_rate"], abs_tol=1e-12)
                    or not math.isclose(hit, row["hit_rate"], abs_tol=1e-12)
                    or row["timely"] != (ready >= 0.95)
                ):
                    raise ValueError(f"weighted grid summary mismatch: {path}")
            checked.append(str(path))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, action="append", default=[])
    parser.add_argument("--cacheoff-reference-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bf16 = read_json(
        Path("docs/perf/results/decoupled-local-20261009-023239-671917/serial_n10.json")
    )
    fp8_path = args.controls_dir / "fp8_ar/serial_n10.json"
    fp8 = read_json(fp8_path) if fp8_path.exists() else None
    cacheoff = (
        read_json(args.cacheoff_reference_results)
        if args.cacheoff_reference_results is not None
        else None
    )
    roots = (args.remote_dir, args.controls_dir, *args.extra_dir)
    directories = sorted(
        {path.parent for root in roots for path in root.rglob("serial_n10.json")}
    )
    cases = [
        audit_case(
            directory=directory,
            bf16_reference=bf16,
            fp8_reference=fp8,
            fp8_cacheoff_reference=cacheoff,
        )
        for directory in directories
    ]
    output = {
        "cases": cases,
        "checked_grid_summaries": checked_grid_summaries(roots=roots, cases=cases),
        "measured_requests": sum(len(case["benchmarks"]) * 10 for case in cases),
        "bf16_reference": "docs/perf/results/decoupled-local-20261009-023239-671917/serial_n10.json",
        "fp8_reference": str(fp8_path) if fp8 is not None else None,
        "fp8_cacheoff_reference": (
            str(args.cacheoff_reference_results) if cacheoff is not None else None
        ),
        "note": (
            "Readiness and hit rates use original warmup-inclusive counters; "
            "speed, parity and main-path costs exclude warmup. Token parity "
            "does not establish broad task accuracy. Remainder is unassigned "
            "wall time, not an independently measured RPC cost."
        ),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(
        f"audited {len(cases)} cases, {output['measured_requests']} measured requests"
    )


if __name__ == "__main__":
    main()
