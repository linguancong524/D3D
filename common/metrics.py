"""Acceptance and timing summary helpers."""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any


def summarize_timing_metrics(per_prompt: list[dict[str, Any]], *, elapsed_s: float) -> dict[str, float]:
    total_output_tokens = sum(int(x.get("output_tokens", 0)) for x in per_prompt)
    total_decode_tokens = sum(int(x.get("decode_tokens", x.get("output_tokens", 0))) for x in per_prompt)
    decode_times = [float(x.get("decode_elapsed_s", 0.0)) for x in per_prompt]
    prompt_tps = [float(x.get("tokens_per_s", 0.0)) for x in per_prompt]
    time_per_output_token = [
        float(x.get("time_per_output_token", 0.0))
        for x in per_prompt
        if int(x.get("decode_tokens", x.get("output_tokens", 0))) > 0
    ]
    total_decode_elapsed_s = sum(decode_times)
    aggregate_decode_tps = total_decode_tokens / total_decode_elapsed_s if total_decode_elapsed_s > 0 else 0.0
    mean_tokens_per_s = statistics.fmean(prompt_tps) if prompt_tps else 0.0
    return {
        "total_decode_elapsed_s": total_decode_elapsed_s,
        "mean_tokens_per_s": mean_tokens_per_s,
        "mean_prompt_decode_tps": mean_tokens_per_s,
        "mean_time_per_output_token": (
            statistics.fmean(time_per_output_token) if time_per_output_token else 0.0
        ),
        "aggregate_decode_tokens_per_s": aggregate_decode_tps,
        "aggregate_decode_tps": aggregate_decode_tps,
        "elapsed_tokens_per_s": total_output_tokens / float(elapsed_s) if elapsed_s > 0 else 0.0,
    }


def headline_metrics(
    d3d_tps: float,
    baseline_tps: float,
    accept_length: float,
    *,
    metric: str = "decode",
) -> dict[str, Any]:
    """Report the two headline numbers: speedup over the single-card AR baseline and accept length."""
    d3d_tps = float(d3d_tps or 0.0)
    baseline_tps = float(baseline_tps or 0.0)
    return {
        "metric": metric,
        "d3d_tps": d3d_tps,
        "baseline_tps": baseline_tps,
        "speedup": d3d_tps / baseline_tps if baseline_tps > 0 else None,
        "accept_length": float(accept_length or 0.0),
    }


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[max(0, min(idx, len(xs) - 1))])


def compact_acceptance(per_sample_accept: list[float], flat_accept: list[int], block_size: int) -> dict[str, Any]:
    hist = Counter(int(x) for x in flat_accept)
    return {
        "num_samples": len(per_sample_accept),
        "num_verify_steps": len(flat_accept),
        "block_size": int(block_size),
        "mean_accept_per_sample": statistics.fmean(per_sample_accept) if per_sample_accept else 0.0,
        "mean_accept_per_verify": statistics.fmean(flat_accept) if flat_accept else 0.0,
        "median_accept_per_verify": statistics.median(flat_accept) if flat_accept else 0.0,
        "p90_accept_per_verify": percentile(flat_accept, 0.90),
        "p95_accept_per_verify": percentile(flat_accept, 0.95),
        "min_accept_per_verify": min(flat_accept) if flat_accept else 0,
        "max_accept_per_verify": max(flat_accept) if flat_accept else 0,
        "histogram": {str(k): hist.get(k, 0) for k in range(int(block_size) + 1)},
    }
