#!/usr/bin/env python3
"""Extract the project-standard serving metrics from sglang.benchmark.serving logs.

Usage:
    python3 docs/perf/scripts/extract_bench_metrics.py docs/perf/logs/bench_serving_*.log

Prints one TSV row per log file with the fields the perf reports tabulate:
throughput (req/s, output tok/s, peak out tok/s), concurrency, accept length,
latency (E2E mean/median, TTFT mean/median, TPOT mean, ITL mean, max ITL).

Project rule: benchmark evidence lives in docs/perf/; this helper only reads it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FIELDS = [
    ("Successful requests", "ok"),
    ("Request throughput (req/s)", "req_s"),
    ("Output token throughput (tok/s)", "out_tok_s"),
    ("Peak output token throughput (tok/s)", "peak_out_tok_s"),
    ("Concurrency", "conc"),
    ("Peak concurrent requests", "peak_conc"),
    ("Accept length", "accept"),
    ("Mean E2E Latency (ms)", "e2e_mean"),
    ("Median E2E Latency (ms)", "e2e_50"),
    ("Mean TTFT (ms)", "ttft_mean"),
    ("Median TTFT (ms)", "ttft_50"),
    ("Mean TPOT (ms)", "tpot_mean"),
    ("Median TPOT (ms)", "tpot_50"),
    ("Mean ITL (ms)", "itl_mean"),
    ("Max ITL (ms)", "itl_max"),
]

NUM = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row: dict[str, str] = {"file": path.name}
    for label, key in FIELDS:
        m = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
        if not m:
            row[key] = "-"
            continue
        n = NUM.search(m.group(1))
        row[key] = n.group(1) if n else m.group(1).strip()
    return row


def main(argv: list[str]) -> int:
    paths: list[Path] = []
    for arg in argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("bench_serving_*.log")))
        else:
            paths.append(p)
    if not paths:
        print(__doc__)
        return 2
    keys = [k for _, k in FIELDS]
    print("\t".join(["file", *keys]))
    for path in paths:
        if not path.exists():
            print(f"# missing: {path}", file=sys.stderr)
            continue
        row = parse(path)
        print("\t".join([row["file"], *(row[k] for k in keys)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
