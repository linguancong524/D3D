#!/usr/bin/env python3
"""Print the Qwen3-8B AR concurrency baseline as a plain terminal table.

Data source: D3D_concurrency/results_baseline/summary_n32_t512.md
  run 20260921_081947_n32_t512_r2 plus the user-requested MT-Bench C=1
  confirmation run 20260921_093838_mtbench_c1_confirm.

Each cell is the pooled actual emitted output tokens / measured wall time.
"""
from __future__ import annotations

BENCHMARKS = ["math500", "humaneval", "mt-bench"]
CONCURRENCY = [1, 4, 8, 16]
TPS = {
    "math500": [39.68, 140.96, 263.18, 465.39],
    "humaneval": [39.33, 159.51, 260.16, 486.32],
    "mt-bench": [41.74, 152.71, 278.63, 436.36],
}


def main() -> None:
    headers = ["benchmark"] + [f"concurrency_{c}_tps" for c in CONCURRENCY]
    rows = [[name] + [f"{value:.2f}" for value in TPS[name]] for name in BENCHMARKS]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def render(row: list[str]) -> str:
        cells = [row[0].ljust(widths[0])] + [cell.rjust(widths[i]) for i, cell in enumerate(row[1:], start=1)]
        return "  ".join(cells)

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


if __name__ == "__main__":
    main()
