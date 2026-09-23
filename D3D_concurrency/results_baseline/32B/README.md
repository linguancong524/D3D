# Qwen3-32B AR decoding baseline (n=32 subset)

Same harness and protocol as `../8B/README.md`: greedy BF16, thinking disabled,
natural EOS, 512-token cap, first 32 prompts per benchmark (MT-Bench first turn
only), 256-token full-workload warmup discarded, two attempts per cell with one
automatic retest when the slower attempt falls below 85% of the faster one.
Single H800, `--pages 96`, `--skip-reference-validation` (only one card is
visible in this container, so no independent HF greedy check was run and
`validation.json` is empty in every run directory).

## Runs

* `20260921_165410_32b_n32_t512_r2` — measured math500, humaneval and mt-bench
  sequentially, 2026-09-21 16:54Z -> 19:42Z, 27 cells, no failures. Raw
  artifacts for all three benchmarks live in `<benchmark>/<run-id>/`.
* `20260922_024600_32b_n32_t512_r2_rerun` — user-cancelled during the math500
  C=1 warmup; no cell was measured, `benchmark.json` is marked `aborted`.
* `20260922_030700_32b_n32_t512_r2_rerun2` — remeasured math500 and humaneval
  after the rerun above; 20 cells, no failures. Referred to below as `runB`
  (`runA` is `20260921_165410_32b_n32_t512_r2`).
* `20260922_061000_32b_n40_t512_r2` — math500 on 40 prompts (not the n=32
  subset), still running; excluded from every table below.

## Tables

`summary_n32_t512_measured.csv` holds the highest single-attempt TPS reached in
each <benchmark, concurrency> cell across all completed n=32/t=512 32B runs
(`runA` plus `runB`; the aborted and the n=40 runs are excluded). It is a
maxima table, not the pooled convention used by `../8B`, and every row carries
the `run/attempt` each value came from in `tps_source`. For reference, the
pooled values it replaced were math500 17.78 / 62.54 / 123.72 / 236.92,
humaneval 17.89 / 37.64 / 110.27 / 183.37, mt-bench 18.92 / 97.63 / 176.55 /
278.78 (C=1/4/8/16). Per-cell repeatability from the raw attempts:

| Benchmark | C=1 | C=4 | C=8 | C=16 |
|---|---:|---:|---:|---:|
| math500 | 17.06 / 18.56 | 62.87 / 62.20 | 132.97 / 115.67 | 246.81 / 227.79 |
| humaneval | 23.62 / 15.54 / 16.39 | 28.62 / 34.66 / 62.89 | 107.76 / 112.90 | 186.40 / 180.43 |
| mt-bench | 13.07 / 23.14 / 25.78 | 97.24 / 98.02 | 174.71 / 178.42 | 281.60 / 276.02 |

Cell trustworthiness, from those repeats: the high-concurrency cells are
repeatable to 1-7% (math500 C=4, mt-bench C=4/C=8/C=16) and humaneval C=1/C=4
are not (2.2x spread at C=4, which also breaks C=1 -> C=4 scaling: 2.1x versus
3.5x for math500 and 4.2x for mt-bench). SM clock stayed at 1980 MHz and
temperatures stayed at 31-39 C throughout, so this is host-side scheduling
drift on a shared 224-core node, not thermal or power throttling; it hits the
low-concurrency cells because the engine is driven step-by-step from Python.

`summary_n32_t512.csv` is **not** a measurement table. At the user's request it
now holds the quiet-node throughput estimate of every cell, derived in
`## Quiet-node estimate` below; `tps_source` marks the rows as estimated and
not measured. It supersedes the earlier version of this file, which copied the
mt-bench row into the math500 and humaneval rows verbatim. Use
`summary_n32_t512_measured.csv` for the values that were actually observed,
keeping in mind that it selects per-cell maxima and therefore mixes the most
favourable window of each cell: its four math500 and humaneval C=1/C=4 cells
are single attempts from drifting low-concurrency cells (up to 2.2x spread at
humaneval C=4), i.e. best-case rather than typical throughput.

## Quiet-node estimate

The estimator is `TPS = mean_batch / step_time`, because the harness flushes
one token per running sequence per decode step and `benchmark.json` records
`decode_batch_sizes` for every step.

* `mean_batch` is deterministic and workload-intrinsic. Greedy decoding on
  fixed prompts reproduces the same output lengths, the same per-step batch
  sizes and the same step count in every repeat, across runs: humaneval C=16 is
  905 steps in both `runA` and `runB`, math500 C=16 is 1017, C=1 is 10428 and
  13276. A FIFO replay of the recorded length distribution reproduces those
  counts to 0.1-0.4% (905 vs 907, 1017 vs 1020, 811 vs 814), so the engine is
  not leaving slots idle for any reason other than the fill/drain of a finite
  request set. That is why `mean_batch` is below C at high concurrency and
  tracks mean output length (416 math500, 327 humaneval, 281 mt-bench):
  longer generations amortise the ramp better.
* `step_time = 39.6 ms` is the quiet-node value. The only sustained quiet
  window in this directory is mt-bench inside `runA`, where all seven
  non-stalled cells read 38.9, 39.0, 39.3, 39.3, 39.6, 40.1 and 40.4 ms
  (mean 39.5 ms, 38.9-40.4 = 3.7% spread, flat from C=1 to C=16 even though
  the batch grows 11x). Flattening against batch size is the signature of a
  step whose cost is a fixed pass over the 65.8 GB of weights plus fixed
  per-step overhead, not of GPU compute: 65.8 GB / 3.35 TB/s = 19.6 ms, so the
  quiet window runs at 2.0x the pure weight-read bound, which is what eager
  PyTorch with per-layer launches and one pipe round trip per step should
  cost.
* Both inputs are therefore independent of the container's contention level:
  the load-dependent part is entirely inside `step_time`. The estimate is a
  ceiling, not a typical value. The same cells measured 52.4-65.2 ms (math500
  C=16 in `runA`), 66.3 ms (math500 C=16 in `runB`, internally repeatable to
  0.2%), 89-122 ms (humaneval C=16 in `runB`) and 97.7 ms (math500 C=1 in the
  in-flight n=40 run), so budget 0.4-1.0x of this table on a contended node.
* Cross-dataset physics is deliberately not modelled: prefill adds 1.5-2.4 ms
  per step and context-length KV reads add ~1 ms per step at C=16, i.e. 1-4%
  of a step, which is far smaller than the window-to-window spread above.
* Sensitivity: the table scales as 1/step_time, so assuming 44 ms or 50 ms
  instead of 39.6 ms multiplies every value by 0.90 or 0.79.

## Reproducing

The documented single-card command is
`RUN_ID=<id> bash scripts/run_32b_baseline_seq.sh [datasets...]`, a serial
variant of the multi-card command in `../README.md` that loads the 62 GB
verifier once per dataset instead of three times at once. Three parallel
datasets on one card OOM during model load.
