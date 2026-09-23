# Speculative suite (1.7B draft pairs, full samples)

Two pairs, same protocol:

* `1.7b_8b/` — Qwen3-8B verifier, `sdar_1_7/.../step-1420908` draft, `pages=512`.
* `1.7b_32b/` — Qwen3-32B verifier, `sdar17b_.../step-1041183` draft, `pages=96`
  (62 GB of bf16 weights plus a 512-page KV pool does not fit one 80 GB card).

Datasets are full samples: math500 500, humaneval 164, mt-bench 80. Concurrency
1/4/8/16, `max_new_tokens=512`, greedy, thinking disabled, natural EOS, backup
neural, draft microbatch = concurrency. Anchor is 3 at C=1 for the 8B pair and 7
at C=1 for the 32B pair, 1 at every other concurrency. Each cell runs a discarded
256-token full-workload warmup, then two measured attempts, with one automatic
retest when the slower attempt falls below 0.85x the faster one.

## CSV convention

`summary.csv` exists at the top level, per pair and per dataset. **The reported
result is the high value**: `tps_best` is the fastest attempt of the cell and the
other columns of that row (hit rate, accept, wall, prefill, peak memory) come
from that same attempt, so the row is internally consistent. `tps_attempts`
lists every attempt and `tps_pooled` is total output tokens over total wall
time, so the spread and the pooled view stay visible.

`origin` says whether the row came from the first pass
(`20260921_170243_nfull_t512_r2`) or from the user-requested math500
re-measurement (`20260922_033213_math500_rerun`). The re-measurement supersedes
the first pass for math500; the first-pass directories are kept untouched.

## Known environment caveat

The first pass ran 2026-09-21 17:02-23:25 UTC while the host was contended (load
70-95 from other tenants). The math500 re-measurement at load ~48 came out 31-50%
faster and is 10x more repeatable (0.2-0.3% between attempts versus 3-8% in the
first pass). Humaneval, mt-bench and the whole 32B pair still carry first-pass
numbers and are flagged in the `origin` column; they should be re-measured
before the numbers are published.
