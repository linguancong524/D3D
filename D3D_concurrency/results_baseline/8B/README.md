# Qwen3-8B AR decoding baseline

## End-to-end concurrency baseline (2026-09-21)

The active suite uses the first **32 prompts per dataset**, per the user's time
budget. Launch with `bash scripts/baseline_suite.sh`; concurrency 1/4/8/16,
512-token cap, two repeats, and at most one automatic outlier retest. These
are subset throughput measurements, not full-dataset results. The initial
full-dataset launch `20260921_081207_full` was cancelled during startup.
`MAX_SAMPLES=64 bash scripts/baseline_suite.sh` overrides the subset size.
Suite driver logs are stored inside each dataset folder.

```bash
bash scripts/baseline_concurrency.sh --concurrency 1 4 8 16 \
  --datasets math500 humaneval mt-bench --max-new-tokens 2048 --repeats 3
```

Results live in `<dataset>/<run-id>`. The standalone command above uses full
datasets; the active suite uses 32 prompts. Greedy BF16, thinking
disabled, natural EOS, MT-Bench first turn only. A single H800 runs the target.
The persistent target process and paged backend match the speculative engine.
Admission is capped at C requests, with refill after completion and prefill
microbatches of at most four. Each concurrency is warmed with the complete 32
request workload and a 256-token cap; all warmup output and time are discarded.
Timing includes prefill, decode, drain and cleanup;
model loading, tokenization, independent HF validation and warmup are excluded.
First output and EOS tokens count toward actual emitted output tokens.

Each cell receives a full-workload 256-token warmup and two measurements. If the
slower attempt is below 85% of the faster attempt's TPS, one additional full
measurement is run automatically with a fresh warmup. Comparisons are within one dataset
and concurrency only. This detects relative outliers, not uniform slowdowns.
All original and repeated attempts remain in the median and pooled TPS; no
fastest-run selection. Persistent variability remains flagged in the summary.

`inputs.json` stores tokenized prompts, `validation.json` stores short independent
HF greedy comparisons, `c*_attempt*.json` stores raw per-request outputs,
latencies, decode batch sizes, memory and GPU snapshots. `benchmark.json` records
configuration, input/source hashes, attempt summaries and completion status.
`table.md` is updated after each concurrency, and `run.log` records progress.
Request latency starts at admission, excluding time waiting for admission.
Sample smoke runs have a `smoke_` run-id and must not be used as full results.
Nonidentical short HF sequences additionally require every actual-path token to
be an independent HF teacher-forced argmax (zero logit regret, allowing exact
BF16 ties). Differences and logits regrets are recorded, not hidden. This is a
short numerical sanity check, not full-dataset accuracy certification.

## Historical decode-only baseline

Run from D3D_concurrency:

```bash
bash scripts/baseline_ar.sh --datasets aime25 --batch-sizes 1 4 8 16 32 \
  --max-new-tokens 2048 --max-samples 32 --repeats 2 \
  --output results_baseline/qwen3_8b_aime25_decode.json
```

Uses the production SSD-derived PyTorch target process and paged backend on
one H800, BF16, greedy, thinking disabled. No draft model. This is not vLLM.
Neural Backup remains the selected speculative policy.

Fixed batch decoding: EOS ignored, 2048 total output tokens per request.
Prefill produces the first token; both its time and token are excluded.
The remaining 2047 steps are timed, including CPU scheduling and IPC.
Model loading, warmup, prefill, cache release and serialization are excluded.
The model stays loaded across warmup and measurements. Each batch size is
warmed independently. Two repeats are pooled by total tokens / total time.

AIME25 has 30 prompts. The first two are repeated to make 32 inputs, identical
across batch sizes. JSON records the exact source indices and input hashes.
An independent HF greedy short-output check precedes throughput measurement.
JSON records every batch duration and KV free-page count. The same-stem Markdown
file contains the final table. Existing output JSON files are never overwritten.

Do not divide the existing Neural end-to-end TPS by these decode-only numbers:
that benchmark includes prefill, natural EOS and dynamic refill. A speedup needs
a matching Neural workload and timing definition. AR uses one target GPU;
Neural additionally uses a draft GPU, which must be disclosed with the ratio.
Qwen3-32B is not measured here.
