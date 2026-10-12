# H800 validation — 2026-10-10

## Scope

Validation used two NVIDIA H800 80 GB GPUs, with target on `cuda:0` and
draft on `cuda:1`. The existing environment provides Python 3.11.13,
PyTorch 2.6.0+cu124, Transformers 4.52.4, datasets 3.6.0,
FlashAttention 2.7.4.post1, and Triton 3.2.0. This was not a clean
Python 3.10 installation test.

The target was the existing local Qwen3-8B checkpoint, and the draft was
a local 1.7B D3D training checkpoint used by the original experiment
scripts. These files have not been matched against a public `lingc8`
snapshot; success here does not validate any published draft ID.
An additional local pair used Qwen3-32B and a local 24-layer pruned D3D
training checkpoint selected by the original 32B experiment script. This
pair is likewise not matched to a public snapshot.
Existing MATH-500/HumanEval JSONL caches were supplied through
`D3D_DATA_CACHE` and `D3D_ANSWER_CACHE`. Neither weights nor datasets were
added to the source export.

## Fixed issues

- With optional Liger installed, single-request draft prefill initially
  failed with `Pointer argument (at 0) cannot be accessed from Triton`.
  The draft tensor was on `cuda:1`, while the calling thread's current
  CUDA device was `cuda:0`. D3D's MLP now enters the tensor's CUDA device
  context before calling Liger and uses its PyTorch path for CPU inputs.
  Regression tests cover both cases and current-device restoration.
- Both benchmark shell scripts previously hardcoded 2048 output tokens,
  ignoring the handoff's `D3D_MAX_NEW_TOKENS=128` smoke setting. They now
  honor the override and retain 2048 as the default. A shell-entry-point
  test checks both defaults and overrides from a different working directory.

## Completed inference checks

- All 19 unit tests passed, including the two new MLP regressions and the
  benchmark-script token-limit regression. Python 3.10 AST parsing of all
  36 Python files, shell syntax checks, and SVG XML parsing also passed.
- `scripts/quickstart_single.sh`: one MATH-500 request, 128 emitted tokens,
  bfloat16, greedy decoding; result and summary JSON written successfully.
- `scripts/quickstart_concurrent.sh`: four MATH-500 requests at concurrency 4,
  128 tokens each, 512 KV pages per device; all four requests active together.
- `serving/bench.py --validate`: four independent target-reference cases
  matched all 32 greedy token IDs exactly. Draft recovery matched, and all
  736 packed token predictions across 46 candidate keys matched the D3D
  reference. Subsequent 128-token runs at concurrency 1 and 4 completed.
- `scripts/benchmark.sh` smoke configuration: two requests each from
  MATH-500 and HumanEval, with the 128-token override recorded in both outputs.
- `scripts/benchmark_concurrent.sh` smoke configuration: 16 MATH-500 requests,
  a 128-token cap, 512 pages, and concurrency 1/4/8/16. All four cells
  completed, each reaching its requested active-request count.
- Local Qwen3-32B plus the 24-layer draft: single-request quick start emitted
  128 tokens; concurrent quick start completed four 128-token requests at
  concurrency 4 with `D3D_PAGES=128`. Reported peak allocated memory was
  74.22 GB on target and 7.09 GB on draft (decimal GB, allocator statistics,
  not total process memory). No simultaneous duplicate-model `--validate`
  was used for this pair. This small run does not establish a safe page
  configuration for 2048 tokens or concurrency 16.

Detailed local artifacts, including logs, configurations, package versions,
source SHA-256 values, acceptance data, and actual output token counts, are
under the git-ignored `outputs/gpu_validation/` directory:

- `single_local_8b_1p7b/result.json` and `summary.json`
- `concurrent_local_8b_1p7b.json`
- `validated_local_8b_1p7b.json`
- `single_smoke/math500.json` and `single_smoke/humaneval.json`
- `concurrent_smoke.json`
- `single_local_32b_d24/result.json` and `summary.json`
- `concurrent_local_32b_d24.json`

These are short local smoke runs, not the formal seven-dataset, 32-sample,
2048-token benchmark. Single-request decode rates and concurrent end-to-end
rates have different timing boundaries and must not be compared directly.
No performance table was added to the README.

## Remaining release gates

- Anonymous Hugging Face metadata requests failed with
  `Network is unreachable` from this execution environment. This is a
  connectivity failure, not evidence that the model repositories are
  missing or private. Anonymous fresh-cache downloads remain unverified.
- The local checkpoint metadata has no resolved Hub revision, and the local
  dataset caches have no Hub revision/fingerprint. The output records a
  hash of selected dataset rows, but it does not establish their upstream
  revision. An independent source commit is also unavailable before the
  standalone repository is initialized. Do not relabel these as pinned
  public-snapshot runs.
- All six advertised public-snapshot pairs still require anonymous download
  and snapshot-identity checks. Pairs beyond the two local smoke pairs,
  32B greedy-reference checks, larger 32B memory/page configurations,
  clean installation, and formal benchmarks remain unverified.
- Author/affiliation placeholders, copyright ownership, and redistribution
  permission for D3D-specific changes still require owner confirmation.
- No independent Git repository, remote, commit, or push was created.
- All inference processes started for these checks exited, and both GPUs
  returned to approximately 1 MiB used after validation.
