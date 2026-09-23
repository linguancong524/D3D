# AIME25 C=1 performance diagnosis

Follow-up: the production process restoration and full AIME measurements are recorded in
[PROCESS_RESTORED.md](PROCESS_RESTORED.md). The diagnosis below describes the preceding threaded version.

2026-09-20. SDAR 1.7B (28 layers) → Qwen3-8B, two H800 GPUs, BF16,
greedy, block 16, anchor 3, all acceptance positions, 30 AIME25 prompts,
max_new_tokens=2048, natural EOS, thinking disabled. All prompt strings and
ordering match the historical run. Original D3DV2 code/results were not edited.

## Matched full-dataset measurements

| Implementation | Output tokens | Decode TPS | Overall TPS | Mean accepted tokens/step |
|---|---:|---:|---:|---:|
| Historical D3DV2, 2026_09_14_08_23_47 | 51,858 | 201.23 | 191.77 | 6.464 |
| Original D3DV2 rerun on current machine | 51,858 | 197.53 | 188.20 | 6.464 |
| Current D3D_concurrency, threaded | 54,915 | not separately recorded | 114.23 | 6.532 |

Current overall throughput is 39.3% below the same-machine original rerun.
Original rerun reproduces all 30 historical token sequences exactly. Current
backend generates different continuations; no full-sequence identity or AIME
accuracy claim is made. Mean accepted length did not decrease, but content and
prefix lengths differ, so use the fixed-workload experiments below for attribution.
Current mean acceptance records requested accepted length before final cap/EOS
clipping; its final partial steps are not precisely identical to the old metric.

Original overall TPS includes tokenization, result processing and JSON output;
current overall TPS includes prefill but excludes model load, warmup, tokenization
and result serialization. Thus these overall metrics are close, not strictly
identical; the new timing boundary is more favorable, and cannot explain its
large slowdown. Historical decode-only vs overall differs by about 4.7%.

CPU settings matter: the first original rerun with default thread settings was
stopped after anomalous latency. Its partial artifacts have the
`original_default_threads_partial` prefix and are NOT used as a full benchmark.
The completed rerun explicitly sets OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4;
the current backend sets torch CPU threads to 4.

## Main finding: ineffective thread overlap

Fixed real AIME token prefixes; target query 16 and packed draft query 752
(46 candidate branches of 16 plus 16 pending tokens). Median wall latency in ms,
20 repetitions after warmup. Separate from throughput measurements.

| Prefix length | Target alone (thread experiment) | Draft alone | Thread overlap | Process overlap |
|---|---:|---:|---:|---:|
| 256 | 25.18 | 25.32 | 46.94 | 29.40 |
| 1024 | 30.42 | 31.87 | 59.62 | 24.60 |
| 2048 | 32.14 | 31.08 | 52.23 | 26.85 |

The process control uses the SAME new Runner/model/attention implementation,
with target in a spawned worker and draft in the parent. Process IPC is included.
Within the process experiment, target-alone times are 23.81/23.55/22.54 ms;
draft-alone times are 30.07/26.44/26.71 ms. Processes recover effective overlap.
Independent runs have CPU scheduling/frequency variation; do not treat the
cross-run difference as an exact additive cost or infer full-run TPS from it.

The current ThreadPoolExecutor architecture is a substantial bottleneck.
Many Python-dispatched eager kernels and shared-process dispatch contention are
consistent with the observations. GIL time itself was not directly measured;
it would be too strong to attribute every lost millisecond specifically to GIL.
Original D3DV2 already isolates verifier in a process. Replacing that with
threads in the concurrency prototype was a performance regression.

## FlashAttention is not supported as the cause

Target-only kernel experiment: B=1, Q=16, H=32, KVH=8, D=128, BF16,
bottom-right causal visibility, 100 repetitions, median wall ms. Dense SDPA
inputs and expanded KV heads are prepared outside timing; paged FlashAttention
uses an already populated cache. Cache writes/model/scheduler are not timed.

| Prefix | Paged FlashAttention | Dense SDPA |
|---|---:|---:|
| 256 | 0.0347 | 0.0656 |
| 1024 | 0.0391 | 0.0924 |
| 2048 | 0.0431 | 0.1352 |

FlashAttention is faster in these tested shapes. This is not a whole-model
attention-backend swap benchmark. Original `sdpa` is a dispatch interface, not
necessarily naive attention. New target uses FlashAttention; packed draft still
uses SDPA because it needs its custom branch mask.

## Secondary costs

For an identical 736-position draft hidden tensor, vocabulary projection plus
argmax and topk takes about 2.10 ms in chunks of 256 versus 1.68 ms in one chunk:
about 0.42 ms per packed step. This is real but far smaller than the overlap loss.
The production implementation also projects the 16 pending positions; original
projects only the candidate suffix. Do not remove argmax in favor of topk[0]:
BF16 ties can produce different greedy tokens.

Draft cache gather/scatter, repeat of grouped KV heads and eager norms/RoPE add
work. Profiling shows many kernel launches and copies, but no isolated old/new
cache-layout ablation was performed, so their exact share is not established.
Torch compile is disabled in the current run; its contribution has not been
isolated. Kernel tables in profile.json are diagnostic profiler output (including
profiler overhead), not directly additive throughput accounting.

## Recommended repair sequence

1. Restore a persistent target worker process, keeping the new batched Runner,
   page allocator and strict concurrency scheduler; use this same architecture
   for all C=1/8/16/32. Do not special-case C=1 by switching to original D3DV2.
2. Validate token outputs against the current backend at C=1, then exercise
   batched admission, rollback/release and recovery at C=8/16/32. Re-run full
   matched AIME TPS before claiming a restored baseline.
3. Separately optimize vectorized branch construction, candidate-only vocabulary
   projection and bounded chunk size. Profile paged draft KV and eager kernels
   before investing in a new attention kernel.
4. Regenerate the complete concurrency table only after the baseline is sound.

No production inference optimization was applied in this diagnosis; the process
control is a standalone microbenchmark, not a ready production implementation.

## Artifacts and reproduction

- `current_c1.json`: complete unchanged new backend measurement with source hashes.
- `original_rerun.json`, `original_rerun.summary.json`: complete original rerun.
- `profile.json`: fixed-state threaded latencies, head and attention comparisons.
- `process_profile.json`: independent process overlap control.
- Corresponding `.log` files retain execution output.
- Historical source: `D3DV2/results_pruning/qwen3_p1.7b_to_8b_d24_d28/raw/d28/2026_09_14_08_23_47/aime25/`.

From the T3D workspace, run serially on idle GPUs:

```bash
D3D_concurrency/run.sh --datasets aime25 --max-samples 0 --concurrency 1 --max-new-tokens 2048 --output results/aime_diagnosis/current_c1.json
bash D3D_concurrency/run_original_aime.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python D3D_concurrency/profile_c1.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python D3D_concurrency/profile_process_c1.py
```

These commands overwrite the named diagnostic outputs if repeated.
