# D3DV2 concurrency throughput: SDAR-1.7B → Qwen3-8B

Each cell reports aggregate output throughput in tok/s. No AR baseline or speedup comparison.

| Concurrency | Math500 |
|---:|---:|
| 1 | 4.3 |
| 2 | 5.5 |

Initial measurement: first 2 prompts per dataset.
Hardware: NVIDIA H800, NVIDIA H800; target on GPU0, draft on GPU1; BF16.
Greedy; thinking disabled; block size 16; anchor count 3; draft microbatch 4.
Maximum output length: 8 tokens; ignore EOS: False; repeats per cell: 1.
MT-Bench uses the first turn, matching the existing D3DV2 loader.
TPS = actual emitted output tokens / full generation wall time, including prefill and final drain; loading, warmup, tokenization and file output excluded.
Repeated measurements are pooled: sum(output tokens) / sum(elapsed seconds).
Concurrency is an admission cap; at most C requests are active. Completed requests are replaced at the next scheduler iteration.
This is a custom SSD-derived engine with SDAR packed attention, not vLLM.

Measured workload:

| Dataset | Requests | Output tokens (C=1) |
|---|---:|---:|
| Math500 | 2 | 16 |

Raw measurements: `benchmark.json`.
