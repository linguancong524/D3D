# D3DV2 concurrency throughput: SDAR-1.7B → Qwen3-8B

Each cell reports aggregate output throughput in tok/s. No AR baseline or speedup comparison.

| Concurrency | Math500 | HumanEval | MT-Bench |
|---:|---:|---:|---:|
| 1 | 160.6 | 139.1 | 80.2 |
| 8 | 317.1 | 275.3 | 167.4 |
| 16 | 348.4 | 297.8 | 178.2 |
| 32 | 363.5 | 311.7 | 183.3 |

Initial measurement: first 64 prompts per dataset.
Hardware: NVIDIA H800, NVIDIA H800; target on GPU0, draft on GPU1; BF16.
Greedy; thinking disabled; block size 16; anchor count 3; draft microbatch 4.
Maximum output length: 256 tokens; ignore EOS: False; repeats per cell: 1.
MT-Bench uses the first turn, matching the existing D3DV2 loader.
TPS = actual emitted output tokens / full generation wall time, including prefill and final drain; loading, warmup, tokenization and file output excluded.
Concurrency is an admission cap; at most C requests are active. Completed requests are replaced at the next scheduler iteration.
This is a custom SSD-derived engine with SDAR packed attention, not vLLM.

Measured workload:

| Dataset | Requests | Output tokens (C=1) |
|---|---:|---:|
| Math500 | 64 | 15926 |
| HumanEval | 64 | 15922 |
| MT-Bench | 64 | 12620 |

Raw measurements: `table_1_7b_8b.json`.
