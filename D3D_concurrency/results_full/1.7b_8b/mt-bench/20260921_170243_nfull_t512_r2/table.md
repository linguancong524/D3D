# 1.7b_8b speculative suite, mt-bench (80 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 3 | 2 | 97.5, 98.4 | 97.97 | 66.2% | 3.73 | 52440 | 535.3 |
| 4 | 1 | 2 | 196.7, 220.7 | 208.01 | 43.7% | 3.77 | 52702 | 253.4 |
| 8 | 1 | 2 | 357.1, 346.3 | 351.63 | 43.0% | 3.74 | 52444 | 149.1 |
| 16 | 1 | 2 | 409.6, 419.7 | 414.61 | 42.7% | 3.72 | 51760 | 124.8 |
