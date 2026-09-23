# dry speculative suite, mt-bench (80 prompts)

max_new_tokens=32, warmup=16 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 1 | 2 | 187.3, 181.0 | 184.07 | 48.2% | 3.93 | 4996 | 27.1 |
