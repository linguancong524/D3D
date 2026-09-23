# dry2 speculative suite, mt-bench (80 prompts)

max_new_tokens=64, warmup=16 discarded, repeats=1 + up to 0 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Decode-only TPS | Hit rate | Mean accept | Prefill s | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1 | 221.3 | 221.25 | 248.4 | 46.0% | 3.81 | 2.4 | 22.1 |
