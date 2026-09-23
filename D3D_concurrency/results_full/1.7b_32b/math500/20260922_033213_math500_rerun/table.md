# 1.7b_32b speculative suite, math500 (500 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Decode-only TPS | Hit rate | Mean accept | Prefill s | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 140.9, 151.3 | 145.94 | 148.4 | 93.2% | 7.60 | 47.9 | 2919.2 |
| 4 | 1 | 3 | 411.3, 250.4, 366.9 | 327.85 | 344.3 | 64.0% | 7.63 | 93.1 | 1947.5 |
