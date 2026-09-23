# 1.7b_32b speculative suite, mt-bench (80 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 66.9, 58.4 | 62.38 | 78.7% | 3.56 | 53090 | 851.1 |
| 4 | 1 | 2 | 185.3, 181.5 | 183.40 | 40.7% | 3.57 | 53440 | 291.4 |
| 8 | 1 | 2 | 289.8, 286.6 | 288.19 | 41.0% | 3.53 | 53116 | 184.3 |
| 16 | 1 | 2 | 381.3, 386.1 | 383.68 | 40.6% | 3.54 | 52982 | 138.1 |
