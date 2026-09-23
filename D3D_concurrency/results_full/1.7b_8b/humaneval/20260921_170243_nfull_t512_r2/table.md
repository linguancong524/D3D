# 1.7b_8b speculative suite, humaneval (164 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 3 | 2 | 191.7, 186.9 | 189.26 | 82.3% | 6.49 | 137662 | 727.4 |
| 4 | 1 | 2 | 459.3, 442.8 | 450.91 | 59.9% | 6.42 | 137374 | 304.7 |
| 8 | 1 | 2 | 607.3, 611.0 | 609.15 | 59.6% | 6.41 | 137616 | 225.9 |
| 16 | 1 | 2 | 685.6, 682.8 | 684.22 | 59.5% | 6.39 | 138054 | 201.8 |
