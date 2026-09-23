# 1.7b_8b speculative suite, math500 (500 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 182.8, 130.2, 185.7 | 161.84 | 84.6% | 7.58 | 654081 | 4041.6 |
| 4 | 1 | 2 | 382.0, 393.4 | 387.61 | 63.7% | 7.59 | 436460 | 1126.0 |
| 8 | 1 | 2 | 532.1, 601.0 | 564.46 | 63.6% | 7.59 | 434836 | 770.4 |
| 16 | 1 | 2 | 590.4, 637.3 | 612.93 | 63.7% | 7.60 | 435138 | 709.9 |
