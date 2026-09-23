# 1.7b_32b speculative suite, math500 (500 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 149.5, 145.9 | 147.69 | 93.2% | 7.60 | 426018 | 2884.5 |
| 4 | 1 | 2 | 382.5, 420.2 | 400.46 | 64.0% | 7.63 | 425658 | 1062.9 |
| 8 | 1 | 2 | 654.9, 701.6 | 677.46 | 64.4% | 7.59 | 426238 | 629.2 |
| 16 | 1 | 2 | 786.2, 795.3 | 790.71 | 64.1% | 7.60 | 425998 | 538.8 |
