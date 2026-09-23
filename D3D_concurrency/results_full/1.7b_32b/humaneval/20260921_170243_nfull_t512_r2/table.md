# 1.7b_32b speculative suite, humaneval (164 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Hit rate | Mean accept | Output tokens | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 129.0, 124.2 | 126.56 | 91.2% | 6.38 | 133662 | 1056.1 |
| 4 | 1 | 2 | 313.6, 337.0 | 324.87 | 58.7% | 6.32 | 134076 | 412.7 |
| 8 | 1 | 2 | 611.3, 608.2 | 609.77 | 58.6% | 6.35 | 134044 | 219.8 |
| 16 | 1 | 2 | 658.9, 654.1 | 656.47 | 58.2% | 6.34 | 133716 | 203.7 |
