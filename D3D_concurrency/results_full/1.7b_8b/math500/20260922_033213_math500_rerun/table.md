# 1.7b_8b speculative suite, math500 (500 prompts)

max_new_tokens=512, warmup=256 discarded, repeats=2 + up to 1 retest (slower < 0.85 x faster), draft microbatch = concurrency, backup=neural.

| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Decode-only TPS | Hit rate | Mean accept | Prefill s | Wall s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 2 | 237.9, 247.5 | 242.59 | 246.3 | 84.6% | 7.58 | 26.8 | 1797.5 |
| 4 | 1 | 2 | 581.0, 579.1 | 580.06 | 604.7 | 63.7% | 7.59 | 30.7 | 752.4 |
| 8 | 1 | 2 | 742.3, 739.2 | 740.76 | 791.2 | 63.6% | 7.59 | 37.4 | 587.0 |
| 16 | 1 | 2 | 859.4, 858.1 | 858.73 | 931.2 | 63.7% | 7.60 | 39.4 | 506.7 |
