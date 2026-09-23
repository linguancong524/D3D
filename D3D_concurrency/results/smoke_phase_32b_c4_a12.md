# draft verify vs drafter smoke v2 (anchor=1, micro = concurrency, 512 new tokens, 4096x4096 fp32 matmul on cuda:1, 2.627 ms/op)

| C | Mode | Injected ms | TPS | Verify mean ms | Draft CPU mean ms | Draft GPU mean ms | Round mean ms | Recover n | Recover mean ms | Hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | measured | 0 | 376.1 | 37.93 | 31.91 | 31.87 | 38.26 | 381 | 17.73 | 54.7% |
| 4 | measured | 0 | 334.9 | 39.67 | 48.81 | 48.77 | 49.65 | 382 | 14.06 | 70.8% |
