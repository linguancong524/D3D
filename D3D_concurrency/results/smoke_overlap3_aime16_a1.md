# draft verify vs drafter smoke v3 interleaved (anchor=1, micro = concurrency, 512 new tokens, 4096x4096 fp32 matmul on cuda:1, 2.645 ms/op)

Per-cell values are the mean over interleaved repeats; TPS range shows host drift.

| C | Mode | Injected ms | TPS (range) | Verify mean ms | Draft CPU mean ms | Draft GPU mean ms | Round mean ms | Recover ms/round | Hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | measured | 0 | 360.7 (331.7-409.2) | 36.25 | 38.54 | 38.49 | 43.25 | 21.90 | 52.9% |
| 4 | forced | 0 | 398.0 (331.9-439.0) | 31.60 | 34.57 | 34.50 | 36.59 | 23.03 | 52.9% |
| 4 | inject | 40 | 224.4 (209.7-249.1) | 42.68 | 36.50 | 75.81 | 44.58 | 29.09 | 52.9% |
| 8 | measured | 0 | 463.3 (310.9-549.7) | 40.89 | 57.95 | 57.80 | 60.65 | 22.78 | 53.4% |
| 8 | forced | 0 | 538.3 (514.4-552.1) | 34.77 | 49.62 | 49.57 | 50.45 | 19.45 | 53.4% |
| 8 | inject | 40 | 304.6 (275.8-357.6) | 51.06 | 54.16 | 93.37 | 64.59 | 24.61 | 53.4% |
| 16 | measured | 0 | 626.3 (599.4-649.3) | 38.50 | 82.26 | 82.21 | 83.89 | 20.88 | 54.8% |
| 16 | forced | 0 | 650.3 (627.3-668.7) | 38.01 | 80.39 | 80.33 | 81.58 | 19.32 | 54.8% |
| 16 | inject | 40 | 432.8 (345.1-478.2) | 40.06 | 79.54 | 118.81 | 83.16 | 35.57 | 54.8% |
