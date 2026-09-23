# micro=C smoke (anchor=1, micro batch = concurrency, 512 new tokens, 16 requests)

| Concurrency | Verify median ms | Verify p90 | Draft median ms | Draft p90 | Round median ms | Exposed draft mean ms | Hidden rounds | Recovery forwards | TPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 27.03 | 37.36 | 32.41 | 39.83 | 33.06 | 6.37 | 15.8% | 349 | 420.66 |
| 8 | 29.82 | 40.45 | 53.36 | 58.49 | 53.67 | 19.15 | 8.2% | 218 | 504.72 |
| 16 | 31.43 | 41.78 | 100.68 | 109.75 | 101.14 | 47.13 | 8.2% | 126 | 645.70 |

## Injected draft-stream delay vs TPS

| Injected ms | C=4 | C=8 | C=16 |
|---:|---:|---:|---:|
| 0 | 421.5 | 522.7 | 679.4 |
| 10 | 408.0 | 564.2 | 686.4 |
| 20 | 437.8 | 570.6 | 691.7 |
| 40 | 456.6 | 560.4 | 668.3 |
