# micro=C smoke (anchor=1, micro batch = concurrency, 32 new tokens, 16 requests)

| Concurrency | Verify median ms | Verify p90 | Draft median ms | Draft p90 | Round median ms | Exposed draft mean ms | Hidden rounds | Recovery forwards | TPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 27.10 | 42.41 | 32.11 | 36.76 | 35.94 | 4.36 | 32.0% | 37 | 171.82 |

## Injected draft-stream delay vs TPS

| Injected ms | C=4 | C=8 | C=16 |
|---:|---:|---:|---:|
| 0 | 214.2 |
