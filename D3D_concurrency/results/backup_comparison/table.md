# Fast vs Neural Backup (SDAR adaptation)

Datasets: ['math500']; first 64 prompts; max output 256; 2 repeats. Natural EOS; greedy; 2 H800; SDAR-1.7B → Qwen3-8B.
One model load; order neural→fast then fast→neural. Same prompts, caches, 46 branches, draft microbatch 4. Timing includes prefill and all recovery/seed work.

| Dataset | C | Neural TPS | Fast TPS | Fast/Neural | Neural tokens/step | Fast tokens/step |
|---|---:|---:|---:|---:|---:|---:|
| math500 | 1 | 239.4 | 220.0 | 0.919× | 7.35 | 6.71 |
| math500 | 8 | 345.5 | 317.7 | 0.920× | 7.34 | 6.70 |
| math500 | 16 | 361.3 | 330.3 | 0.914× | 7.41 | 6.64 |
| math500 | 32 | 370.9 | 335.1 | 0.904× | 7.33 | 6.64 |

TPS pooled over repetitions. Ratios compare backup policies, not an AR baseline.
Fast is a greedy SDAR adaptation: random in-vocabulary suffixes are returned on misses; next-outcome masked anchor seeding runs after target submission. Initial admission uses the same neural bootstrap in both policies.
SSD reference uses AR glue-decode to build next-outcome predictions and random-logit argmax for miss tokens; this code uses SDAR masked seeding and uniform integer random tokens. It does not reproduce SSD AR kernels, miss-specific fanout or its full scheduler.
Exact output agreement and acceptance/miss statistics are in summary.json. BF16 is not promised bitwise identical across different speculative trajectories. These are capped subset performance measurements, not a full accuracy evaluation.
