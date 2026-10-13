# H20 Qwen3-8B decoupled：block 16、每个位点 1 个备选（2026-09-23）

## 测试口径

- 源码：`eed3241bdf412befc83d671f4d11b55dbabb386f`；4× H20，启动前和收尾均确认没有残留 GPU 计算进程。
- GPU0–2：3 个 Qwen3-8B TP=1 Target，各 `max_running_requests=3`；GPU3：hybrid 模式的共享 D3D-1.7B OneStep Draft。local-only 对照使用相同的 3 个 Target，不启动共享 Draft。
- Proposal：`block_size=16`、`accept_counts=1..16`、`num_anchors=1`、`max_branches=16`（前 15 个拒绝位点各 1 个备选，另加 full-accept 分支）。`candidate_topk=4` 是本地候选池大小，不表示每个位点向远端展开 4 个备选。
- 沿用当前栈：Target local prefix reuse、deterministic inference、Draft packed causal pending 与 CUDA Graph；Draft `speculation_batch_size=8`、`batching_window_ms=0`、`mem_fraction_static=0.65`、graph max tokens=1024。Draft 使用 `scripts/decoupled/sdar_block16.yaml`。
- 基准：`sglang.benchmark.decoupled_serving`；random 512 input / 256 output，temperature=0，seed=42。分别跑 n=50/rate=4 和 n=100/request-rate=inf；hybrid 每个工况逐 token 对照同工况 local-only 输出。

## 结果

| 工况 | 模式 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length | token match |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| rate=4, n=50 | local-only | 50/0 | 467.62 | 1.827 | 6.665 / 15.851 / 17.952 | 4.236 | reference |
| rate=4, n=50 | hybrid | 50/0 | 460.18 | 1.798 | 6.790 / 16.193 / 18.284 | 4.236 | 50/50 |
| saturation, n=100 | local-only | 100/0 | 524.12 | 2.047 | 21.648 / 43.373 / 46.115 | 3.575 | reference |
| saturation, n=100 | hybrid | 100/0 | 525.12 | 2.051 | 21.932 / 43.581 / 47.266 | 3.575 | 100/100 |

`accept length = 1 + spec_num_correct_drafts / spec_verify_ct`；rate=4 为 `1+9900/3059`，饱和为 `1+18671/7252`。两种模式在同工况的 verify/correct-draft 计数完全相同。hybrid 相对 local-only 的 output tok/s 为 rate=4 **-1.59%**、饱和 **+0.19%**，均不足以证明稳定提速；两组 hybrid 的 E2E p95 都略高。

远端确实执行了单备选几何：Draft 累计 2358 个 batch item、37728 条分支，恰为 `2358×16`；583 次 physical forward，`error_ct=0`，最终 session 和队列清零。但三张 Target 在两组测试及预热的累计统计中只有 **2 次** `remote_optional_lookup_hit_ct`，几乎全部 verify 仍走 local fallback。这解释了 hybrid 和 local-only 的输出与速度高度接近。Draft 的累计 publish/supersede 为 10431/7935；多数预投机结果未及时用于 Target。

本轮四份 benchmark 均 `summary.valid=true`，hybrid 严格 token match 为 50/50、100/100，日志无未捕获 Python traceback 或致命 CUDA OOM。rate=4 已超过这套三 Target 的完成能力，p95 包含明显排队；饱和工况更适合看容量。各工况只跑一次，约 1% 的差异不应解释为性能收益。此前 block=8 的测试使用旧 commit，不能作为严格同版本 A/B。

## 证据

- local-only：[`decoupled-local-20260923-100449-887913/`](../results/decoupled-local-20260923-100449-887913/)（`run.json`、两个 benchmark JSON/日志、3 个 Target 日志）。
- hybrid：[`decoupled-hybrid-20260923-100953-444782/`](../results/decoupled-hybrid-20260923-100953-444782/)（同上，另有 Draft 日志）。
