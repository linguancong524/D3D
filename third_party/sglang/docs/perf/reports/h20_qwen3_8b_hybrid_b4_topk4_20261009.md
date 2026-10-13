# H20：hybrid block=4 / top-k=4 简测（2026-10-09）

## 结论

**单请求预投机基本能在下轮 lookup 前交付，但当前三 Target 共享 Draft 的并发流程没有明显输出加速。**

- 单请求：远端 GPU 前向均值 **5.78 ms**，batch 服务均值 **7.83 ms**，Target verify 中位 **11.21 ms**；下轮 lookup 时 ready **98.96%**，实际分支命中 **51.92%**。
- 并发：远端 GPU 前向约 **9.3 ms**，batch 服务约 **13.3 ms**，verify 约 **11.3 ms**。GPU 计算仍短于 verify，但服务耗时超过窗口；ready 只有 **12.57%–13.50%**，实际命中 **8.14%–8.75%**。
- 单请求流式 decode 中位速度 **69.71→72.97 tok/s（+4.68%）**，含首 token 的总吞吐仅 **+0.77%**；并发两组吞吐为 **−1.20% / −1.80%**。每种工况仅跑一遍，这些小差异不足以确认提速或显著退化。
- hybrid 三组共 **110/110** 个请求成功，并逐 token 匹配同负载 local-only 对照；Draft `error_ct=0`，各阶段结束 session / queue 清零，停止后 GPU 无残留计算进程。

因此本轮主要确认了两点：**小 batch 下预投机时间足够短；共享 Draft 并发时仍不能稳定及时交付。** 不据此发布性能优化。

## 配置与实际计算量

沿用现有 decoupled hybrid 流程：GPU0–2 各一个 Qwen3-8B TP=1 Target，GPU3 一个共享 D3D-1.7B Draft。单请求组仅向 GPU0 发送请求，其他 Target 空闲。

| 项目 | 配置 |
|---|---|
| Target | `/workspace/models/Qwen3-8B`，bf16，FA3 |
| local / remote Draft | `/workspace/models/d3d-1.7b-b16_for_qwen3-8b`，本地 auto→FP16、远端显式 BF16，FlashInfer（2026-10-10 按运行日志复核修正） |
| GPU | 4×H20，驱动 580.105.08，CUDA 13 |
| 软件 | torch 2.13.0、FlashInfer 0.6.17、sglang-kernel 0.4.6.post1、Triton 3.7.1；完整 metadata 见 `software_versions.json` |
| block / candidate top-k | 4 / 4 |
| 覆盖接受位置 | 1、2、3、4；`num_anchors=4`，`max_branches=12` |
| 实际分支 | 拒绝位置 1–3 各 3 条纠正分支，全接收位置 4 一条分支，共 **10 条** |
| 每请求 packed query | 4 个 pending + 10×4 个分支 token = **44 token**；每分支包含锚点及 3 个后续候选 |
| Target active cap | 每张 3，合计 9 |
| remote batching | batch item≤8，batching window=0，single-row packed causal pending |
| 图捕获 | remote CUDA Graph max tokens=1024；H20 Target prefill 保留 BCG，verify 捕获到 bs=3 |
| 其他 | local prefix reuse、deterministic、poll interval=0.1 ms、lookup wait=0 |
| 输入 | 原始 `docs/scripts/test.jsonl` Math-500；前 50 题 prompt 37–366 token，中位 75 |
| 输出 | 每请求 256 token，temperature=0，ignore_eos；并发发送 seed=42 |

这里的 top-k 是 Draft 的纠正候选集合。现有 planner 在拒绝位点过滤已选中的 top-1，全接收位点使用一条兼容分支，因此不是“只计算 12 个新 token”。本轮按用户确认的全部接受位置覆盖执行，没有更改 planner 语义。

## 预投机耗时与命中

| 工况 | remote item/batch 均值 | GPU 前向均值 | batch 服务均值 | verify 中位 | ready / lookup | hit / lookup | hit / ready |
|---|---:|---:|---:|---:|---:|---:|---:|
| 串行 n=10 | 1.00 | 5.78 ms | 7.83 ms | bs1 11.21 ms | 1235/1248，98.96% | 648/1248，51.92% | 52.47% |
| n=50，rate=4 | 2.73 | 9.31 ms | 13.32 ms | bs3 11.33 ms | 552/4090，13.50% | 358/4090，8.75% | 64.86% |
| n=50，饱和 | 2.70 | 9.29 ms | 13.28 ms | bs3 11.32 ms | 511/4065，12.57% | 331/4065，8.14% | 64.77% |

指标口径：

- GPU 前向由 CUDA Event 测量，按 physical forward 归一；batch 服务来自 `remote_optional_seconds / remote_optional_batch_ct`，包含 batch 内规划、native Engine RPC、前向与结果提取，**不包含 batch 外的 queue、prefix stage 和 Target→Draft→Target 完整往返**。
- verify 由 `--timing-sync` 的 CUDA Event 测量。并发时 remote 与 Target 的 batch shape 不同，上表比较用于解释窗口压力，不是同 shape 算子横比。
- `lookup = ready_lookup_ct + lookup_miss_ct`，不含首轮尚无上一份 proposal 的检查；ready 表示下轮 lookup 时已有结果。实际命中率为 `lookup_hit_ct / lookup`，条件命中率为 `lookup_hit_ct / ready_lookup_ct`。
- 耗时和命中计数包含 benchmark 自带预热；输出速度、verify 次数和 correctness 的 benchmark summary 排除预热。通过每阶段日志 byte offset 和最终请求结束计数提取，避免将下一阶段累计值混入。

串行 ready miss 只有 13 次，说明该负载下时效性已基本满足消费要求。已有 ready outcome 的 587 次分支 miss 均记为 `candidate_rank`，表示纠正 token 不在所观察的 Draft top-4 中。并发两组已有 ready 但分支 miss 分别为 194 / 180 次，也全部是该原因；更主要的问题是 86%–87% 的 lookup 尚未 ready。

**本轮没有采集按请求配对的 publish / verify-start / verify-end / ready 时间线。** ready 统计证明结果在下轮消费前交付，不能解释为严格在 verify GPU 区间内完成的覆盖率。单请求 GPU / batch 服务均短于 verify，加上 98.96% 的及时到达率，支持“大部分时延可被当前流水线掩盖”；并发则不支持稳定覆盖。

## 输出速度与延迟

| 工况 / 指标 | local-only | hybrid | 差异 |
|---|---:|---:|---:|
| 串行 n=10，流式 decode tok/s p50 | 69.71 | 72.97 | +4.68% |
| 串行 n=10，TPOT p50 | 14.35 ms | 13.70 ms | −4.48% |
| 串行 n=10，总输出 tok/s | 71.25 | 71.80 | +0.77% |
| 串行 n=10，TTFT p50 | 21.30 ms | 21.82 ms | +0.52 ms |
| n=50，rate=4，总输出 tok/s | 493.28 | 487.36 | −1.20% |
| n=50，rate=4，E2E p50 / p95 | 11.16 / 15.84 s | 11.17 / 16.00 s | 接近 |
| n=50，饱和，总输出 tok/s | 535.68 | 526.03 | −1.80% |
| n=50，饱和，E2E p50 / p95 | 12.65 / 23.30 s | 13.09 / 23.98 s | 略高 |

串行速度来自真实 SSE 输出，从首个非空输出 chunk 到结束计算后续 token 速度 / TPOT；总输出吞吐则是所有完成 token 除以共同 wall time。并发 E2E 含排队，rate=4 超过本轮系统约 1.9–2.1 req/s 的完成能力，不能当作无排队响应时间。

| 工况 | local verify 次数 | hybrid verify 次数 | local / hybrid 输出 token / verify |
|---|---:|---:|---:|
| 串行 n=10 | 737 | 1149 | 3.47 / 2.23 |
| n=50，rate=4 | 3677 | 3815 | 3.48 / 3.36 |
| n=50，饱和 | 3677 | 3793 | 3.48 / 3.37 |

串行虽然有 52% 的 lookup 能省去本地 required Draft，但 verify 次数增加 **55.9%**，平均每轮输出减少。这是“远端命中很多，端到端收益仍小”的直接测量证据；本轮没有单独隔离远端 proposal 的 token accept rate，不能把整组平均产出的变化都归因于单一原因。

## 证据与复现

- [local-only 实验目录](../results/decoupled-local-20261009-023239-671917/)：`run.json`、worker / launcher 日志及三组结果。
- [hybrid 实验目录](../results/decoupled-hybrid-20261009-023916-785957/)：同上，另含 `phase_boundaries.json`、[analysis.json](../results/decoupled-hybrid-20261009-023916-785957/analysis.json)、`run_benchmarks.py` 和 `analyze.py`。先按 `run.json` 启动服务至 READY，再从仓库根目录运行 `run_benchmarks.py`；本轮产物已存在，复跑需使用新目录以免覆盖证据。
- [block4 YAML](../../../scripts/decoupled/sdar_block4.yaml)、[串行流式测速脚本](../scripts/bench_decoupled_serial.py)。
- 基础代码 commit：`e9124f5`；hybrid 增加的请求结束累计指标日志保存于 `instrumentation.diff`。该日志仅在 `--decoupled-timing-sync` 开启时输出，不改变分支选择或调度。
- 新增 / 修改 Python 通过 Ruff 和语法检查；本轮在线运行验证流式测速、结束计数及三组输出比较。

下一步最有价值的是区分“分支覆盖”与“候选质量”：减少同时计算的接受位置，观察并发 ready 是否改善；并把命中远端后的 correct drafts / verify 与 local fallback 分开记录，解释为何串行每轮输出减少。当前结果不支持直接扩大 top-k 来换取部署收益。
