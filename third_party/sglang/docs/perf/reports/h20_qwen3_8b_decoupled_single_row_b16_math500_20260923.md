# H20 Qwen3-8B decoupled：单行 packed b16 / 单 anchor Math-500（2026-09-23）

## 口径

保持 [旧版 b16 Math-500 测试](h20_qwen3_8b_decoupled_b16_single_anchor_math500_20260923.md)的模型、4×H20 拓扑、原始 `docs/scripts/test.jsonl`、输出 256、seed 42、`block_size=16`、`accept_counts=1..16`、`num_anchors=1`、`max_branches=16`、CUDA Graph max tokens=1024、deterministic、packed causal pending。唯一代码路径变化是开启 `--packed-single-row`，让每个请求的 16 条分支共享一次 pending 前向；第一组实验仍允许每个远端 batch 最多 8 个请求。所有 hybrid 输出均逐 token 对照同 n 值的旧版 local-only JSON。启动前和停止后检查 GPU 无残留计算进程。

## 结果

| 工况 | 配置 | output tok/s | local-only tok/s | 逐 token 匹配 |
| --- | --- | ---: | ---: | ---: |
| n=50，rate=4 | 单行、远端 batch≤8 | 996.94 | 996.18 | 50/50 |
| n=100，饱和 | 单行、远端 batch≤8 | 1098.13 | 1114.32 | 100/100 |
| n=500，饱和 | 单行、远端 batch≤8 | 1144.18 | 1144.05 | 500/500 |
| n=100，饱和 | 单行、远端 batch≤3 | 1098.56 | 1114.32 | 100/100 |

全量 batch≤8：远端 Draft 1765 次 physical forward、10709 个 batch item、171344 条分支，平均 6.07 item/batch。GPU 前向累计 110.764 s，**62.75 ms/forward**；服务累计 139.329 s，**78.94 ms/batch**。`error_ct=0`，结束时 session/队列清零。三张 Target 的 ready lookup 分别 9/1/0，远端 hit 分别 5/1/0，共 **6 次**。全量吞吐几乎等于 local-only，但不是远端预投机带来的可确认加速；相对旧版 b16 hybrid 的 1137.25 tok/s 高约 0.61%，仍属单次运行的小幅差异。

batch≤3 的 n100 实验：613 次 physical forward、1581 个 batch item，平均 2.58 item/batch。GPU 前向 18.617 s，即 **30.37 ms/forward**；服务 23.009 s，即 **37.54 ms/batch**；三张 Target 仅 **2 次 hit**。限制 batch 显著缩短每次 Draft 前向，但没有改善 n100 吞吐。

这次 b16 Target verify 日志共有 8195 次同步计时，p50 **11.034 ms**、p95 **11.198 ms**；上轮 b8 单行日志 p50 **11.002 ms**。b16 并未显著延长 verify 窗口。单行 b16 最少 `16 + 16×16 = 272` 个 query token/item，而 b8 为 `8 + 8×8 = 72`；b16 的每请求计算量约为 b8 的 3.78 倍。batch≤8 时，4 个 b16 请求的最小 1088 token 已超过 1024-token CUDA Graph 上限；batch≤3 消融确认大 batch 是耗时放大因素之一，但图命中率未单独打点，不能把所有差异都归因于 graph 回退。

结论：单行布局比旧版分组 b16 的约 159.70 ms/forward 快很多，但在完整 `1..16` 分支预算下仍远慢于 Target verify，无法稳定在 verify 期间交付远端结果。仅扩大 block 而保持每个位点一条备选，会让 Draft 分支 token 量近似二次增长，verify 时间却基本不变。

## 证据

- [batch≤8 实验目录](../results/decoupled-hybrid-20260923-144957-518368/)与[启动日志](../logs/decoupled_single_row_b16_launch_20260923.log)：n50、n100、n500 benchmark JSON/日志，三张 Target 和 Draft 日志。
- [batch≤3 实验目录](../results/decoupled-hybrid-20260923-145721-286869/)与[启动日志](../logs/decoupled_single_row_b16_batch3_launch_20260923.log)：n100 benchmark JSON/日志。
