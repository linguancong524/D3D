# H20 Qwen3-8B decoupled：block 8、单备选、原始 Math-500（2026-09-23）

## 测试口径

- 源码 commit：`d936e2a`（相对 [同日 b16 测试](h20_qwen3_8b_decoupled_b16_single_anchor_math500_20260923.md)只新增基准归档，没有推理代码变更）。4×H20；每次启动前确认 GPU 无残留计算进程，收尾再次确认 GPU 清空。
- 输入：`docs/scripts/test.jsonl`，500 条 `problem`；SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。使用 `sglang.benchmark.decoupled_serving` 的 `problem` 路径，Qwen3 chat template、关闭 thinking、附加逐步推理与 boxed answer 指令。temperature=0，`max_new_tokens=256`，`ignore_eos=True`，seed=42。
- GPU0–2：3 个 Qwen3-8B TP=1 Target，每个 `max_running_requests=3`；hybrid 在 GPU3 加一个共享 D3D-1.7B OneStep Draft。local-only 使用相同 3 个 Target，但不启动远端 Draft。
- 本轮为 `block_size=8`、`accept_counts=1..8`、`num_anchors=1`、`max_branches=8`；其余保持 b16 测试设置：local prefix reuse、deterministic inference、packed causal pending、Draft CUDA Graph max tokens=1024、Draft mem fraction=0.65、speculation batch size=8、batching window=0。
- 跑 n=50/rate=4、n=100/request-rate=inf 和全量 n=500/request-rate=inf。hybrid 各组逐 token 对照同组 local-only；这是速度与输出一致性测试，不计算答案正确率。

## 结果

| 工况 | 模式 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length | token match |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| n=50，rate=4 | local-only | 50/0 | 814.23 | 3.181 | 4.247 / 5.654 / 5.871 | 5.614 | reference |
| n=50，rate=4 | hybrid | 50/0 | 816.64 | 3.190 | 4.265 / 5.687 / 5.836 | 5.587 | 50/50 |
| n=100，饱和 | local-only | 100/0 | 879.66 | 3.436 | 15.414 / 27.553 / 28.608 | 5.622 | reference |
| n=100，饱和 | hybrid | 100/0 | 865.83 | 3.382 | 15.490 / 27.811 / 28.833 | 5.607 | 100/100 |
| n=500，饱和 | local-only | 500/0 | 912.89 | 3.566 | 69.599 / 130.975 / 136.772 | 5.673 | reference |
| n=500，饱和 | hybrid | 500/0 | 894.31 | 3.493 | 70.809 / 133.684 / 138.551 | 5.669 | 500/500 |

`accept length = 1 + spec_num_correct_drafts / spec_verify_ct`。hybrid 相对 local-only 的 output tok/s 差值依次为 **+0.30%**、**-1.57%**、**-2.04%**。各工况只测一次，不能把约 1–2% 的差距断言为稳定回退；目前看不到端到端吞吐收益。饱和 E2E 分位数包含一次性提交造成的排队。

远端确实开始被消费：三张 Target 在三组 benchmark 和预热期间累计 **639 次** ready lookup、**309 次** branch hit，约占全部 29603 次 lookup 的 **1.04%**；同口径 b16 测试为 0 次。Draft 共处理 24785 个 batch item、198280 条分支（恰为每条 8 个），3946 次 physical forward，`error_ct=0`，最终 session 和队列清零。Draft GPU 前向累计 133.912 s，平均 **33.94 ms/forward**；服务总耗时 176.689 s，平均 **44.78 ms/batch**。Target verify 日志中位约 **11.00 ms**。因此 b8 比 b16 的约 159.70 ms Draft 前向快很多，但远端结果仍多数赶不上下一轮 lookup。

全量 500 题 hybrid 比 b16 hybrid 的 1137.25 tok/s 低 **21.4%**；主要差异是 b8 的接受长度约 5.67，b16 约 7.50，Target 需要更多 verify。两个 block 的 hybrid 分别使用各自的同 block local-only reference，不能把跨 block 的逐 token 输出差异误判为错误。六份 benchmark 均 `summary.valid=true`，hybrid 三组逐 token 对照全部通过，服务日志无未捕获 Python traceback 或致命 CUDA OOM。

## 证据

- local-only：[`decoupled-local-20260923-105141-096952/`](../results/decoupled-local-20260923-105141-096952/)（`run.json`、三组 benchmark JSON/日志、Target 日志）。
- hybrid：[`decoupled-hybrid-20260923-105830-995075/`](../results/decoupled-hybrid-20260923-105830-995075/)（同上，另有 Draft 日志）。
