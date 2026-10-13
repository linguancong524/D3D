# H20 Qwen3-8B decoupled：b16 单 anchor 的 1+1 与等待远端 A/B（2026-09-23）

## 口径

输入为 `docs/scripts/test.jsonl`（SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`），Qwen3 chat template、关闭 thinking、temperature=0、输出 256 token、ignore_eos、seed=42。GPU0 跑 Qwen3-8B Target（TP=1、max_running_requests=3），hybrid 的 GPU3 跑 D3D-1.7B Draft；GPU1/2 空闲。local-only 只启动 GPU0。所有运行前后检查无残留 GPU 进程；Target 启动时可用显存 94.60 GB，未发生致命 OOM。

保持 `block_size=16`、`num_anchors=1`、`accept_counts=1..16`、`max_branches=16`、`candidate_topk=4`、本地 prefix reuse、deterministic、远端单行 packed + causal pending、Draft bf16/FlashInfer、CUDA Graph max tokens=1024、mem fraction=0.65。等待实验只改变 Target 的 lookup 策略。启动命令、完整日志和 JSON 分别在下列运行目录的 `run.json`、`target-0.log`、`drafter.log`、benchmark 文件中；所有 hybrid 输出均逐 token 对照 local-only。

## 1+1：减少 Draft 争用后能否加速

| 工况 | 模式 | output tok/s | 相对 local-only | 平均 accept length | token match |
| --- | --- | ---: | ---: | ---: | ---: |
| n=50，rate=1 | local-only | 286.87 | — | 7.397 | 参考 |
| n=50，rate=1 | 1+1 零等待 | 285.69 | -0.4% | 7.273 | 50/50 |
| n=100，饱和 | local-only | 398.37 | — | 7.384 | 参考 |
| n=100，饱和 | 1+1 零等待 | 368.99 | -7.4% | 7.281 | 100/100 |
| n=500，饱和 | local-only | **405.41** | — | 7.499 | 500/500 |
| n=500，饱和 | 1+1 零等待 | **383.05** | **-5.5%** | 7.369 | 500/500 |

全部请求成功，benchmark `summary.valid=true`。与 [3+1 单行 b16 全量运行](h20_qwen3_8b_decoupled_single_row_b16_math500_20260923.md)相比，1+1 的 Draft 平均合批从约 6.07 请求降至约 2.9，GPU 前向从约 62.75 ms/批降至约 31.6 ms/批，服务从约 78.94 降至约 39.7 ms/批。远端 hit 从 3+1 全量约 6 次增至 1+1 全量约 379 次；但 379/17665 次 verify 仅约 2.1%，尚不能弥补远端开销与略低的 proposal 接受长度。单次 Target verify 仍约 11 ms。rate=1 的低负载组无明确吞吐差异，饱和 500 题显示稳定负收益。

## anchor 覆盖时等待远端

`DraftClient.lookup` 在上一轮 proposal 的接受位点、Target correction 与候选 anchor 匹配时，才可能使用远端分支。此前 `lookup_wait=0`，结果未就绪就立即走本地 Draft；已有 `lookup_wait_ms=50` 会在收到 inflight 信号后等待，但在 `min(发布时刻+50 ms, Draft 预计完成时刻)` 到点就回退。新增**默认关闭**的 `--wait-for-covered-result` 实验开关只去掉预计完成时刻的提前截断，仍以发布时刻后 50 ms 为绝对上限；anchor 未覆盖或尚无 inflight 信号时仍立即回退，远端故障不会无限阻塞。

相同 n=100 饱和 Math-500 的 A/B：

| 模式 | output tok/s | 远端 hit* | inflight 等待次数 | 累计阻塞 | 本地 fallback 项 | verify 次数 | 平均 accept length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local-only | **398.37** | 0 | 0 | 0 | 3561 | 3527 | 7.384 |
| 1+1 零等待 | 368.99 | 约 62 | 0 | 0 | 3548 | 3576 | 7.281 |
| 1+1 预计时刻截断、50 ms 上限 | 257.12 | 246 | 1118 | 30.32 s | 3511 | 3712 | 7.009 |
| 1+1 等结果优先、50 ms 上限 | **208.20** | **1968** | 1280 | **36.22 s** | 3181 | **5101** | **5.090** |

\* Target 日志累计计数，包含每次 benchmark 的少量预热。n=100 的四组结果均 100/100 完成；三个 hybrid 均 100/100 逐 token 匹配。

有界等待比零等待多约 179 次 hit，但只少处理 37 个本地 fallback 项：远端 proposal 的接受长度降低，让 verify 多了 136 次；等待的 30.32 s 几乎等于整组完成时间增加的 30.18 s。真正等结果优先后，hit 增至 1968，但 verify 比零等待多 **1525 次（+42.6%）**；即使本地 fallback 项少了 367 个，其调用次数从 1268 增至 1635，累计本地 fallback 时间从约 52.8 增至 64.1 s，加上 36.22 s 等待，吞吐进一步下降。两种等待方式在当前远端 proposal 质量下均不划算，不能只凭 hit 数判断有效。

## 结论与后续方向

1+1 确实减轻共享 Draft 争用，远端结果不再几乎全错过；但在完整 16 分支预算下，Draft 仍慢于单次 verify，零等待的全量吞吐比本地兜底低 5.5%。仅在 anchor 覆盖时硬等远端也不解决问题：等待阻塞 Target，而且大量使用远端 proposal 使接受长度显著下降。该实验开关默认关闭，不应作为部署建议；无需再对明显负收益的硬等模式跑 500 题。下一步应先对照远端与本地 proposal 的逐位置正确率/接受长度，定位质量差异，再考虑更少分支、更快的远端前向或基于剩余完成时间与 proposal 质量的选择策略。

## 证据与复现说明

- [local-only 短测](../results/decoupled-local-20260923-154015-981105/)与[启动日志](../logs/decoupled_1plus1_local_launch_20260923.log)；[local-only 全量](../results/decoupled-local-20260923-155721-447870/)与[启动日志](../logs/decoupled_1plus1_local_full_launch_20260923.log)。
- [1+1 零等待短测及全量](../results/decoupled-hybrid-20260923-154457-327156/)与[启动日志](../logs/decoupled_1plus1_hybrid_launch_20260923.log)。
- [1+1 预计时刻截断等待](../results/decoupled-hybrid-20260923-160555-799076/)与[启动日志](../logs/decoupled_1plus1_hybrid_wait50_launch_20260923.log)。
- [1+1 等结果优先](../results/decoupled-hybrid-20260923-161723-856954/)与[启动日志](../logs/decoupled_1plus1_hybrid_hardwait50_launch_20260923.log)。

前三组启动时 `run.json` 的 commit 字段为当时 HEAD `9360e44`，1+1 launcher 改动尚在工作树；相同 launcher 改动和默认关闭的实验等待开关随后提交为 `6de4769`。最后一组直接运行 `6de4769`。前三组推理主路径仍为既有实现；各组的完整 worker 命令保存在对应 `run.json`，以命令和归档日志为复现依据。
