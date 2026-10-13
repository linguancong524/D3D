# H20 hybrid：单请求 block / top-k 逐点搜索（2026-10-09）

后续 FP8 对照见 [2026-10-10 报告](h20_qwen3_8b_hybrid_serial_fp8_comparison_20261010.md)：包含远端 FP8 的更大 top-k 预算、Target FP8 的速度 / 缓存保真，以及完整数学答案质量对照。

## 结论

按 **block=4、5、6、7、8，top-k=4、3、2** 跑完 15 个组合，以用户指定的 **及时到达率 ≥95%** 筛选。**扩大 block 没有提高预投机命中率；减少 top-k 能降低服务耗时，让更大的 block 保持及时交付，但会损失候选覆盖。**

- top-k=4：block=4–6 的及时到达率为 98.52%–99.12%；block=7、8 分别降到 0.19%、0%，重测同样失败。本次范围内最大达标 block 为 **6**。
- top-k=3、2：block=4–8 全部达标，及时到达率为 98.22%–99.10%。**没有测出它们的上界**，不能把 block=8 当作全局最佳值。
- top-k=4 的实际命中率在 block=4→6 从 51.92% 降到 50.11%；top-k=3 在 block=4→8 从 50.37% 降到 47.61%；top-k=2 从 48.88% 降到 43.88%。这批输入不支持“增大 block 会提高命中率”的假设。
- 达标配置中首轮最快为 **block=8 / top-k=2**：decode p50 **112.13 tok/s**，及时到达率 **98.40%**，实际命中率 **43.88%**。独立重启复核为 **112.85 tok/s / 98.28% / 44.25%**。

相同 block 的 local-only 为 **117.14 tok/s**；候选 hybrid 复核比它低 **3.66%**，总输出吞吐低 **3.71%**。因此增大 block 提高了整体输出速度，但本轮没有证明 remote 预投机带来额外提速。hybrid 每轮平均输出 **3.64 token**，local-only 为 **5.55 token**，解释了为什么及时到达与分支命中尚不能直接兑现为速度收益。这个短负载搜索不作为发布性能优化的依据。

## 实验设置与口径

基础源码 commit 为 `be801f29dc50a48d6fdc19daf5c6424cfa4f0827`。使用 Qwen3-8B Target 与 `d3d-1.7b-b16_for_qwen3-8b` Draft：Target 为 BF16，远端 Draft 显式 BF16，本地 required Draft 在 `dtype=auto` 下实际加载为 FP16；Target FA3，Draft FlashInfer。本地精度标签于 2026-10-10 按[原始日志](../results/decoupled-serial-search-exact-h20-20261009-031600/b4_k4/target-0.log)复核修正，其中两份 KV 池分别为 `torch.bfloat16` 与 `torch.float16`。GPU 为 4×H20，驱动 580.105.08，cu130 软件栈沿用[上一轮记录](h20_qwen3_8b_hybrid_b4_topk4_20261009.md)。hybrid 仅启动 GPU0 一个 Target（含本地 required Draft）和 GPU3 一个 remote optional Draft，单请求、active cap=1、remote batch / speculation batch=1。local-only 只启动 GPU0。

每点使用 `docs/scripts/test.jsonl` 原始 Math500 前 10 题，每请求生成 256 token，temperature=0、ignore_eos；预热 1 个请求。SSE 流式测速采用[现有串行脚本](../scripts/bench_decoupled_serial.py)。所有输出 ID 与上一轮 local-only 逐 token 对齐，旧结果只用于正确性比较。速度比较使用本轮搜索与本轮 local-only。

覆盖 block 内全部接受位置 `1..B`，保持 deterministic、local prefix reuse、packed single-row causal pending、poll=0.1 ms、batching window=0、lookup wait=0；没有通过等待远端来抬高及时到达率。Target prefill 保留 H20 BCG，各点统一捕获到 512 token；remote CUDA Graph max tokens=1024。

planner 在拒绝位置过滤已选中的 top-1，全接收位置使用一条兼容分支，因此实际分支数为 **`(B−1)×(K−1)+1`**，packed query 为 **`B×(实际分支数+1)`**（另含 B 个 pending token）。本轮 `max_branches` 精确匹配实际分支数，避免额外上限引入不必要的图 padding。例如 B4/K4 为 10 条分支、44 个 query token；B8/K2 为 8 条、72 token；B8/K4 为 22 条、184 token。这仍是全部位置覆盖，并非只生成 12 个新 token。

指标定义：

- **及时到达率**：`ready_lookup_ct / (ready_lookup_ct + lookup_miss_ct)`，表示下轮消费时已有结果；不包含首轮尚无上一份 proposal 的检查。
- **实际命中率**：`lookup_hit_ct / lookup`；已及时到达但不匹配所选纠正分支也计为不命中。它不是 remote proposal 后续 token 的接受率。
- **GPU 前向**：CUDA Event 累计前向时间 / physical forward 数；**服务耗时**：`remote_optional_seconds / remote_optional_batch_ct`，含 batch 内规划、Engine RPC、前向与结果提取，不含 batch 外 queue、prefix stage 及完整网络往返。
- **Target verify**：`--timing-sync` CUDA Event 的中位数。没有采集逐轮配对的 publish / verify / ready 时间线，故及时到达率不能等同于严格在 verify GPU 区间内完成的比例。
- 耗时与 lookup 计数包含预热；SSE decode / TTFT、总吞吐、verify 次数及输出正确性排除预热。日志 byte offset 界定每次测量，采用请求结束的最终 client 累计计数和 Draft service 增量。

每点先跑一次；及时到达率低于 95% 或距阈值不超过 2 个百分点时重测。最终候选另行重启复核。表中 B7/K4、B8/K4 的命中与及时率合并两次计数，其他为首轮；速度和耗时列均为首轮。

## 全部 15 个组合

| block | top-k | 实际分支 / query token | 及时到达率 | 实际命中率 | decode p50 tok/s | GPU 均值 ms | 服务均值 ms | verify p50 ms | ≥95% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 4 | 10 / 44 | 99.12% | 51.92% | 74.21 | 5.69 | 7.70 | 10.81 | 是 |
| 4 | 3 | 7 / 32 | 99.10% | 50.37% | 73.48 | 5.30 | 7.02 | 10.82 | 是 |
| 4 | 2 | 4 / 20 | 99.09% | 48.88% | 74.92 | 5.22 | 6.62 | 10.81 | 是 |
| 5 | 4 | 13 / 70 | 98.77% | 50.38% | 86.03 | 7.43 | 9.87 | 10.83 | 是 |
| 5 | 3 | 9 / 50 | 98.95% | 49.62% | 85.42 | 5.69 | 7.69 | 10.83 | 是 |
| 5 | 2 | 5 / 30 | 98.91% | 47.28% | 83.75 | 5.29 | 6.80 | 10.83 | 是 |
| 6 | 4 | 16 / 102 | 98.52% | 50.11% | 94.48 | 7.62 | 10.33 | 10.88 | 是 |
| 6 | 3 | 11 / 72 | 98.82% | 48.39% | 94.53 | 7.43 | 9.62 | 10.84 | 是 |
| 6 | 2 | 6 / 42 | 98.67% | 46.18% | 93.95 | 5.65 | 7.35 | 10.84 | 是 |
| 7 | 4 | 19 / 140 | 0.19% | 0.19% | 110.00 | 10.21 | 13.41 | 10.88 | 否 |
| 7 | 3 | 13 / 98 | 98.22% | 47.74% | 102.05 | 7.59 | 10.02 | 10.86 | 是 |
| 7 | 2 | 7 / 56 | 98.63% | 43.66% | 101.85 | 5.73 | 7.53 | 10.86 | 是 |
| 8 | 4 | 22 / 184 | 0.00% | 0.00% | 110.27 | 10.42 | 13.96 | 10.86 | 否 |
| 8 | 3 | 15 / 128 | 98.49% | 47.61% | 108.34 | 7.74 | 10.38 | 10.86 | 是 |
| 8 | 2 | 8 / 72 | 98.40% | 43.88% | 112.13 | 7.43 | 9.34 | 10.86 | 是 |

B7/K4 重测及时率 / 命中率仍为 0.19% / 0.19%，decode 112.55 tok/s；B8/K4 重测为 0% / 0%，decode 117.28 tok/s。它们主要走 local fallback，输出较快不能作为 remote 预投机有效的证据，也不能进入达标候选排名。

top-k=4 从 B6 到 B7，服务均值从 10.33 ms 跨到 13.41 ms，超过约 10.9 ms 的 verify 窗口，及时到达率随之骤降。GPU 前向 10.21 ms 仍小于 verify，说明仅比较 GPU 前向不足以判断预投机能否及时交付。降低 top-k 后，B8/K3、B8/K2 服务均值分别为 10.38、9.34 ms，及时率仍接近 98.5%。这些均值与消费时刻计数支持该解释，但不提供逐轮因果时间线。

在达标点中，已有 ready 但未命中的记录均为 `candidate_rank`，没有 length / prefix / missing-branch 错误。更长 block 没有扩大纠正 token 的 top-k 集合，实测也未提升分支命中。命中率随 block 变化同时受实际访问的接受位置、纠正 token 及后续生成轨迹影响，本轮没有将这些因素逐个隔离。

## 同 block 的速度对照

| 配置 / 指标 | local-only | hybrid | hybrid 相对 local |
|---|---:|---:|---:|
| B4/K4，decode p50 tok/s | 68.76 | 74.21 | +7.94% |
| B4/K4，总输出 tok/s | 70.93 | 73.03 | +2.96% |
| B4/K4，verify 次数 | 737 | 1149 | +55.90% |
| B8/K2，decode p50 tok/s | 117.14 | 112.85 | −3.66% |
| B8/K2，总输出 tok/s | 113.82 | 109.59 | −3.71% |
| B8/K2，TPOT p50 ms | 8.55 | 8.87 | +0.32 ms |
| B8/K2，TTFT p50 ms | 21.55 | 21.20 | −0.35 ms |
| B8/K2，verify 次数 | 461 | 704 | +52.71% |
| B8/K2，输出 token / verify | 5.55 | 3.64 | −34.52% |

B4/K4 的 hybrid 来自正式网格首轮；B8/K2 使用独立重启的候选复核。local-only 均在候选复核后重启运行，同样 active cap=1、BCG=512、相同 prompt、256 token、相同权重。SSE decode 从首个非空输出 chunk 到结束测量，首个 chunk 可能含多个 token；总吞吐包含所有测量请求的完整 wall time。

与 B4/K4 hybrid 74.21 tok/s 相比，B8/K2 hybrid 首轮 112.13 tok/s 提高 51.09%。但是 local-only 随 block=4→8 也从 68.76 提高到 117.14 tok/s。不能将跨 block 的速度变化归为 remote 预投机收益。B8/K2 的首轮与复核速度相差仅 0.64%，及时到达率均远高于 95%，说明该候选在本轮两次运行中稳定达标；同 block 对照没有显示额外速度收益。

remote 命中能够跳过部分本地 required Draft，但 verify 次数更多、每轮输出更少；这是候选质量成本的整组测量证据。尚未按 remote-hit / fallback 分组记录后续 token 接受量，因此不能把整组产出的下降精确归因到某类分支。local-only 只测一次，网格失效点的重复测速也有约 6% 波动；不将 3.66% 的差异称为已经确认的稳定退化，也不将较小 top-k 间的速度差异称为显著提升。

## 后续诊断：为什么及时到达仍没有额外提速

对 B8/K2 的独立复核与 local-only，按每个请求的结束日志排除预热，再累加本地 Draft wall time 与同步 CUDA Event verify 时间。两者都是串行主路径的独立区间；结果保存于 [cost_breakdown.json](../results/decoupled-serial-search-exact-h20-20261009-031600/cost_breakdown.json)，由同目录的 `analyze_cost_breakdown.py` 从已有日志重算，没有增加推理实验。

**主要的已知成本是 verify 轮数增加，省掉的本地 Draft 总时间不足以抵消它。** 及时到达只保证可以消费分支，并不保证分支的后续 token 与本地 Draft 有相同接受质量。

| 10 个请求、共 2560 输出 token，排除预热 | local-only | hybrid | hybrid − local |
|---|---:|---:|---:|
| 本地 Draft 调用次数 | 461 | 396 | −65 |
| Target verify 次数 | 461 | 704 | +243 |
| 有效输出 token / verify | 5.55 | 3.64 | −34.52% |
| 本地 Draft 总耗时 | 16.492 s | 14.294 s | **−2.198 s** |
| Target verify 总耗时 | 5.005 s | 7.659 s | **+2.654 s** |
| 未单独计时部分（wall − 上述两项） | 0.995 s | 1.407 s | **+0.411 s** |
| 全段 wall time | 22.492 s | 23.360 s | **+0.868 s** |

这笔账闭合：**增加 2.654 s verify + 增加 0.411 s 其余时间 − 节省 2.198 s Draft = 净增加 0.868 s**。摊到每轮的全段 wall time 确实从 48.79 ms 降到 33.18 ms，但每轮有效输出下降得更多，所以每 token 的成本没有降低。

hybrid 的 704 轮中，308 轮使用远端命中结果，396 轮仍需本地 Draft；后者包含每个请求首轮。本地调用数因此只比基线 461 次少 65 次，不能把“约 44% 的 hybrid lookup 命中”解释为“省掉基线 44% 的 Draft”。在本次测量请求中，lookup 共 694 次，其中 ready 682、hit 308；报告前面的 98.28% / 44.25% 包含预热，这里为排除预热后的同段成本核算。

单次本地 Draft 均值接近：local 35.77 ms，hybrid fallback 36.10 ms；单次 verify 同样接近：10.857 / 10.879 ms。日志没有显示单次操作突然出现大额耗时。源码中，连续前缀缓存只在真正调用本地 Draft 时更新；连续 remote hit 后 fallback 需要补齐缺失的已提交前缀，这可能增加部分本地调用成本，但本轮没有单独计时该补齐，不能把 0.32 ms 差值全归因于它。

其余多出的 0.411 s 包含 prefill、verify 前的准备与 H2D、采样、KV commit、scheduler、日志及客户端流式收发等未拆开的时间。额外 243 轮本身也会增加这些重复步骤。**这是剩余差额，不是已定位的 RPC 或网络开销**。本轮没有 CPU 分段 trace，不能精确分配到某个步骤。全部测量的主动 `lookup_block_seconds=0`；`ready_wait_seconds` 实际累计的是结果 ready 后到消费的驻留时间，并非 Target 主动等待远端。

remote 端确有规划、top-k、D2H 结果提取及 RPC 等 GPU 以外的步骤。B8/K2 复核 GPU 均值 7.42 ms，完整 batch 服务均值 9.37 ms，这些成本已经体现在服务时间里，且及时到达率仍约 98%。此外，`packed_postprocess_seconds` 的计时从前向发射前开始，包含等待 GPU 结果；它与 GPU Event 时间重叠，不能直接把两者相加当成额外成本，定义见 [io_struct.py](../../../python/sglang/srt/managers/io_struct.py#L2254)。

### 候选质量的一个具体待验证差异

local Draft 每次实际使用 **`[真实 bonus token, MASK, …]`** 作为下一 block 输入。当前 legacy Proposal 的 top-token 行数只有 B，故全接收位点在 [select_branches](../../../python/sglang/srt/speculative/decoupled/protocol.py#L133) 中退为 `Branch(B, None)`。其 packed 分支首位在前向时仍为 MASK；[select_outcome](../../../python/sglang/srt/speculative/decoupled/protocol.py#L248) 消费时才将首 token 替换为真实 Target bonus，其余 token 不重新计算。

因此该分支的后续 token **没有在生成时以真实 bonus 为条件**，与本地 required Draft 不等价。这是值得优先检查的接受质量差异，并不影响最终输出正确性，因为 Target 仍逐轮验证。但没有按 remote-hit / fallback、全接收 / 拒绝分支记录下一轮接受长度，不能认定它已经解释全部额外 243 次 verify。

下一步应优先做同一已提交前缀的 remote / local shadow 候选对齐，分别记录全接收与纠正分支的后续接受长度；诊断运行的 shadow 计算会改变速度，应与性能测速分开。其次再为 lookup / publish、准备、采样与 KV commit 补充 CPU 分段计时，定位剩余 0.411 s。仅继续降低 remote GPU 耗时，无法直接解决每轮有效输出减少的问题。

## 证据、有效性与复现

- [正式运行目录](../results/decoupled-serial-search-exact-h20-20261009-031600/)：各 `bN_kN/` 下保存启动命令、YAML、worker 日志、SSE 原始结果、输出 ID、byte offset 与分析；[summary.json](../results/decoupled-serial-search-exact-h20-20261009-031600/summary.json) 汇总 15 点，[validation/](../results/decoupled-serial-search-exact-h20-20261009-031600/validation/) 保存独立复核及 local-only。
- 正式网格共 170 个测量请求（15×10 + 两个失败点各重测 10 个），均成功且逐 token 匹配。各运行 Draft `error_ct=0`。初始化由 launcher 检查所用 GPU 空闲；Qwen3-8B 的内存池与 Qwen3.5 GDN 基线不同，不套用 9B mamba slot 判据。
- 独立候选复核 10 个、local-only 两组共 20 个请求也全部通过，共 **200/200** 个正式测量请求成功并逐 token 匹配，每个恰好 256 token。所有服务停止后 GPU 无残留计算进程；有效性审计保存于 `audit.json`。
- [初步运行](../results/decoupled-serial-search-h20-20261009-030600/) 使用较宽 `max_branches=B×K`，完成两点后发现额外 padding；已标为 superseded，不混入正式矩阵。正式 B5/K2 的一次未完成初始化保存在 `interrupted_b5_k2_before_cache/`，没有产生有效测速结果。
- 前五点从 `/workspace/models` 加载；后续为减少 OSS 重复加载时间，改用本地 `/tmp/sglang-hybrid-model-cache` 的相同模型。全部模型文件在拷贝时计算源 SHA256 并重新读取目标校验，见 [model_cache_manifest.json](../results/decoupled-serial-search-exact-h20-20261009-031600/model_cache_manifest.json)。启动与图捕获均在测速外；模型缓存不属于实验输出，也不提交权重。
- [搜索脚本](../scripts/search_decoupled_serial.py) 和归档的 `search_runner.py` / `search_runner_resume.py`、`execution.json` / `resume_execution.json` 保存实际版本及 SHA256。`validate_candidates.py` 保存复核过程。停止 worker 时日志可能记录子进程 SIGTERM（exit −15）；这些发生在完整结果落盘后的清理阶段。
- 本轮软件 metadata 与 GPU inventory 另见 `software_versions.json`；新搜索脚本及复核脚本通过 Ruff、格式和语法检查，实际在线运行验证了完整网格、阈值复核、流式测速与结果比较。

复现网格从仓库根目录运行，使用新的输出目录；默认直接加载 `/workspace/models`，若使用缓存须先验证相同权重：

```bash
set -o pipefail
.venv/bin/python3 docs/perf/scripts/search_decoupled_serial.py \
    --output-dir docs/perf/results/decoupled-serial-search-new-run \
    --min-block 4 --max-block 8 --topks 4 3 2 --minimum-ready 0.95 \
    2>&1 | tee docs/perf/logs/decoupled_serial_search_new_run.log
```

本轮每个点仅 10 个相同数学输入，未覆盖并发、长上下文和更广泛任务；不同 top-k 的小速度差异不能据此判为稳定收益。严格的 verify 区间覆盖率还需要配对时间线，remote 命中后的 token 接受质量还需要单独分组计数。
