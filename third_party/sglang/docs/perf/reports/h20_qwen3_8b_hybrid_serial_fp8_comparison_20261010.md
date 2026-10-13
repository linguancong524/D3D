# H20 Qwen3-8B hybrid：远端 FP8 与全模型 FP8 对照（2026-10-10）

## 结论

**远端 FP8 能放宽候选预算；Target FP8 大幅提速，但本轮出现最终答案回退，不能作为无损优化。** 已完成原 15 点网格的两类 FP8 对照、同日原精度控制、top-k 扩展、独立候选复测、缓存诊断与完整答案小测。

- **保留 Target 原精度，仅远端 FP8**：B4/K11 初测 / 独立复测及时率 **97.85% / 98.54%**，实际命中 **55.92% / 55.39%**。对旧网格最高命中 51.92% 提高 3.47–4.00 pp；对同日扩大预算后的原精度 K8（54.35%）提高 1.04–1.57 pp。原精度 K11 两轮仅约 5% 及时率，FP8 让这个点可用；K12 两种精度均失败。仅远端 FP8 共 **230/230** 个测量请求保持原输出。
- **同配置速度**：远端 FP8 K11 hybrid 复测 **74.24 tok/s**，相对匹配 local 两轮 decode +5.48%–9.61%、完整输出吞吐 +5.12%–6.46%。但最快达标的 B8/K3 hybrid 复测 **107.19 tok/s**，仍慢于匹配 local 的 109.91 / 117.79，未证明整体速度最优点有额外 remote 收益。
- **Target / 本地 / 远端支持线性层均 FP8**：普通 AR 由约 **94 提高到 166 tok/s，约 +76%**。Target verify 缩短到约 6.2–6.3 ms，原缓存设置仅 3/15 点及时达标；初始同精度 AR 参考自身也不稳定。关闭 Target radix cache 后，独立重启 AR、本地与两个 hybrid 代表点恢复逐 token 一致；B4/K3、B5/K2 分别为 **92.55 / 105.83 tok/s**，仍低于普通 FP8 AR。
- **模型答案质量**：相同 10 题的完整答案对照，原精度 **10/10**，在线 FP8 **9/10**。第 10 题正确答案为 4，FP8 最终答成 1；已排除截断、LaTeX 提取和数学等价问题。样本不足以估计完整 Math500 的准确率损失，但已经观察到一例实际回退。

若优先满足本轮的保真目标，当前证据支持 **Target 保留 BF16，远端 FP8，B4/K11** 作为本次离散范围内的最高命中达标选择。Target 在线 FP8 的速度收益值得继续研究，但需先做量化质量修复与更广任务验证。本轮短串行负载不作为默认部署配置或 release 发版依据。

## 实验设置与精度

本轮接续 [10 月 9 日 block / top-k 搜索](h20_qwen3_8b_hybrid_serial_block_topk_search_20261009.md)，分别测试远端 Draft 单独 FP8，以及 Target、本地 required Draft、远端 optional Draft 的支持线性层全部 FP8。未修改推理核心源码；`python/sglang/` 与原实验基准 `be801f29dc50a48d6fdc19daf5c6424cfa4f0827` 相同。

| 实验 | Target 线性层 | 本地 Draft 线性层 | 远端 Draft 线性层 | 未量化部分 / KV 的原始 dtype |
|---|---|---|---|---|
| 原配置、同日控制组 | BF16 | FP16 | BF16 | Target / 远端 BF16，本地 FP16 |
| 仅远端 FP8 | BF16 | FP16 | FP8 W8A8 | 同原配置 |
| 全模型 FP8 | FP8 W8A8 | FP8 W8A8 | FP8 W8A8 | 同原配置 |

原报告将本地 Draft 写成 BF16；本轮核对旧日志中 `KV Cache ... dtype=torch.float16` 后修正。启动参数保持本地 `dtype=auto`，没有额外改成 BF16。FP8 使用同一份未量化 checkpoint，加载时在线量化权重，并按 token 动态量化激活；没有量化 KV、embedding、norm 或未支持的输出层。它不表示整个模型的所有运算和数据均为 FP8。

历史归档已有 [9 月 24 日远端 Draft FP8 实验](h20_qwen3_8b_decoupled_b16_fp8_math500_20260924.md)，其中 Target 与本地 Draft 未量化；其他报告中的 `--kv-cache-dtype fp8_e4m3` 仅指 KV 存储精度。所选 10 月 9 日搜索没有启用 FP8 线性层。这次新增了 Qwen3-8B Target 与本地 Draft 的 FP8 测速。

仍用 Math500 原始前 10 题、串行 active cap=1、每请求 256 token、temperature=0、ignore_eos、预热 1 次。GPU0 为 Target / 本地 Draft，GPU3 为远端 Draft；其余 GPU 空闲。Target FA3、Draft FlashInfer，H20 prefill 保留 BCG，捕获到 512 token；远端图上限 1024。全部接受位置 `1..B`，实际分支 `(B−1)×(K−1)+1`，query 为 `B×(分支数+1)`，零主动等待，local prefix reuse 和 packed causal pending 沿用原设置。

及时到达率与实际命中率沿用旧定义：ready / lookup 与 hit / lookup，不等于候选后续 token 接受率。远端服务耗时和 ready / hit 计数包含预热；测速、逐 token 校验、verify 计数及主路径成本排除预热。未采集逐轮配对的 publish / verify / ready 时间线，因此不能把及时率解释为严格的 verify GPU 区间覆盖率。低于 95% 或距阈值不超过 2 个百分点的点重测；合并计数判断 timely，表中的速度及耗时列采用首轮。

FP8 Target 的输出可以与原精度不同，因此全模型 FP8 另建同精度普通自回归（AR）参考，同时核对该参考自身的重复性。所有不匹配的原始结果与 `valid=false` 均保留；成功完成请求与通过保真校验分别统计。10 个输入、256 token 截断的逐 token 比较不能代替完整数学答案准确率评估。

## 仅远端 FP8：原网格对照

完整 15 点中，达标区域与旧实验一致：K4 最大达标 B6，K3 / K2 的 B4–8 均达标。B7/K4 合并两轮及时率 0.10%，B8/K4 为 0%；同日原精度 B7/K4、B8/K4 两轮也均为 0%。170/170 个测量请求保持原输出。

| B | K | FP8 及时率 | FP8 实际命中率 | 相对旧命中率 pp | decode p50 tok/s | 旧 / FP8 服务均值 ms | ≥95% |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 4 | 99.12% | 51.84% | -0.08 | 72.98 | 7.70 / 7.63 | 是 |
| 4 | 3 | 99.03% | 50.93% | +0.57 | 73.29 | 7.02 / 6.33 | 是 |
| 4 | 2 | 99.08% | 48.32% | -0.56 | 74.75 | 6.62 / 6.01 | 是 |
| 5 | 4 | 98.87% | 50.52% | +0.14 | 85.52 | 9.87 / 8.88 | 是 |
| 5 | 3 | 98.94% | 49.38% | -0.24 | 86.61 | 7.69 / 7.53 | 是 |
| 5 | 2 | 98.91% | 46.73% | -0.55 | 86.01 | 6.80 / 6.16 | 是 |
| 6 | 4 | 98.63% | 49.68% | -0.42 | 94.03 | 10.33 / 9.39 | 是 |
| 6 | 3 | 98.82% | 48.55% | +0.16 | 93.35 | 9.62 / 8.63 | 是 |
| 6 | 2 | 98.79% | 46.15% | -0.03 | 93.53 | 7.35 / 7.13 | 是 |
| 7 | 4 | 0.10% | 0.10% | -0.10 | 109.51 | 13.41 / 13.21 | 否 |
| 7 | 3 | 98.47% | 48.23% | +0.49 | 100.25 | 10.02 / 9.06 | 是 |
| 7 | 2 | 98.65% | 44.73% | +1.07 | 101.34 | 7.53 / 7.33 | 是 |
| 8 | 4 | 0.00% | 0.00% | +0.00 | 111.60 | 13.96 / 13.90 | 否 |
| 8 | 3 | 98.61% | 47.73% | +0.12 | 110.07 | 10.38 / 9.48 | 是 |
| 8 | 2 | 98.55% | 44.14% | +0.25 | 107.91 | 9.34 / 8.34 | 是 |

原网格最高命中仍为 B4/K4，51.84% 对旧值 51.92%；更长 block 没有提高命中率。B6/K4 服务从旧 10.33 ms 降至 9.39 ms，但 B7/K4 仍约 13.21 ms，未跨回达标区域。小 top-k / 小 query 下 FP8 收益有限；本轮不把不到 1 个百分点的命中浮动称为稳定改善。网格内最快达标点为 B8/K3，110.07 tok/s；它与 B8/K2 的小差异需要独立复测及同 K 本地对照。

## B4 top-k 扩展：FP8 放宽了高覆盖预算

选择 K8、K11、K12：对应 22 / 31 / 34 条分支与 92 / 128 / 140 个 query token，覆盖原网格外的更大候选预算。以下都是零主动等待、同日同 B / K 对照。未搜索所有整数 top-k 或其他 block，K11 仅为本次离散范围的最高命中达标点。

| K | 原精度及时率 | FP8 及时率 | 原精度实际命中 | FP8 实际命中 | 原精度 / FP8 GPU 均值 ms | 原精度 / FP8 服务均值 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 98.75% | 99.15% | 54.35% | 55.04% | 7.56 / 6.58 | 10.83 / 9.78 |
| 11 | 5.10% | 97.85% | 4.37% | 55.92% | 7.74 / 6.77 | 12.01 / 10.99 |
| 12 | 0.00% | 0.00% | 0.00% | 0.00% | 10.16 / 10.06 | 14.64 / 14.65 |

K11 原精度两轮及时率 5.81% / 4.38%，FP8 首轮 97.85%；GPU 前向从 7.74 降到 6.77 ms，完整服务从 12.01 降到 10.99 ms。这组同配置证据表明，较小的服务时间节省能改变边界点的实际可用性。K12 两种精度均为 0% 及时率，FP8 没有恢复它。

FP8 K11 首轮 55.92% 实际命中，对原网格 B4/K4 的 51.92% 高 4.00 pp；但同日原精度 K8 已达到 54.35%，因此对本轮扩展后的原精度达标点，差值为 **1.57 pp**。前者同时包含扩大 top-k 的收益，不能全归于 FP8。这个小差值只描述相同 10 题下的观察，不作为广泛负载的显著性结论。独立重启结果另见下节。

## 候选独立重启与同配置本地对照

| 候选 | FP8 及时率 / 实际命中 | FP8 hybrid decode tok/s | 原精度 local 首轮 / 复测 tok/s | hybrid 相对 local 两轮 |
|---|---:|---:|---:|---:|
| B4/K11 | 98.54% / 55.39% | 74.24 | 67.73 / 70.38 | +9.61% / +5.48% |
| B8/K3 | 98.63% / 47.95% | 107.19 | 109.91 / 117.79 | -2.48% / -8.99% |

K11 初测与独立复测分别为 97.85% / 98.54% 及时率、55.92% / 55.39% 实际命中，均逐 token 保持原输出。相对旧网格最高命中 51.92%，提高 3.47–4.00 pp；对扩展后的同日原精度 K8（54.35%），提高 1.04–1.57 pp。K11 hybrid 的完整输出吞吐对匹配 local 两轮提高 6.46% / 5.12%。这支持它在本轮样本中以更高覆盖跳过一部分 Draft，但没有证明广泛负载下的稳定提升。

最快达标的 B8/K3 复测比匹配 local 两轮均慢，完整输出吞吐为 −4.58% / −6.41%。同日原精度 hybrid B8/K3 为 109.22 tok/s，远端 FP8 为 107.19 tok/s；这个小差异不支持 FP8 带来整体加速。较高命中 K11 的约 74 tok/s 也低于 B8 本地方案，因此它是本次最高命中的达标选择，不能称为整体速度最优。

以下用独立复测 hybrid 与匹配 local 首轮核算主路径成本，均为 10 请求 / 2560 输出 token，排除预热：

| 成本 | K11 local | K11 hybrid | B8/K3 local | B8/K3 hybrid |
|---|---:|---:|---:|---:|
| 本地 Draft 调用次数 | 737 | 537 | 461 | 392 |
| Target verify 次数 | 737 | 1195 | 461 | 745 |
| 有效输出 token / verify | 3.474 | 2.142 | 5.553 | 3.436 |
| 本地 Draft 总耗时 s | 27.444 | 19.376 | 17.207 | 14.758 |
| Target verify 总耗时 s | 7.970 | 12.948 | 5.008 | 8.103 |
| 其余未单独计时 s | 1.456 | 2.309 | 1.016 | 1.485 |
| 完整 wall time s | 36.871 | 34.633 | 23.231 | 24.345 |

两点都省下了部分本地 Draft，但有效输出 token / verify 下降，增加了 verify 轮数。K11 的 Draft 节省仍能覆盖增加的成本，B8/K3 则无法覆盖。当前候选结构仍有旧报告指出的全接收分支条件差异；本轮没有按 remote-hit / fallback 分组记录后续接受长度，不能将全部差额精确归因到某类分支。FP8 没有自动消除这个候选质量成本。

## Target / 本地 FP8：同日速度与参考重复性

| 模式 | 原精度 decode 首轮 / 复测 tok/s | FP8 decode 首轮 / 复测 tok/s | FP8 匹配同精度 AR 首轮参考 |
|---|---:|---:|---:|
| 普通 AR | 93.62 / 93.85 | 164.99 / 165.81 | 首轮为参考，复测 6/10 |
| 本地 B4/K4 | 69.22 / 71.90 | 82.91 / 83.67 | 两轮均 9/10 |
| 本地 B8/K4 | 115.28 / 118.40 | 137.05 / 138.76 | 两轮均 9/10 |

FP8 普通 AR 的 decode p50 首轮相对同日原精度首轮提高 **76.24%**，复测相对原精度复测提高 **76.69%**，约 165 对 94 tok/s。固定相同输入和输出长度，但生成 token 并非完全相同，故它是这组负载的部署速度对照，不能视为同一生成轨迹的逐算子加速比。FP8 local B8 约 138 tok/s，仍慢于同日 FP8 普通 AR。

原精度 AR 和本地 B4 / B8 的两轮均为 10/10 原输出匹配。FP8 AR 首轮与原精度仅 1/10 请求完整一致；第二轮与它自己的首轮仅 6/10，一致性失败出现在请求 4、5、7、9，首个差异位置分别为 11、40、76、173。FP8 本地 B4 / B8 两轮彼此 10/10 一致，但对 FP8 AR 首轮均为 9/10，请求 0 在位置 5 分歧。

因此，默认缓存配置下，同精度 AR 参考自身尚不稳定；全 FP8 网格即使有及时率达标点，也不能凭这个参考宣布严格保真通过。原始结果保留为诊断测速，后面的 Target cache-off 实验单独检验缓存条件的影响。

## 全模型 FP8：完整 15 点

完成 27 轮、270 个测量请求，全部运行成功、每请求 256 token；每轮对 FP8 AR 首轮参考均为 9/10，分歧仍是请求 0、token 位置 5。因此，表中“及时达标”只表示 ready ≥95%，不表示通过严格保真。默认缓存配置下没有同时满足这两个条件的全 FP8 点。

| B | K | 及时到达率 | 实际命中率 | decode p50 tok/s | 远端服务均值 ms | verify p50 ms | 及时 ≥95% |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 4 | 0.86% | 0.74% | 83.12 | 7.60 | 6.18 | 否 |
| 4 | 3 | 98.79% | 50.65% | 92.74 | 6.35 | 6.17 | 是 |
| 4 | 2 | 98.58% | 47.87% | 92.02 | 6.01 | 6.18 | 是 |
| 5 | 4 | 0.15% | 0.15% | 97.28 | 8.90 | 6.25 | 否 |
| 5 | 3 | 3.29% | 2.05% | 99.88 | 7.57 | 6.23 | 否 |
| 5 | 2 | 98.54% | 46.30% | 105.51 | 6.15 | 6.22 | 是 |
| 6 | 4 | 0.09% | 0.09% | 115.79 | 9.40 | 6.24 | 否 |
| 6 | 3 | 0.00% | 0.00% | 115.83 | 8.73 | 6.24 | 否 |
| 6 | 2 | 79.86% | 37.89% | 118.41 | 7.21 | 6.25 | 否 |
| 7 | 4 | 0.00% | 0.00% | 125.20 | 13.15 | 6.25 | 否 |
| 7 | 3 | 0.00% | 0.00% | 125.03 | 9.11 | 6.27 | 否 |
| 7 | 2 | 19.64% | 10.63% | 127.91 | 7.34 | 6.27 | 否 |
| 8 | 4 | 0.00% | 0.00% | 135.47 | 13.82 | 6.26 | 否 |
| 8 | 3 | 0.00% | 0.00% | 136.60 | 9.46 | 6.26 | 否 |
| 8 | 2 | 0.20% | 0.20% | 137.60 | 8.37 | 6.26 | 否 |

相比原网格 13/15 及时达标，全 FP8 仅 **3/15**：B4/K3、B4/K2、B5/K2。及时达标点中最高实际命中是 B4/K3 的 50.65%，最快是 B5/K2 的 105.51 tok/s，仍低于 FP8 普通 AR 的约 165 tok/s。B8 各点虽约 135–138 tok/s，但及时率接近 0%，主要是本地回退，不能计为远端预投机收益。

Target verify 从原配置约 10.8 ms 降至 6.2–6.3 ms，远端服务的相对下降较小。这与及时达标区域收缩一致；没有配对时间线，不能仅用两项均值证明逐轮先后关系。例如 B4/K3 服务均值 6.35 ms 略高于 verify 6.17 ms，仍有 98.79% 及时率，说明消费时刻还受 verify 前后步骤影响，均值不是精确的 deadline 不等式。

全 FP8 的 AR 参考自身复测仅 6/10，因此原设置的候选验证明确记为参考不稳定，未把这些诊断点列为保真候选。后面的 cache-off 结果另列，保留原网格的所有失败记录。

## Cache-off 诊断与同精度保真复核

只关闭 Target radix cache，保持本地 Draft prefix reuse、所有模型精度和其余启动参数。实际初始化日志核对：原 AR / local / hybrid 的 `disable_radix_cache=False`，本组为 `True`。远端 Engine 原先就关闭 radix cache，不是本次变化项。

| 模式，Target cache-off | decode 首轮 / 复测 tok/s | 输出参考与匹配 |
|---|---:|---|
| 原精度 AR | 93.77 / 94.24 | 两轮均 10/10 匹配原精度旧参考 |
| FP8 AR | 165.51 / 165.86 | 首轮建立参考；复测 10/10 |
| FP8 AR 独立重启 | 166.21 / 166.16 | 两轮均 10/10 匹配 cache-off FP8 AR |
| FP8 local B4/K4 | 83.78 / 83.77 | 两轮均 10/10 匹配 cache-off FP8 AR |
| FP8 local B8/K4 | 130.29 / 132.82 | 两轮均 10/10 匹配 cache-off FP8 AR |

FP8 AR 的复测与独立重启三轮均对首轮参考 10/10，速度仍约 166 tok/s。关闭缓存后，本地 B4 / B8 也与同精度 AR 完全对齐。这支持缓存配置影响本轮 FP8 的数值一致性；尚未做逐层 logits / KV 误差 trace，不能定性某一个内核的具体原因。原精度 AR 两轮仍保持旧输出。

原 cached FP8 AR 首轮对 cache-off FP8 AR 仅 6/10，cached local 与已完成 hybrid 首轮样例为 7/10；因此不能简单把旧网格换成新参考就宣布通过。旧结果的 `valid=false` 与原始参考均保留。以下是独立运行的 cache-off hybrid，使用已验证重复性的同精度参考：

| 点 | 及时率 | 实际命中率 | hybrid decode tok/s | 同 B/K FP8 local 两轮 tok/s | hybrid 相对 local 两轮 | 保真 |
|---|---:|---:|---:|---:|---:|:---:|
| B4/K3 | 98.79% | 50.81% | 92.55 | 82.87 / 82.83 | +11.67% / +11.73% | 10/10 |
| B5/K2 | 98.72% | 46.17% | 105.83 | 101.59 / 100.63 | +4.18% / +5.18% | 10/10 |

两点均通过及时率与同 FP8 Target AR 的逐 token 保真，且比匹配的 FP8 local 更快；但它们的 92.55 / 105.83 tok/s 仍低于普通 FP8 AR 的约 166 tok/s。只复测了最高命中和最快及时的两个代表点，没有重跑 cache-off 全矩阵，不能将它们称为该新设置的全局上界。

独立审计的 Target FP8 cache-off 组共有 140 个请求，含参考自身 10 个，全部直接 token-list 比较匹配稳定参考；其中 hybrid 20/20、本地组 80/80、AR 自身复测 / 独立重启 30/30。所有请求仍是同 10 个输入的重复运行。**这是对同精度 Target 的投机保真，不表示 FP8 与 BF16 的生成轨迹相同**：cache-off FP8 AR 与旧 BF16 仅 1/10 全序列一致，最终答案质量另做完整答案小对照。

## 本地主路径成本：Target FP8 收益为何被 Draft 稀释

下表使用本地 B8/K4 各自首轮、相同 10 个输入和 2560 输出 token，排除预热；两组输出轨迹不同，成本核算说明固定负载的构成，不能精确归因每个 token 的数值变化。

| 主路径成本 | 原配置 local B8 | FP8 local B8 |
|---|---:|---:|
| 本地 Draft 总 wall time s | 16.852 | 16.247 |
| Target verify GPU 总时间 s | 5.009 | 2.914 |
| 其余未单独计时部分 s | 1.010 | 0.961 |
| 完整 wall time s | 22.872 | 20.122 |
| verify 次数 | 461 | 465 |
| 有效输出 token / verify | 5.553 | 5.505 |

FP8 local B8 的本地 Draft 仍占主路径 wall time **80.7%**。Target verify 明显变快，但本地 Draft 总时间仅小幅下降，故普通 AR 的约 76% decode 收益在本地投机模式下被稀释为约 18%。远端若无法及时命中，不能减少这部分 Draft 成本；即使命中，仍需核对每轮接受质量与增加的 verify 次数。

主路径成本由原始日志的 11 次 request-finish 边界排除预热，独立核对 `fallback_count + hit_count = verify_count` 与请求内 verify 总计。其余部分为 wall 减去两项计时的剩余，包含未拆分步骤；不将它直接命名为 RPC 或网络成本。远端服务时间与 Target verify 重叠，也不加到主路径总和。

## 完整答案小对照：观察到一例 Target FP8 回退

速度实验的 256 token 输出均没有完整 boxed 最终答案，因此另开普通 AR、Target cache-off 的质量组：相同前 10 题、相同 `enable_thinking=False` 模板、temperature=0，允许 EOS 停止，初始上限 2048。两组各有 9 题完成且答案正确，第 10 题均截断；仅这道题对两组统一延长到 4096。原始截断记录保留。

| Target 精度 | 前 9 题 | 第 10 题，完整答案 | 最终正确数 / 同 10 题 |
|---|---:|---|---:|
| BF16 | 9/9 | 2075 token，答案 `4`，正确 | **10/10** |
| 在线 FP8 | 9/9 | 2785 token，答案 `1`，错误 | **9/10** |

第 10 题要求在 `2×3×4×5+1` 中插入括号，统计不同结果。标准解列出 **121、122、126、144**，共 4 种；独立枚举所有保持项与运算符顺序的二叉括号表达式也得到这四个值。FP8 生成结尾只考虑乘积的结合顺序，遗漏将 `+1` 包进乘法子表达式，最终明确输出 `\boxed{1}`。这是实质错答，非格式差异。

两组延长运行的前 2048 token 均与原截断输出完全相同；全部 22 次答案生成的前 256 token 均匹配对应的性能参考。完整答案试验与速度网格分开，未把它的延迟或生成长度混进 tok/s 对照。评分采用最后一个完整 boxed 答案的规范化精确匹配，唯一错例又按标准解与独立算术枚举核对。

原始回答、输出 ID、EOS / length finish reason、标准答案和审计见 [质量运行目录](../results/decoupled-serial-fp8-quality-h20-20261010/) 与 [quality_audit.json](../results/decoupled-serial-fp8-quality-h20-20261010/quality_audit.json)。这 10 题不是独立随机抽样，不能把 10/10→9/10 外推为整个 Math500 下降 10 pp；但本轮在线 FP8 已不能被描述为模型能力无损。cache-off 恢复同精度投机保真，也不能消除 Target 量化的这个答案变化。

## FP8 内核证据

本轮 [FP8 AR trace](../profiles/decoupled_serial_fp8_h20_20261010/fp8_ar/fp8_ar-TP-0.trace.json.gz) 包含 `per_token_quant_fp8_cta_kernel<__nv_bfloat16>` 与 `cutlass_3x_gemm_sm90_fp8`；[FP8 local B4 trace](../profiles/decoupled_serial_fp8_h20_20261010/fp8_local_b4/fp8_local_b4-TP-0.trace.json.gz) 同时包含 Target 的 BF16 输入量化和本地 Draft 的 `__half` 输入量化。确认实际执行原生 FP8 GEMM。profile 请求在测速结束后执行，不计入 benchmark 时间或计数。

NVIDIA 的 [NIM 支持表](https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-models.html) 也列出 H20 的 FP8 部署配置。原生 FP8 计算能力可用于这些线性层，但完整部署收益仍取决于 Draft 成本、verify 窗口与候选质量；本报告以实际测速判断。

## 证据与复现

模型缓存沿用 `/tmp/sglang-hybrid-model-cache`，启动前重新核对 39 个文件，SHA256 全部匹配旧实验 manifest。模型缓存是相同权重的加载优化，不是实验输出，也不提交权重。启动、模型加载、图捕获和 profiler 均在测速外。

软件为 Python 3.12.14、torch 2.13.0（cu130）、FlashInfer 0.6.17、sglang-kernel 0.4.6.post1、Triton 3.7.1、Transformers 5.12.1；4×H20，驱动 580.105.08。editable SGLang 的包 metadata 为 `0.5.19.dev39+g91a20e1`，可早于源码，比较以实际核心树及逐点启动命令为准。完整记录见 [software_versions.json](../results/decoupled-serial-remote-fp8-h20-20261010/software_versions.json)。

最终独立审计覆盖 **61 个配置目录、950 个性能测量请求**，每个成功生成 256 token，无远端服务错误。直接 token-list 比较、原始 request-finish 计数、加权 ready / hit、95% 分类、请求级 SSE p50 / 吞吐和主路径计数均已重算；各份 JSON 的原始 `valid` 状态保留。所有性能与完整答案运行结束后，GPU 无残留计算进程。

| 性能样本组 | 请求数，均复用相同 10 题 | 直接逐 token 比较 |
|---|---:|---|
| 原精度控制 / 本地匹配对照 | 250 | 250/250 匹配旧原精度参考 |
| 仅远端 FP8 | 230 | 230/230 匹配旧原精度参考 |
| Target FP8，默认缓存组 | 330 | 初始参考自身不稳定；保留诊断结果 |
| Target FP8，cache-off 组 | 140 | 140/140 匹配稳定同精度参考，含参考自身 10 个 |

另有 **22 次完整答案生成**，与性能样本分开统计。审计见 [final_audit.json](../results/decoupled-serial-fp8-controls-h20-20261010/final_audit.json)、[audit_summary.json](../results/decoupled-serial-fp8-controls-h20-20261010/audit_summary.json) 与质量组的 `quality_audit.json`。95 个测速轮次各有 1 个额外预热请求；这些预热不计入 950，但按旧口径纳入服务耗时及 lookup 比率。

运行期间新增或完善了性能辅助脚本，推理核心保持与 `be801f2` 一致。父控制进程加载版本由 `suite.json` 和 [loaded_helpers_manifest.json](../results/decoupled-serial-fp8-controls-h20-20261010/loaded_helpers_manifest.json) 固定；逐点 `run.json` 的 commit 是启动当时仓库 HEAD，可能不同于已在内存加载的父辅助脚本。归档保存实际父脚本快照、最终审计入口、启动命令与 SHA256；最终入口快照见 [final_source_manifest.json](../results/decoupled-serial-fp8-controls-h20-20261010/final_source_manifest.json)。结束时的 [GPU 清单](../results/decoupled-serial-fp8-controls-h20-20261010/gpu_inventory_after.txt) 与 [计算进程清单](../results/decoupled-serial-fp8-controls-h20-20261010/gpu_compute_apps_after.txt) 均已保存。清理时的 SIGTERM / NCCL 退出告警发生在结果落盘后。

- [远端 FP8 运行目录](../results/decoupled-serial-remote-fp8-h20-20261010/)：模型校验、GPU / 软件版本、完整网格和 top-k 覆盖扩展。
- [同日控制组与全模型 FP8 网格](../results/decoupled-serial-fp8-controls-h20-20261010/)：普通 AR、本地投机、原精度 hybrid 控制、全 FP8 网格。
- [独立候选复测](../results/decoupled-serial-fp8-validation-h20-20261010/)：最高命中 / 最快达标候选以及同 B / K 的本地基线。
- [缓存诊断](../results/decoupled-serial-fp8-cache-probe-h20-20261010/)：单独关闭 Target radix cache，保持本地 Draft prefix reuse；不混入原设置矩阵。
- [完整答案质量对照](../results/decoupled-serial-fp8-quality-h20-20261010/)：2048 上限初测、共同截断题的 4096 配对复查及独立算术核对。

从仓库根目录复现，必须使用新的输出目录；若直接使用 `/workspace/models`，将 `--model-cache` 改为该目录：

```bash
set -o pipefail
mkdir -p docs/perf/results/decoupled-serial-remote-fp8-new
.venv/bin/python3 docs/perf/scripts/search_decoupled_serial.py \
  --output-dir docs/perf/results/decoupled-serial-remote-fp8-new/grid \
  --model-cache /tmp/sglang-hybrid-model-cache \
  --remote-draft-quantization fp8 \
  2>&1 | tee docs/perf/logs/decoupled_serial_remote_fp8_new.log

.venv/bin/python3 docs/perf/scripts/run_decoupled_serial_fp8_controls.py \
  --output-dir docs/perf/results/decoupled-serial-fp8-controls-new \
  --model-cache /tmp/sglang-hybrid-model-cache \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_controls_new.log

.venv/bin/python3 docs/perf/scripts/search_decoupled_serial.py \
  --output-dir docs/perf/results/decoupled-serial-remote-fp8-new/coverage_extension \
  --min-block 4 --max-block 4 --topks 8 11 12 \
  --model-cache /tmp/sglang-hybrid-model-cache --remote-draft-quantization fp8 \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_coverage_new.log

.venv/bin/python3 docs/perf/scripts/validate_decoupled_serial_fp8.py \
  --remote-dir docs/perf/results/decoupled-serial-remote-fp8-new \
  --controls-dir docs/perf/results/decoupled-serial-fp8-controls-new \
  --output-dir docs/perf/results/decoupled-serial-fp8-validation-new \
  --model-cache /tmp/sglang-hybrid-model-cache \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_validation_new.log

.venv/bin/python3 docs/perf/scripts/run_decoupled_serial_fp8_cache_controls.py \
  --output-dir docs/perf/results/decoupled-serial-fp8-cache-probe-new \
  --model-cache /tmp/sglang-hybrid-model-cache \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_cache_probe_new.log

.venv/bin/python3 docs/perf/scripts/validate_decoupled_serial_fp8_cache.py \
  --probe-dir docs/perf/results/decoupled-serial-fp8-cache-probe-new \
  --controls-dir docs/perf/results/decoupled-serial-fp8-controls-new \
  --model-cache /tmp/sglang-hybrid-model-cache \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_cache_validation_new.log
```

第二个命令建立 FP8 AR 参考，运行同日原配置控制组，并搜索全模型 FP8。诊断模式允许完整保存输出不匹配的测速结果，不会将它们标为保真通过。搜索、控制、候选复测与审计脚本分别为 [search_decoupled_serial.py](../scripts/search_decoupled_serial.py)、[run_decoupled_serial_fp8_controls.py](../scripts/run_decoupled_serial_fp8_controls.py)、[validate_decoupled_serial_fp8.py](../scripts/validate_decoupled_serial_fp8.py)、[analyze_decoupled_serial_fp8.py](../scripts/analyze_decoupled_serial_fp8.py)。

完整答案小对照使用 [check_decoupled_serial_fp8_answer_quality.py](../scripts/check_decoupled_serial_fp8_answer_quality.py)，给新的输出目录、相同 model cache 和稳定的 cache-off FP8 AR 参考；共同截断题另存子目录：

```bash
set -o pipefail
.venv/bin/python3 docs/perf/scripts/check_decoupled_serial_fp8_answer_quality.py \
  --output-dir docs/perf/results/decoupled-serial-fp8-quality-new \
  --model-cache /tmp/sglang-hybrid-model-cache \
  --fp8-reference-results docs/perf/results/decoupled-serial-fp8-cache-probe-new/fp8_ar_cacheoff/serial_n10.json \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_quality_new.log

.venv/bin/python3 docs/perf/scripts/check_decoupled_serial_fp8_answer_quality.py \
  --output-dir docs/perf/results/decoupled-serial-fp8-quality-new/extended_max4096 \
  --model-cache /tmp/sglang-hybrid-model-cache \
  --fp8-reference-results docs/perf/results/decoupled-serial-fp8-cache-probe-new/fp8_ar_cacheoff/serial_n10.json \
  --max-output-tokens 4096 --indices 9 \
  2>&1 | tee docs/perf/logs/decoupled_serial_fp8_quality_extended_new.log
```

每个配置最多 10 个相同输入，未覆盖并发、长上下文、完整 Math500 或其他任务；小幅速度和命中差异不作广泛显著性结论。实验已归档于 main，未改变默认部署精度或发布 release。
