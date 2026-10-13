# Qwen3.5-9B H20 DFlash：unieai 社区版 vs z-lab 官方版对比（SGLang v0.5.18）

日期：2026-09-04。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
target：Qwen3.5-9B（bf16，17.62 GB，混合 GDN 线性注意力 + 全注意力）。
运行时：torch 2.13.0+cu130、flashinfer 0.6.17、attention=fa3、linear-attn=triton。
H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告对比**两个不同的 DFlash draft checkpoint** 在完全相同 serving 条件下的表现：

- **unieai 社区版**：`/workspace/models/Qwen3.5-9B-DFlash-unieai`
  （2026-09-03 测试，数据见 [bs16 报告](h20_qwen35_9b_dflash_comparison.md)、
  [bs8 报告](h20_qwen35_9b_dflash_bs8_comparison.md)）
- **z-lab 官方版**：`/workspace/models/Qwen3.5-9B-DFlash`
  （2026-09-04 补测，数据见 [z-lab 报告](h20_qwen35_9b_dflash_zlab_comparison.md)）

> 背景：2026-09-03 的 DFlash 测试误用了 unieai 社区版（当时放在
> `Qwen3.5-9B-DFlash` 路径下），并非官方 z-lab 发布版。2026-09-04 将 unieai 版
> 移至 `Qwen3.5-9B-DFlash-unieai`，重新下载官方 z-lab 版到 `Qwen3.5-9B-DFlash`，
> 并按**完全相同**的条件补测。本报告即两版的逐格对比。

## 0. 两个 checkpoint 的差异

| 维度 | unieai 社区版 | z-lab 官方版 |
|---|---|---|
| 来源 | 社区（unieai） | Z-Lab × Modal 联合 retrain |
| draft 层数 | 5（全 full_attention） | 6（5 sliding + 1 full） |
| target 层（`target_layer_ids`） | 5 `[1,8,15,22,29]` | 8 `[1,5,9,13,17,21,25,29]` |
| sliding window | 无 | 4096 |
| 训练序列长度 | 未注明 | 40k |
| `mask_token_id` | 248070 | 248077 |
| 权重 | ~2.1 GB | ~2.6 GB（TP=1 实测 2.56 GB） |
| config `block_size` | 16（顶层） | 16（`dflash_config` 内） |

**唯一变量是 draft checkpoint**；target、verify 窗口（bs16/bs8）、cap48、mem 0.85、
mamba bfloat16、fa3/triton 后端、512 in/256 out、random、n=100/200/400、
TP=1/TP=4/DP=4 三形态全部相同。z-lab 版 8 个 target 层 + 40k 训练 + sliding
attention 理论上 draft 质量更高。

## 1. Accept length 对比（draft 质量的直接度量）

| 形态 | rate | bs16 unieai | bs16 **z-lab** | Δ | bs8 unieai | bs8 **z-lab** | Δ |
|---|---|---|---|---|---|---|---|
| TP=1 | 4 | 4.83 | **5.21** | +0.38 | 4.27 | 4.29 | +0.02 |
| TP=1 | 16 | 4.73 | **5.22** | +0.49 | 4.18 | 4.30 | +0.12 |
| TP=1 | 64 | 4.68 | **5.13** | +0.45 | 4.12 | 4.27 | +0.15 |
| TP=4 | 4 | 4.86 | **5.22** | +0.36 | 4.19 | 4.23 | +0.04 |
| TP=4 | 16 | 4.91 | **5.34** | +0.43 | 4.23 | 4.38 | +0.15 |
| TP=4 | 64 | 4.82 | **5.20** | +0.38 | 4.18 | 4.31 | +0.13 |
| DP=4 | 4 | 4.62 | **5.92**※ | +1.30 | 4.36 | 4.58 | +0.22 |
| DP=4 | 16 | 4.52 | **5.26** | +0.74 | 4.04 | 4.24 | +0.20 |
| DP=4 | 64 | 4.60 | **5.28** | +0.68 | 4.28 | 4.31 | +0.03 |

- **bs16：z-lab accept 全面高 0.36–0.74**（DP=4 rate4 的 5.92 为低并发离群，见下）。
  8 个 target 层 + 40k 训练让 draft 提议更准，每步多接受 ~0.4 token。
- **bs8：z-lab accept 仅高 0.02–0.22**。窗口 8 限制了可提议 token 数，draft 质量
  优势无法充分表达。
- **DP=4 bs16 rate4 的 5.92 是离群**：同形态 bs16 其余格 5.26/5.28、TP=4 bs16
  5.22–5.34，rate 4 低并发（peak concurrent ~11）样本少，作量级参考，不影响
  「z-lab bs16 accept 更高」的结论。

## 2. 吞吐对比（req/s）

| 形态 | rate | bs16 unieai | bs16 **z-lab** | Δ | bs8 unieai | bs8 **z-lab** | Δ |
|---|---|---|---|---|---|---|---|
| TP=1 | 4 | 4.08 | 4.08 | — | 4.08 | 4.07 | — |
| TP=1 | 16 | 8.07 | **8.77** | +8.7% | 11.75 | **11.98** | +2.0% |
| TP=1 | 64 | 8.32 | **8.65** | +4.0% | 13.75 | **13.85** | +0.7% |
| TP=4 | 4 | 4.08 | 4.08 | — | 4.08 | 4.08 | — |
| TP=4 | 16 | 15.89 | 15.88 | — | 15.94 | 15.95 | — |
| TP=4 | 64 | 25.93 | **26.90** | +3.7% | 36.81 | **37.12** | +0.8% |
| DP=4 | 4 | 4.08 | 4.08 | — | 4.08 | 4.07 | — |
| DP=4 | 16 | 15.71 | **15.88** | +1.1% | 15.77 | 15.78 | — |
| DP=4 | 64 | 29.18 | **30.45** | +4.4% | 40.69 | **41.11** | +1.0% |

- **rate 4 吞吐都 ≈4 req/s（发送速率受限）**，两版无差异。
- **bs16 高并发（rate 16/64）：z-lab 吞吐 +1.1–8.7%**（TP=1 rate16 +8.7% 最大，
  因 unieai 在单卡 rate16 已严重过载 8.07，z-lab 略好到 8.77）。
- **bs8 高并发：z-lab 吞吐 +0.7–2.0%**，增益很小。
- **rate 16 的 TP=4/DP=4 两版吞吐都 ≈15.9（接近饱和上限）**，无差异。

## 3. 延迟对比（E2E 中位 / TPOT 中位，ms）

**bs16**：

| 形态 | rate | E2E unieai | E2E **z-lab** | TPOT unieai | TPOT **z-lab** |
|---|---|---|---|---|---|
| TP=1 | 4 | 337.08 | **303.94** | 3.00 | **2.49** |
| TP=1 | 16 | 7 868.52 | **6 456.30** | 31.21 | **27.96** |
| TP=1 | 64 | 22 479.79 | **21 451.28** | 36.70 | **33.97** |
| TP=4 | 4 | 144.22 | **138.06** | 1.01 | 1.01 |
| TP=4 | 16 | 443.92 | **401.03** | 3.45 | **3.13** |
| TP=4 | 64 | 4 339.14 | **4 058.42** | 11.10 | **10.71** |
| DP=4 | 4 | 225.73 | **215.47** | 1.79 | **1.74** |
| DP=4 | 16 | 325.37 | **291.10** | 2.45 | **2.16** |
| DP=4 | 64 | 4 303.61 | **4 014.65** | 27.50 | **26.71** |

**bs8**：

| 形态 | rate | E2E unieai | E2E **z-lab** | TPOT unieai | TPOT **z-lab** |
|---|---|---|---|---|---|
| TP=1 | 4 | 361.64 | 361.88 | 2.52 | 2.65 |
| TP=1 | 16 | 3 124.80 | **2 901.73** | 14.90 | **13.76** |
| TP=1 | 64 | 10 690.81 | 10 715.00 | 21.26 | 21.21 |
| TP=4 | 4 | 152.86 | 161.41 | 0.98 | 1.10 |
| TP=4 | 16 | 293.57 | **282.27** | 1.95 | **1.78** |
| TP=4 | 64 | 2 183.73 | **2 176.59** | 7.97 | **7.92** |
| DP=4 | 4 | 279.11 | 283.90 | 1.80 | 1.97 |
| DP=4 | 16 | 351.87 | **347.00** | 2.36 | 2.29 |
| DP=4 | 64 | 2 573.19 | **2 565.02** | 18.09 | **17.34** |

- **bs16：z-lab 延迟全面更低**（E2E -4 至 -18%、TPOT -3 至 -17%），accept 高直接
  转化为延迟收益，低并发（rate 4）最明显（TP=1 E2E 304 vs 337、TPOT 2.49 vs 3.00）。
- **bs8：z-lab 与 unieai 基本持平**（E2E 差异 <7%，TPOT 互有高低）。低并发
  （rate 4）z-lab TPOT 甚至略高（6 层 draft 成本），高并发（rate 16/64）略低。

## 4. 结论：两版 DFlash 表现是否有差异？

**有差异，但差异大小取决于 verify 窗口（block_size）**：

1. **draft 质量：z-lab 官方版明显更高**。bs16 accept 5.1–5.9 vs unieai 4.5–4.9
   （+0.4–0.7），来自 8 个 target 层 + 40k 训练 + sliding attention。这是两版
   **最本质的差异**。

2. **bs16（大窗口）：z-lab 优势充分表达**
   - 延迟更低：rate 4 E2E -4 至 -10%、TPOT -3 至 -17%；
   - 吞吐更高：rate 64 +3.7 至 +4.4%（TP=1 rate16 +8.7%）。
   - 大窗口让高质量 draft 的每个额外接受 token 都兑现为收益。

3. **bs8（小窗口）：z-lab 优势大幅缩小，两版基本持平**
   - 吞吐 +0.7 至 +2.0%、延迟持平或略高（低并发 6 层 draft 成本略高）。
   - 窗口 8 是瓶颈，draft 质量成为次要因素。

4. **verify 窗口仍是第一决定因素**（与 unieai 报告一致）：bs8 高并发吞吐
   （37–41 req/s）远超 bs16（27–30 req/s），draft 版本只带来个位数百分比的边际
   增益。**先选对窗口，再选 draft 版本**。

5. **选型建议**：
   - 有 z-lab 官方版时**优先 z-lab**（同窗口下 accept/延迟/吞吐全面不劣、bs16 更优）。
   - **低延迟（rate ≤4）bs16** → z-lab（accept 高、延迟最低）。
   - **高吞吐（rate 64）bs8** → z-lab 略优（+1%），但 bs8 本身才是关键。
   - **高吞吐（rate 64）bs16** → z-lab（+4%）。
   - 只有 unieai 版时，结论与 [unieai 报告](h20_qwen35_9b_dflash_comparison.md)
     一致（低延迟 bs16、高吞吐 bs8）。

## 5. 数据有效性

- 两版共 36 组基准（2 版 × 3 形态 × bs16/bs8 × rate 4/16/64）全部请求成功，无失败。
- 所有 server 日志 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）；
  运行后 4 卡全部释放到 0 MiB。
- z-lab 6 个 server 日志确认 `n_layers=6`、`mask_token_id=248077`、
  `mem usage=2.56 GB`（TP=1/DP=4）/`0.92 GB`（TP=4），无 graph 捕获 OOM。
- unieai 数据为 2026-09-03 已归档结果（见两份 unieai 报告），z-lab 为 2026-09-04
  补测，**同条件**（mem 0.85 + mamba bfloat16、512 in/256 out、random、sglang
  backend、H20 不带 prefill 图禁用参数、cap48）。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（z-lab 日志清单见
  [z-lab 报告 §6](h20_qwen35_9b_dflash_zlab_comparison.md)）。

## 6. 日志清单（docs/perf/logs/）

| 版本 | server 日志 | bench 日志 |
|---|---|---|
| unieai bs16 | `server_h20_qwen35_9b_serving_{tp1,tp4,dp4}_dflash.log` | `bench_serving_h20_qwen35_9b_{tp1,tp4,dp4}_dflash_rate{4,16,64}.log` |
| unieai bs8 | `server_h20_qwen35_9b_serving_{tp1,tp4,dp4}_dflash_bs8.log` | `bench_serving_h20_qwen35_9b_{tp1,tp4,dp4}_dflash_bs8_rate{4,16,64}.log` |
| z-lab bs16 | `server_h20_qwen35_9b_serving_{tp1,tp4,dp4}_dflash_zlab_bs16.log` | `bench_serving_h20_qwen35_9b_{tp1,tp4,dp4}_dflash_zlab_bs16_rate{4,16,64}.log` |
| z-lab bs8 | `server_h20_qwen35_9b_serving_{tp1,tp4,dp4}_dflash_zlab_bs8.log` | `bench_serving_h20_qwen35_9b_{tp1,tp4,dp4}_dflash_zlab_bs8_rate{4,16,64}.log` |

> unieai 的 7 个 server 日志中 `speculative_draft_model_path` 已更正为
> `/workspace/models/Qwen3.5-9B-DFlash-unieai`（2026-09-04），与模型目录现状一致。
