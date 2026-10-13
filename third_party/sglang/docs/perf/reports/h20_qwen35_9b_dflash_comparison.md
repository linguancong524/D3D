# Qwen3.5-9B H20 DFlash 投机解码对比（SGLang v0.5.18）

> **⚠️ 2026-09-04 更正**：本报告（及续篇 bs8 报告）所用 draft 模型是
> **unieai 社区版** `Qwen3.5-9B-DFlash`（5 层 full_attention、5 个 target 层
> `[1,8,15,22,29]`、~2.1 GB），**并非官方 z-lab 发布版**。该 checkpoint 现已移至
> `/workspace/models/Qwen3.5-9B-DFlash-unieai`。官方 z-lab 版（6 层、含 sliding
> attention、8 个 target 层、~2.6 GB）的补测与两版对比见
> [z-lab 官方 DFlash 补测](h20_qwen35_9b_dflash_zlab_comparison.md) 与
> [unieai vs z-lab 对比](h20_qwen35_9b_dflash_unieai_vs_zlab.md)。
> 下文所有「DFlash」均指 **unieai 版**，数据不变。

日期：2026-09-03。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力）。
运行时：torch 2.13.0+cu130、flashinfer 0.6.17、attention=fa3、linear-attn=triton。
H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告在 [TP=1/TP=4/DP=4 三形态对比](h20_qwen35_9b_tp1_tp4_dp4_comparison.md)
（2026-09-02 非 MTP 基线）与 [MTP 投机解码对比](h20_qwen35_9b_mtp_comparison.md)
（2026-09-03）基础上，叠加 **DFlash 投机解码**，与 MTP 同条件对照。

## 0. DFlash 是什么 / 怎么开

**DFlash**（[z-lab/dflash](https://github.com/z-lab/dflash)）用轻量 **block diffusion**
模型并行 draft 多个 token，官方称最高 4.4× 加速。本报告所用 draft 模型为
**unieai 社区版** `Qwen3.5-9B-DFlash`（`DFlashDraftModel`，5 层 full_attention，
`block_size: 16`，`target_layer_ids [1,8,15,22,29]`，权重 ~2.1 GB），需与 target
`Qwen3.5-9B` 配对。本 fork 内置 `DFlashDraftModel`
（`python/sglang/srt/models/dflash.py`），架构名直接匹配，无需
`--trust-remote-code`。

启动参数（三形态通用，叠加在各自 `--tp-size`/`--dp-size` 之上）：

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path /workspace/models/Qwen3.5-9B-DFlash-unieai \
--speculative-num-draft-tokens 16 \
--mamba-scheduler-strategy extra_buffer
```

- `--speculative-num-draft-tokens 16` = draft config 的 `block_size`（verify 窗口长度）。
  DFLASH 强制 `num_steps=1, topk=1`（`_handle_dflash`）。
- `--mamba-scheduler-strategy extra_buffer`（`--mamba-radix-cache-strategy` 的别名，
  取值 `auto/no_buffer/extra_buffer/extra_buffer_lazy`）：README 推荐，配合 DFLASH
  的 mamba 状态跟踪。
- DFLASH 不支持 dp-attention，但 DP=4 用数据并行（`--dp-size 4`）不触发该限制。
- `max_running_requests` 默认重置 **48**（同 MTP），是 rate 64 吞吐对比的混淆变量。

**与 MTP 的核心差异——verify 窗口**：

| | MTP (EAGLE) | DFlash |
|---|---|---|
| draft 方式 | 自回归 3 步 × topk1 | block diffusion 并行 16 token |
| verify 窗口（`num_draft_tokens`） | **4** | **16** |
| Accept length（实测） | 3.21–3.36 | **4.52–4.91** |
| 每步 target forward token 数 | 4 × bs | **16 × bs** |
| draft 模型 | 1 层 MTP 头（复用 target） | 5 层 full_attention（独立 2 GB） |

**这是理解 DFlash 性能曲线的钥匙**：verify 阶段是 target model 的 forward，处理
`verify_window × bs` 个 token。DFlash 窗口 16 是 MTP 的 4 倍，每步 target forward
计算量 4×。低并发（bs 小）时这成本被隐藏，高并发（bs 大）时成为瓶颈。

## 1. rate 4（标准工况，n=100）：DFlash 延迟全场最低

| 指标 | TP=1 非MTP | TP=1 MTP | TP=1 **DFlash** | TP=4 非MTP | TP=4 MTP | TP=4 **DFlash** | DP=4 非MTP | DP=4 MTP | DP=4 **DFlash** |
|---|---|---|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 3.95 | 4.04 | 4.08 | 4.06 | 4.08 | 4.08 | 3.96 | 4.05 | 4.08 |
| E2E 中位 (ms) | 1 203.07 | 580.55 | **337.08** | 470.55 | 305.21 | **144.22** | 964.65 | 440.47 | **225.73** |
| E2E p99 (ms) | 2 216.88 | 1 282.00 | 1 716.24 | 876.12 | 685.44 | **652.61** | 1 705.29 | 957.13 | 925.03 |
| TTFT 中位 (ms) | 59.40 | 67.98 | 68.53 | 31.97 | 40.64 | 38.04 | 52.43 | 57.12 | 55.44 |
| TPOT 中位 (ms) | 7.75 | 3.83 | 3.00 | 2.97 | 1.92 | **1.01** | 6.38 | 2.75 | **1.79** |
| Accept length | — | 3.22 | 4.83 | — | 3.21 | 4.86 | — | 3.36 | 4.62 |

**rate 4 下吞吐都 ≈4 req/s（发送速率受限），DFlash 延迟全场最低**：

| 形态 | 非MTP→MTP→DFlash E2E 中位 | DFlash vs MTP | DFlash vs 非MTP |
|---|---|---|---|
| TP=1 | 1 203→581→**337** ms | 1.72× | **3.57×** |
| TP=4 | 471→305→**144** ms | 2.12× | **3.26×** |
| DP=4 | 965→440→**226** ms | 1.95× | **4.27×** |

- **DFlash 比 MTP 再快 ~1.7–2.1×**（accept 4.8 vs 3.2，每步多接受 ~1.6 token）。
- **TP=4+DFlash 全场最低延迟**：E2E 144 ms、TPOT 1.01 ms（MTP 305/1.92、非 MTP 471/2.97）。
- TTFT 三者接近（38–69 ms），DFlash 的 draft 阶段未明显拖慢首 token。

## 2. rate 16（n=200，server 开始饱和）：TP=4/DP=4 DFlash 延迟仍最优，TP=1 崩

| 指标 | TP=1 非MTP | TP=1 MTP | TP=1 **DFlash** | TP=4 非MTP | TP=4 MTP | TP=4 **DFlash** | DP=4 非MTP | DP=4 MTP | DP=4 **DFlash** |
|---|---|---|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 13.84 | 13.53 | **8.07** | 15.53 | 15.78 | **15.89** | 14.83 | 15.57 | 15.71 |
| E2E 中位 (ms) | 3 457.40 | 2 155.65 | **7 868.52** | 748.83 | 444.86 | 443.92 | 1 115.37 | 530.99 | **325.37** |
| E2E p99 (ms) | 7 595.10 | 6 246.87 | 15 551.53 | 1 646.78 | 1 241.76 | 2 794.75 | 2 131.75 | 1 177.71 | 1 342.96 |
| TPOT 中位 (ms) | 23.11 | 13.01 | **31.21** | 4.74 | 2.68 | 3.45 | 7.60 | 3.43 | **2.45** |
| Accept length | — | 3.26 | 4.73 | — | 3.26 | 4.91 | — | 3.25 | 4.52 |

- **TP=4 / DP=4：DFlash 吞吐与 MTP 持平**（15.89/15.71 vs 15.78/15.57），**延迟仍最优**
  （E2E 444/325 ms，TPOT 3.45/2.45 ms）。TP=4 的 16-token verify 窗口被 4 卡算力摊薄，
  高并发下尚未成为瓶颈。
- **TP=1：DFlash 严重过载**（8.07 req/s，E2E 7 869 ms、TPOT 31.21 ms，比 MTP 13.53/
  2 156/13.01 差 ~40%）。单卡 9B 算力有限，16-token verify 窗口（bs 堆到 121 时
  16×121≈1 936 token/step）单卡扛不住，MTP 的 4-token 窗口（4×78≈312 token）则轻松。
  **DFlash 在单卡高并发下尤其吃亏**。

## 3. rate 64（n=400，逼近饱和）：DFlash 全面落后

| 指标 | TP=4 非MTP | TP=4 MTP | TP=4 **DFlash** | DP=4 非MTP | DP=4 MTP | DP=4 **DFlash** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 42.91 | 34.80 | **25.93** | 47.71 | 48.01 | **29.18** |
| E2E 中位 (ms) | 4 161.20 | 2 774.72 | 4 339.14 | 2 086.22 | 1 853.62 | 4 303.61 |
| E2E p99 (ms) | 8 297.68 | 5 588.31 | 9 592.20 | 4 781.27 | 4 750.51 | 10 728.93 |
| TPOT 中位 (ms) | 34.85 | 9.58 | 11.10 | 16.21 | 14.35 | 27.50 |
| Accept length | — | 3.23 | 4.82 | — | 3.28 | 4.60 |

（TP=1 rate 64：非 MTP ~13.8 饱和、MTP 17.32、**DFlash 8.32** req/s，单卡瓶颈最严重。）

**rate 64 下 DFlash 吞吐全面落后**：

| 形态 | 非MTP | MTP (cap48) | DFlash (cap48) | DFlash vs MTP |
|---|---|---|---|---|
| TP=1 | ~13.8 | 17.32 | 8.32 | **0.48×** |
| TP=4 | 42.91 | 34.80 | 25.93 | **0.75×** |
| DP=4 | 47.71 | 48.01 | 29.18 | **0.61×** |

原因：高并发下 bs 大，DFlash 的 16-token verify 窗口让每步 target forward 处理
`16×bs` token（MTP 仅 `4×bs`），target forward 成为瓶颈。虽然 DFlash 每步接受更多
token（accept 4.8 vs 3.2），但 target forward 成本 4×，**净吞吐反而下降**。
MTP 的 4-token 窗口在高并发下 target forward 成本低，吞吐更稳。

## 4. DFlash 的显存墙（比 MTP 更低）

DFlash 的 verify 窗口 16 让 decode CUDA graph 捕获的 token 数是 MTP 的 4 倍
（16×bs vs 4×bs），显存压力更大：

| 配置 | cap | 结果 |
|---|---|---|
| TP=4+MTP | 256 | 正常（4×256=1 024 verify token） |
| **TP=4+DFlash** | **256** | **graph 捕获 OOM**（16×256=4 096 verify token） |

TP=4+DFlash cap256 在 CUDA graph 捕获阶段 OOM（`Capture cuda graph failed:
Tried to allocate 24 MiB, 18.62 MiB free`，GPU 2 占满 95.05/95.07 GiB）。
**即 DFlash 在 TP=4 的安全 cap 上限低于 MTP 的 256**（16-token 窗口的 graph 激活
显存更大）。cap48 是 DFlash 在 TP=4 高并发的安全值（25.93 req/s）。

> 注：DFlash draft 模型仅 5 层 2 GB（比 MTP 复用 target 的 1 层 MTP 头显存占用方式
> 不同），但 verify 窗口 16 的 graph 激活显存是主导项，故 DFlash 的 cap 上限反而更低。

## 5. 关键结论

1. **DFlash 是「低并发延迟利器，高并发吞吐拖累」**：
   - **rate 4**：延迟全场最低（TP=4 E2E 144 ms/TPOT 1.01 ms），比 MTP 再快 1.7–2.1×
     （accept 4.8 vs 3.2）。
   - **rate 16**：TP=4/DP=4 延迟仍最优、吞吐与 MTP 持平；**TP=1 严重过载**
     （8.07 req/s，单卡扛不住 16-token verify）。
   - **rate 64**：吞吐全面落后 MTP（TP=4 25.93 vs 34.80、DP=4 29.18 vs 48.01），
     16-token verify 窗口高并发 target forward 成本 4×。
2. **verify 窗口是 DFlash 性能曲线的决定因素**：窗口 16（MTP 4）让每步 target
   forward 处理 4× token。低并发隐藏成本、高并发暴露瓶颈。
3. **DFlash 显存墙比 MTP 更低**：TP=4 cap256 graph 捕获 OOM（16×256=4 096 token），
   MTP cap256 正常（4×256=1 024 token）。
4. **DFlash vs MTP 选型**：
   - **极致低延迟单流**（rate ≤4）→ **DFlash**（TP=4 E2E 144 ms，全场最低）。
   - **中并发（rate 16）TP=4/DP=4** → DFlash 与 MTP 相当（延迟 DFlash 略优）。
   - **高吞吐（rate 64）** → **MTP**（DFlash 吞吐拖累 25–40%）。
   - **单卡高并发** → **MTP**（DFlash 单卡过载最严重）。

## 6. 部署建议（9B / 4× H20 + 投机解码）

| 场景 | 推荐 | 理由 |
|---|---|---|
| **极致低延迟单流 / 实时对话** | **TP=4 + DFlash** | E2E 144 ms、TPOT 1.01 ms（rate 4），全场最低 |
| **低中并发、重延迟（TP=4/DP=4）** | DFlash 或 MTP | rate 16 延迟 DFlash 略优，吞吐持平 |
| **高吞吐 serving** | **DP=4 + MTP** | 48 req/s（DFlash 仅 29），MTP 无副作用 |
| **单卡 / 边缘（高并发）** | **TP=1 + MTP** | DFlash 单卡 rate16 即过载（8 req/s） |
| **单卡 / 边缘（低并发）** | TP=1 + DFlash | rate 4 延迟 337 ms（MTP 581） |

**一句话**：DFlash 的 16-token verify 窗口让它**低并发延迟碾压 MTP**（accept 4.8 vs
3.2），但**高并发吞吐被 4× target forward 成本拖垮**（rate 64 落后 MTP 25–40%），
且**显存墙更低**（TP=4 cap256 graph OOM）。**低延迟选 DFlash，高吞吐选 MTP**。

## 7. 数据有效性

- 所有运行 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后确认 4 卡全部释放到 0 MiB。
- 9 组主基准（3 形态 × rate 4/16/64）全部 100%/200%/400% 请求成功，无失败。
- Accept length 稳定 4.52–4.91（跨形态、跨速率一致，显著高于 MTP 的 3.21–3.36）。
- TP=4+DFlash cap256 的 graph 捕获 OOM 日志保留为证据
  （`server_..._tp4_dflash_r64_cap256.log`）。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §8 日志清单）。
- 与 MTP/非 MTP 基线同条件：mem 0.85 + `--mamba-ssm-dtype bfloat16`、512 in/256 out、
  random、sglang backend、H20 不带 prefill 图禁用参数。

## 8. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `server_h20_qwen35_9b_serving_tp1_dflash.log` | TP=1+DFlash server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp1_dflash_rate4/16/64.log` | TP=1+DFlash 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_tp4_dflash.log` | TP=4+DFlash server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp4_dflash_rate4/16/64.log` | TP=4+DFlash 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_dp4_dflash.log` | DP=4+DFlash server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_dp4_dflash_rate4/16/64.log` | DP=4+DFlash 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_tp4_dflash_r64_cap256.log` | TP=4+DFlash cap256（**graph 捕获 OOM 崩溃**，证据） |

## 9. 环境坑（DFlash 专属）

1. **ninja PATH**：同 MTP/非 MTP，启动 server 需 `PATH="$PWD/.venv/bin:$PATH"`。
2. **verify 窗口 16 的显存墙**：TP=4 cap256 在 decode graph 捕获 OOM
   （16×256=4 096 verify token），安全 cap 上限低于 MTP 的 256。高并发 TP=4 用 cap48。
3. **单卡高并发过载**：TP=1+DFlash rate16 即严重过载（8 req/s），16-token verify
   窗口单卡扛不住；单卡高并发应选 MTP。
4. **cap 混淆**：投机解码默认把 `max_running_requests` 重置为 48，吞吐对比须显式
   对齐 cap（本报告主矩阵 cap48 与 MTP 主矩阵完全对齐，rate 64 对比公平）。
5. **`--mamba-scheduler-strategy extra_buffer`**：README 推荐，是
   `--mamba-radix-cache-strategy` 的弃用别名（取值 `auto/no_buffer/extra_buffer/
   extra_buffer_lazy`），配合 DFlash 的 mamba 状态跟踪。
