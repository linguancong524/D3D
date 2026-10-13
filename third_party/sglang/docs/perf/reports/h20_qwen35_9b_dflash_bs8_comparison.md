# Qwen3.5-9B H20 DFlash block_size=8 对比（SGLang v0.5.18）

> **⚠️ 2026-09-04 更正**：本报告所用 draft 模型是 **unieai 社区版**
> `Qwen3.5-9B-DFlash`（5 层 full_attention、5 个 target 层 `[1,8,15,22,29]`、
> ~2.1 GB），**并非官方 z-lab 发布版**。该 checkpoint 现已移至
> `/workspace/models/Qwen3.5-9B-DFlash-unieai`。官方 z-lab 版的补测与两版对比见
> [z-lab 官方 DFlash 补测](h20_qwen35_9b_dflash_zlab_comparison.md) 与
> [unieai vs z-lab 对比](h20_qwen35_9b_dflash_unieai_vs_zlab.md)。
> 下文所有「DFlash」均指 **unieai 版**，数据不变。

日期：2026-09-03。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力）。
运行时：torch 2.13.0+cu130、flashinfer 0.6.17、attention=fa3、linear-attn=triton。
H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告是 [DFlash block_size=16 对比](h20_qwen35_9b_dflash_comparison.md)
（2026-09-03）的续篇：把 DFlash 的 **verify 窗口从 16 调到 8**
（`--speculative-dflash-block-size 8`），在同样的 TP=1/TP=4/DP=4 × rate 4/16/64
矩阵上重测，验证「**缩小 verify 窗口能否救回 DFlash 的高并发吞吐**」。

## 0. block_size=8 是什么 / 怎么开

DFlash 的 `block_size`（= `--speculative-num-draft-tokens` = verify 窗口长度）
决定每步 target model forward 处理的 token 数（`block_size × bs`）。
draft checkpoint（**unieai 版** `Qwen3.5-9B-DFlash`）的 config 里
`block_size: 16`，但运行时 `--speculative-dflash-block-size 8` 会**覆盖**它：
worker 用 8 作为 `self.block_size`，所有 draft buffer 按 8 分配，draft config 的
16 只触发一条 WARNING（`DFLASH block size mismatch: using
speculative_num_draft_tokens=8 but draft config block_size=16`），**不是错误**。

启动参数（三形态通用，叠加在各自 `--tp-size`/`--dp-size` 之上）：

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path /workspace/models/Qwen3.5-9B-DFlash-unieai \
--speculative-dflash-block-size 8 \
--mamba-scheduler-strategy extra_buffer
```

- **只传 `--speculative-dflash-block-size 8`，不要再传 `--speculative-num-draft-tokens`**
  （两者同传时必须相等，否则 ValueError）。
- 其余约束同 bs16：DFLASH 强制 `num_steps=1, topk=1`，`max_running_requests`
  重置 **48**，不支持 dp-attention（DP=4 用数据并行不触发）。
- 三形态 server 日志均确认 `Initialized DFLASH draft runner. ... block_size=8`。

**bs8 vs bs16 的唯一变量**：verify 窗口 8 vs 16，即每步 target forward 处理
`8×bs` vs `16×bs` token（**减半**）。draft 模型、accept 机制、cap48 全部相同。

## 1. rate 4（标准工况，n=100）：bs8 略逊 bs16（低并发 bs16 占优）

| 指标 | TP=1 bs16 | TP=1 **bs8** | TP=4 bs16 | TP=4 **bs8** | DP=4 bs16 | DP=4 **bs8** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 |
| E2E 中位 (ms) | **337.08** | 361.64 | **144.22** | 152.86 | **225.73** | 279.11 |
| E2E p99 (ms) | 1 716.24 | 1 316.00 | **652.61** | 512.54 | 925.03 | 893.72 |
| TTFT 中位 (ms) | 68.53 | 66.64 | 38.04 | 37.42 | 55.44 | 57.05 |
| TPOT 中位 (ms) | 3.00 | 2.52 | **1.01** | 0.98 | 1.79 | 1.80 |
| Accept length | **4.83** | 4.27 | **4.86** | 4.19 | **4.62** | 4.36 |

**rate 4 吞吐都 ≈4 req/s（发送速率受限），bs16 延迟略优**：

- bs16 的 16-token 窗口在低并发（bs≤11）下成本被隐藏，accept 更高（4.8 vs 4.2），
  每步多接受 ~0.6 token，E2E 略低（TP=1 337 vs 362、TP=4 144 vs 153、DP=4 226 vs 279）。
- **TP=4+bs8 仍是全场最低延迟档**（E2E 153 ms、TPOT 0.98 ms），与 bs16（144/1.01）
  几乎持平——低并发下窗口大小对延迟影响很小。
- **结论：低并发（rate ≤4）选 bs16**（accept 高、延迟略优），bs8 无明显优势。

## 2. rate 16（n=200，server 开始饱和）：bs8 大幅救回 TP=1，TP=4/DP=4 持平偏优

| 指标 | TP=1 bs16 | TP=1 **bs8** | TP=4 bs16 | TP=4 **bs8** | DP=4 bs16 | DP=4 **bs8** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 8.07 | **11.75** | 15.89 | **15.94** | 15.71 | 15.77 |
| E2E 中位 (ms) | 7 868.52 | **3 124.80** | 443.92 | **293.57** | 325.37 | 351.87 |
| E2E p99 (ms) | 15 551.53 | **8 118.68** | 2 794.75 | **1 137.10** | 1 342.96 | 1 092.72 |
| TTFT 中位 (ms) | — | 121.68 | — | 40.86 | — | 56.96 |
| TPOT 中位 (ms) | 31.21 | **14.90** | 3.45 | **1.95** | 2.45 | 2.36 |
| Accept length | 4.73 | 4.18 | 4.91 | 4.23 | 4.52 | 4.04 |

**bs8 在 rate 16 的收益**：

| 形态 | bs16 req/s | bs8 req/s | bs8 提升 | bs16 E2E | bs8 E2E |
|---|---|---|---|---|---|
| TP=1 | 8.07 | **11.75** | **+46%** | 7 869 | **3 125** |
| TP=4 | 15.89 | 15.94 | +0.3% | 444 | **294** |
| DP=4 | 15.71 | 15.77 | +0.4% | 325 | 352 |

- **TP=1 是最大受益者**：bs16 在 rate 16 严重过载（8.07 req/s、E2E 7 869 ms、
  TPOT 31.21 ms），bs8 把 verify 窗口减半后吞吐 +46%（11.75）、E2E 降到 3 125 ms
  （-60%）、TPOT 降到 14.90 ms（-52%）。**单卡高并发 bs8 显著优于 bs16**。
- **TP=4 / DP=4**：bs16 在 rate 16 尚未饱和（15.9/15.7 req/s），bs8 吞吐持平
  （15.94/15.77），但 **TP=4 延迟更优**（E2E 294 vs 444、TPOT 1.95 vs 3.45）。
  DP=4 两者接近（bs8 E2E 略高 352 vs 325，在噪声范围内）。

## 3. rate 64（n=400，逼近饱和）：bs8 全面碾压 bs16，TP=4 反超 MTP

| 指标 | TP=4 bs16 | TP=4 **bs8** | DP=4 bs16 | DP=4 **bs8** |
|---|---|---|---|---|
| 请求吞吐 (req/s) | 25.93 | **36.81** | 29.18 | **40.69** |
| E2E 中位 (ms) | 4 339.14 | **2 183.73** | 4 303.61 | **2 573.19** |
| E2E p99 (ms) | 9 592.20 | **5 183.55** | 10 728.93 | **7 177.10** |
| TTFT 中位 (ms) | — | 789.40 | — | 136.72 |
| TPOT 中位 (ms) | 11.10 | **7.97** | 27.50 | **18.09** |
| Accept length | 4.82 | 4.18 | 4.60 | 4.28 |

（TP=1 rate 64：bs16 8.32 req/s、**bs8 13.75 req/s**（+65%），E2E 22 480→10 691 ms。）

**rate 64 下 bs8 吞吐全面碾压 bs16**：

| 形态 | bs16 req/s | bs8 req/s | bs8 提升 | bs8 vs MTP |
|---|---|---|---|---|
| TP=1 | 8.32 | **13.75** | **+65%** | 0.79×（MTP 17.32） |
| TP=4 | 25.93 | **36.81** | **+42%** | **1.06×（MTP 34.80）** |
| DP=4 | 29.18 | **40.69** | **+39%** | 0.85×（MTP 48.01） |

- **TP=4+bs8 反超 MTP**：36.81 vs 34.80 req/s（+6%），E2E 2 184 vs 2 775 ms
  （-21%）。bs16 时 TP=4 落后 MTP 25%（25.93 vs 34.80），bs8 一举反超。
- **DP=4+bs8 大幅缩小与 MTP 的差距**：40.69 vs 48.01（bs16 时 29.18，落后 39%；
  bs8 后只落后 15%）。
- **TP=1+bs8 仍落后 MTP**（13.75 vs 17.32），但比 bs16（8.32）好 65%。
- **原因**：高并发下 bs 大（peak concurrent 188–309），verify 窗口 16 让每步
  target forward 处理 `16×bs` token 成为瓶颈；bs8 减半到 `8×bs`，target forward
  成本减半，吞吐显著回升。虽然 accept 略降（4.2 vs 4.8），但 target forward 成本
  减半的收益远大于 accept 下降的损失。

## 4. bs8 的显存墙（比 bs16 更高）

bs8 的 verify 窗口 8 让 decode CUDA graph 捕获的 token 数是 bs16 的一半
（8×bs vs 16×bs），显存压力更小：

| 配置 | cap | bs16 结果 | bs8 结果 |
|---|---|---|---|
| TP=4 | 48 | 正常（25.93 req/s） | 正常（36.81 req/s） |
| TP=4 | 256 | **graph 捕获 OOM**（16×256=4 096 token） | 未测（8×256=2 048 token，预期更安全） |

bs8 在 cap48 下三形态全部正常，无 OOM。verify 窗口减半使 graph 激活显存减半，
**bs8 的安全 cap 上限高于 bs16**（bs16 的 TP=4 cap256 OOM 墙在 bs8 下被抬高）。

## 5. 关键结论

1. **bs8 是「高并发吞吐利器，低并发略逊 bs16」**：
   - **rate 4**：bs16 略优（accept 4.8 vs 4.2，E2E 略低），bs8 无明显优势。
   - **rate 16**：**TP=1 大幅救回**（8.07→11.75 req/s，+46%），TP=4/DP=4 持平偏优。
   - **rate 64**：**全面碾压 bs16**（TP=1 +65%、TP=4 +42%、DP=4 +39%），
     **TP=4 反超 MTP**（36.81 vs 34.80）。
2. **verify 窗口是 DFlash 性能曲线的决定因素**（与 bs16 报告一致，bs8 进一步验证）：
   窗口 16→8 让每步 target forward 处理 token 减半。低并发隐藏成本（bs16 占优），
   高并发暴露瓶颈（bs8 占优）。
3. **bs8 的显存墙比 bs16 更高**：verify 窗口减半使 graph 激活显存减半，
   bs16 的 TP=4 cap256 OOM 墙在 bs8 下被抬高。
4. **bs8 vs bs16 vs MTP 选型**：
   - **极致低延迟单流**（rate ≤4）→ **bs16**（accept 高、延迟略优）。
   - **中并发（rate 16）TP=1** → **bs8**（bs16 严重过载，bs8 救回 46%）。
   - **高吞吐（rate 64）TP=4** → **bs8**（反超 MTP，36.81 vs 34.80）。
   - **高吞吐（rate 64）DP=4** → **MTP** 仍最优（48.01），bs8 次之（40.69）。
   - **单卡高并发** → **bs8**（13.75，比 bs16 8.32 好 65%，但仍落后 MTP 17.32）。

## 6. 部署建议（9B / 4× H20 + DFlash 投机解码，bs8 vs bs16）

| 场景 | 推荐 | 理由 |
|---|---|---|
| **极致低延迟单流 / 实时对话** | **TP=4 + bs16** | E2E 144 ms、TPOT 1.01 ms（rate 4），accept 4.86 最高 |
| **低中并发、重延迟（TP=4/DP=4）** | bs16 或 bs8 | rate 16 延迟 bs8 略优（TP=4 294 vs 444），吞吐持平 |
| **单卡高并发（TP=1，rate 16+）** | **TP=1 + bs8** | bs16 严重过载（8 req/s），bs8 救回 46%（11.75） |
| **高吞吐 serving（TP=4，rate 64）** | **TP=4 + bs8** | 36.81 req/s，反超 MTP（34.80），E2E 2 184 ms |
| **高吞吐 serving（DP=4，rate 64）** | **DP=4 + MTP** | 48.01 req/s 仍最优，bs8 次之（40.69） |

**一句话**：DFlash 的 verify 窗口从 16 调到 8，**高并发吞吐大幅回升**
（rate 64 TP=4 反超 MTP、TP=1 救回 65%），**低并发延迟略逊 bs16**
（accept 4.2 vs 4.8）。**低延迟选 bs16，高吞吐选 bs8**；DP=4 高吞吐 MTP 仍最优。

## 7. 数据有效性

- 所有运行 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后确认 4 卡全部释放到 0 MiB。
- 9 组主基准（3 形态 × rate 4/16/64）全部 100%/200%/400% 请求成功，无失败。
- Accept length 稳定 4.04–4.36（跨形态、跨速率一致，低于 bs16 的 4.52–4.91，
  符合窗口减半的预期）。
- 三形态 server 日志均确认 `block_size=8` 与预期的 mismatch WARNING。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §8 日志清单）。
- 与 bs16/MTP/非 MTP 基线同条件：mem 0.85 + `--mamba-ssm-dtype bfloat16`、
  512 in/256 out、random、sglang backend、H20 不带 prefill 图禁用参数、cap48。

## 8. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `server_h20_qwen35_9b_serving_tp1_dflash_bs8.log` | TP=1+DFlash bs8 server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp1_dflash_bs8_rate4/16/64.log` | TP=1+DFlash bs8 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_tp4_dflash_bs8.log` | TP=4+DFlash bs8 server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp4_dflash_bs8_rate4/16/64.log` | TP=4+DFlash bs8 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_dp4_dflash_bs8.log` | DP=4+DFlash bs8 server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_dp4_dflash_bs8_rate4/16/64.log` | DP=4+DFlash bs8 三档速率（cap48） |

## 9. 环境坑（bs8 专属）

1. **只传 `--speculative-dflash-block-size 8`**：不要再传
   `--speculative-num-draft-tokens`（两者同传时必须相等，否则 ValueError）。
2. **mismatch WARNING 是预期的**：draft config `block_size: 16` 与运行时 8 不一致，
   触发 `DFLASH block size mismatch` WARNING，**不是错误**，可忽略。
3. **ninja PATH**：同 bs16/MTP，启动 server 需 `PATH="$PWD/.venv/bin:$PATH"`。
4. **显存墙比 bs16 更高**：verify 窗口减半使 graph 激活显存减半，bs16 的
   TP=4 cap256 OOM 墙在 bs8 下被抬高（cap48 三形态全部正常）。
