# Qwen3.5-9B H20 官方 z-lab DFlash 补测（SGLang v0.5.18）

日期：2026-09-04。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力）。
运行时：torch 2.13.0+cu130、flashinfer 0.6.17、attention=fa3、linear-attn=triton。
H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告补测**官方 z-lab 发布版** DFlash draft 模型
（`z-lab/Qwen3.5-9B-DFlash`，现位于 `/workspace/models/Qwen3.5-9B-DFlash`），
与 [unieai 社区版 bs16 对比](h20_qwen35_9b_dflash_comparison.md)、
[unieai 社区版 bs8 对比](h20_qwen35_9b_dflash_bs8_comparison.md)
（2026-09-03，draft 已移至 `/workspace/models/Qwen3.5-9B-DFlash-unieai`）
**完全同条件**对照。两版模型差异与逐格对比见
[unieai vs z-lab 对比](h20_qwen35_9b_dflash_unieai_vs_zlab.md)。

## 0. z-lab 官方版是什么 / 怎么开

z-lab 官方版（[z-lab/dflash](https://github.com/z-lab/dflash)，Z-Lab × Modal 联合
retrain，40k 序列长度训练 + sliding-window attention）与 unieai 社区版是**两个不同
checkpoint**：

| | unieai 社区版 | **z-lab 官方版** |
|---|---|---|
| draft 层数 | 5（全 full_attention） | **6（5 sliding + 1 full）** |
| target 层数（`target_layer_ids`） | 5 `[1,8,15,22,29]` | **8 `[1,5,9,13,17,21,25,29]`** |
| sliding window | 无 | **4096** |
| `mask_token_id` | 248070 | **248077** |
| 权重 | ~2.1 GB | **~2.6 GB**（TP=1 实测 2.56 GB） |
| config `block_size` | 16（顶层） | 16（`dflash_config` 内） |

本 fork 内置 `DFlashDraftModel`（`python/sglang/srt/models/dflash.py`）同时支持
`full_attention` 与 `sliding_attention` 层、`dflash_config` 内/顶层 `block_size`、
8 个 target 层，架构名直接匹配，**无需 `--trust-remote-code`**。server 日志确认
`n_layers=6`、`mask_token_id=248077`、`mem usage=2.56 GB`。

启动参数（三形态通用，与 unieai 测试**完全一致**，仅 draft 路径不同）：

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path /workspace/models/Qwen3.5-9B-DFlash \
--speculative-num-draft-tokens 16 \        # bs16
--speculative-dflash-block-size 8 \        # bs8（二选一，勿同传）
--mamba-scheduler-strategy extra_buffer
```

- 其余约束同 unieai：DFLASH 强制 `num_steps=1, topk=1`，`max_running_requests`
  重置 **48**，不支持 dp-attention（DP=4 用数据并行不触发）。
- bs8 触发预期的 `DFLASH block size mismatch` WARNING（config 16 vs 运行时 8），
  **不是错误**。
- 三形态 × bs16/bs8 共 6 个 server 日志均确认 `n_layers=6` 与对应 `block_size`。

**与 unieai 测试的唯一变量**：draft 模型 checkpoint（unieai 5 层 5-target vs
z-lab 6 层 8-target）。target、verify 窗口（bs16/bs8）、cap48、mem 0.85、
mamba bfloat16、fa3/triton 后端、512 in/256 out、random、n=100/200/400 全部相同。

## 1. rate 4（标准工况，n=100）：z-lab 延迟更低（bs16 明显，bs8 持平）

**bs16**（z-lab accept 5.2–5.9，显著高于 unieai 4.5–4.9）：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 |
| E2E 中位 (ms) | 337.08 | **303.94** | 144.22 | **138.06** | 225.73 | **215.47** |
| E2E p99 (ms) | 1 716.24 | 1 602.51 | 652.61 | 648.99 | 925.03 | 954.09 |
| TTFT 中位 (ms) | 68.53 | 67.55 | 38.04 | 37.77 | 55.44 | 57.78 |
| TPOT 中位 (ms) | 3.00 | **2.49** | 1.01 | 1.01 | 1.79 | **1.74** |
| Accept length | 4.83 | **5.21** | 4.86 | **5.22** | 4.62 | **5.92** |

**bs8**（z-lab accept 4.2–4.6，仅略高于 unieai 4.0–4.4）：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 4.08 | 4.07 | 4.08 | 4.08 | 4.08 | 4.07 |
| E2E 中位 (ms) | 361.64 | 361.88 | 152.86 | 161.41 | 279.11 | 283.90 |
| E2E p99 (ms) | 1 316.00 | 1 163.40 | 512.54 | 541.74 | 893.72 | 859.50 |
| TTFT 中位 (ms) | 66.64 | 69.18 | 37.42 | 38.14 | 57.05 | 57.32 |
| TPOT 中位 (ms) | 2.52 | 2.65 | 0.98 | 1.10 | 1.80 | 1.97 |
| Accept length | 4.27 | 4.29 | 4.19 | 4.23 | 4.36 | 4.58 |

- **bs16 rate 4：z-lab 延迟全面更低**（E2E TP=1 304 vs 337、TP=4 138 vs 144、
  DP=4 215 vs 226；TPOT TP=1 2.49 vs 3.00）。accept 5.2–5.9 让每步多接受 ~0.4–1.3
  token，低并发下直接转化为更低延迟。
- **bs8 rate 4：z-lab 与 unieai 基本持平**（E2E 差异 <6%，TPOT 略高 5–12%）。
  窗口 8 限制了 draft 可提议 token 数，z-lab 的 draft 质量优势无法充分表达，
  6 层 draft 的略高成本反而让 TPOT 略升。
- **DP=4 bs16 rate 4 的 accept 5.92 是单格离群**（其余 DP=4 bs16 格 5.26/5.28，
  TP=4 bs16 5.22–5.34），rate 4 低并发（peak concurrent ~11）下样本少，作量级参考。

## 2. rate 16（n=200，server 开始饱和）：z-lab 略优（bs16 明显）

**bs16**：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 8.07 | **8.77** | 15.89 | 15.88 | 15.71 | **15.88** |
| E2E 中位 (ms) | 7 868.52 | **6 456.30** | 443.92 | **401.03** | 325.37 | **291.10** |
| E2E p99 (ms) | 15 551.53 | 14 143.27 | 2 794.75 | 2 584.45 | 1 342.96 | 1 201.39 |
| TPOT 中位 (ms) | 31.21 | **27.96** | 3.45 | **3.13** | 2.45 | **2.16** |
| Accept length | 4.73 | **5.22** | 4.91 | **5.34** | 4.52 | **5.26** |

**bs8**：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 11.75 | **11.98** | 15.94 | 15.95 | 15.77 | 15.78 |
| E2E 中位 (ms) | 3 124.80 | **2 901.73** | 293.57 | **282.27** | 351.87 | 347.00 |
| E2E p99 (ms) | 8 118.68 | 7 922.06 | 1 137.10 | 1 225.14 | 1 092.72 | 1 125.69 |
| TPOT 中位 (ms) | 14.90 | **13.76** | 1.95 | **1.78** | 2.36 | 2.29 |
| Accept length | 4.18 | 4.30 | 4.23 | 4.38 | 4.04 | 4.24 |

- **bs16 rate 16：z-lab 全面略优**（TP=1 吞吐 8.77 vs 8.07 +9%、E2E 6 456 vs
  7 869；TP=4/DP=4 吞吐持平、E2E 更低 401/291 vs 444/325）。
- **bs8 rate 16：z-lab 与 unieai 基本持平**（吞吐差异 <2%，E2E 略低）。
- TP=1 bs16 在 rate 16 仍过载（8.77 req/s，E2E 6.5 s），与 unieai（8.07）同量级——
  单卡 16-token verify 窗口的瓶颈与 draft 版本无关。

## 3. rate 64（n=400，逼近饱和）：z-lab 吞吐略高（bs16 +4%，bs8 +1%）

**bs16**：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 8.32 | **8.65** | 25.93 | **26.90** | 29.18 | **30.45** |
| E2E 中位 (ms) | 22 479.79 | **21 451.28** | 4 339.14 | **4 058.42** | 4 303.61 | **4 014.65** |
| E2E p99 (ms) | 41 671.65 | 39 772.11 | 9 592.20 | 8 986.45 | 10 728.93 | 10 292.40 |
| TPOT 中位 (ms) | 36.70 | **33.97** | 11.10 | **10.71** | 27.50 | **26.71** |
| Accept length | 4.68 | **5.13** | 4.82 | **5.20** | 4.60 | **5.28** |

**bs8**：

| 指标 | TP=1 unieai | TP=1 **z-lab** | TP=4 unieai | TP=4 **z-lab** | DP=4 unieai | DP=4 **z-lab** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 13.75 | **13.85** | 36.81 | **37.12** | 40.69 | **41.11** |
| E2E 中位 (ms) | 10 690.81 | 10 715.00 | 2 183.73 | **2 176.59** | 2 573.19 | **2 565.02** |
| E2E p99 (ms) | 22 798.24 | 22 538.26 | 5 183.55 | 5 184.62 | 7 177.10 | 6 955.29 |
| TPOT 中位 (ms) | 21.26 | 21.21 | 7.97 | **7.92** | 18.09 | **17.34** |
| Accept length | 4.12 | 4.27 | 4.18 | 4.31 | 4.28 | 4.31 |

**rate 64 吞吐对比（z-lab vs unieai）**：

| 形态 | bs16 unieai | bs16 **z-lab** | bs8 unieai | bs8 **z-lab** |
|---|---|---|---|---|
| TP=1 | 8.32 | **8.65** (+4.0%) | 13.75 | **13.85** (+0.7%) |
| TP=4 | 25.93 | **26.90** (+3.7%) | 36.81 | **37.12** (+0.8%) |
| DP=4 | 29.18 | **30.45** (+4.4%) | 40.69 | **41.11** (+1.0%) |

- **bs16 rate 64：z-lab 吞吐 +3.7–4.4%**（accept 5.1–5.3 vs 4.7–4.8，每步多接受
  ~0.4 token 在高并发下累积为吞吐增益）。
- **bs8 rate 64：z-lab 吞吐 +0.7–1.0%**（accept 4.3 vs 4.1–4.3，增益很小）。
- 高并发下 bs8 仍显著优于 bs16（TP=4 37 vs 27、DP=4 41 vs 30），与 unieai 结论一致：
  **verify 窗口是吞吐的决定因素，draft 版本是次要因素**。

## 4. 关键结论

1. **z-lab 官方版 accept length 全面高于 unieai 社区版**：bs16 5.1–5.9 vs 4.5–4.9
   （+0.4–1.3），bs8 4.2–4.6 vs 4.0–4.4（+0.1–0.2）。8 个 target 层 + 40k 训练让
   draft 质量更高。
2. **z-lab 的优势集中在 bs16**：大窗口让高质量 draft 充分表达，转化为
   - rate 4 延迟更低（E2E -4 至 -10%、TPOT -3 至 -17%）；
   - rate 64 吞吐更高（+3.7 至 +4.4%）。
3. **bs8 下 z-lab 优势大幅缩小**：小窗口是瓶颈，draft 质量次要，z-lab 与 unieai
   基本持平（吞吐 +0.7–1.0%，延迟持平或略高）。
4. **verify 窗口仍是第一决定因素**（与 unieai 报告一致）：bs8 高并发吞吐（37–41）
   远超 bs16（27–30），draft 版本只带来个位数百分比的边际增益。
5. **选型**：
   - **低延迟（rate ≤4）bs16** → **z-lab**（accept 高、延迟更低）。
   - **高吞吐（rate 64）bs8** → z-lab 略优（+1%），但 bs8 本身才是关键。
   - **高吞吐（rate 64）bs16** → z-lab（+4%）。
   - 若只能用 unieai 版，结论与 [unieai 报告](h20_qwen35_9b_dflash_comparison.md)
     一致；有 z-lab 版则优先 z-lab。

## 5. 数据有效性

- 6 个 server 日志 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后确认 4 卡全部释放到 0 MiB。
- 18 组基准（3 形态 × bs16/bs8 × rate 4/16/64）全部 100%/200%/400% 请求成功，无失败。
- 6 个 server 日志均确认 `n_layers=6`、`mask_token_id=248077`、`mem usage=2.56 GB`
  （TP=1/DP=4）或 `0.92 GB`（TP=4 分片），无 graph 捕获 OOM。
- Accept length 稳定：bs16 5.13–5.92、bs8 4.23–4.58（跨形态、跨速率一致）。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §6 日志清单）。
- 与 unieai/MTP/非 MTP 基线同条件：mem 0.85 + `--mamba-ssm-dtype bfloat16`、
  512 in/256 out、random、sglang backend、H20 不带 prefill 图禁用参数、cap48。

## 6. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `server_h20_qwen35_9b_serving_tp1_dflash_zlab_bs16.log` | TP=1+z-lab bs16 server |
| `bench_serving_h20_qwen35_9b_tp1_dflash_zlab_bs16_rate4/16/64.log` | TP=1+z-lab bs16 三档速率 |
| `server_h20_qwen35_9b_serving_tp1_dflash_zlab_bs8.log` | TP=1+z-lab bs8 server |
| `bench_serving_h20_qwen35_9b_tp1_dflash_zlab_bs8_rate4/16/64.log` | TP=1+z-lab bs8 三档速率 |
| `server_h20_qwen35_9b_serving_tp4_dflash_zlab_bs16.log` | TP=4+z-lab bs16 server |
| `bench_serving_h20_qwen35_9b_tp4_dflash_zlab_bs16_rate4/16/64.log` | TP=4+z-lab bs16 三档速率 |
| `server_h20_qwen35_9b_serving_tp4_dflash_zlab_bs8.log` | TP=4+z-lab bs8 server |
| `bench_serving_h20_qwen35_9b_tp4_dflash_zlab_bs8_rate4/16/64.log` | TP=4+z-lab bs8 三档速率 |
| `server_h20_qwen35_9b_serving_dp4_dflash_zlab_bs16.log` | DP=4+z-lab bs16 server |
| `bench_serving_h20_qwen35_9b_dp4_dflash_zlab_bs16_rate4/16/64.log` | DP=4+z-lab bs16 三档速率 |
| `server_h20_qwen35_9b_serving_dp4_dflash_zlab_bs8.log` | DP=4+z-lab bs8 server |
| `bench_serving_h20_qwen35_9b_dp4_dflash_zlab_bs8_rate4/16/64.log` | DP=4+z-lab bs8 三档速率 |
| `run_zlab_dflash.sh` / `run_zlab_dflash_all.sh` | 复现脚本（server 启动 + 三档速率压测） |

## 7. 环境坑（z-lab 版专属）

1. **config 布局不同但兼容**：z-lab 的 `block_size` 在 `dflash_config` 内（unieai 在
   顶层），`parse_dflash_draft_config` 两者都支持，无需改代码。
2. **sliding_attention 层**：z-lab 6 层中 5 层 sliding（window 4096），
   `get_dflash_attention_sliding_window_size` 返回 `sliding_window - 1`，fa3 后端
   正常处理，无额外配置。
3. **draft 显存略高**：2.56 GB（unieai ~2.1 GB），TP=1/DP=4 占 2.56 GB、TP=4 分片
   0.92 GB，对 KV/mamba 池影响可忽略（pool end avail 29–40 GB）。
4. **ninja PATH**：同 unieai/MTP，启动 server 需 `PATH="$PWD/.venv/bin:$PATH"`。
