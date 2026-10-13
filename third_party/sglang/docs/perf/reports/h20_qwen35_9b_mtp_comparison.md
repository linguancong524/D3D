# Qwen3.5-9B H20 MTP 投机解码加速对比（SGLang v0.5.18）

日期：2026-09-03。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力，
`mtp_num_hidden_layers: 1`）。运行时：torch 2.13.0+cu130、flashinfer 0.6.17、
attention=fa3、linear-attn=triton。H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告在 [TP=1/TP=4/DP=4 三形态对比](h20_qwen35_9b_tp1_tp4_dp4_comparison.md)
（2026-09-02 非 MTP 基线）基础上，叠加 **MTP 投机解码**，量化其对延迟与吞吐的影响。

## 0. MTP 是什么 / 怎么开

Qwen3.5-9B 自带 **1 层 MTP（Multi-Token Prediction）头**（config
`mtp_num_hidden_layers: 1`，checkpoint 含 15 个 `mtp.*` 权重键）。SGLang 通过
**EAGLE 投机解码**复用这层 MTP 头做 draft：把 draft 架构
`Qwen3_5ForConditionalGeneration` 重写为 `Qwen3_5ForCausalLMMTP`
（`python/sglang/srt/configs/model_config.py`），因此**同一个模型路径同时充当
target 与 draft，无需单独 draft checkpoint**（解决了 CLAUDE.md §8 的未决前置项）。

启动参数（三形态通用，叠加在各自 `--tp-size`/`--dp-size` 之上）：

```bash
--speculative-algorithm EAGLE \
--speculative-draft-model-path /workspace/models/Qwen3.5-9B \
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

（`(3, 1, 4)` 与 `_auto_choose_speculative_params` 的 else 分支一致，也与
Qwen3-Coder-Next 测试配方相同。）

**接受长度（Accept length）稳定在 3.21–3.36**（每步 4 个 draft token 中平均接受
~3.2 个），即每个 target forward 平均多产出 ~2.2 个 token，是 MTP 收益的来源。

## 1. MTP 的内存副作用（关键约束）

MTP 不是「免费午餐」，draft 权重 + 投机状态会挤占显存，带来两个副作用：

1. **`max_running_requests` 被投机解码逻辑重置为 48**（非 MTP 的 cap 是
   TP=1=241、TP=4=1155、DP=4=241/rank）。这是 rate 64 吞吐对比的**混淆变量**，
   需单独用「对齐 cap」重跑隔离（见 §4）。
2. **draft 权重压缩 mamba 池 → 有效 cap 进一步下降**：
   - TP=1：显式 `--max-running-requests 241` 后，mamba 池只剩 623 slot → 有效 cap **124**
   - TP=4：显式 `--max-running-requests 1155` 后，mamba 池 3161 slot → 有效 cap **632**
3. **draft CUDA graph 捕获在高 bs 下 OOM**：TP=4 用 cap 1155 时，EAGLE draft worker
   的 decode graph 捕获在 bs=1155 处 OOM（`eagle_draft_cuda_graph_runner.py`，
   `Tried to allocate 244 MiB`）。规避：`--cuda-graph-max-bs-decode 256`（bs>256 走 eager）。
4. **高并发 forward OOM 墙**：TP=4 cap 632 时，rate 64 压测中途 batch 涨高，
   EAGLE verify（每请求 4 draft token）+ draft 激活超出 15% 激活预算
   （`Tried to allocate 1.32 GiB, 1.02 GiB free`）→ 进程被 OOM kill。
   **即 TP=4+MTP 的安全 cap 上限约 256**（非 MTP TP=4 能跑 cap 1155，MTP 版 cap 632 就 OOM）。

## 2. rate 4（标准工况，n=100）：MTP 延迟收益最大

| 指标 | TP=1 非MTP | TP=1 **MTP** | TP=4 非MTP | TP=4 **MTP** | DP=4 非MTP | DP=4 **MTP** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 3.95 | 4.04 | 4.06 | 4.08 | 3.96 | 4.05 |
| E2E 中位 (ms) | 1 203.07 | **580.55** | 470.55 | **305.21** | 964.65 | **440.47** |
| E2E p99 (ms) | 2 216.88 | **1 282.00** | 876.12 | **685.44** | 1 705.29 | **957.13** |
| TTFT 中位 (ms) | 59.40 | 67.98 | 31.97 | 40.64 | 52.43 | 57.12 |
| TPOT 中位 (ms) | 7.75 | **3.83** | 2.97 | **1.92** | 6.38 | **2.75** |
| ITL 中位 (ms) | 6.48 | **2.49** | 2.73 | **1.30** | 6.31 | **2.37** |
| Accept length | — | 3.22 | — | 3.21 | — | 3.36 |

**rate 4 下吞吐都 ≈4 req/s（发送速率受限），MTP 的收益全在延迟**：

| 形态 | E2E 中位加速 | TPOT 加速 |
|---|---|---|
| TP=1 | 1 203→581 ms（**2.07×**） | 7.75→3.83 ms（**2.02×**） |
| TP=4 | 471→305 ms（**1.54×**） | 2.97→1.92 ms（**1.55×**） |
| DP=4 | 965→440 ms（**2.19×**） | 6.38→2.75 ms（**2.32×**） |

- **TP=1 / DP=4 收益最大（~2.1–2.3×）**：单卡/单副本 decode 是带宽受限，MTP 每步
  多接受 ~2.2 token 直接摊薄每 token 的权重读取。
- **TP=4 收益较小（~1.5×）**：TP=4 本身已把权重分片到 4 卡、decode 很快（TPOT 3 ms），
  MTP 的相对增益被压缩；且 MTP 的 draft forward 也要跨 4 卡 allreduce，摊薄了收益。
- **TTFT 略升**（TP=1 59→68、TP=4 32→41 ms）：MTP 的 draft 阶段增加了首 token 前的
  计算，但绝对值仍很小（<70 ms），可忽略。

## 3. rate 16（n=200，server 开始饱和）

| 指标 | TP=1 非MTP | TP=1 **MTP** | TP=4 非MTP | TP=4 **MTP** | DP=4 非MTP | DP=4 **MTP** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 13.84 | 13.53 | 15.53 | **15.78** | 14.83 | 15.57 |
| E2E 中位 (ms) | 3 457.40 | **2 155.65** | 748.83 | **444.86** | 1 115.37 | **530.99** |
| E2E p99 (ms) | 7 595.10 | **6 246.87** | 1 646.78 | **1 241.76** | 2 131.75 | **1 177.71** |
| TPOT 中位 (ms) | 23.11 | **13.01** | 4.74 | **2.68** | 7.60 | **3.43** |
| Accept length | — | 3.26 | — | 3.26 | — | 3.25 |

| 形态 | E2E 中位加速 | TPOT 加速 |
|---|---|---|
| TP=1 | 3 457→2 156 ms（**1.60×**） | 23.11→13.01 ms（**1.78×**） |
| TP=4 | 749→445 ms（**1.68×**） | 4.74→2.68 ms（**1.77×**） |
| DP=4 | 1 115→531 ms（**2.10×**） | 7.60→3.43 ms（**2.22×**） |

rate 16 下 MTP 仍带来 **1.6–2.2× 延迟改善**，吞吐基本持平（TP=4+MTP 15.78 甚至略超
非 MTP 15.53）。TP=1 单卡饱和最严重（非 MTP TPOT 23 ms），MTP 把它压到 13 ms。

## 4. rate 64（n=400，逼近饱和）：MTP 的显存墙显现

rate 64 是 MTP 收益与显存代价的拉锯点。直接对比受 **cap 混淆**影响
（MTP 默认 cap 48 vs 非 MTP cap 241/1155），故额外做了「对齐 cap」重跑。

### 4.1 默认 cap（MTP cap 48）

| 指标 | TP=4 非MTP | TP=4 **MTP**(cap48) | DP=4 非MTP | DP=4 **MTP**(cap48) |
|---|---|---|---|---|
| 请求吞吐 (req/s) | 42.91 | 34.80 | **47.71** | **48.01** |
| E2E 中位 (ms) | 4 161.20 | **2 774.72** | 2 086.22 | **1 853.62** |
| TPOT 中位 (ms) | 34.85 | **9.58** | 16.21 | **14.35** |
| Accept length | — | 3.23 | — | 3.28 |

- **DP=4+MTP 吞吐反超非 MTP**（48.01 vs 47.71 req/s），且延迟更好
  （E2E 1 854 vs 2 086 ms、TPOT 14.35 vs 16.21 ms）。DP=4 每 rank cap 48×4=192
  总并发，已接近非 MTP 的 241/rank 行为，MTP 的延迟收益直接转化为吞吐。
- **TP=4+MTP 吞吐低于非 MTP**（34.80 vs 42.91）：cap 48 限制了并发，是混淆变量，
  需对齐 cap 重跑（§4.2）。

### 4.2 对齐 cap 重跑（隔离 MTP 真实吞吐上限）

| 配置 | cap | 请求吞吐 (req/s) | E2E 中位 (ms) | TPOT 中位 (ms) | 结果 |
|---|---|---|---|---|---|
| TP=1+MTP | 241（有效 124） | 15.08 | 15 164.54 | 53.13 | 深度过载，饱和 ~15 req/s |
| TP=4+MTP | 632 | — | — | — | **OOM 崩溃**（forward 激活超预算） |
| TP=4+MTP | 256（=graph 上限） | 39.08 | 4 328.04 | 35.79 | 安全上限 |

- **TP=1+MTP**：cap 提到 241（mamba 池限制后有效 124），rate 64 仍深度过载
  （E2E 15 s、TPOT 53 ms），饱和吞吐 ~15 req/s，与非 MTP TP=1 的 ~13.8 相当。
  单卡 9B 的瓶颈是算力/带宽，MTP 救不了高并发吞吐。
- **TP=4+MTP**：cap 632 直接 OOM（§1.4 的 forward 激活墙）；安全 cap 256 下吞吐
  **39.08 req/s，仍低于非 MTP TP=4 的 42.91**。即 **TP=4+MTP 受显存限制，
  高并发吞吐上限（~39）低于非 MTP（~43）**——draft 权重压缩 mamba 池
  （cap 1155→632），cap 提到 632 又触发 forward OOM，两头受限。

### 4.3 rate 64 小结

| 形态 | 非MTP 吞吐 | MTP 吞吐（安全 cap） | MTP 吞吐结论 |
|---|---|---|---|
| TP=1 | ~13.8（饱和） | ~15（cap124） | 持平，单卡瓶颈 |
| TP=4 | 42.91 | 39.08（cap256） | **MTP 更低**（显存墙） |
| DP=4 | 47.71 | 48.01（cap48） | **MTP 持平/略高** |

## 5. 关键结论

1. **MTP 是延迟优化，不是吞吐优化**：在 rate 4/16（低中并发），MTP 带来
   **1.5–2.3× 的 E2E/TPOT 延迟改善**（accept ~3.2 token/step），吞吐基本不变
   （受发送速率或单卡算力限制）。
2. **收益随形态而异**：TP=1 / DP=4（单副本、带宽受限）收益最大（~2.1–2.3×）；
   TP=4（已分片、decode 快）收益较小（~1.5×），因 draft forward 也要 allreduce。
3. **MTP 有显存代价**：draft 权重压缩 mamba 池 → 有效 cap 下降（TP=1 241→124、
   TP=4 1155→632）；高并发下 EAGLE verify + draft 激活触发 forward OOM
   （TP=4 cap 632 崩溃，安全上限 ~256）。
4. **高并发（rate 64）下 MTP 吞吐分化**：
   - **DP=4+MTP 吞吐持平/略超非 MTP**（48.01 vs 47.71）且延迟更好 → 高吞吐场景 MTP 无副作用。
   - **TP=4+MTP 吞吐低于非 MTP**（39.08 vs 42.91）→ 显存墙限制，高吞吐场景 MTP 反而拖累。
   - **TP=1+MTP 吞吐持平**（~15 vs ~13.8）→ 单卡瓶颈，MTP 救不了。
5. **TTFT 略升**（<10 ms 量级），可忽略。

## 6. 部署建议（9B / 4× H20 + MTP）

| 场景 | 推荐 | 理由 |
|---|---|---|
| **低延迟单流 / 实时对话** | **TP=4 + MTP** | TPOT 1.92 ms、E2E 305 ms（rate 4），全场最低延迟 |
| **低中并发、重延迟** | **DP=4 + MTP** | E2E 440 ms（rate 4）、TPOT 2.75 ms，延迟收益最大（2.2×） |
| **高吞吐 serving** | **DP=4 + MTP** | 吞吐 48.01 req/s（≥非 MTP 47.71）且延迟更好，MTP 无副作用 |
| **高吞吐 + 必须 TP=4** | 非 MTP TP=4 | MTP 版受显存墙限制（39 vs 43 req/s），高吞吐下不开 MTP 更好 |
| **单卡 / 边缘** | TP=1 + MTP（低并发） | 低并发延迟 2× 改善；高并发单卡瓶颈，MTP 无吞吐收益 |

**一句话**：MTP 对 9B 是**延迟利器**（低中并发 1.5–2.3×），**高吞吐场景只在
DP=4 下无副作用**；TP=4 高吞吐下 MTP 因显存墙反而拖累，应关闭。

## 7. 数据有效性

- 所有运行 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后确认 4 卡全部释放到 0 MiB。
- 9 组主基准（3 形态 × rate 4/16/64）全部 100%/200%/400% 请求成功，无失败。
- Accept length 稳定 3.21–3.36（跨形态、跨速率一致）。
- TP=4+MTP cap632 的 OOM 崩溃日志保留为证据（`server_..._tp4_mtp_r64_cap632.log`）。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §8 日志清单）。

## 8. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `server_h20_qwen35_9b_serving_tp1_mtp.log` | TP=1+MTP server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp1_mtp_rate4/16/64.log` | TP=1+MTP 三档速率 |
| `server_h20_qwen35_9b_serving_tp4_mtp.log` | TP=4+MTP server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp4_mtp_rate4/16/64.log` | TP=4+MTP 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_dp4_mtp.log` | DP=4+MTP server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_dp4_mtp_rate4/16/64.log` | DP=4+MTP 三档速率（cap48） |
| `server_h20_qwen35_9b_serving_tp1_mtp_r64_cap241.log` | TP=1+MTP cap241（有效124）server |
| `bench_serving_h20_qwen35_9b_tp1_mtp_rate64_cap241.log` | TP=1+MTP cap241 rate 64 |
| `server_h20_qwen35_9b_serving_tp4_mtp_r64_cap1155.log` | TP=4+MTP cap1155（**graph 捕获 OOM 崩溃**，证据） |
| `server_h20_qwen35_9b_serving_tp4_mtp_r64_cap632.log` | TP=4+MTP cap632（**forward OOM 崩溃**，证据） |
| `server_h20_qwen35_9b_serving_tp4_mtp_r64_cap256.log` | TP=4+MTP cap256（安全上限）server |
| `bench_serving_h20_qwen35_9b_tp4_mtp_rate64_cap256.log` | TP=4+MTP cap256 rate 64 |

## 9. 环境坑（MTP 专属）

1. **ninja PATH**：同非 MTP（§8 of 对比报告），启动 server 需
   `PATH="$PWD/.venv/bin:$PATH"`。
2. **draft graph 捕获 OOM**：TP=4 高 cap 时 EAGLE draft worker 的 decode graph
   捕获 OOM，需 `--cuda-graph-max-bs-decode 256` 限制捕获上限（bs>256 走 eager）。
3. **forward 激活 OOM 墙**：TP=4+MTP cap 632 在 rate 64 中途 OOM（verify + draft
   激活超 15% 预算），安全 cap 上限 ~256。这是 MTP 在 TP=4 高并发的真实限制。
4. **cap 混淆**：投机解码默认把 `max_running_requests` 重置为 48，做吞吐对比时
   必须显式对齐 cap，否则 MTP 吞吐被低估。
