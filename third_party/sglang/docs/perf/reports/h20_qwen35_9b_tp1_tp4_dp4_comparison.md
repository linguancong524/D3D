# Qwen3.5-9B H20 部署形态对比：单卡 / 4卡TP / 4卡并发（SGLang v0.5.18）

日期：2026-09-02。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB（sm90，
HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力）。
运行时：torch 2.13.0+cu130、flashinfer 0.6.17、attention=fa3、linear-attn=triton
（FLA chunk GDN）。H20 平台**不带** `--cuda-graph-backend-prefill=disabled`
（prefill 走 BCG 图捕获，见 CLAUDE.md §4.1）。

本报告对比三种 4 卡 H20 部署形态（均为 9B）：

| 形态 | 含义 | 启动参数 |
|---|---|---|
| **单卡（TP=1）** | 1 卡 1 副本，其余 3 卡空闲 | `--tp-size 1` |
| **4卡TP（TP=4）** | 4 卡拼 1 个大副本（张量并行） | `--tp-size 4` |
| **4卡并发（DP=4）** | 4 卡各 1 个独立副本（数据并行，SGLang 内置 DP） | `--tp-size 1 --dp-size 4` |

> 4卡并发用 SGLang 内置 `--dp-size 4`（每卡一个 TP=1 副本，server 内部做请求分发），
> 等价于项目部署模型「TP=1 一卡一副本 × 4，前置 LB」（CLAUDE.md §1）。本 pod 无独立
> router 二进制，故用内置 DP 复现该形态。

## 1. 内存布局与有效性

| 形态 | Load weight avail | pool end avail | max_running_requests cap |
|---|---|---|---|
| one_batch TP=1 | 94.60 GB | 14.15 GB | 443 |
| one_batch TP=4 | 93.58 GB | 32.37 GB | 443 |
| serving TP=1 | 94.60 GB | 14.00 GB | 241 |
| serving TP=4 | 93.54–93.77 GB | ~13.0 GB | 1155 |
| serving DP=4（每 rank） | 94.60 GB | 14.02 GB | 241/rank |

- 所有运行 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净，无残留进程）。
- serving 带 `--mamba-ssm-dtype bfloat16`（ssm 池减半、mamba 上限翻倍），故 cap 241
  （单卡）/ 1155（TP=4，权重分片后每卡空闲显存更多 → mamba 池更大）。
- one_batch 不带 bfloat16，cap 443（与 CLAUDE.md §5 H20 9B@0.85 一致）。
- **TP=1 one_batch bs32/il4096 格 OOM**（见 §2.2），其余 8/9 格有效；TP=4 9/9 完整。

## 2. 离线单批（one_batch，output=128，mem 0.85，median）

one_batch 的 prefill 是**单发全量 extend（无 chunked prefill）**，GDN 临时 buffer
随 token 总量线性增长（~155 KB/tok，9B）。

### 2.1 TP=1（单卡）vs TP=4（4卡TP）

| bs | input | TP=1 prefill | TP=4 prefill | TP=1 decode | TP=4 decode |
|---|---|---|---|---|---|
| 1 | 128 | 23.6 ms (5 423) | 23.2 ms (5 528) | 6.84 ms/tok (146) | 3.43 ms/tok (291) |
| 1 | 1024 | 131.8 ms (7 771) | 111.7 ms (9 166) | 6.86 ms/tok (146) | 3.31 ms/tok (302) |
| 1 | 4096 | 519.1 ms (7 891) | 147.4 ms (27 788) | 7.00 ms/tok (143) | 3.32 ms/tok (302) |
| 8 | 128 | 178.4 ms (5 739) | 50.8 ms (20 160) | 7.27 ms/tok (1 100) | 3.52 ms/tok (2 274) |
| 8 | 1024 | 876.0 ms (9 352) | 283.6 ms (28 891) | 7.55 ms/tok (1 059) | 3.57 ms/tok (2 244) |
| 8 | 4096 | 3 560.8 ms (9 202) | 1 595.0 ms (20 544) | 8.77 ms/tok (912) | 3.76 ms/tok (2 125) |
| 32 | 128 | 451.6 ms (9 070) | 134.2 ms (30 511) | 8.52 ms/tok (3 758) | 4.35 ms/tok (7 353) |
| 32 | 1024 | 3 409.6 ms (9 611) | 965.9 ms (33 926) | 9.56 ms/tok (3 347) | 4.62 ms/tok (6 920) |
| 32 | 4096 | **OOM** | 3 832.5 ms (34 200) | — | 5.87 ms/tok (5 450) |

（prefill 列为「延迟 ms (吞吐 tok/s)」，decode 列为「ms/tok (tok/s)」）

### 2.2 关键观察

1. **decode bs=1：TP=4 比 TP=1 快 ~2×**（6.84→3.43 ms/tok）。decode 是带宽受限，
   TP=4 把权重分片到 4 卡，每 token 每卡只读 1/4 权重 → 聚合带宽 ~4×，扣除 allreduce
   后净 ~2×。bs=1 的 ctx 无关性（GDN 状态 O(1)/step）在两种形态下都保持。
2. **prefill：TP=4 快 ~2–3.5×，且 token 越多优势越大**。bs1/il4096：519→147 ms
   （3.5×）；bs8/il4096：3561→1595 ms（2.2×）。prefill 是算力受限，TP=4 提供 ~4×
   算力，扣除 allreduce 后净 2–3.5×。
3. **TP=1 bs32/il4096 OOM**：131 072 token 单发 extend，GDN 临时 buffer
   ~155 KB/tok × 131 072 ≈ 20 GB，超过 TP=1 pool-end 空闲 14.15 GB。这是单卡
   9B 的**真实限制**（CLAUDE.md §7.3 根因），非 bug。TP=4 把 buffer 分片到 4 卡
   （每卡 ~5 GB）故能跑通。server 端 `chunked_prefill_size=8192` 可把峰值压到
   ~1.3 GB/chunk 规避，但 one_batch 是单发 extend 不切块。

## 3. 在线 serving（512 in / 256 out，random，sglang backend）

三档发送速率：rate 4（n=100，项目标准工况）、rate 16（n=200）、rate 64（n=400，
逼近饱和）。

### 3.1 rate 4（标准工况，n=100）

| 指标 | TP=1 | TP=4 | DP=4 |
|---|---|---|---|
| 请求吞吐 (req/s) | 3.95 | 4.06 | 3.96 |
| 输出吞吐 (tok/s) | 532.90 | 547.94 | 533.32 |
| 峰值并发 | 13 | 10 | 12 |
| E2E 中位 (ms) | 1 203.07 | **470.55** | 964.65 |
| E2E p99 (ms) | 2 216.88 | **876.12** | 1 705.29 |
| TTFT 中位 (ms) | 59.40 | **31.97** | 52.43 |
| TTFT p99 (ms) | 655.29 | **169.74** | 79.03 |
| TPOT 中位 (ms) | 7.75 | **2.97** | 6.38 |
| ITL 中位 (ms) | 6.48 | **2.73** | 6.31 |

**rate 4 下三者吞吐都 ≈4 req/s**（受发送速率限制，单卡已能跟上，server 不是瓶颈）。
差异全在延迟：**TP=4 延迟最低**（E2E 中位 470 ms、TPOT 3.0 ms），适合低延迟/单流；
TP=1 延迟最高（单卡）；DP=4 居中（请求分散到 4 卡，但每卡单副本）。

### 3.2 rate 16（n=200，server 开始成为瓶颈）

| 指标 | TP=1 | TP=4 | DP=4 |
|---|---|---|---|
| 请求吞吐 (req/s) | 13.84 | **15.53** | 14.83 |
| 输出吞吐 (tok/s) | 1 958.37 | **2 197.82** | 2 097.93 |
| 峰值并发 | 105 | 40 | 45 |
| E2E 中位 (ms) | 3 457.40 | **748.83** | 1 115.37 |
| E2E p99 (ms) | 7 595.10 | **1 646.78** | 2 131.75 |
| TTFT 中位 (ms) | 88.32 | **37.13** | 52.86 |
| TPOT 中位 (ms) | 23.11 | **4.74** | 7.60 |

> DP=4 列于 2026-09-03 重跑确认（`_v2` 日志），与 2026-09-02 首跑一致（吞吐 14.83 vs
> 14.84、E2E 中位 1 115 vs 1 107 ms、TPOT 7.60 vs 7.52 ms，均在 ~1% 内）。

rate 16 下 server 开始饱和：**TP=4 吞吐与延迟双优**（15.53 req/s、E2E 749 ms、
TPOT 4.7 ms）。TP=1 单卡饱和最严重（并发堆到 105、TPOT 23 ms、E2E 3.5 s）。
DP=4 介于两者之间。

### 3.3 rate 64（n=400，逼近饱和）

| 指标 | TP=4 | DP=4 |
|---|---|---|
| 请求吞吐 (req/s) | 42.91 | **47.71** |
| 输出吞吐 (tok/s) | 5 618.14 | **6 245.64** |
| 峰值并发 | 355 | 237 |
| E2E 中位 (ms) | 4 161.20 | **2 086.22** |
| E2E p99 (ms) | 8 297.68 | **4 781.27** |
| TTFT 中位 (ms) | 86.76 | **65.83** |
| TPOT 中位 (ms) | 34.85 | **16.21** |

（TP=1 在 rate 16 已严重饱和，未再测 rate 64；其饱和吞吐 ~13.8 req/s。）

**高吞吐下 DP=4（4卡并发）反超 TP=4**：吞吐 47.71 vs 42.91 req/s（+11%），
延迟 E2E 中位 2 086 vs 4 161 ms（~2× 更好）、TPOT 16.2 vs 34.9 ms（~2× 更好）。
原因：DP=4 是 4 个独立副本，每副本拥有**完整模型带宽**（无 per-token allreduce
开销），聚合 decode 带宽 ~4×；TP=4 是 1 个大副本，每 token 都要跨 4 卡 allreduce，
高并发下 allreduce 成为瓶颈。

## 4. 关键结论

1. **decode 单流延迟**：TP=4 比 TP=1 快 ~2×（权重分片，聚合带宽 ~4× 扣 allreduce）。
   bs=1 ctx 无关性（GDN O(1)/step）两种形态都保持。
2. **prefill**：TP=4 快 ~2–3.5×（算力受限，token 越多优势越大）。
3. **单卡 9B 限制**：one_batch bs32/il4096（131k token 单发 extend）OOM，GDN 临时
   buffer ~20 GB 超单卡空闲。server 端 chunked prefill 可规避，one_batch 不能。
4. **serving 形态选择**：
   - **低延迟 / 单流 / 小并发** → **TP=4**（rate 4/16 延迟最低，TPOT 3–5 ms）。
   - **高吞吐 serving** → **DP=4（4卡并发）**（rate 64 吞吐 +11%、延迟 ~2× 更好，
     无 allreduce 瓶颈）。
   - 这与项目部署模型「TP=1 一卡一副本 × N，前置 LB」（CLAUDE.md §1）一致：
     4 卡并发（DP=4）是 9B 高吞吐 serving 的最优形态。
5. **rate 4 标准工况下三者吞吐相同**（≈4 req/s，发送速率受限），选型看延迟：
   TP=4 延迟最低。

## 5. 部署建议（9B / 4× H20）

| 场景 | 推荐形态 | 理由 |
|---|---|---|
| 低延迟单流 / 实时对话 | TP=4 | TPOT 3 ms、E2E 470 ms（rate 4） |
| 高吞吐批量 serving | **DP=4（4卡并发）** | 吞吐 47.7 req/s、延迟 ~2× 优于 TP=4（rate 64） |
| 单卡部署 / 边缘 | TP=1 | 1 卡 1 副本，但 bs32 长上下文 prefill 会 OOM |

## 6. 数据有效性

- 所有运行 `Load weight begin. avail mem` ≈ 93.5–94.6 GB（GPU 干净）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后确认 4 卡全部释放到 0 MiB。
- one_batch TP=1 8/9 格（bs32/il4096 OOM，已标注）；TP=4 9/9 完整。
- serving 三形态 × 三档速率全部 100%/200%/400% 请求成功，无失败。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §7 日志清单）。

## 7. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `bench_one_batch_h20_qwen35_9b_tp1.log` | one_batch TP=1（8/9，bs32/4096 OOM） |
| `bench_one_batch_h20_qwen35_9b_baseline.log` | one_batch TP=4（9/9，2026-09-01 基线） |
| `bench_serving_h20_qwen35_9b_tp1.log` | serving TP=1 rate 4 |
| `bench_serving_h20_qwen35_9b_tp4.log` | serving TP=4 rate 4 |
| `bench_serving_h20_qwen35_9b_dp4.log` | serving DP=4 rate 4 |
| `bench_serving_h20_qwen35_9b_tp1_rate16.log` | serving TP=1 rate 16 |
| `bench_serving_h20_qwen35_9b_tp4_rate16.log` | serving TP=4 rate 16 |
| `bench_serving_h20_qwen35_9b_dp4_rate16.log` | serving DP=4 rate 16（2026-09-02 首跑） |
| `bench_serving_h20_qwen35_9b_dp4_rate16_v2.log` | serving DP=4 rate 16（2026-09-03 重跑，§3.2 采用） |
| `bench_serving_h20_qwen35_9b_tp4_rate64.log` | serving TP=4 rate 64 |
| `bench_serving_h20_qwen35_9b_dp4_rate64.log` | serving DP=4 rate 64 |
| `server_h20_qwen35_9b_serving_tp1.log` / `_tp1_r16.log` | TP=1 server 日志 |
| `server_h20_qwen35_9b_serving_tp4.log` / `_tp4_r16.log` / `_tp4_r64.log` | TP=4 server 日志 |
| `server_h20_qwen35_9b_serving_dp4.log` / `_dp4_r64.log` | DP=4 server 日志 |
| `server_h20_qwen35_9b_serving_dp4_r16_v2.log` | DP=4 server 日志（2026-09-03 rate 16 重跑） |

## 8. 环境坑（ninja PATH）

TP=4 / DP=4 server 首次启动在 CUDA graph 捕获阶段报
`FileNotFoundError: 'ninja'`——FlashInfer AllReduce Fusion（backend=trtllm）JIT 编译
经 `sglang/kernels/jit/utils/compile/ninja.py` 需要 `ninja` 在 PATH 上。ninja 在
`.venv/bin/ninja` 但默认不在 PATH。**修复**：启动 server 时
`PATH="$PWD/.venv/bin:$PATH"`。TP=1 未触发（其 kernel 变体已 JIT 缓存）。
