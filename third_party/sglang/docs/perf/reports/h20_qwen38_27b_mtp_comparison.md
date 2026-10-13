# Qwen3.8-27B H20 4×96GB：并发/吞吐基线 与 MTP(4/8) 投机解码对比

日期：2026-09-15。硬件：4× NVIDIA H20 96GB（sm90，HBM3e ~4.0 TB/s，bf16 dense
峰值 ~148 TFLOPS），K8s dev-pod。模型：`/workspace/models/Qwen3.8-27B`（BF16，
18 分片 ~52.8 GB，dense 混合 GDN 线性注意力 + 全注意力 VL 模型，48 线性 + 16 全
注意力层，`mtp_num_hidden_layers: 1` → 自带 1 层 MTP 头，15 个 `mtp.*` 权重键）。
运行时：sglang `0.5.19.dev39+g91a20e1`（`72414d7`）、torch 2.13.0+cu130、
flashinfer 0.6.17、sgl-kernel 0.4.6.post1、attention=fa3（默认）、
linear-attn=triton、mamba_backend=triton。H20 平台**不带**
`--cuda-graph-backend-prefill=disabled`（prefill 走 BCG 图捕获）。

本报告回答两个问题：**(1) 4 卡 H20 用仓库当前 sglang 直接起 27B，能撑多少并发、
多少输出吞吐；(2) 打开 checkpoint 自带 MTP（mtp=4 / mtp=8）后并发与吞吐怎么变。**

复现脚本：`docs/perf/scripts/run_qwen38_27b_h20.sh`（模式 `tp4 | tp4_mtp4 | tp4_mtp8 |
tp4_mtp4_rss | dp4 | dp4_mtp4`，可选 `CAP=` / `RATES=` 覆盖）。

## 0. 结论速览

1. **4 卡能跑，标准形态就是 TP=4**：权重 13.04 GB/卡，剩 ~68 GB 给池子，
   KV 池 228.9 万 token（≈3.5 个满 262k 会话）、mamba 态池 1718 slot →
   **`max_running_requests` 天然上限 343**；瓶颈是 mamba 态池，不是 KV、也不是显存。
2. **饱和输出（512 in/256 out 名义，实测 EOS 后 ~131 tok out/req）：
   21.4 req/s、2 805 output tok/s**（TP=4，rate 64，平均并发 251、峰值 394 排队）。
   Peak output 6 167 tok/s。
3. **MTP 是延迟优化，不是吞吐优化**（与 9B 结论一致，27B 上更极端）：
   低并发（rate 4，~5 并发）**TPOT 9.40 → 4.18 ms（2.25×）**、E2E 1 326 → 645 ms
   （2.06×）；高并发饱和时 MTP 的 throughput 反而**低于**基线。
4. **MTP 会把 `max_running_requests` 重置为 48**（`speculative_hook.py`：
   "Max running requests is reset to 48 for speculative decoding"）。做吞吐对比必须
   显式 `--max-running-requests` 对齐，否则 MTP 被严重低估。
5. **MTP 的并发代价来自 verify 中间态**：`intermediate_ssm_state_cache` 吃掉
   13.15 GB（MTP4）/18.14 GB（MTP8）mamba 池 → 有效上限
   **343 → 186（MTP4）→ 128（MTP8）**；KV 池基本不变。
6. **MTP=8 不值**：accept length 3.01 → 3.90（+30%），但每步 verify 成本几乎等比例
   上升，**每 token 延迟与 MTP=4 持平（4.33 vs 4.18 ms），高并发反而更差**
   （rate 64：18.25 → 14.18 req/s）。
7. **DP=4（一卡一副本）不如 TP=4**：decode 每步要读整份 53 GB 权重（TP=4 每卡只读
   13 GB），饱和吞吐 16.65 < 21.42 req/s。DP=4 唯一亮点是 **DP=4+MTP4**：
   吞吐 16.83 ≈ 基线，TPOT 53.25 → 24.90 ms（2.1×），是"开 MTP 不掉吞吐"的组合。
8. **`--enable-linear-replayssm-spec` 能救并发**：verify 中间态改走固定 ring
   （`intermediate_ssm_state_cache = 0.00GB`），上限回到 **322**（≈基线的 94%），
   饱和吞吐 19.12 req/s（比普通 MTP4 的 18.25 好，仍低于基线 21.42）。

一句话：**要吞吐 → TP=4 基线（21.4 req/s）；要延迟 → TP=4+MTP4（TPOT 2.25×，
并发上限 186）；高并发下想两者兼顾 → DP=4+MTP4 或 TP=4+MTP4+ReplaySSM。**

## 1. 启动命令与池子/并发上限

基线（"当前 sglang 直接起"，除模型自身需要外不加任何调优 flag）：

```bash
.venv/bin/python3 -m sglang.launch_server \
    --model-path /workspace/models/Qwen3.8-27B \
    --host 0.0.0.0 --port 30000 \
    --tp-size 4 --mem-fraction-static 0.85 --mamba-ssm-dtype bfloat16
```

MTP（用 checkpoint 自带 MTP 头，draft = 同一个模型路径，无需单独 draft checkpoint）：

```bash
    ... --speculative-algorithm EAGLE \
        --speculative-draft-model-path /workspace/models/Qwen3.8-27B \
        --speculative-num-steps 3 --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4            # 即本文的 "mtp=4"
        # mtp=8：--speculative-num-steps 7 --speculative-num-draft-tokens 8
```

（`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` 的 H200/SM90 配方即 `(3,1,4)`；
H20 与 H200 同为 SM90，配方通用。）

各启动配置的池子与并发上限（TP=4 时 4 卡数值一致，取 TP0/DP0）：

| 配置 | KV 池 (token) | KV 显存 | mamba slot | ssm_state | **verify 中间态** | cap |
|---|---|---|---|---|---|---|
| **TP=4 基线** | 2 288 771 | 17.46+17.46 GB | 1718 | 30.22 GB | — | **343** |
| TP=4 + MTP4（默认 cap） | 2 729 719 | — | 931 | 16.38 GB | 13.15 GB | 48（spec 默认重置） |
| TP=4 + MTP4（`--max-running-requests 343`） | 2 107 842 | 16.08+16.08 GB | 931 | 16.38 GB | 13.15 GB | **186** |
| TP=4 + MTP8（`343`） | 2 112 348 | 16.12+16.12 GB | 643 | 11.32 GB | 18.14 GB | **128** |
| TP=4 + MTP4 + ReplaySSM（`343`） | 2 106 400 | 16.07+16.07 GB | 1614 | 28.39 GB | **0.00 GB** | **322** |
| DP=4 基线（每 rank=每卡） | 252 252 | 7.70+7.70 GB | 188 | 13.29 GB | — | **37/rank（148 总）** |
| DP=4 + MTP4（每 rank） | 194 961 | 5.95+5.95 GB | 82 | 5.84 GB | 4.78 GB | **16/rank（64 总）** |

要点：

- 每卡权重 13.04 GB（TP=4）；`mem-fraction-static 0.85` → 池子 ~68 GB。
- 上限由 server 日志的这句给出（可直接抄进任何新配置的自检）：
  `max_running_requests is capped to 343 by the mamba state cache
  (max_mamba_cache_size=1718, 5 state slots per request)`。
- **DP=4 的 KV 池只有 25.2 万 token**：满 262k 的单个会话都装不下，
  长上下文场景不要用 DP=4（BF16 权重 53 GB/卡把池子挤没了）。

## 2. 基准矩阵（512 in / 256 out，random dataset，与项目历次基准同工况）

三档速率对应不同并发水平：rate 4（n=100，平均并发 3–11）、rate 16（n=200，平均并发
39–69）、rate 64（n=400，**饱和**，平均并发 158–251）。所有配置生成 token 总数完全
相同（rate4: 13 484、rate16: 28 299、rate64: 52 367），说明 MTP 输出与基线逐 token
一致（lossless），横向可比。

### 2.1 rate 4（低并发 / 延迟工况，n=100）

| 配置 | req/s | out tok/s | 并发 | E2E50 (ms) | E2E 均值 | TTFT50 | TTFT 均值 | **TPOT** | ITL | accept |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线** | 3.94 | 531 | 5.2 | 1 403 | 1 326 | 74.1 | 90.2 | **9.40** | 9.24 | — |
| TP=4 + MTP4 | 4.06 | 548 | 2.6 | 605 | 645 | 82.4 | 98.2 | **4.18** | 4.09 | 3.01 |
| TP=4 + MTP8 | 4.08 | 550 | 2.6 | 539 | 647 | 87.2 | 99.5 | **4.33** | 4.09 | 3.90 |
| TP=4 + MTP4 + ReplaySSM | 4.06 | 548 | 2.6 | 602 | 645 | 79.7 | 101.0 | **4.17** | 4.07 | 2.99 |
| DP=4 基线 | 3.63 | 489 | 10.7 | 3 122 | 2 944 | 159.8 | 161.7 | **20.73** | 20.79 | — |
| DP=4 + MTP4 | 3.94 | 531 | 5.2 | 1 347 | 1 318 | 174.1 | 168.7 | **8.57** | 8.59 | 3.05 |

### 2.2 rate 16（中并发，n=200）

| 配置 | req/s | out tok/s | 并发 | E2E50 | E2E 均值 | TTFT50 | TTFT 均值 | TPOT | ITL | accept |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线** | 13.15 | 1 860 | 53.9 | 3 995 | 4 102 | 96.2 | 110.3 | **31.77** | 28.41 | — |
| TP=4 + MTP4（cap48） | 12.52 | 1 772 | 44.3 | 3 555 | 3 535 | 186.3 | 769.5 | **21.96** | 19.68 | 3.04 |
| TP=4 + MTP4（cap186） | 12.78 | 1 808 | 55.3 | 4 065 | 4 328 | 131.0 | 155.9 | **35.22** | 29.69 | 3.04 |
| TP=4 + MTP8（cap128） | 10.88 | 1 540 | 69.1 | 6 395 | 6 346 | 227.8 | 289.4 | **52.00** | 43.11 | 3.98 |
| TP=4 + MTP4 + ReplaySSM（322） | 12.15 | 1 719 | 61.0 | 4 834 | 5 018 | 145.8 | 221.5 | **42.28** | 34.14 | 3.04 |
| DP=4 基线 | 11.69 | 1 654 | 60.4 | 5 257 | 5 163 | 175.2 | 178.1 | **36.79** | 35.48 | — |
| DP=4 + MTP4（64） | 12.94 | 1 831 | 38.5 | 2 915 | 2 978 | 229.8 | 315.5 | **19.37** | 18.95 | 2.93 |

### 2.3 rate 64（饱和 / 高并发，n=400）

| 配置 | req/s | out tok/s | 并发 | Peak out/s | E2E50 | TTFT50 | TTFT 均值 | TPOT | ITL | accept |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线** | **21.42** | **2 805** | 250.8 | 6 167 | 12 046 | 246.7 | 1 448 | 114.76 | 78.96 | — |
| TP=4 + MTP4（cap48） | 15.38 | 2 013 | 157.6 | 3 080 | 9 577 | 6 505 | 7 385 | 22.98 | 22.04 | 3.06 |
| TP=4 + MTP4（cap186） | 18.25 | 2 389 | 210.1 | 4 582 | 13 304 | 1 127 | 3 286 | 71.30 | 63.34 | 3.05 |
| TP=4 + MTP4 + ReplaySSM（322） | 19.12 | 2 504 | 247.1 | 5 259 | 13 921 | 458 | 1 501 | 119.04 | 87.91 | 3.05 |
| TP=4 + MTP8（cap128） | 14.18 | 1 857 | 196.8 | 3 219 | 14 632 | 5 548 | 6 288 | 64.03 | 58.41 | 4.02 |
| DP=4 基线 | 16.65 | 2 180 | 189.1 | 4 266 | 12 175 | 4 437 | 5 109 | 53.25 | 48.10 | — |
| DP=4 + MTP4（64/rank） | 16.83 | 2 203 | 167.4 | 2 964 | 9 806 | 6 621 | 6 826 | 24.90 | 24.02 | 3.06 |

> rate 64 的 TTFT 是"排队时间"而非首 token 算力（例如 TP=4 基线 TTFT50 246 ms；
> MTP4 cap48 因并发被压到 48，队列排到 6.5 s）。看排队公平性请用 cap 对齐后的行。

## 3. 延迟收益（MTP 加速比）

| 工况 | 配置 | TPOT | vs 同形态基线 | E2E50 | vs 基线 |
|---|---|---|---|---|---|
| rate 4 | TP=4 基线 → MTP4 | 9.40 → **4.18 ms** | **2.25×** | 1 403 → 605 ms | **2.32×** |
| rate 4 | TP=4 基线 → MTP8 | 9.40 → 4.33 ms | 2.17× | 1 403 → 539 ms | 2.60× |
| rate 4 | DP=4 基线 → MTP4 | 20.73 → **8.57 ms** | **2.42×** | 3 122 → 1 347 ms | 2.32× |
| rate 16 | TP=4 基线 → MTP4(cap48) | 31.77 → 21.96 ms | 1.45× | 3 995 → 3 555 ms | 1.12× |
| rate 16 | DP=4 基线 → MTP4 | 36.79 → 19.37 ms | **1.90×** | 5 257 → 2 915 ms | **1.80×** |
| rate 64 | TP=4 基线 → MTP4(cap186) | 114.76 → 71.30 ms | 1.61× | 12 046 → 13 304 ms | 0.91×（更差） |
| rate 64 | DP=4 基线 → MTP4 | 53.25 → **24.90 ms** | **2.14×** | 12 175 → 9 806 ms | **1.24×** |

规律（与 9B 报告一致，但 27B 更陡）：

- **decode 带宽/算力越紧张、并发越低，MTP 收益越大**：DP=4（每卡读整份 53 GB）
  > TP=4（每卡读 13 GB）。
- **饱和后 MTP 变负收益**：高并发下计算成为瓶颈，每个目标 forward 要 verify
  `num_draft_tokens` 个 token，成本按 verify 宽度线性上涨，而 accept 只有 3.0–3.9 个
  → 净收益 ≈ 1.0×，同时 verify 中间态把并发上限从 343 砍到 186/128。
- **TTFT 轻微上升**（rate 4：90 → 98 ms，+9%），量级可忽略。

## 4. MTP=4 vs MTP=8：为什么 8 不值

| 指标 | MTP4 (3/1/4) | MTP8 (7/1/8) |
|---|---|---|
| accept length（rate 4/16/64） | 3.01 / 3.04 / 3.05 | **3.90 / 3.98 / 4.01** |
| rate 4 TPOT | **4.18 ms** | 4.33 ms |
| rate 4 E2E50 | 605 ms | **539 ms** |
| 每步耗时（=TPOT×accept） | 12.6 ms | 16.9 ms |
| 有效上限 | 186 | 128 |
| rate 64 req/s | **18.25** | 14.18 |
| verify 中间态 | 13.15 GB | **18.14 GB** |

draft 从 4 加到 8 只把 accept 从 3.0 提到 3.9（**边际接受率 22%**：draft 5→8
几乎全被拒），而每步 verify 宽度翻倍、中间态 +38% → 每 token 延迟持平、并发与饱和
吞吐都变差。**结论：该 MTP 头的甜点在 3–4 个 draft token，不要上 8。**

## 5. 高并发下的"两全"选项

| 方案 | 饱和吞吐 | TPOT@rate64 | 上限 | 代价 |
|---|---|---|---|---|
| TP=4 基线（无 MTP） | **21.42 req/s** | 114.76 ms | 343 | 延迟最差 |
| TP=4 + MTP4 | 18.25 | 71.30 | 186 | 吞吐 -15% |
| **TP=4 + MTP4 + `--enable-linear-replayssm-spec`** | 19.12 | 119.04（≈基线） | **322** | 并发回来了，但高并发延迟收益归零 |
| **DP=4 + MTP4** | 16.83 | **24.90** | 64 | 绝对吞吐低（decode 读全量权重） |

- ReplaySSM 把 verify 中间态放到固定 ring（cookbook 记作 `D = 0`），
  mamba 池 931 → 1614 slot、上限 186 → 322、饱和吞吐 +5%，但**高并发下 TPOT
  回落到基线水平**（119 vs 115 ms）——ring 写/折回的开销在 compute-bound 区吃掉了
  spec 收益。低并发（rate 4）仍保持 4.17 ms，与普通 MTP4 相同。
- **DP=4 + MTP4 是唯一"高并发不掉吞吐还降延迟"的组合**：16.83 vs 16.65 req/s
  （+1%），TPOT 24.90 vs 53.25 ms（**2.1×**）。前提是能接受每 rank 仅 16 并发、
  KV 池 19.5 万 token/rank（短上下文场景）。

## 6. 部署建议（Qwen3.8-27B / 4×H20，512/256 短工况）

| 场景 | 推荐 | 依据 |
|---|---|---|
| 高吞吐 serving / 批任务 | **TP=4 基线**（不开 MTP） | 21.42 req/s、2 805 out tok/s、343 并发 |
| 低延迟对话 / 小并发（≤10 并发） | **TP=4 + MTP4** | TPOT 4.18 ms、E2E50 605 ms（2.25×） |
| 中并发且重延迟（~40–60 并发） | **DP=4 + MTP4** | rate 16：19.37 ms TPOT（1.90×），吞吐还略升 |
| 高并发又要低延迟 | **DP=4 + MTP4**，或 TP=4+MTP4+RSS 后用 rate 控制 | 前者吞吐持平+2.1× 延迟；后者并发 322 |
| 长上下文（>32k / agent） | **TP=4**（MTP 亦可，但并发折半） | DP=4 的 KV 池只有 25 万 token，装不下长会话 |
| **不要** | MTP8、以及不带 `--max-running-requests` 的 MTP 吞吐对比 | accept 边际低；spec 默认把 cap 重置为 48 |

调优先手（若要继续压榨并发）：`--mamba-radix-cache-strategy extra_buffer_lazy`
（S=5→4，上限 +25%）、`--mamba-full-memory-ratio`（cookbook 的 ratio 计算器）、
`--kv-cache-dtype fp8_e4m3`（KV 池翻倍，当前 KV 不是瓶颈故收益小）。

## 7. 数据有效性

- 8 次启动全部 `Load weight begin. avail mem` = 93.54–94.60 GB（GPU 干净，符合
  项目判据）；每次运行前后 `nvidia-smi` 4 卡 0 MiB。
- 8 个 server 日志 **0 个 Traceback / 0 个 CUDA out of memory / 无进程死亡**；
  7×3=21 组基准全部 100% 成功（100/200/400 请求全成功）。
- 所有配置生成 token 数逐速率完全一致（13 484 / 28 299 / 52 367），
  MTP 与基线输出一致（投机解码 lossless），可比。
- 已知混淆变量已隔离：MTP 版默认 cap 48，另跑 cap 对齐行；DP=4 与 TP=4 的
  cap 不同（148 vs 343），比较时以"同 cap/同形态"为准。
- 日志噪声（忽略）：torchcodec libnvrtc、sarashina2_vision import error、
  NCCL destroy_process_group 未调用。

## 8. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `run_qwen38_27b_h20.sh` | 复现脚本（6 个模式，`CAP`/`RATES`/`SAT` 可覆盖） |
| `server_h20_qwen38_27b_serving_tp4.log` | TP=4 基线 server |
| `bench_serving_h20_qwen38_27b_tp4_rate{4,16,64}.log` | TP=4 基线三档 |
| `server_h20_qwen38_27b_serving_tp4_mtp4.log` + `..._tp4_mtp4_rate{4,16,64}.log` | MTP4，默认 cap 48 |
| `server_h20_qwen38_27b_serving_tp4_mtp4_cap343.log` + `..._tp4_mtp4_cap343_rate{4,16,64}.log` | MTP4，cap 343→有效 186 |
| `server_h20_qwen38_27b_serving_tp4_mtp8.log` + `..._tp4_mtp8_rate{4,16,64}.log` | MTP8，默认 cap 48 |
| `server_h20_qwen38_27b_serving_tp4_mtp8_cap343.log` + `..._tp4_mtp8_cap343_rate{4,16,64}.log` | MTP8，cap 343→有效 128 |
| `server_h20_qwen38_27b_serving_tp4_mtp4_rss_cap343.log` + `..._tp4_mtp4_rss_cap343_rate{4,16,64}.log` | MTP4 + `--enable-linear-replayssm-spec`，有效 322 |
| `server_h20_qwen38_27b_serving_dp4.log` + `..._dp4_rate{4,16,64}.log` | DP=4 基线（37/rank） |
| `server_h20_qwen38_27b_serving_dp4_mtp4.log` + `..._dp4_mtp4_rate{4,16,64}.log` | DP=4 + MTP4（16/rank） |
| `runner_h20_qwen38_27b_*.log` | 各模式的 runner/时间线日志 |
