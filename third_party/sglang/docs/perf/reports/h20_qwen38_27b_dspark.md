# Qwen3.8-27B H20 4×96GB：DSpark 外部 draft checkpoint（Qwen3.8-27B-DSpark）实测

日期：2026-09-16。硬件：4× NVIDIA H20 96GB（sm90，HBM3e ~4.0 TB/s，bf16 dense
峰值 ~148 TFLOPS/卡），K8s dev-pod。目标模型：`/workspace/models/Qwen3.8-27B`
（BF16，18 分片 ~52.8 GB）。Draft：`/workspace/models/Qwen3.8-27B-DSpark`
（`DSparkDraftModel`，bf16 3.54 GB，5 层全注意力 + VanillaMarkov rank 256 +
confidence head，`block_size: 7`，target aux layers 5/19/33/47/61，1.86B 参数）。
运行时：sglang `0.5.19.dev39+g91a20e1`（包元数据，editable 安装时的 commit；当前工作树
=`v0.5.18-78-g15911f4-dirty`，其中 -dirty 是他人未提交的 decoupled drafter 改动，
与本报告走的 DSPARK 路径无关）、torch 2.13.0+cu130、
attention=fa3（target）/ flashinfer（draft）、linear-attn=triton、H20 **不带**
`--cuda-graph-backend-prefill=disabled`。

上游模型卡给的是 **GB300（DP4×TP1，NVFP4 target，accept 3.43）/ H200（FP8 target）**
的数字；本次是**同一份 draft 对着 BF16 target 跑在 4×H20**。目的：把
`docs/perf/reports/h20_qwen38_27b_mtp_comparison.md`（TP=4 基线 + 自带 MTP4/MTP8）的矩阵
补上"外部 DSpark draft"这一列。

复现脚本：`docs/perf/scripts/run_qwen38_27b_h20.sh` 的 `tp4_dspark` 模式
（`DRAFT_MODEL=` / `DSPARK_GAMMA=` / `CAP=` / `TAGSUF=` 可覆盖）。

## 0. 结论速览

1. **能跑、且逐 token 无损**：DSPARK + 外部 draft 在本仓库直接起得来（无需改码、
   无需 SPS 表、无需 compact ragged-verify）；6 次启动三档速率共 18 组基准全部
   100% 成功，生成 token 总数与基线/MTP **逐档完全一致**
   （13 484 / 28 299 / 52 367）→ 投机解码 lossless。
2. **draft 本身是好的**：稳态 accept length **3.36 / 3.42 / 3.44**（rate 4/16/64），
   比自带 MTP4 的 3.01/3.04/3.06 高 **+11~12%**，与上游模型卡在 GB300 上给的
   聚合值 3.4286 基本一致 → draft 训练/接线没问题。
3. **但在 H20 上仍打不过自带 MTP**：原生 γ7（verify 窗口 8）下
   饱和吞吐 **12.98 req/s**，仅为 TP=4 基线（21.92）的 **59%**，也低于 MTP4
   cap48（15.38）与 MTP4 cap186（18.25）；低并发 TPOT **5.06 ms** vs MTP4
   4.18 ms（慢 21%），对基线的延迟收益 1.81×（MTP4 是 2.19×）。
4. **瓶颈不在接受率，在每步成本**：verify 窗口 8 = MTP4 的 2 倍，而 H20 是
   **算力稀缺**平台（bf16 峰值仅 H800 的 ~1/6.7）；同 bs≈48 下每步
   89.6 ms（γ7）vs 70.3 ms（MTP4），每步只多产出 12% 的 token → 净负收益。
   此外外部 draft 每步要多做一次 1.86B 前向（低并发时约 +2.4~3.4 ms/步）。
5. **γ 有权重甜点：窗口越窄越划算**。把 `--speculative-dspark-block-size` 从
   checkpoint 的 7 压到 **3（窗口 4，与 MTP4 同宽）**：饱和吞吐 12.98 → **16.26**
   req/s（+25%），rate 16 12.48 ≈ MTP4 cap48 的 12.52，E2E50 3 546 ms 优于
   MTP4 的 3 555 ms；代价是低并发变差（accept 3.36 → 2.65，rate 4 E2E50
   627 → 739 ms）。γ5（窗口 6）居中：14.15 req/s。
6. **提高 cap 对 γ7 无用**：48 → 128 吞吐 12.98 → 13.11（+1%），但 TPOT
   26.05 → 70.00 ms、E2E50 11 545 → 15 916 ms（verify 已算力饱和，加并发只加
   排队）。对 γ3 有效但有限：cap 48 → 186 饱和吞吐 16.26 → **17.49**（+7.6%），
   仍低于 MTP4 cap186 的 18.25。
7. **并发/显存代价符合 cookbook 的 `D = window` 模型**：verify 中间态按窗口吃
   mamba 池 —— `max_mamba_cache_size` 644（γ7）/ 762（γ5）/ **932（γ3，≈MTP4 的
   931）**，中间态 6.89 / 5.17 / **3.45 GB（γ3 = MTP4 的 3.45）**。γ7 下把 cap
   开到 128 时中间态 6.89 → 18.14 GB，KV 池 230 万 → 171 万 token。
8. **一句话**：4×H20 上 Qwen3.8-27B 的投机解码**首选仍是自带 MTP4**；要用
   DSpark draft，就**必须把 verify 窗口压到 4（γ3）**，否则是在算力瓶颈上做亏本
   买卖。γ3 + cap186 可作为"中并发折中"（17.49 req/s、TPOT 71.56@bs206）。

## 1. 启动命令

基线 / MTP / DP=4 的启动命令见 `h20_qwen38_27b_mtp_comparison.md` §1。本次新增：

```bash
.venv/bin/python3 -m sglang.launch_server \
    --model-path /workspace/models/Qwen3.8-27B \
    --host 0.0.0.0 --port 30000 \
    --tp-size 4 --mem-fraction-static 0.85 --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /workspace/models/Qwen3.8-27B-DSpark \
    --speculative-draft-attention-backend flashinfer \
    --speculative-dspark-block-size 7          # γ=窗口-1；省略则从 draft config 推 7
    # 变体：--speculative-dspark-block-size 3  → verify 窗口 4
    # 变体：--max-running-requests 128|186     → spec hook 默认会把 cap 重置为 48
```

要点：

- `SGLANG_RAGGED_VERIFY_MODE` **保持默认 `static`**（上游模型卡也用 static）；
  这种模式下 `--speculative-dspark-sps-table-path` 是 no-op，本次未使用 SPS/STS
  校准表（`dspark_sps_profiler` / `dspark_sts_fit`），即"无预算调度"的纯 verify 形态。
- `--speculative-num-draft-tokens` 不用给：hook 会按 `gamma + 1` 强制解出
  （γ7→8、γ5→6、γ3→4），给错会直接报错。
- 启动日志确认 draft 接线正确（每个变体都打这行）：
  `Initialized DSpark draft runner. attention_backend=flashinfer, model=DSparkDraftModel,
  gamma=7, verify_num_draft_tokens=8, mask_token_id=248070, markov_head=VanillaMarkov`
- DSPARK 的 MTP 头 **不参与**（draft 是独立 checkpoint），`--speculative-algorithm
  EAGLE` 那套（steps/topk/draft-tokens）不适用。

### 1.1 官方 cookbook recipe 对照（H200 / BF16 / single node / DSPARK / high-throughput / float32）

上游页面 `docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B`（本仓库对应
`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` + `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`）
在该选择下生成的命令（H200 是 141 GB 单卡，TP=1，不带任何并行参数）：

```bash
sglang serve --trust-remote-code --model-path Qwen/Qwen3.8-27B \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --attention-backend flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-draft-attention-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype float32 \
  --mamba-full-memory-ratio 11.01 \
  --host 0.0.0.0 --port 30000
```

- DSPARK 只有 **3 个 flag**（算法 + draft 路径 + draft attention 后端），γ 自动推 7；
  **不含** SPS/STS/confidence 相关参数（即 verify-all，零 confidence 裁切）。
- `--mamba-full-memory-ratio 11.01` 由页面计算器给出：`r = (S + D) × state_bytes /
  (L × kv_bytes_per_token)` = `(4 + 8) × 153.9 MB / (32.8 KB × 5120)`（S=4 来自
  high-throughput 档的 `extra_buffer_lazy`，D=γ+1=8，L 取计算器默认 5120
  input+output）；等价 pin 是 `--max-mamba-cache-size 256`（默认目标并发 64 × S=4）。
- 页面自己也强调：`--mamba-full-memory-ratio` 是 hybrid GDN 模型唯一影响 throughput
  的 sizing flag，默认 0.9 会把 KV 池供过头、静默把并发卡住。
- 与本报告 H20 实测的差异：H20 只能 TP=4（96 GB 装不下整份 bf16），且本次用 bf16 KV、
  bf16 SSM、默认 `extra_buffer`（S=5）、默认 ratio 0.9、chunked-prefill 8192、fa3；
  因此上面的 cap 数字是「默认 sizing」下的（按本工况 L≈643 重算，balanced ratio
  约 24，可把并发上限抬上去）。换官方 sizing 重跑是后续可选项。

各配置的池子与并发上限（TP=4，四卡数值一致，取 TP0）：

| 配置 | KV 池 (token) | KV (K+V) | mamba slot | ssm_state | **verify 中间态** | **cap** |
|---|---|---|---|---|---|---|
| **TP=4 基线** | 2 288 771 | 17.46+17.46 GB | 1718 | 30.22 GB | — | **343**（mamba 上限） |
| TP=4 + MTP4 | 2 729 719 | — | 931 | 16.38 GB | 3.45 GB | 48（spec 默认）→ 186 |
| TP=4 + MTP4（cap343） | 2 107 842 | 16.08+16.08 GB | 931 | 16.38 GB | 13.15 GB | 186 |
| **DSPARK γ7** | 2 298 769 | 17.54+17.54 GB | 644 | 11.34 GB | 6.89 GB | 48（spec 默认） |
| DSPARK γ7（cap128） | 1 712 693 | 13.07+13.07 GB | 644 | 11.34 GB | **18.14 GB** | 128 |
| DSPARK γ5 | 2 278 133 | 17.38+17.38 GB | 762 | 13.41 GB | 5.17 GB | 48 |
| **DSPARK γ3** | 2 212 469 | 16.88+16.88 GB | **932** | 16.40 GB | **3.45 GB** | 48 |
| DSPARK γ3（cap186） | 1 709 045 | 13.04+13.04 GB | 932 | 16.40 GB | 13.15 GB | 186 |

- 每次启动 `Load weight begin. avail mem` = 93.22–93.77 GB（GPU 干净，符合项目判据），
  draft 加载行为 `Load weight begin. avail mem=80.5 GB`（target 装完后）。
- γ7 的 cap 由 spec hook 重置为 48，但 **mamba 池（644 slot / 5）其实允许 128**，
  故另跑了 cap128 行；γ7 的 mamba 池比 MTP4 小，正是 verify 中间态从 3.45 GB
  涨到 6.89 GB 挤出来的（D=8）。

## 2. 基准矩阵（512 in / 256 out，random dataset；rate 4=n100、16=n200、64=n400）

> "本次"= 2026-09-16 同会话复跑；"归档"= 09-15 数据。基线两轮逐档相差 <2.3%
> （21.92 vs 21.42 req/s，rate 4/16 几乎完全一致）→ **归档的 MTP 数字可直接横比**。
> `每步 (ms) = TPOT × accept`，**只在同 bs（运行批大小）下可比**：cap48 组 bs≈48，
> cap186 组 bs≈186，故 cap 不同的行不要拿这一列互比。

### 2.1 rate 4（低并发 / 延迟工况，n=100）

| 配置 | req/s | out tok/s | 并发 | E2E50 (ms) | TTFT50 | **TPOT** | ITL | Max ITL | accept | 每步 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线**（本次） | 3.94 | 531 | 5.1 | 1 389 | 73 | **9.17** | 9.15 | 162 | — | — |
| TP=4 基线（归档） | 3.94 | 531 | 5.2 | 1 403 | 74 | **9.40** | 9.24 | 151 | — | — |
| TP=4 + MTP4 (3/1/4) cap48 | 4.06 | 548 | 2.6 | 605 | 82 | **4.18** | 4.09 | 159 | 3.01 | 12.6 |
| TP=4 + MTP4 cap343→186 | 4.06 | 548 | 2.6 | 614 | 80 | **4.29** | 4.07 | 154 | 2.99 | 12.8 |
| TP=4 + MTP8 (7/1/8) cap343→128 | 4.08 | 550 | 2.7 | 560 | 81 | **4.47** | 4.21 | 262 | 3.90 | 17.4 |
| **DSPARK γ7 (win8) cap48** | 4.08 | 550 | 3.0 | 627 | 83 | **5.06** | 4.71 | 206 | 3.36 | 17.0 |
| DSPARK γ7 cap128 | 4.07 | 549 | 3.0 | 600 | 85 | **5.20** | 4.79 | 353 | 3.34 | 17.4 |
| DSPARK γ5 (win6) cap48 | 4.07 | 549 | 3.5 | 701 | 87 | **6.13** ※ | 5.50 | 515 | 3.02 | 18.5 |
| **DSPARK γ3 (win4) cap48** | 4.05 | 546 | 3.2 | 739 | 84 | **5.63** | 5.22 | 261 | 2.65 | 14.9 |
| DSPARK γ3 cap186 | 4.05 | 547 | 3.0 | 675 | 80 | **5.19** | 4.87 | 255 | 2.67 | 13.9 |

※ γ5 的 rate4 单发比 γ3/γ7 都差（TPOT 6.13、Max ITL 515 ms），与 γ 单调性不符，
按项目惯例标 ※：单发噪声，未复核（低并发 3 档配置各只跑 1 次）。

### 2.2 rate 16（中并发，n=200）

| 配置 | req/s | out tok/s | 并发 | E2E50 | TTFT50 | **TPOT** | ITL | Max ITL | accept | 每步 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线**（本次） | 13.17 | 1 864 | 54.8 | 4 170 | 103 | **32.29** | 28.77 | 490 | — | — |
| TP=4 基线（归档） | 13.15 | 1 860 | 53.9 | 3 995 | 96 | **31.77** | 28.41 | 593 | — | — |
| TP=4 + MTP4 cap48 | 12.52 | 1 772 | 44.3 | 3 555 | 186 | **21.96** | 19.68 | 344 | 3.04 | 66.8 |
| TP=4 + MTP4 cap343→186 | 12.78 | 1 808 | 55.3 | 4 065 | 131 | **35.22** | 29.69 | 738 | 3.04 | 107.1 |
| TP=4 + MTP8 cap343→128 | 10.88 | 1 540 | 69.1 | 6 395 | 228 | **52.00** | 43.11 | 1 629 | 3.98 | 207.0 |
| **DSPARK γ7 cap48** | 10.65 | 1 507 | 54.3 | 5 134 | 1 016 | **26.96** | 24.88 | 296 | 3.42 | 92.2 |
| DSPARK γ7 cap128 | 10.67 | 1 510 | 70.6 | 6 852 | 183 | **48.73** | 45.02 | 892 | 3.41 | 166.2 |
| DSPARK γ5 cap48 | 11.48 | 1 624 | 49.8 | 4 362 | 451 | **24.31** | 22.70 | 271 | 3.14 | 76.3 |
| **DSPARK γ3 cap48** | **12.48** | 1 766 | 44.6 | **3 546** | 168 | **20.93** | 19.91 | 291 | 2.70 | 56.5 |
| DSPARK γ3 cap186 | 12.37 | 1 751 | 57.2 | 4 513 | 125 | **37.56** | 31.74 | 652 | 2.70 | 101.4 |

### 2.3 rate 64（饱和 / 高并发，n=400）

| 配置 | req/s | out tok/s | Peak out/s | 并发 | E2E50 | TTFT50 | **TPOT** | ITL | accept | 每步 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **TP=4 基线**（本次） | **21.92** | **2 870** | 6 523 | 246.9 | 11 510 | 324 | **113.10** | 78.00 | — | — |
| TP=4 基线（归档） | 21.42 | 2 805 | 6 167 | 250.8 | 12 046 | 247 | **114.76** | 78.96 | — | — |
| TP=4 + MTP4 cap48 | 15.38 | 2 013 | 3 080 | 157.6 | 9 577 | 6 505 | **22.98** | 22.04 | 3.06 | 70.3 |
| TP=4 + MTP4 cap343→186 | 18.25 | 2 389 | 4 582 | 210.1 | 13 304 | 1 127 | **71.30** | 63.34 | 3.05 | 217.5 |
| TP=4 + MTP8 cap343→128 | 14.18 | 1 857 | 3 219 | 196.8 | 14 632 | 5 548 | **64.03** | 58.41 | 4.02 | 257.4 |
| **DSPARK γ7 cap48** | 12.98 | 1 700 | 2 412 | 160.7 | 11 545 | 8 246 | **26.05** | 24.61 | 3.44 | 89.6 |
| DSPARK γ7 cap128 | 13.11 | 1 716 | 2 475 | 197.5 | 15 916 | 6 035 | **70.00** | 62.81 | 3.43 | 240.1 |
| DSPARK γ5 cap48 | 14.15 | 1 852 | 2 729 | 156.0 | 10 189 | 6 767 | **23.86** | 22.80 | 3.15 | 75.2 |
| **DSPARK γ3 cap48** | **16.26** | **2 128** | 3 278 | 146.9 | 7 964 | 5 201 | **20.43** | 19.85 | 2.70 | 55.2 |
| **DSPARK γ3 cap186** | **17.49** | **2 289** | 4 110 | 206.0 | 13 367 | 661 | **71.56** | 64.85 | 2.71 | 193.9 |

> rate 64 的 TTFT 是"排队时间"：cap48 的 DSPARK 启动队列排到 5–8 s
> （server 日志 `#queue-req` 峰值 244–273，cap186 为 163），只看同 cap 行。

## 3. 延迟收益（vs 同形态基线）

| 工况 | 配置 | TPOT | vs 基线 | E2E50 | vs 基线 |
|---|---|---|---|---|---|
| rate 4 | 基线 → MTP4 | 9.17 → **4.18 ms** | **2.19×** | 1 389 → 605 ms | **2.30×** |
| rate 4 | 基线 → DSPARK γ7 | 9.17 → **5.06 ms** | **1.81×** | 1 389 → 627 ms | **2.22×** |
| rate 4 | 基线 → DSPARK γ3 | 9.17 → 5.63 ms | 1.63× | 1 389 → 739 ms | 1.88× |
| rate 16 | 基线 → MTP4(cap48) | 32.29 → 21.96 ms | 1.47× | 4 170 → 3 555 ms | 1.17× |
| rate 16 | 基线 → DSPARK γ7(cap48) | 32.29 → 26.96 ms | 1.20× | 4 170 → 5 134 ms | 0.81×（更差） |
| rate 16 | 基线 → DSPARK γ3(cap48) | 32.29 → **20.93 ms** | **1.54×** | 4 170 → **3 546 ms** | **1.18×** |
| rate 64 | 基线 → MTP4(cap186) | 113.10 → 71.30 ms | 1.59× | 11 510 → 13 304 ms | 0.87×（更差） |
| rate 64 | 基线 → DSPARK γ3(cap186) | 113.10 → 71.56 ms | 1.58× | 11 510 → 13 367 ms | 0.86×（更差） |
| rate 64 | 基线 → DSPARK γ7(cap48) | 113.10 → **26.05 ms** | **4.34×** ※ | 11 510 → 11 545 ms | 1.00× |

※ 该行的"加速比"不可当收益看：cap48 把并发压到 48，TPOT 是在小批下测的，
吞吐同时掉到 12.98 req/s（基线的 59%）。**投机解码的延迟/吞吐权衡要同 cap 同看。**

## 4. 为什么 H20 上 DSPARK 打不过自带 MTP

三条实测证据：

1. **接受率不是问题，窗口成本才是**。accept length：DSPARK 3.36/3.42/3.44
   > MTP4 3.01/3.04/3.06（+11~12%）。但同 bs≈48（cap48 行）每步耗时：
   **γ7 89.6 ms、γ5 75.2 ms、γ3 55.2 ms、MTP4 70.3 ms**。
   γ7 每步多花 27% 的时间，只换来 12% 的 token → 饱和吞吐 12.98 vs 15.38。
2. **窗口宽度直接定价**：γ7→γ5→γ3（窗口 8→6→4）每步 89.6→75.2→55.2 ms，
   近似线性；每步产出 3.44→3.15→2.70 token。**效率（token/ms）= 0.0384 / 0.0419 /
   0.0489**，即窗口越窄越划算；MTP4 是 0.0435。这条在 H20（算力/带宽比最差的
   平台之一）上是决定性的，在 H200/GB300（BF16/FP8 算力高 6~10 倍）上则不然。
3. **外部 draft 的固定开销**：低并发（bs≈3）每步 γ7 17.0 ms、γ3 14.9 ms、
   MTP4 12.8 ms、基线 9.2 ms → 外部 draft 每次前向约 +2.4~3.4 ms/步（5 层
   full-attention + 248k vocab 的 lm_head/采样），MTP 头只占 ~3.4 ms/步（1 层，
   但要跑 3 个自回归 step）。这解释了低并发下 DSPARK 反而比 MTP4 慢。

另外两条工程侧观察：

- **spec hook 把 cap 重置为 48**（与 MTP 一样），做吞吐对比必须显式
  `--max-running-requests`；但 γ7 即使放到 128 也不涨吞吐（第 5 节）。
- **中间态占用随窗口线性**：γ3 的 mamba 池 932 slot / 中间态 3.45 GB 与 MTP4
  完全一致（cookbook 的 `D = window`），所以"γ3 就是 MTP4 宽度"的直觉成立。

## 5. γ（verify 窗口）与 cap 扫描

| γ (窗口) | accept (rate4/16/64) | 饱和吞吐 cap48 | 饱和吞吐 高 cap | 低并发 TPOT | 中间态 @cap48 | mamba slot |
|---|---|---|---|---|---|---|
| 7 (8) | 3.36 / 3.42 / 3.44 | 12.98 | 13.11 @128（+1%） | **5.06 ms** | 6.89 GB | 644 |
| 5 (6) | 3.02 / 3.14 / 3.15 | 14.15 | —— | 6.13 ms ※ | 5.17 GB | 762 |
| **3 (4)** | 2.65 / 2.70 / 2.70 | **16.26** | **17.49 @186（+7.6%）** | 5.63 ms | **3.45 GB** | **932** |

- **checkpoint 原生 γ7 = H20 上最差的选择**（吞吐 59% 基线）；
  窄窗口 γ3 用 -21% 的 accept 换来 +25% 的吞吐和 +45% 的 mamba 池（644→932 slot）。
  按项目口径这属于**配置级调优，不是改模型权重**：γ 只是 draft 一次产出多少
  个候选位置，输出仍与基线逐 token 一致（token 总数三档全等 → lossless 未被破坏）。
- γ3 的低并发（rate 4）比 γ7 差（E2E50 739 vs 627 ms），因为那里是**带宽区**
  （bs≈3，每步要读整份 target 权重），accept 才是主导；γ7 的宽窗口在那时几乎免费。
- **γ 的最优值随并发移动**：低并发选大窗口（γ7 甚至 γ7+），高并发/算力紧张选
  小窗口（γ3）。若是生产部署，应按目标并发选 γ，而不是照抄 checkpoint 的 7。
- cap：γ7 加 cap 无收益（验证了"verify 已饱和"）；γ3 加 cap 有 +7.6%，
  但仍追不上 MTP4 cap186（18.25）。**DSPARK 在这台机器上不存在"高并发不掉
  吞吐"的组合**（对照 MTP4+ReplaySSM/DP4+MTP4 都有）。

## 6. 部署建议（Qwen3.8-27B / 4×H20 / 512-256 短工况）

| 场景 | 推荐 | 依据 |
|---|---|---|
| 高吞吐 serving | **TP=4 基线**（不开投机） | 21.92 req/s、2 870 out tok/s |
| 低延迟小并发（≤10） | **TP=4 + MTP4** | TPOT 4.18 ms（2.19×）、E2E50 605 ms |
| 中并发（~40–60）且要延迟 | **TP=4 + DSPARK γ3** 或 MTP4 | 12.48 req/s、TPOT 20.93 ms、E2E50 3 546 ms（略优于 MTP4 的 3 555） |
| 高并发又要低延迟 | DP=4+MTP4（见 MTP 报告） | DSPARK 无对应组合 |
| 想用 DSpark draft | **必须 `--speculative-dspark-block-size 3`**（+ 需要时 cap186） | γ7 只到基线吞吐的 59%；γ3 到 79%（cap186） |
| confidence 裁切（compact/cap-accept + SPS 表） | **暂不在本模型开** | compact 崩、cap-accept 吞吐 -21%，见 `h20_qwen38_27b_dspark_confidence.md` |
| **不要** | γ7 原生窗口做吞吐对比、以及不带 `--max-running-requests` 的对比 | 每步成本 2×；spec 默认把 cap 重置为 48 |

DSPARK 的价值在这台机器上没有被证明：**除非你的目标并发很小、且能接受
"比 MTP4 慢 20% 的 TPOT"来换"少一个额外 checkpoint 的接受率余量"**，否则
4×H20 上没有理由用外部 DSpark draft。上游模型卡的 3.16× 是 GB300/NVFP4 的
数字，不能外推到 H20 + BF16。

## 7. 数据有效性

- 6 次启动（`tp4_dspark` / `_cap128` / `_g5` / `_g3` / `_cap186_g3` / `tp4_v2`）
  全部 `Load weight begin. avail mem` = 93.22–93.77 GB（GPU 干净，符合项目判据）；
  每次运行前后 `nvidia-smi` 4 卡 0 MiB。
- 6 个 server 日志 **0 个 Traceback / 0 个 CUDA out of memory / 无进程死亡**；
  5 次 DSPARK 启动的 draft 初始化行都打印了正确的
  `gamma / verify_num_draft_tokens`（7/8、5/6、3/4）。
- 18 组基准（3 配置×3 速率 + cap 变体）全部 100% 成功
  （100/200/400 全成功）；生成 token 逐速率与基线/MTP 完全一致
  （13 484 / 28 299 / 52 367）→ 投机解码 lossless。
- 漂移控制：基线同会话复跑（21.92/13.17/3.94）vs 09-15 归档
  （21.42/13.15/3.94）相差 ≤2.3% → 归档的 MTP4/MTP8 数字可用于横比。
- 服务器侧交叉验证：`Decode batch ... accept len` 在 bs≥32 的稳态采样均值
  = γ7 3.428（n=13）/ γ5 3.163（n=13）/ γ3 2.711（n=15）/ MTP4 3.065（n=11），
  与客户端 `Accept length` 一致。
- 已知混淆变量：① cap 未对齐的行不能比吞吐（已分别列 cap48 / cap128 / cap186）；
  ② rate4 每组只跑 1 次，γ5 那格标 ※；③ γ3 是**覆盖 checkpoint 训练窗口**的
  配置，accept 下降即为其代价，不算数据问题。

## 8. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `run_qwen38_27b_h20.sh` | 复现脚本（新增 `tp4_dspark` 模式；`DSPARK_GAMMA`/`CAP`/`TAGSUF` 覆盖） |
| `extract_bench_metrics.py` | 从 `bench_serving_*.log` 提表的小工具（本次表格由此生成） |
| `server_h20_qwen38_27b_serving_tp4_dspark.log` + `bench_..._tp4_dspark_rate{4,16,64}.log` | DSPARK γ7，默认 cap48 |
| `server_..._tp4_dspark_cap128.log` + `bench_..._tp4_dspark_cap128_rate{4,16,64}.log` | DSPARK γ7，cap128 |
| `server_..._tp4_dspark_g5.log` + `bench_..._tp4_dspark_g5_rate{4,16,64}.log` | DSPARK γ5（窗口 6），cap48 |
| `server_..._tp4_dspark_g3.log` + `bench_..._tp4_dspark_g3_rate{4,16,64}.log` | DSPARK γ3（窗口 4），cap48 |
| `server_..._tp4_dspark_cap186_g3.log` + `bench_..._tp4_dspark_cap186_g3_rate{4,16,64}.log` | DSPARK γ3，cap186 |
| `server_..._tp4_v2.log` + `bench_..._tp4_v2_rate{4,16,64}.log` | TP=4 基线同会话复跑（漂移控制） |
| `runner_h20_qwen38_27b_tp4_dspark*.log`、`runner_h20_qwen38_27b_tp4_v2.log` | 各次 runner/时间线日志 |

复现（H20，约 9 分钟/配置）：

```bash
cd /workspace/projects/sglang-speedup-nvidia
bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark                       # γ7, cap48
DSPARK_GAMMA=3 bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark        # win4, cap48
DSPARK_GAMMA=3 CAP=186 TAGSUF=_g3 bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark
TAGSUF=_v2 bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4                   # 基线
.venv/bin/python3 docs/perf/scripts/extract_bench_metrics.py docs/perf/logs/bench_serving_h20_qwen38_27b_tp4_dspark*.log
```

## 9. 未决 / 可选下一步

- **DSpark 的 confidence 裁切（ragged-verify budget scheduler）已实测，另开一份报告**：
  `docs/perf/reports/h20_qwen38_27b_dspark_confidence.md`。结论：实现齐备（1D SPS 表已在 H20
  画像成功，`docs/perf/results/dspark_sps_g7.json`），但 `compact` 在 Qwen3.8-27B（VL/mrope）上
  必崩（`compute_spec_mrope_positions` 只接受均匀布局），`cap-accept` 虽能服务却
  吞吐 -21%；官方 cookbook recipe 也不开这套。
- `--enable-linear-replayssm-spec` 与 DSPARK/DFLASH + static 是允许的组合
  （`server_args.py` 里显式放行），能把 verify 中间态搬到固定 ring、拿回 mamba
  池；但本报告已证明 γ7 的瓶颈不是池子而是每步算力，故未测。若要用 γ7 跑长上下文
  并发，值得补测。
- rate4 低并发档建议补 2~3 次重复，确认 γ5 的 ※ 异常与 γ 单调性。
