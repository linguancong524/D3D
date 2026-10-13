# H20 部署 Qwen3.5-9B 容量估算(100 并发)

日期:2026-08-25 · 方法:基于本仓库 H800 实测基线(docs/perf/logs/bench_*_qwen35_9b_baseline_v*.log,
v0.5.18)按 H20 规格缩放 + 离散事件仿真(脚本 `h20_capacity_sim.py`,可复跑)。
非 H20 实测;上线前需在真卡上跑一轮 serving 基准验证(见 §6)。

## 1. 结论

**推荐 8 张 H20;6 张是满足体验 SLO 的最低值;4-5 张在基准工况下会出现明显可感知的
首 token 等待(2-2.5s)与 token 间隔尖峰(~0.4-0.9s),不建议。**

- 部署形态:TP=1,一卡一副本 × N,前置 LB(9B bf16 单卡装得下,TP 无收益反增开销)
- 8 卡 = 每卡 12-13 并发:TTFT ≈ 1.2-1.7s,ITL p99 ≤ 10ms,单用户 ≥ 150 tok/s,
  且对 prefill 模型不确定性和 ~1.5× 流量增长都有余量
- 6 卡 = 每卡 16-17 并发:TTFT ≈ 1.6-2.0s,ITL p99 ≤ ~100ms,单用户 130-150 tok/s
  —— 三个 prefill 假设场景下都压线通过,零余量

**修正(agent 任务,262k 上下文,详见 §8)**:长上下文下瓶颈从 prefill 算力变成 **KV 显存**。
262k 上下文每用户占 ~7.7 GiB(KV 7.45 + mamba 0.23),96GB 卡每卡仅容 ~8 个满上下文
并发:100 用户全满 262k → **14×H20-96G(0.85)或 8×H20-3E 141G**;若 262k 只是尾部
(多数会话 <64k),8×96G 仍够(显存利用率 ~57%)。

## 2. H20 vs H800(本仓库基线硬件)

| 项 | H800 | H20 | 影响 |
|---|---|---|---|
| HBM 带宽 | 3.35 TB/s(实测有效 ~73%) | 4.0 TB/s(源:[iotworld](https://www.iotworld.com.cn/html/News/202401/52da19aad1c61b42.shtml)) | decode 提速 ~19% |
| BF16 算力 | ~990 TFLOPS | 148 TFLOPS(=1/6.7) | prefill 大幅变慢(本 workload 的瓶颈) |
| 显存 | 80 GB | 96 GB | KV/mamba 池更大,100 并发显存远未打满(每卡 ~2-3 GB 状态) |
| NVLink | 400 GB/s | 900 GB/s | TP=1 方案用不到 |

## 3. 锚点(H800 实测,9B bf16,v0.5.18)

- 权重 17.62 GB;KV 30.52 KB/token;mamba 状态读取 ~115 MB/req(bs1→bs32 边际拟合)
- decode:bs1 7.17 ms/tok;bs32 8.66-10.05 ms/tok(ctx 128→4096)
- prefill(单发 extend,one_batch):bs8×1024 = 27.9k tok/s;bs8×4096 = 36.6k;bs32×4096(131k tok)= 34.0k tok/s(≈27-36 µs/tok)
- serving(512/256,rate 4):TTFT 中位 70.5 ms、TPOT 中位 8.33 ms —— 模型校验点
- 小 batch(单请求)prefill 低效:512 tok ≈ 55-60 ms(≈110 µs/tok)

## 4. H20 上的两个 regime

**Decode(带宽主导,非瓶颈)**:S(c,ctx) = max(显存读, 计算)。权重 17.62 GB +
c×(115 MB mamba + ctx×30.5 KB KV),有效带宽按 73% 取 2.92 TB/s;计算项按 148×35% MFU。
→ c=17: 6.9 ms/step(145 tok/s/人);c=25: 7.2 ms(138);c=34: 9.9 ms(101,计算开始压上)。
单卡带 17-25 个并发用户,单用户码率远高于"体验底线"(~20-30 tok/s)。

**Prefill(计算主导,瓶颈所在)**:H800 的 27 µs/tok 工作点需要 ~55-67% H800 算力;
H20 只有 1/6.7 的算力 → 批量 prefill ≈ 27 µs/tok × 990/148 ≈ 180-250 µs/tok
(4-5.5k tok/s/card,chunked_prefill 8192 工况)。100 并发一波 = 51,200 prefill tokens
≈ 8-13 card·s —— TTFT ≈ 该值 / N。单请求小 batch prefill 87-295 ms(内核低效项,
按绝对时间或算力比缩放的区间)。

## 5. 仿真结果(TTFT p50/p90/p99,ms;ITL p50/p90/p99/max,ms;单用户 tok/s)

工况:100 并发 ×(512 in/256 out),t=0 齐发 + 完成后立即补发(持续满载),
每卡独立 SGLang 调度模型(prefill chunk 8192 与 decode 混批)。

| N | S2 基准(5.1k tok/s, 87ms 小) | S3 悲观(4.0k, 295ms 小) | S5 乐观算力(6.7k) |
|---|---|---|---|
| 3 | TTFT 3.2/3.2/3.3s, ITL max 1.7s, min 44 tok/s | — | 2.5/2.5/2.6s, max 1.3s |
| 4 | 1.6/2.5/2.5s, max 0.9s, 93 | 2.0/3.4/3.4s, max 1.4s, 36-79 | 1.2/1.9/1.9s, max 0.7s, 35-100 |
| 5 | 1.6/2.0/2.0s, max 0.39s, 103-118 | 2.0/2.8/2.8s, max 0.8s | 1.2/1.6/1.6s, max 0.34s |
| **6** | **1.6/1.6/1.6s, max 0.10s, 134-148** | **2.0/2.0/2.0s, max 7ms, 148** | 1.2/1.2/1.2s, max 0.10s |
| **8** | **1.2/1.2/1.3s, max 7ms, 151-152** | **1.7/1.7/1.8s, max 7ms, 151-152** | 0.9/0.9/1.0s, max 7ms |
| 10 | 1.0/1.0/1.0s, max 6.5ms, 154 | 1.5/1.5/1.5s, 154 | 0.8/0.8/0.8s |

E2E p90 ≈ TTFT + 256×ITL:6 卡 ≈ 3.3-3.8s;8 卡 ≈ 2.6-3.4s。
经验式:基准工况下 ≈ 每卡 12-17 个并发用户;200 并发 → 12-16 卡(同上表外推)。

## 6. 主要不确定性与验证建议

1. **H20 prefill 每 token 成本是最大不确定项**(算力比缩放是估计,不是实测):
   若 78 SM 上 Triton GDN/FA 内核 MFU 偏低,S3 悲观档成立 → 结论不变(6 卡仍压线通过,
   8 卡更稳);若高于预期,4-5 卡可用,但不建议以乐观假设做生产容量。
2. 小 batch(单请求)prefill 的低效项在 H20 上如何缩放未知(87-295 ms 区间)。
3. 上下文长度:上表按 512 in。若真实输入 4k(8×),prefill 需求 8×,同 TTFT 下卡数
   约 2×(或接受 TTFT ~10s);输出更长则 decode 占比上升,对卡数影响较小(带宽富余)。
4. 建议验证(真卡,~2h):
   a. `launch_server` + `benchmark.serving` 512/256,n=100,max_concurrency=100 跑
      N=6 与 N=8,对照 §5 的 TTFT/ITL;
   b. 顺带测一次 4k in 工况标定 prefill 斜率;
   c. 命令注意:节点 e01-cn-c6k4blkz60c 驱动已由管理员升级(2026-08-26,
      nvidia-smi 确认 driver_version ≥580),启动命令可去掉
      `--cuda-graph-backend-prefill=disabled`,prefill 走 BCG 图捕获(若确认
      <580 则沿用该参数);跑前 nvidia-smi 清残留,日志 tee 进 docs/perf/logs/。

## 7. 能省卡的优化(按性价比)

- **FP8(W8A8)**:H20 FP8 296 TFLOPS,权重 8.8 GB → prefill ~2×、decode 更宽裕,
  同体验 8 卡 → 4-6 卡。前提:Qwen3.5-9B 有可用 FP8 checkpoint 且 GDN 线性注意力
  层的 FLA 内核支持 FP8(当前 v0.5.18 未见支持,需先验证)。
- **PD 分离**(v0.5.18 支持):prefill 节点吃算力、decode 节点吃带宽,尾延迟更平滑,
  总卡数相近但 SLO 达标率更高;100 并发规模下收益中等。
- **投机解码(dflash/dspark,仓库第 8 节工作项)**:decode 提速 1.5-2×,但本 workload
  decode 非瓶颈,对卡数贡献有限;主要价值在长上下文/长输出场景。
- 调度侧:chunked_prefill_size 从 8192 调小可把 ITL 尖峰从 ~1s 压到 ~0.4s,代价是
  TTFT 上升(仿真 S4),默认值已较优。

## 8. Agent 任务修正:262k 上下文(2026-08-25 补充)

用户场景:agent 多轮任务,上下文逐轮累积,需支撑满 262,144 token。
§1-§7 按 512/256 短工况计算;262k 下**瓶颈从 prefill 算力转移到 KV 显存**,重算如下。

### 8.1 瓶颈转移:KV 显存

KV 单价 = 30.52 KB/token(仓库实测 17.48 GiB / 572,747 tok);mamba 状态 O(1)/req
≈ 0.23 GiB(5 slot × 48.1 MiB,与上下文长度无关 —— GDN 架构的优势)。
每卡池(0.85):96GB 卡 ≈ 62 GiB;H20-3E 141GB 卡 ≈ 100 GiB。

| 每用户上下文 L | KV/user | +mamba | 96G 每卡并发 | 141G 每卡并发 | 100 用户需 96G 卡数 |
|---|---|---|---|---|---|
| 32k | 0.93 GiB | 1.17 | 53(显存不绑,decode 限 ~25-30) | 86(同上) | 2(实际被 §5 prefill 波顶到 6-8) |
| 64k | 1.86 | 2.10 | 30 | 48 | 4 |
| 128k | 3.73 | 3.96 | 16 | 25 | 7 |
| **262k** | **7.45** | **7.69** | **8** | **13** | **13** |

**卡数 ≈ ceil(100 × (KV(L)+0.234 GiB) / 62 GiB)**(96G、0.85)——即由"同时处于
长上下文的用户数"决定,而不是总并发数 100 本身。

- 硬性口径(100 会话同时满 262k):**14×H20-96G**(0.90 压线 12 张;留 margin 16 张)
- **H20-3E 141G**(HBM3e,同算力 148 TFLOPS/同 4.0 TB/s,2025-01 发布,源:
  flopper.io):每卡 13 个满上下文 → **8 张,是 262k 场景更优的采购选项**(留 margin 10)
- 现实 agent 分布(如 60%@32k / 25%@128k / 15%@262k,均值 2.85 GiB/user):
  8×96G 的 ~496 GiB 池只用 ~57%,每卡 ~12 并发、decode ~55 tok/s/user → **8 张仍够**

### 8.2 显存绑定时 decode 速度(与 L 无关的恒值)

显存打满时每步读取量 = 权重 17.62 GiB + 整池 ≈ 80 GiB →
**ITL ≈ 27 ms(37 tok/s/user,96G c=8);141G c=13 时 ITL ≈ 40 ms(25 tok/s)**。
agent 任务(人不盯流式输出)可接受;若要求交互级 ≥50 tok/s,需每卡限 5 个满上下文
并发(20 张 96G),不建议为此买单。

### 8.3 Prefill:增量 vs 冷启动

长 ctx 下每 token prefill FLOPs ≈ 16(GDN+proj+FFN)+ 4×L×d×n_full(注意力,
≈8 个全注意力层、d≈4096 估):L=32k → ~20 GFLOP/tok(≈4.9k tok/s);
L=262k → ~50 GFLOP/tok(≈2.0k tok/s);按 H20 有效算力 ~99 TFLOPS(沿用 H800
实测 ~67% MFU 按算力比缩放):

- **冷启动整段 262k prefill ≈ 80-120 s/请求**(H800 上约 13 s)——必须设计为罕见事件
- **agent 正常一轮**(前缀走 radix cache,只 prefill 增量):10k 增量 @200k ctx ≈ 4.3 s;
  1k 增量 @32k ctx ≈ 0.2-0.3 s。agent 循环里工具执行占大头,4-5 s 的 turn TTFT
  不伤任务吞吐
- 100 用户的 prefill 速率受 agent 节奏限制(约 1-3k tok/s/卡),**不是瓶颈**
- **真正的风险是 radix cache 驱逐**:262k 会话被驱逐 = 下次重付 80-120 s。
  池要留 20-30% 余量,必要时对长会话 pin KV

### 8.4 部署配置(262k 场景)

- `--context-length 262144`(先确认远端 config.json 的 max_position_embeddings)
- `--mem-fraction-static 0.85`(满载 262k 场景可 0.90;chunked prefill 峰值 ~1.3 GB)
- `--mamba-ssm-dtype bfloat16`(mamba 池减半,零成本)
- 客户端每轮超时 ≥ 3 min(覆盖冷 262k prefill 尾部)
- radix cache 保持开启 —— agent 前缀复用是本场景唯一的 prefill 优化

### 8.5 修订后的采购结论

| 口径 | 96GB 卡 | 141GB(H20-3E)卡 |
|---|---|---|
| 100 会话,262k 仅尾部(均值 <170k) | **8**(同 §1) | 4-6 |
| 100 会话同时满 262k(硬保证) | **14**(压线 12 @0.90;留 margin 16) | **8**(留 margin 10) |
| 200 会话同时满 262k | 28 | 16 |

验证建议(真卡,~2h):先跑 **1 卡 × 8 并发 × 262k** 小压测,验证每卡并发上限、
27 ms ITL、驱逐行为,再按 §8.5 定最终卡数。
