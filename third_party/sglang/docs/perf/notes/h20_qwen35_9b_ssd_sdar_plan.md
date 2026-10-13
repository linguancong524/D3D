# Qwen3.5-9B H20：离体式 SDAR Drafter（SSD 并行投机）可行性结论与接入规划

日期：2026-09-07。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
target：Qwen3.5-9B（bf16，17.62 GB，混合 GDN 线性注意力 + 全注意力）。

> 本文档是**规划与可行性结论**，不是基准报告。所有"预估"数字在 Phase 0
> 实测落地前只作量级参考；实测数据另出报告（命名沿用
> `bench_serving_h20_qwen35_9b_*` 约定，`tee` 进 `docs/perf/logs/`）。

## 0. 方案概述

自训 **离体式 drafter**：以 [SDAR](https://arxiv.org/abs/2510.06303)
（JetLM 的 diffusion + AR 混合 LM，[模型系列](https://huggingface.co/collections/JetLM/sdar)）
为基础、Qwen3.5-9B 为 teacher 蒸馏，做纯 token 条件的独立 draft 模型
（**不同步 hidden state**），部署形态 **3 卡 target + 1 卡 drafter**（共 4 卡 H20）。
draft 与 verify 在不同卡上**并行**，并行思想参考
[SSD: Speculative Speculative Decoding](https://github.com/tanishqkumar/ssd)
（ICLR 2026，Kumar / Tri Dao / May）。

与既有 DFlash/MTP 路线的本质区别：

| | DFlash / MTP（现有） | SDAR 离体式（本方案） |
|---|---|---|
| drafter 位置 | 与 target 同卡同进程 | **独立卡、独立进程** |
| drafter 输入 | target hidden states（EAGLE/DFLASH 式） | **纯 token id**（离体，蒸馏对齐） |
| draft 生成 | K 次串行 decode（或 MTP 单步多头） | **1-step block-diffusion：K-token 窗口 = 1 次 forward** |
| draft 成本落在哪 | target 卡（与 verify 抢算力） | drafter 卡（独立带宽/算力预算） |
| 回滚 | draft KV 逐步截断 | **截断 KV + 1 次 forward 重生成整块**（最便宜） |

**实测 draft 质量（用户提供，待 Phase 0 按模型分测确认）**：
accept length **9.82 @ K=16**、**12.24 @ K=32**
（对照 z-lab DFlash：5.1–5.3 @ bs16、4.2–4.4 @ bs8，见
[unieai vs z-lab 报告](../reports/h20_qwen35_9b_dflash_unieai_vs_zlab.md)）。

## 1. SSD 算法要点（移植参照）

SSD 的并行不是"drafter 自己往前跑"，而是 **fan-out 预投机**：

1. verify 步 t 完成后，verifier 经独立 NCCL group 把 key
   `(seq_id, accepted_len−1, recovery_token)` 发给 drafter；
2. drafter 维护 tree cache：提前对最可能的 f 个 verify 结果
   （预测接受长度 × 候选 recovery token，官方 b=1 配置 f=3、k=7）
   各跑一遍 draft 并缓存分支；
3. key 命中 → K 个 draft token **零延迟返回**（draft 成本完全藏在
   verify 背后）；未命中 → `jit_speculate` 按需补跑（退化为同步 SD）。

两个约束事实：

- 官方旗舰配置是 **b=1**（70B target 4×H100 + 1B drafter 1×H100）——
  这是低中并发延迟技术，不是高吞吐技术；
- 其 roadmap 第一条是 "Draft data parallel … to avoid getting
  compute bound"——作者自认 draft 卡会先撞算力墙。

**对 1-step block-diffusion drafter 的简化**：每个分支的预计算从 K 次
串行 decode 降为 1 次 extend-like forward，fan-out 成本变为 f 次
forward——SSD 的 fan-out 从"昂贵"变"买得起"。但更简单的
**draft-ahead 流水线**（draft 步 t+1 与 verify 步 t 并行，1–5 ms 的
draft 工作量藏进 verify）即可拿到大部分收益，fan-out 降为可选项
（见 §5 Phase 2b）。

## 2. 本树 v0.5.18 已有基础（利好）

上游 SGLang 的"解耦投机"（drafter/verifier 分离引擎）协议层已进树：

| 组件 | 状态 | 位置 |
|---|---|---|
| 协议：`DraftSync`（verifier→drafter 开请求、对齐 committed prefix）、`VerifyCommit`（连续提交段；drafter 需 truncate/reprefill 对齐 = 回滚语义，与 1-step diffusion 的"截断+重生成"一一映射）、`DraftClose`、`DraftTailStreamOutput`（drafter→verifier 逐 token 流式，带 `base_committed_len` 过期校验，为异步 draft-ahead 而设计） | ✅ 完整 + 单测 | `python/sglang/srt/speculative/decoupled_spec_io.py` |
| 多 verifier → 单 drafter 路由（`DraftReqKey(src_verifier_rank, request_id)`）——DP target 共享 drafter 拓扑原生支持 | ✅ | 同上 |
| CLI：`--decoupled-spec-role {verifier,drafter}`、`--decoupled-spec-bind-endpoint/--connect-endpoints/--rank`（ZMQ IPC mesh）、`--spec-trace-dir` | ✅ | `srt/server_args.py:2266-2292` |
| **引擎层**：ZMQ 传输线程、drafter 调度循环、verifier 侧 DraftTailBuffer 接入 | ❌ 缺失（代码内 TODO 提及 "phase 5.c"，上游分阶段开发中） | — |
| standalone draft 加载路径（独立权重、不共享 embed/lm_head；**硬性要求与 target 同 vocab**——同 teacher 蒸馏的 SDAR 天然满足，Phase 0 确认 config） | ✅ 参照 | `srt/speculative/standalone_worker_v2.py` |
| bs 分桶自适应调窗（`--speculative-adaptive`，EAGLE 系）——高负载关桶机制可复用其思路 | ✅ | `srt/speculative/adaptive_spec_params.py` |

**动手前建议先查 sgl-project/sglang 上游 main 是否已合入引擎层**；
若已合入，rebase 比手写 Phase 1 的传输/调度划算得多。

## 3. 成本模型与分负载结论

### 3.1 基本公式

每投机步（窗口 K、平均接受 accept）：target 做 b×(K+1) token 的 verify
forward，drafter 做 1 次 b×K mask-token 的 1-step diffusion forward。

- **算力受限区（高并发）吞吐比** = `accept / [(K+1) + (X/9)·K]`
  （X = drafter 参数量，9 = target 参数量；1-step 下 drafter 项为
  (X/9)·K 个 target-token 当量）。drafter 免费时上限 = `accept/(K+1)`。
- **带宽受限区（低并发）TPOT 下界** = `每步墙钟 / accept`，
  每步墙钟 = 顺序模式 `T_draft + T_verify`，重叠模式
  `max(T_draft, T_verify)`。

> 勘误：本项目早期分析（对话记录）曾给出"1.7B+K16 高负载 1.41× /
> 55.7 req/s"的表格，系把 per-token 开销重复除以 accept 的算术错误，
> 已作废。正确值见下表（0.49×）。

### 3.2 高负载（rate 64，compute-bound）：结构性劣势，必须关窗

| 配置 | 纯算力上限 | 经验估计* |
|---|---|---|
| 1.7B, K=16, accept 9.82 | **0.49×** | ~0.7× |
| 8B-1step, K=16 | **0.34×** | ~0.55× |
| 1.7B, K=32, accept 12.24 | **0.31×** | ~0.6× |
| （对照）DFlash-8 @ rate64 | 理论 0.48× | **实测 0.86×**（41.11 vs 47.71） |

*按 DFlash-8 实测/理论比（0.86/0.48）折中带宽重叠外推。

verify 每步读 17 个 token 只产出 9.82 个，tokens/FLOP = 0.577 < 1——
**drafter 再快也救不了**。rate 64 下 3+1 投机 ≈ 0.6–0.7 × TP=3 非投机
（~39.5 req/s 锚点）≈ **24–28 req/s，低于 DP=4 非投机 47.71**。
→ 高负载必须 bs 分桶关窗退非投机；这不是保险，是本方案的**架构必需组件**。

### 3.3 低负载（rate 4，bs≈1，TP=3 verify 17 tok ≈ 2.5 ms，预估）

| 配置 | 每步墙钟 | TPOT | vs 现最优 DFlash-16 TP=4（1.01 ms / E2E 138 ms） |
|---|---|---|---|
| 1.7B 1-step，顺序 | ~1–1.5 + 2.5 ≈ 4 ms | **0.41 ms** | 2.5× 更好 |
| 1.7B 1-step，+draft-ahead 重叠 | max(1.5, 2.5) = 2.5 ms | **0.25 ms** | **4× 更好** |
| 8B 1-step，+重叠 | max(~5, 2.5) = 5 ms | 0.51 ms | 2× 更好 |

→ rate 4 E2E 预估 **73–94 ms（−32~47% vs 138 ms）**。1-step 把 draft
从 16 次串行 decode 变 1 次 extend forward，简单 draft-ahead 即可藏住，
**不依赖 SSD fan-out**。

### 3.4 中负载（rate 16）

TP=3 接近饱和：吞吐持平（~15.5–16 req/s 量级），步数少 1.73 倍 →
队列排空更快，E2E/TPOT 明显改善（对照 DFlash-8 TP=4 的 282 ms，
预估 150–200 ms，需实测）。

### 3.5 模型选型

- **1.7B 首选**：draft 最快（低负载 TPOT 最好）、关窗前的过渡区最短、
  显存最省。
- 8B 仅在 **accept 显著更高**时翻盘：若 9.82/12.24 是 8B 蒸馏版测的、
  而 1.7B 版 accept 低 1.5 以上，8B 低负载仍可能胜出
  （TPOT ∝ T_draft/accept）。**Phase 0 必须把 accept 按模型分测。**

### 3.6 一句话定位

**3+1 SSD/SDAR 是低中负载延迟利器（低负载 TPOT 0.25–0.41 ms，
E2E −32~47%；中负载 E2E 明显更好、吞吐持平）；高负载维持 DP=4
非投机（47.71 req/s），靠自适应关窗衔接。** 4 卡按负载画像二选一；
8 卡后可 2 组 3+1 或混合池。

## 4. 前置检查（拓扑成立性）

1. **TP=3 整除性**：Qwen3.5-9B 的 `hidden_size`、`num_attention_heads`、
   `num_key_value_heads`（若 MoE 还要专家数）须均被 3 整除。
   **任一不满足 → TP=3 出局**，退路 **DP=3（3×TP=1）+ 1 张共享
   drafter**——decoupled 协议原生支持多 verifier 共享单 drafter，且更贴
   项目"TP=1×N"部署模型。H20 上 `cat config.json` 即定。
2. **vocab 一致**：STANDALONE 路径硬性要求（`standalone_worker_v2.py`
   有校验）；确认 SDAR checkpoint `vocab_size` = Qwen3.5 词表。
3. **drafter 架构确认**：1.7B/8B 是 dense 还是 MoE（影响 §3.2 的
   FLOP 比）、context 长度、mask_token_id 配置。
4. 非投机 TP=3（或 DP=3）冒烟：起 server + 小压测通过
   （ninja PATH 坑、H20 不带 prefill 图禁用参数，见 CLAUDE.md §4/§8）。

## 5. 分阶段实施规划

### Phase 0 — H20 实测标定（1–2 天，零移植代码，go/no-go）

| # | 项目 | 产出 / 判据 |
|---|---|---|
| 0.1 | 两个 drafter + target 的 config 核对 | TP=3 整除性、vocab、dense/MoE、mask_token_id → 定拓扑（TP=3 vs DP=3+1） |
| 0.2 | **drafter 1-step 窗口 forward 实测**：b×16 mask-token 单次 extend forward 墙钟，bs=1/16/64/240 | §3.3 的 0.25/0.41 ms 结论锚点；drafter 卡吞吐上限 |
| 0.3 | **accept 按模型分测**：1.7B 与 8B 各自 K=16/32 的 mean accept，bs=1 与 bs≈240 两档（512 in/256 out 采样，target 判定） | 确认 9.82/12.24 归属；高并发漂移（§3.2 经验估计的命门）；模型选型 |
| 0.4 | drafter 显存：bs=350 × 800 ctx 的 KV 池 + mask-token slot 预算 + 权重 | 1 卡 96GB 装得下（mem-fraction 按 CLAUDE.md §4） |
| 0.5 | TP=3（或 DP=3）非投机冒烟 + rate 4/64 小压测 | ~39.5 req/s 锚点实测值 + 冒烟通过 |

**go/no-go 判据**：低负载预测 TPOT（用 0.2/0.3 实测值代入
`max 或 sum 公式 / accept`）≤ ~0.7 ms 且 accept ≥ ~5 → go；
否则回到 4 卡同卡方案（DFlash/MTP）或改 8 卡规划。

### Phase 1 — 顺序式解耦（移植本体，估 2–3 周）

目标：端到端正确 + 真实 serving 下的 drafter 延迟/accept 回归基线。
**此阶段不做 overlap。**

1. **先查上游**：sgl-project/sglang main 是否已合入 decoupled-spec
   引擎层（ZMQ 传输 + drafter 调度）；合入 → rebase，未合入 → 自研。
2. **drafter 侧（独立轻量进程，不进 SGLang ModelRunner）**：
   SDAR 的 mask-token block denoising + 独立 KV + 截断重推是自成一套的
   采样循环，硬塞进 SGLang runner 收益低。独立进程（或 JetEngine）
   实现：消费 `DraftControlInbox`（`DraftSync`/`VerifyCommit`/
   `DraftClose`）→ 1-step block forward → 流式回
   `DraftTailStreamOutput`。CUDA graph 按 bs 桶捕获（extend-like
   固定形状，友好）。
3. **verifier 侧（target TP=3，SGLang 内）**：以
   `dflash_worker_v2.py` 的 draft→verify 结构为模板，把进程内 draft
   forward 换成"发 DraftSync/VerifyCommit + 从 DraftTailBuffer 拉
   proposal（含 `base_committed_len` 过期校验）"；verify 走标准路径，
   target 的 mamba 状态回滚复用现有 spec worker 机制
   （注意 `mamba_track_interval ≥ num_draft_tokens` 约束）。
4. **ZMQ 传输**：按 `server_args.py` 注释的通道约定
   （verifier: result PULL；drafter: control PULL）实现收发线程；
   注意 K8s `/dev/shm` 坑（CLAUDE.md §3）。
5. vocab 一致性校验沿用 STANDALONE 路径的检查。

### Phase 2 — 重叠（估 1–2 周）

- **2a（默认路线）draft-ahead**：drafter 沿 committed prefix 持续向前
  block 生成，`VerifyCommit` 到达时截断 + 重生成对齐。协议即为此设计，
  无预测逻辑。预期拿到 §3.3 重叠行（TPOT 0.25 ms，1.7B）。
- **2b（可选）SSD fan-out tree cache**：预测 top-f 个
  (accepted_len, recovery_token) 分支预跑（每分支 1 次 forward，1-step
  下便宜）。仅当 2a 实测 miss/回滚占比高才做。
- **自适应关窗（必需）**：bs 分桶，超阈值降窗至 0 退非投机
  （机制参照 `adaptive_spec_params.py` 的桶/滞回参数，接入 SDAR 路径）。

### Phase 3 — 验收矩阵

标准工况 512 in / 256 out × rate 4/16/64（n=100/200/400），
`tee` 进 `docs/perf/logs/`（CLAUDE.md §4 全部约束：残留进程检查、
prefill 图参数、mem 0.85 + mamba bfloat16）。对照组：
TP=4+DFlash-16/8（z-lab）、TP=4+MTP、DP=4 非投机、TP=3 非投机。
判据：rate 4 E2E/TPOT 是否破 138 ms / 1.01 ms；rate 16 E2E vs 282 ms；
rate 64 关窗后与 DP=4 非投机持平（±2%）。有效性判据沿用 CLAUDE.md §5。

## 6. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| TP=3 整除性不成立 | 拓扑作废 | Phase 0.1 首查；退路 DP=3+1（协议原生支持） |
| 9.82/12.24 是高并发下测不准 / 1.7B 版 accept 显著更低 | 全部收益估算下修 | Phase 0.3 双模型 × 双 bs 分测，go/no-go 用实测值 |
| 高并发 accept 漂移大 | rate 16 结论不稳 | 0.3 的 bs≈240 档专门覆盖 |
| drafter 卡显存装不下 bs=350 工作集 | 并发上限低于 target 侧 | 0.4 先算；必要时 KV 量化或降 max-running |
| SDAR 1-step 质量对 remasking/温度敏感 | serving（temp>0）accept 掉 | 0.3 同时测 temp 0 / 1；必要时 serving 侧 greedy |
| 上游 decoupled-spec 引擎层进度 | Phase 1 工作量 2–3 周 → 更短或更长 | Phase 1 第 1 步先查上游，决定 rebase vs 自研 |
| target mamba 状态在 K=16 投机下的回滚边界 | 正确性 | 复用现有 spec worker 的 mamba 追踪；Phase 1 单测覆盖部分接受/全拒两路径 |

## 7. 参考资料

- SSD 仓库与论文：[github.com/tanishqkumar/ssd](https://github.com/tanishqkumar/ssd)、
  [arXiv 2603.03251](https://arxiv.org/abs/2603.03251)
- SDAR：[arXiv 2510.06303](https://arxiv.org/abs/2510.06303)、
  [JetLM/SDAR-8B-Chat](https://huggingface.co/JetLM/SDAR-8B-Chat)、
  [JetLM SDAR collection](https://huggingface.co/collections/JetLM/sdar)、
  [github.com/JetAstra/SDAR](https://github.com/JetAstra/SDAR)
- 本项目既有数据：[三形态对比](../reports/h20_qwen35_9b_tp1_tp4_dp4_comparison.md)、
  [MTP 对比](../reports/h20_qwen35_9b_mtp_comparison.md)、
  [DFlash unieai vs z-lab](../reports/h20_qwen35_9b_dflash_unieai_vs_zlab.md)
