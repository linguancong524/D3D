# Qwen3.5-9B H20 MTP + `--speculative-adaptive` 对比（SGLang v0.5.18）

日期：2026-09-07。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3.5-9B（bf16，权重 17.62 GB，混合 GDN 线性注意力 + 全注意力，
`mtp_num_hidden_layers: 1`）。运行时：torch 2.13.0+cu130、flashinfer 0.6.17、
attention=fa3、linear-attn=triton。H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

本报告在 [MTP 投机解码三形态对比](h20_qwen35_9b_mtp_comparison.md)（2026-09-03，
静态 MTP，steps=3）基础上，叠加 **`--speculative-adaptive`**（自适应投机解码），
量化其相对静态 MTP 的效果。所有运行 cap=48、mem-fraction 0.85、mamba bfloat16、
512 in / 256 out，与静态 MTP 基线完全对齐（唯一变量是 adaptive 开关）。

## 0. `--speculative-adaptive` 是什么 / 怎么开

静态 MTP 固定 `--speculative-num-steps 3`（每步 4 个 draft token）。
**adaptive 在运行时按 batch-size 分档、按观测接受率（EMA）动态调整 `num_steps`**：
draft 一直被接受就加深步数、一直早拒就减步数，避免「draft 白算」。

启动参数（在静态 MTP 参数之上加两行）：

```bash
--speculative-algorithm EAGLE \
--speculative-draft-model-path /workspace/models/Qwen3.5-9B \
--speculative-num-steps 3 --speculative-eagle-topk 1 \
--speculative-adaptive \
--speculative-adaptive-config docs/perf/configs/adaptive_mtp_config.json
```

- 仅支持 EAGLE/EAGLE3 + topk=1（MTP 走 EAGLE，✓）；不支持 dp-attention
  （DP=4 数据并行 `enable_dp_attention=False`，不触发，✓）。
- 本测试用**默认分档配置**（`adaptive_mtp_config.json`，与代码
  `DEFAULT_ADAPTIVE_CONFIG` 一致，仅把 bs1 档补上 0 便于观察）：
  - bs1 → `[0,1,3,7]`（单请求可探到 7 步）
  - bs8 → `[0,1,3]`
  - bs32 → `[0,1]`
  - bs64 → `[0]`（高并发基本关掉 draft）
- 启动时为**每个候选步数**各建一套 CUDA graph 运行时状态（steps=0/1/3/7），
  运行时按 EMA 决策原子切换。

## 1. 前置修复：adaptive MTP 启动即崩（logits buffer 尺寸 bug）

首次运行 TP=1+adaptive 在启动建 steps=7 状态时崩溃：

```
AssertionError: shared logits buffer holds 192 rows but caller needs 384 (vocab_size=248320)
```

**根因**：共享 logits buffer（`GraphSharedOutput`）在 `ModelRunner.max_decode_logits_rows()`
里按**初始** draft-token 数定尺寸（steps=3 → 4 token × bs48 = 192 行），但 adaptive
运行时会涨到 steps=7（8 token × bs48 = 384 行），steps=7 的 target-verify graph
申请 384 行时越界。

**修复**（`python/sglang/srt/model_executor/model_runner.py`）：`max_decode_logits_rows()`
改用 `max_speculative_num_draft_tokens()`（adaptive 时返回 `max(candidate_steps)+1`，
静态时返回原值）来定尺寸。静态 MTP / 非投机路径行为不变（无回归），adaptive 路径
buffer 按最大候选步数预留。修复后 steps=0/1/3/7 四套状态全部建成，server 正常起。

> 这是 adaptive 投机解码路径的一个真实 bug（非本测试特有）：任何「初始步数 < 最大候选
> 步数」的 adaptive 配置都会在启动建最大步数状态时触发该断言。

## 2. rate 4（标准工况，n=100）：adaptive 延迟收益最大

| 指标 | TP=1 静态 | TP=1 **adaptive** | TP=4 静态 | TP=4 **adaptive** | DP=4 静态 | DP=4 **adaptive** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 4.04 | 4.06 | 4.08 | 4.08 | 4.05 | 4.06 |
| Accept length | 3.22 | **3.98** | 3.21 | **3.92** | 3.36 | **4.13** |
| E2E 中位 (ms) | 580.55 | **508.27** | 305.21 | **222.56** | 440.47 | **399.14** |
| TTFT 中位 (ms) | 67.98 | 68.90 | 40.64 | 40.64 | 57.12 | 57.17 |
| TPOT 中位 (ms) | 3.83 | **3.58** | 1.92 | **1.66** | 2.75 | **2.72** |
| ITL 中位 (ms) | 2.49 | 2.39 | 1.30 | 1.25 | 2.37 | 2.16 |

**rate 4 吞吐都 ≈4 req/s（发送速率受限），adaptive 的收益全在延迟，且全面优于静态 MTP**：

| 形态 | E2E 中位（静态→adaptive） | TPOT（静态→adaptive） |
|---|---|---|
| TP=1 | 581→508 ms（**1.14×**） | 3.83→3.58 ms（**1.07×**） |
| TP=4 | 305→223 ms（**1.37×**） | 1.92→1.66 ms（**1.16×**） |
| DP=4 | 440→399 ms（**1.10×**） | 2.75→2.72 ms（**1.01×**） |

- **TP=4 收益最大（E2E −27%）**：单请求时 adaptive 把 bs1 档探到 steps=7（8 draft
  token），accept 从 3.21 提到 3.92，每步多产 ~0.7 token，直接摊薄 TPOT。
- **accept 普遍 +0.6~0.8**：bs1 档在 steps=3↔7 间振荡（见 §4），净效果是单请求
  阶段更多时间跑在 steps=7，接受长度高于静态的固定 3 步。

## 3. rate 16（n=200）：adaptive 仍全面占优

| 指标 | TP=1 静态 | TP=1 **adaptive** | TP=4 静态 | TP=4 **adaptive** | DP=4 静态 | DP=4 **adaptive** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 13.53 | 13.56 | 15.78 | **15.89** | 15.57 | 15.64 |
| Accept length | 3.26 | 2.84 | 3.26 | **3.83** | 3.25 | **4.19** |
| E2E 中位 (ms) | 2155.65 | **2021.52** | 444.86 | **369.32** | 530.99 | **482.57** |
| TPOT 中位 (ms) | 13.01 | **12.05** | 2.68 | **2.57** | 3.43 | **3.22** |

| 形态 | E2E 中位（静态→adaptive） | TPOT（静态→adaptive） |
|---|---|---|
| TP=1 | 2156→2022 ms（**1.07×**） | 13.01→12.05 ms（**1.08×**） |
| TP=4 | 445→369 ms（**1.20×**） | 2.68→2.57 ms（**1.04×**） |
| DP=4 | 531→483 ms（**1.10×**） | 3.43→3.22 ms（**1.07×**） |

rate 16 下 adaptive 三形态延迟**全部优于**静态 MTP，吞吐持平或略高。注意 TP=1 的
accept 降到 2.84（低于静态 3.26）但延迟仍更好——rate 16 下 TP=1 已接近单卡饱和，
adaptive 在 bs8 档（候选 `[0,1,3]`）用更浅步数省下的 draft 计算，抵消了 accept 下降。

## 4. rate 64（n=400，逼近饱和）：分化——DP=4 持平、TP=1 持平、TP=4 略降

| 指标 | TP=1 静态 | TP=1 **adaptive** | TP=4 静态 | TP=4 **adaptive** | DP=4 静态 | DP=4 **adaptive** |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 17.32 | 17.26 | 34.80 | 32.31 | 48.01 | **48.19** |
| Accept length | 3.23 | 2.25 | 3.23 | 2.49 | 3.28 | 3.31 |
| E2E 中位 (ms) | 7446.02 | 7666.50 | 2774.72 | 3256.82 | 1853.62 | 1865.47 |
| TPOT 中位 (ms) | 19.01 | 19.13 | 9.58 | 10.86 | 14.35 | 14.72 |

- **DP=4：持平/略优**（48.19 vs 48.01 req/s，accept 3.31≈3.28）。DP=4 每 rank 独立
  调度，rate 64 下仍有相当比例的小 batch（bs1/bs8 档），adaptive 在这些档探深步数
  的收益保留，高 bs 档减步数的代价被摊薄 → 高吞吐场景 adaptive 无副作用。
- **TP=1：持平**（17.26 vs 17.32 req/s）。单卡算力/带宽瓶颈，accept 从 3.23 降到
  2.25（bs32 档候选 `[0,1]`，高 bs 基本只用 1 步）但吞吐不变——单卡已饱和，
  减 draft 步数省的计算正好补上 accept 下降的损失。
- **TP=4：略降**（32.31 vs 34.80 req/s，**−7.2%**）。TP=4 在 rate 64 是吞吐受限
  （非单卡瓶颈），accept 从 3.23 降到 2.49 直接减少每步产出 token → 吞吐下降。
  这是**默认 adaptive 配置在高 bs 档过于保守**所致（见 §6 建议）。

## 5. adaptive 的运行时行为（日志观测）

- **bs1 档在 steps=3↔7 间振荡**（TP=1 31↔31、TP=4 44↔44、DP=4 117↔115 次切换）。
  机制：bs1 时 accept ~2.5–2.8 > 升档阈值（step3 的 3−0.5=2.5）→ 升到 7；steps=7
  时深层 draft 接受率低、EMA 回落到降档阈值（3−0.5−0.25=2.25）以下 → 降回 3。
  每次切换是预建状态的指针交换（廉价），无错误，但偏频繁（~1 次/秒）。
- **高 bs 档用更浅步数**：bs8 档最多 3 步、bs32 档最多 1 步、bs64 档 0 步（不 draft）。
  这是 rate 64 accept 下降的直接原因，也是设计意图（高并发时 verify 是瓶颈，
  省 draft 计算）。
- 启动时为 4 个候选步数各建一套 graph 状态（steps=0/1/3/7），TP=1 下 steps=7 状态
  占 ~1.9 GB（比 steps=3 多 ~1.5 GB），是 adaptive 的显存代价（cap 仍 48，未触发墙）。

## 6. 关键结论

1. **adaptive 是「低并发延迟利器」，且比静态 MTP 更强**：rate 4/16 下三形态
   E2E/TPOT **全面优于**静态 MTP（TP=4 rate4 E2E −27% 最亮眼），因为单请求时
   bs1 档探到 steps=7，accept 提到 3.9–4.2（静态 3.2）。
2. **高并发（rate 64）分化**：
   - **DP=4 持平/略优**（48.19 vs 48.01）→ 高吞吐场景 adaptive 无副作用，可放心开。
   - **TP=1 持平**（17.26 vs 17.32）→ 单卡瓶颈，adaptive 无吞吐影响。
   - **TP=4 略降**（32.31 vs 34.80，−7.2%）→ 默认配置高 bs 档过保守，accept 下降
     拖累吞吐。
3. **默认配置对 Qwen3.5-9B MTP 偏保守**：MTP draft 只有 1 层（很便宜），但默认
   配置假设「高 bs 时 draft 昂贵」而把 bs32/bs64 档压到 1/0 步。对这种廉价 draft，
   高 bs 档保留更深步数（如 bs32 档 `[0,1,3]`）大概率能挽回 TP=4 rate64 的吞吐。
4. **前置 bug 已修**：adaptive MTP 原本启动即崩（logits buffer 按初始步数定尺寸，
   最大步数状态越界），已修 `max_decode_logits_rows()` 按最大候选步数预留（§1）。

## 7. 部署建议（9B / 4× H20 + MTP + adaptive）

| 场景 | 推荐 | 理由 |
|---|---|---|
| **低延迟单流 / 实时对话** | **TP=4 + MTP + adaptive** | E2E 223 ms / TPOT 1.66 ms（rate4），比静态 MTP 再快 1.37×，全场最低 |
| **低中并发、重延迟** | **DP=4 + MTP + adaptive** | E2E 399 ms（rate4）、TPOT 2.72 ms，延迟优于静态 MTP |
| **高吞吐 serving** | **DP=4 + MTP + adaptive** | 48.19 req/s（≥静态 48.01）且低并发延迟更好，无副作用 |
| **高吞吐 + 必须 TP=4** | 静态 MTP（或调优 adaptive 配置） | 默认 adaptive 配置 TP=4 rate64 降 7.2%；调高 bs32 档步数可挽回 |
| **单卡 / 边缘** | TP=1 + MTP + adaptive | 低并发延迟优于静态；高并发单卡瓶颈，adaptive 无吞吐影响 |

**一句话**：adaptive 让 MTP 在**低并发延迟上再上一个台阶**（TP=4 rate4 E2E −27%），
**DP=4 高吞吐无副作用**；唯一短板是**默认配置在 TP=4 高并发下偏保守**（−7.2%），
可通过调高 bs32/bs64 档候选步数解决。

## 8. 数据有效性

- 3 个 adaptive server 日志 `Load weight begin. avail mem`：TP=1 94.60 GB、
  TP=4 93.58 GB、DP=4 94.60 GB（GPU 干净）；`max_running_requests=48`（与静态 MTP 对齐）。
- 每次运行前 `nvidia-smi` 确认无残留进程；运行后 4 卡全部释放到 0 MiB。
- 9 组（3 形态 × rate 4/16/64）全部 100%/200%/400% 请求成功，无失败、无 OOM/traceback。
- 4 个候选步数（0/1/3/7）的 graph 状态全部建成（修复后），无 graph 捕获 OOM。
- 所有基准输出 `tee` 进 `docs/perf/logs/`（见 §9 日志清单）。

## 9. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `adaptive_mtp_config.json` | 本测试 adaptive 分档配置（默认档，bs1 补 0） |
| `run_mtp_adaptive.sh` | 复现脚本（`run_mtp_adaptive.sh <tp\|tp4\|dp4>`，单形态 3 档速率） |
| `server_h20_qwen35_9b_serving_tp1_mtp_adaptive.log` | TP=1+adaptive server（rate 4/16/64 共用） |
| `bench_serving_h20_qwen35_9b_tp1_mtp_adaptive_rate4/16/64.log` | TP=1+adaptive 三档速率 |
| `server_h20_qwen35_9b_serving_tp4_mtp_adaptive.log` | TP=4+adaptive server |
| `bench_serving_h20_qwen35_9b_tp4_mtp_adaptive_rate4/16/64.log` | TP=4+adaptive 三档速率 |
| `server_h20_qwen35_9b_serving_dp4_mtp_adaptive.log` | DP=4+adaptive server |
| `bench_serving_h20_qwen35_9b_dp4_mtp_adaptive_rate4/16/64.log` | DP=4+adaptive 三档速率 |

静态 MTP 对照数据见 [h20_qwen35_9b_mtp_comparison.md](h20_qwen35_9b_mtp_comparison.md)
（`bench_serving_h20_qwen35_9b_{tp1,tp4,dp4}_mtp_rate{4,16,64}.log`）。

## 10. 环境坑（adaptive 专属）

1. **logits buffer 尺寸 bug**（§1）：adaptive MTP 启动建最大步数状态时越界崩溃，
   已修 `max_decode_logits_rows()`。未修复前任何 adaptive MTP 配置都无法启动。
2. **启动更慢**：为每个候选步数各建一套 graph 状态（4 套），TP=1 下比静态 MTP 多
   ~10–15 s 启动时间，steps=7 状态多占 ~1.5 GB 显存。
3. **bs1 档振荡**：steps=3↔7 频繁切换（~1 次/秒），日志偏噪但功能正常、切换廉价。
   若在意可用 `--speculative-adaptive-config` 调 `up_hysteresis`/`down_hysteresis`
   加大滞回、降低切换频率。
