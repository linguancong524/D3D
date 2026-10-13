# Qwen3-8B H20 DP=4 部署基线（SGLang v0.5.18，非 MTP）

日期：2026-09-08。基线树：139e531（tag v0.5.18）。硬件：4× NVIDIA H20 96GB
（sm90，HBM3e ~4.0 TB/s，bf16 dense 峰值 ~148 TFLOPS），K8s dev-pod。
模型：Qwen3-8B（bf16，标准 Qwen3 dense，36 层、hidden 4096、GQA 32:8，
权重 ~16.5 GB，**无 MTP 头**）。运行时：torch 2.13.0+cu130、flashinfer
0.6.17、attention=fa3。H20 平台**不带** `--cuda-graph-backend-prefill=disabled`。

背景：Qwen3-8B 作为**离体式异步投机解码的暂时开发对象**（Qwen3.5-9B 的
DFlash/DSPARK draft 尚未训好）。本报告按 9B 现有 DP=4 测试协议
（[三形态对比](h20_qwen35_9b_tp1_tp4_dp4_comparison.md)、
[MTP 对比](h20_qwen35_9b_mtp_comparison.md)）测 **DP=4 非 MTP** 的
输出与负载情况，作为后续投机解码实验的对照基线。

## 0. MTP 在本 checkpoint 上不可用（前置结论）

**Qwen3-8B 无法开 MTP 投机解码**，两条独立证据：

1. **checkpoint 无 MTP 头**：`config.json` 无 `mtp_num_hidden_layers`；
   `model.safetensors.index.json` 的 399 个权重键中无任何 `mtp.*` 键
   （Qwen3.5-9B 自带 1 层 MTP——15 个 `mtp.*` 键——才能用
   `EAGLE + 同模型路径 draft` 复用 MTP 头做 draft）。
2. **SGLang MTP 白名单不覆盖 dense Qwen3**：`python/sglang/srt/configs/
   model_config.py` 的 draft 架构重写只覆盖 Qwen3Next/Qwen3Moe/Qwen3_5/
   MiMo/Step 等带 MTP 的架构，`Qwen3ForCausalLM`（dense）不在其中。

→ 若需 MTP 对照，必须另行提供 Qwen3-8B 的 MTP 变体 checkpoint 或独立
EAGLE draft（`/workspace/models/` 中目前没有）。**本次仅测非 MTP。**

## 1. 配置与内存布局（有效性）

启动参数（DP=4，对齐 9B 协议；纯 KV 模型，无需 `--mamba-ssm-dtype`）：

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python3 -m sglang.launch_server \
    --model-path /workspace/models/Qwen3-8B \
    --host 0.0.0.0 --port 30000 \
    --tp-size 1 --dp-size 4 --mem-fraction-static 0.85
```

压测（512 in / 256 out，random，sglang backend）：

```bash
.venv/bin/python3 -m sglang.benchmark.serving --backend sglang \
    --model /workspace/models/Qwen3-8B \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts {100,200,400} --request-rate {4,16,64}
```

| 项 | 值（每 rank） |
|---|---|
| Load weight begin. avail mem | 94.60 GB（4 rank 一致，GPU 干净） |
| Memory pool end. avail mem | 13.42 GB |
| max_total_num_tokens | 474 135（~463k tok/rank） |
| max_running_requests cap | 4096（纯 KV，无 mamba 池限制；9B 为 241/rank） |
| decode 图捕获 | 36 档到 bs 256 完整 |
| 基准成功率 | 100%/200%/400% 全部成功，server 日志 0 OOM / 0 traceback |

与 9B 的差异：9B（混合 GDN）的并发 cap 被 mamba 池压到 241/rank；
8B 是纯全注意力 KV 模型，pool 只受显存限制，cap 4096/rank、KV 池 ~474k
tok/rank，高并发余量远大于 9B。

## 2. 输出与负载（DP=4，非 MTP）

### 2.1 三档发送速率总览

| 指标 | rate 4（n=100） | rate 16（n=200） | rate 64（n=400） |
|---|---|---|---|
| 请求吞吐 (req/s) | 3.97 | 14.92 | **48.42** |
| 输出吞吐 (tok/s) | 534.89 | 2 111.56 | **6 339.08** |
| 总吞吐 (tok/s) | 1 614.03 | 6 121.73 | **19 183.44** |
| 峰值并发 | 11 | 43 | 199 |
| E2E 中位 (ms) | 906.29 | 1 048.16 | 1 731.21 |
| E2E p99 (ms) | 1 592.81 | 1 976.32 | 3 906.23 |
| TTFT 中位 (ms) | 46.33 | 45.92 | 51.83 |
| TTFT p99 (ms) | 113.80 | 70.44 | 116.92 |
| TPOT 中位 (ms) | 6.06 | 7.07 | 13.20 |
| ITL 中位 (ms) | 5.95 | 6.25 | 7.70 |
| ITL max (ms) | 139.31 | 62.49 | 133.78 |
| 压测时长 (s) | 25.21 | 13.40 | 8.26 |

### 2.2 与 Qwen3.5-9B DP=4（非 MTP）对比（同 4 卡 H20、同工况）

| 指标 | 8B rate 4 | 9B rate 4 | 8B rate 16 | 9B rate 16 | 8B rate 64 | 9B rate 64 |
|---|---|---|---|---|---|---|
| 请求吞吐 (req/s) | 3.97 | 3.96 | 14.92 | 14.83 | **48.42** | 47.71 |
| 输出吞吐 (tok/s) | 534.89 | 533.32 | 2 111.56 | 2 097.93 | **6 339.08** | 6 245.64 |
| E2E 中位 (ms) | **906** | 965 | **1 048** | 1 115 | **1 731** | 2 086 |
| TPOT 中位 (ms) | **6.06** | 6.38 | **7.07** | 7.60 | **13.20** | 16.21 |

（9B 数据源：`h20_qwen35_9b_mtp_comparison.md` §2/§3/§4.1 非 MTP 列。）

- **8B 全档略优于 9B（~5–17%）**：模型更小（8.2B vs 9B 混合架构）、
  纯 KV 池无 mamba 状态开销；rate 64 延迟差最大（E2E 1 731 vs 2 086 ms、
  TPOT 13.2 vs 16.2 ms），因为 9B 的 GDN 状态读取 + 更重的全注意力层
  在高并发下摊得更薄。
- **吞吐上限接近**（48.42 vs 47.71 req/s，+1.5%）：rate 64 已逼近
  prefill 算力/总带宽墙，两者都打到 ~19k tok/s 总吞吐。

### 2.3 负载分布（rate 4+16+64 合计，server 日志）

| rank | prefill batches | decode batches |
|---|---|---|
| DP0 | 176 | 126 |
| DP1 | 172 | 137 |
| DP2 | 175 | 131 |
| DP3 | 174 | 132 |

内置 DP（`load_balance_method=round_robin`）分发均匀：prefill 4 rank 之间
差 <2.4%，decode 差 ~9%（正常抖动，单卡 batch 粒度不同）。无 rank 饥饿
或过载。

## 3. 关键结论

1. **DP=4 非 MTP 基线确立**：8B 在 4× H20 上 rate 64 饱和吞吐
   **48.42 req/s / 6 339 输出 tok/s**，TPOT 中位 13.2 ms、TTFT 中位
   51.8 ms 保持健康；这是离体式投机解码实验的对照线。
2. **MTP 补测被 checkpoint 阻塞**：Qwen3-8B（HF 官方版）无 MTP 头，
   SGLang 也不支持 dense Qwen3 走 MTP draft。要做「带 MTP」对照，
   需要 Qwen3-8B-MTP 变体或 EAGLE draft checkpoint（待用户提供/训练）。
3. **并发余量大**：cap 4096/rank、KV 池 ~474k tok/rank（9B 仅 241/rank），
   高并发实验不受池限制——投机解码引入的 draft KV 开销在 8B 上更容易吸收。
4. 8B 全档延迟/吞吐 ≥ 9B DP=4 非 MTP，作为**临时开发对象性能足够**，
   后续 DFlash/DSPARK 收益可先在 8B 上量化，再迁移 9B。

## 4. 数据有效性

- 运行前 `nvidia-smi` 确认 4 卡 0 MiB；运行后确认全部释放到 0 MiB。
- 4 rank `Load weight begin. avail mem` = 94.60 GB（GPU 干净）。
- 三档基准 100/200/400 全部成功，无失败请求。
- server 日志 0 OOM、0 traceback；decode 图捕获 36 档完整。
- 所有输出 `tee` 进 `docs/perf/logs/`（见 §5）。

## 5. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `server_h20_qwen3_8b_serving_dp4.log` | DP=4 server（三档速率共用） |
| `bench_serving_h20_qwen3_8b_dp4.log` | rate 4（n=100） |
| `bench_serving_h20_qwen3_8b_dp4_rate16.log` | rate 16（n=200） |
| `bench_serving_h20_qwen3_8b_dp4_rate64.log` | rate 64（n=400） |

## 6. 未决项

- **Qwen3-8B 带 MTP 对照**：等 MTP 变体 / EAGLE draft checkpoint 落地后
  按同协议补测（rate 4/16/64），并与 §2.1 对比。
- 离体式（DECOUPLED）异步投机解码在 8B 上的实验：以本报告为基线，
  验收口径对齐 CLAUDE.md §8（吞吐/尾延迟不劣化 + 延迟收益）。
