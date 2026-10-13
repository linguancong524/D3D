# Qwen3.5-2B 基线性能（SGLang v0.5.18 / H800）

日期：2026-08-24
基线版本：v0.5.18（本仓库 139e531 基线树 + 文档提交；tag v0.5.18 = 上游 ff4c6e64）

## 环境

| 项 | 值 |
|----|----|
| GPU | NVIDIA H800 (sm90)，tp_size=1 |
| CUDA / 驱动 | CUDA 12.9 工具链；驱动 575.x 代（只支持到 CUDA 12.9，CUDA 13 运行时不可用） |
| Python 栈 | torch 2.13.0+cu129，sglang-kernel 0.4.6.post1，sgl-deep-gemm 0.1.5.post3，flashinfer 0.6.17 (cu129) |
| 关键启动参数 | --mem-fraction-static 0.85 --cuda-graph-backend-prefill=disabled（驱动限制，见 SETUP.md） |
| 模型 | Qwen3.5-2B（hybrid GDN 线性注意力 + 全注意力层，bf16，权重 4.27 GB） |
| 注意力/采样后端 | attention=fa3（Hopper 默认），linear attn=triton，sampling=flashinfer |

内存布局（server 运行）：权重 4.27 GB；mamba ssm_state 28.46 GB + conv_state 1.0 GB；
KV cache 32.76 GB（2.86M tokens, bf16）；剩余 ~11.5 GB。
max_running_requests=323（被 mamba 状态池上限：1618 slots × 5 slot/req）。

## 1. 离线单批延迟（sglang.benchmark.one_batch，output-len=128）

命令见文末"复现"。warmup 段（首次 JIT，prefill 1.13s）不计入。

| bs | input | prefill 延迟 | prefill 吞吐 | decode 中位延迟 | decode 吞吐 | 总延迟 |
|----|-------|-------------|-------------|----------------|------------|--------|
| 1 | 128 | 48.7 ms | 2 630 tok/s | 2.93 ms/tok | 341.7 tok/s | 0.421 s |
| 1 | 1024 | 178.1 ms | 5 749 tok/s | 2.94 ms/tok | 340.0 tok/s | 0.638 s |
| 1 | 4096 | 357.0 ms | 11 473 tok/s | 2.97 ms/tok | 336.9 tok/s | 0.735 s |
| 8 | 128 | 49.5 ms | 20 703 tok/s | 3.13 ms/tok | 2 553.3 tok/s | 0.475 s |
| 8 | 1024 | 209.8 ms | 39 051 tok/s | 3.18 ms/tok | 2 517.3 tok/s | 0.631 s |
| 8 | 4096 | 999.4 ms | 32 789 tok/s | 3.17 ms/tok | 2 524.4 tok/s | 1.404 s |
| 32 | 128 | 36.4 ms | 112 468 tok/s | 3.56 ms/tok | 8 986.2 tok/s | 0.500 s |
| 32 | 1024 | 198.7 ms | 164 917 tok/s | 3.70 ms/tok | 8 642.0 tok/s | 0.674 s |
| 32 | 4096 | 948.1 ms | 138 247 tok/s | 4.26 ms/tok | 7 510.1 tok/s | 1.743 s |

## 2. 在线压测（sglang.benchmark.serving）

参数：--dataset-name random --random-input-len 512 --random-output-len 256
--num-prompts 100 --request-rate 4（100/100 成功，duration 24.57 s）。

| 指标 | 数值 |
|------|------|
| 请求吞吐 | 4.07 req/s（跟上注入速率，无积压） |
| 输入/输出吞吐 | 1 107.4 / 548.9 tok/s（输出峰值 1 270） |
| 总吞吐 | 1 656.3 tok/s |
| 平均/峰值并发 | 2.15 / 13 |
| E2E 延迟 ms | 均值 528.3 / 中位 462.8 / P90 1 001.4 / P99 2 109.6 |
| TTFT ms | 均值 132.4 / 中位 50.6 / P90 88.6 / P95 555.6 / P99 1 457.9 |
| TPOT ms | 均值 3.06 / 中位 2.57 / P90 4.05 / P99 8.39 |
| ITL ms | 均值 2.96 / 中位 2.20 / P99 34.5 / 最大 520.7 |

## 3. 分析

1. **decode 与上下文长度几乎无关**：bs=1 时输入 128→4096，decode 中位延迟仅
   2.93→2.97 ms/tok。混合 GDN 架构每步只读固定 mamba 状态，KV 成本近似常数——
   长上下文对 decode 近乎免费，是该模型的核心优势，优化时须保持此特性。
2. **prefill 是主要优化空间**：
   - bs=1/4096 仅 ~11.5k tok/s ≈ H800 bf16 峰值（~990 TFLOPS）的 5%，
     小 batch 下 kernel launch/延迟 bound（当前 prefill CUDA graph 被禁用）；
   - bs=32 时 ~138k tok/s ≈ 59% 峰值，接近 compute-bound；
   - bs=8 与 bs=32 的 4096 格吞吐均低于 1024 格（32.8k<39.1k、138k<165k）：
     32k/131k 总 token 需多轮 chunked prefill（8192/块），块间固定开销 +
     GDN prefill Triton 内核在大块上效率偏低，是后续首要怀疑对象。
3. **decode bs=1 带宽利用率 ~43%**：4.27 GB 权重 / 2.93 ms ≈ 1.46 TB/s
   （H800 HBM3 ≈ 3.35 TB/s）→ 瓶颈在 kernel（Triton GDN decode + 小 GEMM +
   launch 开销）而非带宽。dflash/dspark 类 decode 加速（fuse/batch 每步 kernel
   序列）最可能在此见效。
4. **在线尾延迟存在 prefill/decode 干扰**：TTFT P99 1.46 s、单点 ITL 520 ms，
   出现在峰值并发 13 的请求上（chunked prefill 插入 decode 流）。rate=4 属轻载，
   中位指标健康；吞吐上限需更高注入率（16~32 req/s 或 --max-concurrency）另测。
5. **并发上限受 mamba 状态池约束**（323 请求）：高并发测试受限可用
   --mamba-ssm-dtype bfloat16 将 ssm 池（28.5 GB）减半、上限翻倍；基线阶段不改。
6. 已知无害异常：bs=32/4096 格 decode 第 0 步 255 ms（其后恢复至 4.3 ms）及一条
   CUDACachingAllocator OOM 警告，均瞬时自愈，中位数不受影响；warmup 段为首次 JIT。

## 4. 复现

```bash
# 离线单批（需停 server，独占 GPU；日志落 `docs/perf/logs/`）
python3 -m sglang.benchmark.one_batch \
    --model-path $MODEL_PATH \
    --batch-size 1 8 32 --input-len 128 1024 4096 --output-len 128 \
    --cuda-graph-backend-prefill=disabled \
    --result-filename docs/perf/results/bench_one_batch_qwen35_2b_baseline.jsonl \
    2>&1 | tee docs/perf/logs/bench_one_batch_qwen35_2b_baseline.log

# 启动 server
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port 30000 --mem-fraction-static 0.85 \
    --cuda-graph-backend-prefill=disabled

# 在线压测
python3 -m sglang.benchmark.serving \
    --backend sglang --port 30000 --model $MODEL_PATH \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 100 --request-rate 4 \
    --output-file docs/perf/results/bench_serving_qwen35_2b_baseline.jsonl \
    2>&1 | tee docs/perf/logs/bench_serving_qwen35_2b_baseline.log
```

## 5. 原始日志

- bench_one_batch_qwen35_2b_baseline.log
- bench_serving_qwen35_2b_baseline.log
