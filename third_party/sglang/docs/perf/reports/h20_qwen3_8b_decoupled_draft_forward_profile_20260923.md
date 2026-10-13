# H20 decoupled Draft 前向耗时剖析（2026-09-23）

## 问题与结论

在 Qwen3-8B Target、D3D/SDAR 1.7B Draft、`block_size=16`、每个接受位点 1 个 anchor 的 Math-500 运行中，远端 Draft 结果几乎无法赶上 Target verify。根因不是 16 条分支逐条调用模型：当前单行 packed 路径已经把同一请求的 pending 和 16 条分支合成 **一次**模型前向。问题是一次前向仍包含每请求 `16 + 16×16 = 272` 个 query token；3 个请求即 816 token，6 个请求即 1632 token。所谓“1.7B 模型几毫秒一次”的单 token decode 直觉不适用于这次多 token、多分支的 extend。

完整 16 分支时，单请求 GPU 前向也需 18.54 ms，超过实测 Target verify p50 11.03 ms。仅把三张 Target 的请求拆给更多 Draft 副本，并不能让单请求在这一个 verify 窗口内完成。多请求时还叠加图覆盖范围与队列问题。

## 测量方式

使用 [固定形状剖析脚本](../../../scripts/decoupled/profile_packed_forward.py)：在空闲 H20 GPU 3 上加载同一 Draft checkpoint，预先 stage 约 512 token 前缀，构造 16-token proposal，单行 packed + causal pending，`bf16`、FlashInfer、CUDA Graph max tokens=1024。每种配置启动一个 engine 并捕获 graph，在 `speculate_many` 上记录一次 profiler trace 和调度器 CUDA event 的 `model_forward_seconds`。同一配置的 batch=6 数字为累计 GPU 时间减去先前 batch=3；这是固定形状单次剖析，适合定位结构性瓶颈，不是稳定性基准。batch=1 的完整 16 分支单独启动 engine 测量。

| 分支数/请求 | 所选接受位点 | query token/请求 | 1 请求 GPU 前向 | 3 请求 GPU 前向 | 6 请求 GPU 前向 | 6 请求是否图回放 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 32 | — | 10.90 ms | 12.84 ms | 是 |
| 2 | 1, 8 | 48 | — | 13.48 ms | 16.92 ms | 是 |
| 4 | 1, 4, 8, 16 | 80 | — | 15.30 ms | 24.72 ms | 是 |
| 16 | 1–16 | 272 | 18.54 ms | 36.75 ms | 64.07 ms | **否** |

3 请求的完整 16 分支 trace 中，DeepGEMM 矩阵乘约 22.97/37.16 ms（62%），FlashInfer 注意力约 4.64 ms（12%），top-k/argmax 相关 kernel 约 4.27 ms（12%，发生在前向后的结果提取，不能与 `model_forward_seconds` 相加）。6 请求的相应矩阵乘约 41.95/68.44 ms（61%），注意力约 8.45 ms（12%）。Trace 的 CUDA kernel `graph id` 证实：16 分支、3 请求处于图回放；16 分支、6 请求全为 eager。6×272=1632 超过 1024-token graph 上限；其余较小分支预算的 6 请求均命中图。

作为量级校验，模型配置为 28 层、hidden=2048、intermediate=6144、vocab=151936。仅 transformer 线性层加全词表输出头就约有 1.72B 个乘法权重；按项目 H20 bf16 理论峰值 148 TFLOPS，816 token 的理想 GEMM 算力下界已约 19 ms，还未计注意力、显存访问、kernel/调度开销。这不是单 token decode 的工作量。

在线 Math-500 b16 完整运行的 1765 次 physical forward 平均 6.07 请求/批、GPU 前向 62.75 ms/批、远端服务 78.94 ms/批；Target verify p50 11.03 ms。固定形状剖析的 6 请求 64.07 ms 与在线量级一致。既有报告记录全量 500/500 逐 token 匹配，但远端 hit 只有 6 次，不能将接近 local-only 的吞吐归因于成功隐藏 Draft。

## 源码对应与优化取舍

- `scheduler.py` 的 `dllm_packed_single_row` 令每个请求只建一行，随后一次调用 `forward_dllm_packed`；分支结果提取在其后进行。已不存在“每条分支单独前向”的主要问题。
- `sdar.py` 设 `return_full_logits=True`，`LogitsProcessor._get_dllm_logits` 对全部 query hidden state 做全词表输出头。只对分支行投影可去掉 pending 的 16/272≈5.9% 输出头 token，但不能去掉 256 个分支 token 的 28 层主体计算，预计只是次级收益，需单独测量。
- `gather_shared_pending_kv` 只搬运注意力 K/V；单行布局已经避免重复 pending token 前向。它不能消除分支 Q/MLP/全词表投影；当前注意力仅约 12% kernel 时间，因此单独接入不可能填平 18–64 ms 与 11 ms 的差距。
- 把图捕获上限扩到至少 1632 token 可避免 6 请求 eager，但 3 请求已命中图仍需 36.75 ms；扩大图只处理次要因素，还要验证捕获显存与收益。
- **优先考虑减少实际分支计算量**：以 Target 接受长度/命中位置分布选择少量最有价值的接受位点，或按时限自适应调节分支数；然后在同一 Math-500 口径验证远端 hit、逐 token 输出和吞吐。2–4 分支的 3 请求仍需 13.48–15.30 ms；单分支 3 请求的 GPU 前向才接近 verify 窗口，完整远端服务时间还会更长。因此单纯稀疏分支也不能保证在当前窗口内命中，可能需要更早启动/跨多个 verify 窗口的流水线，或更轻量的 Draft/低精度方案。稀疏分支会改变预投机覆盖范围，不能直接当成完整 16 分支的等价加速。
- 目前每分支独立 top-k/argmax 约占 trace kernel 时间的一成，批量 top-k、只返回必要 logits、减少 host/device 同步可缩短服务尾部，但完整 16 分支的**模型前向本身**已超过 verify，不能把这类后处理优化当成主解。

## 证据

- [完整 16 分支固定形状日志](../logs/decoupled_draft_b16_profile_v2_20260923.log)、[对应 traces](../profiles/decoupled_draft_b16_profile_v2_20260923/)；[单请求日志](../logs/decoupled_draft_b16_batch1_profile_20260923.log)与[trace](../profiles/decoupled_draft_b16_batch1_profile_20260923/)。
- [4 分支日志](../logs/decoupled_draft_b16_4branch_profile_20260923.log)与[traces](../profiles/decoupled_draft_b16_4branch_profile_20260923/)；[2 分支日志](../logs/decoupled_draft_b16_2branch_profile_20260923.log)与[traces](../profiles/decoupled_draft_b16_2branch_profile_20260923/)；[1 分支日志](../logs/decoupled_draft_b16_1branch_profile_20260923.log)与[traces](../profiles/decoupled_draft_b16_1branch_profile_20260923/)。
- [在线 Math-500 b16 单行报告](h20_qwen3_8b_decoupled_single_row_b16_math500_20260923.md)。
