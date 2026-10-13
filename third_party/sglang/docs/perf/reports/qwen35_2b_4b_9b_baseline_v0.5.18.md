# Qwen3.5 2B/4B/9B 基线性能汇总（SGLang v0.5.18 / H800）

日期：2026-08-24。基线树：139e531（tag v0.5.18）。单机 H800 (sm90)，tp=1。
共同配置：torch 2.13.0+cu129，attention=fa3，linear-attn=triton（FLA chunk GDN），
sampling=flashinfer；所有运行均带 --cuda-graph-backend-prefill=disabled
（驱动 575.x/cu129 限制，见 SETUP.md）。
2B 单模型明细另见 qwen35_2b_baseline_v0.5.18.md。

数据版本（最终采信）：
- one_batch：2B=v1（0.88，9/9 完整）；4B=v3（0.65，9/9 完整）；9B=v3（0.65，9/9 完整）。
- serving：2B=v1（0.85）；4B=v2（0.75）；9B=v2（0.75）。
- v1 4B/9B one_batch 被 OOM 污染、v2 缺 bs32/4096 格，均已由 v3 取代
  （完整修正史见第 5 节）。

## 1. 模型与内存布局

| | 2B | 4B | 9B |
|---|---|---|---|
| 权重 (bf16) | 4.27 GB | 8.62 GB | 17.62 GB |
| mamba ssm_state (server 0.85) | 28.46 GB | 24.70 GB | 19.78 GB |
| KV cache (server 0.85) | 32.76 GB / 2.86M tok | 28.12 GB / 922k tok | 22.52 GB / 738k tok |
| max_running_requests（mamba 池上限） | 323 (server) | 105 | 84 |
| GDN 单请求 state（5 slot 合计） | ~18 MB | ~47 MB | ~47 MB |

注：mamba 每请求 5 个 state slot；4B/9B 单 slot ≈ 47 MB，2B ≈ 18 MB。
4B/9B 高并发测试先撞 mamba 上限，需要时可 --mamba-ssm-dtype bfloat16 减半。

## 2. 离线单批（one_batch，output=128；median）

注意：
- one_batch 的 prefill 是**单发全量 extend（无 chunked prefill）**，GDN 临时
  buffer 随 token 总量线性增长（见第 4.4 节）。
- 2B=v1（0.88）、4B/9B=v3（0.65）。v3 两模型均 9/9 格完整、无致命 OOM。
- **prefill 单发方差大**：bs1/bs8 中小 token 格（1k–8k tok）跨版本差异
  1.5–3.4×（见 4.3）；decode 跨版本差 ≤1.3%、bs32 prefill 行跨版本差 ≤5%，
  两者可信。
- 4B v3 实际执行过两个实例（09:36 与 10:01），仓库归档 10:01 实例；
  两实例 decode 差 ≤1%、bs32 prefill 差 ≤5%，与上述稳定性结论一致。

| 模型 | bs | input | prefill 延迟 | prefill 吞吐 | decode 延迟 | decode 吞吐 |
|---|---|---|---|---|---|---|
| 2B (v1) | 1 | 128 | 48.7 ms | 2 630 | 2.93 ms/tok | 341.7 |
| 2B (v1) | 1 | 1024 | 178.1 ms | 5 749 | 2.94 ms/tok | 340.0 |
| 2B (v1) | 1 | 4096 | 357.0 ms | 11 473 | 2.97 ms/tok | 336.9 |
| 2B (v1) | 8 | 128 | 49.5 ms | 20 703 | 3.13 ms/tok | 2 553 |
| 2B (v1) | 8 | 1024 | 209.8 ms | 39 051 | 3.18 ms/tok | 2 517 |
| 2B (v1) | 8 | 4096 | 999.4 ms | 32 789 | 3.17 ms/tok | 2 524 |
| 2B (v1) | 32 | 128 | 36.4 ms | 112 468 | 3.56 ms/tok | 8 986 |
| 2B (v1) | 32 | 1024 | 198.7 ms | 164 917 | 3.70 ms/tok | 8 642 |
| 2B (v1) | 32 | 4096 | 948.1 ms | 138 247 | 4.26 ms/tok | 7 510 |
| 4B (v3) | 1 | 128 | 43.1 ms | 2 969 | 4.70 ms/tok | 212.7 |
| 4B (v3) | 1 | 1024 | 323.4 ms | 3 167 | 4.74 ms/tok | 210.8 |
| 4B (v3) | 1 | 4096 | 224.7 ms | 18 231 | 4.78 ms/tok | 209.2 |
| 4B (v3) | 8 | 128 | 176.2 ms ※ | 5 811 | 5.14 ms/tok | 1 557.8 |
| 4B (v3) | 8 | 1024 | 328.1 ms | 24 971 | 5.25 ms/tok | 1 524.7 |
| 4B (v3) | 8 | 4096 | 759.6 ms | 43 141 | 5.54 ms/tok | 1 444.5 |
| 4B (v3) | 32 | 128 | 70.1 ms | 58 424 | 6.25 ms/tok | 5 117.5 |
| 4B (v3) | 32 | 1024 | 479.1 ms | 68 389 | 6.63 ms/tok | 4 741.5 |
| 4B (v3) | 32 | 4096 | 2 072.9 ms | 63 230 | 7.63 ms/tok | 4 194.5 |
| 9B (v3) | 1 | 128 | 44.0 ms | 2 907 | 7.17 ms/tok | 139.4 |
| 9B (v3) | 1 | 1024 | 314.8 ms | 3 253 | 7.22 ms/tok | 138.6 |
| 9B (v3) | 1 | 4096 | 302.5 ms | 13 541 | 7.24 ms/tok | 138.1 |
| 9B (v3) | 8 | 128 | 57.4 ms | 17 833 | 7.63 ms/tok | 1 049.2 |
| 9B (v3) | 8 | 1024 | 293.5 ms | 27 914 | 7.74 ms/tok | 1 033.1 |
| 9B (v3) | 8 | 4096 | 894.1 ms | 36 648 | 7.96 ms/tok | 1 005.5 |
| 9B (v3) | 32 | 128 | 110.3 ms | 37 129 | 8.66 ms/tok | 3 695.2 |
| 9B (v3) | 32 | 1024 | 794.8 ms | 41 229 | 8.99 ms/tok | 3 560.0 |
| 9B (v3) | 32 | 4096 | 3 854.9 ms | 34 001 | 10.05 ms/tok | 3 184.9 |

※ 4B v3 bs8/128 格（1024 tok）176.2 ms 疑为单发离群：同格 v2=52.5 ms，
2B/9B 同格 49.5/57.4 ms（4B 应 ~55 ms）。该格横向比较建议用 v2 值；
其余 v3 值未做替换（见 4.3 方差讨论）。

### decode bs=1 有效带宽（权重 / ms per token）

| 模型 | ms/tok | 有效带宽 | 占 H800 HBM3 (~3.35 TB/s) |
|---|---|---|---|
| 2B | 2.93–2.97 | ~1.45 TB/s | ~43% |
| 4B | 4.70–4.78 | ~1.82 TB/s | ~54% |
| 9B | 7.17–7.24 | ~2.44 TB/s | ~73% |

### prefill 峰值效率（bs=32/in=1024，32k tokens；按 2×参数量/tok 估算）

| 模型 | 吞吐 | 估算利用率（bf16 峰值 ~990 TFLOPS） |
|---|---|---|
| 2B | 164 917 tok/s | ~71% |
| 4B | 68 389 tok/s | ~61% |
| 9B | 41 229 tok/s | ~73% |

### 131 072-token 单发 extend（bs32/in4096，v3 补齐）

| 模型 | prefill 延迟 | 吞吐 | 相对 2B |
|---|---|---|---|
| 2B (v1) | 948.1 ms | 138 247 tok/s | 1.00× |
| 4B (v3) | 2 072.9 ms | 63 230 tok/s | 2.19×（参数量比 2.0×） |
| 9B (v3) | 3 854.9 ms | 34 001 tok/s | 4.07×（参数量比 4.5×，略快于纯参数缩放） |

三模型 131k 格吞吐随参数量近似线性下降，自洽；该格是 one_batch 模式下
唯一能打到 GDN 全量瞬时峰值（~20–24 GB）的场景（见 4.4）。

## 3. 在线压测（serving：random 512 in / 256 out，rate=4，n=100，全部成功）

| 指标 | 2B (v1) | 4B (v2) | 9B (v2) |
|---|---|---|---|
| 请求吞吐 (req/s) | 4.07 | 4.01 | 3.94 |
| 输出吞吐 (tok/s) | 548.9 | 540.8 | 531.3 |
| 平均/峰值并发 | 2.15 / 13 | 3.15 / 12 | 4.84 / 13 |
| E2E 中位 (ms) | 463 | 791.6 | 1 284 |
| E2E P99 (ms) | 2 110 | 1 528 | 2 390 |
| TTFT 中位 (ms) | 50.6 | 66.1 | 70.6 |
| TTFT P99 (ms) | 1 458 | 245.2 | 203.4 |
| TPOT 中位 (ms) | 2.57 | 5.23 | 8.33 |
| ITL 最大 (ms) | 521 | 183.7 | 172.7 |

4B serving v2（0.75）完全干净：吞吐跟上注入 4.0，尾延迟正常 ——
v1（0.85）的尾部异常未复现（v1 细节见第 5 节）。9B serving v2 与 v1
一致（v1 本就干净）。

## 4. 分析

1. **decode 随规模接近带宽上限**：bs=1 有效带宽 2B 43% → 4B 54% → 9B 73%。
   越大模型 GEMM 算术强度越高，越接近 HBM 峰值。含义：**decode kernel
   优化（dflash/dspark 方向）在 2B/4B 上空间最大（理论 ~2.3×/~1.8×
   带宽利用率上限），9B decode 已 73%，纯带宽型收益有限（~1.4×），9B 的
   decode 收益主要来自 launch 开销/状态读写优化**。验证优化建议
   先在 2B/4B 上做。
2. **decode 与上下文长度的关系分两档（混合 GDN 架构特征）**：
   - bs=1 与上下文长度无关：2B 2.93→2.97、4B 4.70→4.78、
     9B 7.17→7.24 ms/tok（128→4096 输入），状态读取 O(1)/step。
     **优化时必须保住这一特性**。
   - bs=32 随上下文长度上升：2B +20%（3.56→4.26）、4B +22%
     （6.25→7.63）、9B +16%（8.66→10.05）ms/tok（128→4096）；
     bs=8 介于两者（+1%~+8%）。来源：全注意力层 KV 读取 ∝ bs×ctx。
     大并发长上下文场景的 decode 优化需把这部分 KV 读取计入收益模型。
3. **prefill：单发测量方差是本次基线的主要测量学发现**：
   - decode 行跨 v2/v3 差 ≤1.3%、bs32 prefill 行差 ≤5%（稳定、可信）；
   - bs1/bs8 的 1k–8k token 格跨版本差 1.5–3.4×（如 4B bs8/128：
     v2 52.5 ms vs v3 176.2 ms；9B bs1/4096：v2 475.6 vs v3 302.5 ms），
     且"偏快/偏慢"的格在两次运行间会移动 —— 典型单发噪声，非系统差异。
     bs1/1024-token 格在 4B/9B 上三个版本反复偏慢，值得单独关注
     （GDN Triton 内核在该形状上的配置敏感，疑似 chunk/分块形状效应）。
   - 峰值利用率排序 9B 73% > 2B 71% > 4B 61%（bs32/1024 行）：
     **4B 是 prefill 低谷**（中间规模 GEMM + GDN 内核开销未被摊薄），
     与 GDN chunk 内核 + 形状效应相关，需单独 profile。
   - 结论：小格 prefill 数字只作量级参考，横向/版本比较以 bs32 行为准；
     若后续要精确的 prefill 效率曲线，需给 one_batch prefill 加多次
     平均或直接用 serving 端 chunked prefill 指标。
4. **GDN prefill 临时 buffer 是 v1/v2 全部 OOM 的根因（源码级定位）**：
   - gdn_backend.py 中 extend 路径把整个 extend batch 拍平成
     **B=1, T=总 token 数** 的张量传给 FLA chunk 内核（fused_qkv_split /
     view(1, seq_len, H, D)），因此 FLA 内部 buffer 按总 token 数分配：
     h = (1, NT, H, V, K)，NT = T/64（chunk_delta_h.py:349），
     v_new/o = (1, T, H, V)（chunk_o.py:146）。
   - 4B/9B GDN v-head 32×128、k-head 128（由 OOM 报错的精确尺寸反推：
     131 072-token extend 的 o buffer = 131072×32×128×2 = 恰好 1 GiB，
     h buffer = (1, 2048, 32, 128, 128) = 恰好 2 GiB，均与日志一致）。
   - 实测整层 forward（含 MLP/attention）的瞬时峰值 ≈ **155 KB/token
     （4B/9B）、~58 KB/token（2B）**：bs32/in4096 = 131 072 token 的
     单发 extend 需要 ~20.3 GB（4B）/ ~24 GB（9B，由 v3 日志中 6 GiB
     分配失败时 free=3.6 GiB 反推：27.5 − 3.6 ≈ 23.9 GB 已在分配）。
   - **one_batch 不做 chunked prefill**（one_batch.py extend → 单次
     model_runner.forward），所以大格直接顶到这个峰值；而 server 端
     chunked_prefill_size=8192 把峰值压到 ~1.3 GB/chunk。
   - 对后续 decode 加速工作无直接影响（decode 路径 buffer 与 T 无关），
     但提示任何"长 prompt 单请求 + one_batch 式全量 extend"的压测
     必须先算这个峰值。
5. **v1 → v2 → v3 one_batch 数据修正史**（4B/9B）：
   - v1（默认 0.88）：图捕获后空闲仅 665 MB/1.3 GB，bs32/4096 格触发
     未捕获 OOM，进程终止于 cell 9；且 4B bs1/4096（820.9 ms，7× 污染）、
     bs8/4096（1 438.3 ms）等格被 OOM 重试膨胀。弃用。
   - v2（0.75）：空闲 ~19.6 GB，8/9 格干净，仍死于 bs32/4096
     （需 ~20.3 GB）。8 格曾采信，现作为 v3 的交叉验证基准。
   - v3（0.65）：空闲 ~27.5 GB，**两模型 9/9 格完整、无致命 OOM**，
     为最终数据。交叉验证：decode 各格 vs v2 差 ≤1.3%；bs32 prefill
     vs v2 差 ≤5%（4B 69.8→70.1、476.2→479.1 ms；9B 105.1→110.3、
     791.8→794.8 ms）。4B 曾有两个 v3 实例（09:36/10:01），归档 10:01，
     两实例 decode 差 ≤1%。
   - 9B v3 bs32/4096 格期间有一条**非致命** allocator OOM 告警
     （申请 6 GiB 时 free=3.6 GiB，释放缓存块后重试成功，无异常抛出，
     9/9 完成，数值与跨模型缩放自洽）：说明 9B 在 0.65 下跑 131k
     单发 extend 已贴近空闲上沿，余量薄（4B 同格无任何告警）。
6. **4B serving v1 尾部异常（v2 未复现，根因未定）**：v1（0.85）
   3.23 req/s < 4.0、E2E P90 7.3 s、ITL max 7.7 s；v2（0.75）全部
   干净。无 v1 server log 可查。候选解释：(a) 16 GB 空闲下的 allocator
   碎片/重试压力；(b) serving 特有的 prefill 形状冷 JIT（v1 为 4B
   serving 首跑，v2 复用 /root/.cache/sglang 编译缓存）。如需要定论，
   可做 A/B：0.85 + 热 JIT 缓存重跑 4B serving（见第 6 节可选）。
7. **9B serving 最干净**：TTFT 中位 70.6 ms / P99 203 ms；E2E 略慢于
   4B 纯由 decode 更慢（TPOT 8.33 vs 5.23 ms）贡献。
8. **并发上限**：mamba 池限制 2B 323 / 4B 105 / 9B 84 请求（server 0.85）。
   one_batch v3（0.65）对应 4B 83 / 9B 65（本测试 ≤32 请求，未触顶）。
   后续高并发吞吐测试（4B/9B）建议 --mamba-ssm-dtype bfloat16
   （ssm 池减半、上限翻倍）。
9. **部署注意**：单卡 80GB 跑 4B/9B 时，server 端 8192-token prefill
   chunk 的瞬时峰值 ~1.3 GB，0.75 分位（~19 GB 空闲）下安全；
   one_batch 式 131k token 单发 extend 需 ≥22 GB（4B）/ ≥24 GB（9B）
   空闲，即 mem-fraction ≤ 0.68（4B）/ ≤ 0.65（9B，实测贴边）。

## 5. 已知问题与数据有效性

- **one_batch v3（最终）**：4B/9B 均 9/9 格完整。4B v3 全程无任何
  allocator 告警；9B v3 仅 1 条非致命 6 GiB 分配告警（bs32/4096 格，
  自动恢复，见第 4.5 节）。v3 池布局：4B mamba 415 slot（ssm 19.50 +
  conv 0.46 GB）+ KV 727 751 tok（22.2 GB），pool end 空闲 27.48 GB；
  9B mamba 326 slot（ssm 15.33 + conv 0.36 GB）+ KV 572 747 tok
  （17.5 GB），pool end 空闲 27.46 GB。
- **2B v1**：9/9 格完整（0.88，空闲 16.0 GB > 2B 峰值 ~7.6 GB）。
  init 阶段有 2 条非致命 allocator 告警（512 MB / 3 GiB 申请失败后
  恢复），不影响任何 cell 数值；与 9B v3 同属本栈在显存紧张时的
  已知良性现象。
- v1/v2 日志保留作历史与交叉验证（4B/9B v2 的 8 格干净，用于 4.5 的
  v2↔v3 一致性验证）。
- v2 4B/9B 日志中 warmup 后的 OOM traceback 为 cell 9 的（stdout 缓冲
  使结果块在文件中晚于 traceback 出现，勿误读为 warmup 失败）。
- serving：三模型 v1/v2 数据均可信（见第 3 节）。
  4B serving v1 尾延迟指标已被 v2 取代，不再引用。

## 6. 复现命令（v3，已完成；留作复现）

```bash
# 4B：mem-fraction 0.65 → 图捕获后空闲 ≈ 27.3 GB > 20.3 GB 峰值（7 GB 余量）
python3 -m sglang.benchmark.one_batch \
    --model-path /inspire/ssd/project/sais-bio/public/Chinese_civilization/models/Qwen/Qwen3.5-4B \
    --batch-size 1 8 32 --input-len 128 1024 4096 --output-len 128 \
    --mem-fraction-static 0.65 \
    --cuda-graph-backend-prefill=disabled \
    --result-filename docs/perf/results/bench_one_batch_qwen35_4b_baseline_v3.jsonl \
    2>&1 | tee docs/perf/logs/bench_one_batch_qwen35_4b_baseline_v3.log

# 9B：同上（model-path 换 Qwen3.5-9B，日志名换 9b）
```

说明：0.65 只缩小 KV/mamba 池（本测试 ≤131k token、≤32 请求，远小于池
容量），不改变 kernel 路径，与 v2 的 8 格可比。注意：跑之前必须确认
GPU 无残留进程（09:33 的无效 4B v3 即因残留进程占 61 GB 导致池塌缩：
avail 17.6 GB → mamba 25 slot → bs8 格 alloc_req_slots 失败）。

可选（仅当需要给 4B serving v1 异常定论）：0.85 + 热 JIT 缓存重跑
4B serving（先跑一次 one_batch 4B 预热 /root/.cache/sglang，再启动
0.85 server 跑 serving）。干净 → v1 异常是冷 JIT；仍卡 → 内存压力。

## 7. 日志清单

- bench_one_batch_qwen35_{2b,4b,9b}_baseline.log（v1；2B 完整采信，
  4B/9B 被 OOM 污染，弃用）
- bench_one_batch_qwen35_{4b,9b}_baseline_v2.log（0.75；8/9 格干净，
  缺 bs32/4096；作 v3 交叉验证基准）
- bench_one_batch_qwen35_{4b,9b}_baseline_v3.log（0.65；9/9 完整，
  **最终采信**）
- bench_serving_qwen35_2b_baseline.log（0.85，采信）
- bench_serving_qwen35_4b_baseline.log（0.85，v1，尾延迟受污染，已弃用）
- bench_serving_qwen35_{4b,9b}_baseline_v2.log（0.75，干净，采信）
