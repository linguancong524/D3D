# H20 性能优化当前进展快照（2026-09-23）

本文汇总截至 2026-09-23 工作区内尚未归档的 H20 测试证据，并把“可用于结论的有效
结果”“中间微基准”和“失败诊断”分开。当前分支为
`feat/async-decoupled-drafter`，硬件为 4×NVIDIA H20 96GB。

测试文件主要生成于 2026-09-15 至 2026-09-19，覆盖两个相互独立的工作流：

1. Qwen3.8-27B 的 TP=4、TP=4+MTP4/8、DP=4+MTP4 serving 复测；
2. Qwen3-8B target + D3D-1.7B drafter 的异步 decoupled speculative decoding。

## 1. 本次归档范围与有效性

- 归档原工作区中 298 个未跟踪 `docs/perf/` 文件，合计约 15 MiB。
- 其中 83 份 JSON 含 benchmark summary：76 份 `summary.valid=true`，7 份为失败或
  不完整运行；明确保存 correctness 判断的结果中，32 份通过、6 份失败。
- Qwen3.8-27B 的 16 组 serving benchmark 全部请求成功；对应 4 份 server 日志也一并
  归档。
- decoupled 日志保留了成功验收、性能微基准和已知失败路径。失败日志是定位过程的证据，
  不能解释为当前实现仍然复现同一故障。
- 本次只归档测试结果和本文，不包含工作区内未跟踪的源码文件
  `python/sglang/kernels/ops/kvcache/shared_pending.py`。

## 2. Qwen3.8-27B：输出速度、延迟与并发

### 2.1 配置和并发上限

所有 benchmark 都是 random 512 input / 256 output，temperature=0，seed=42。文件名中的
`bs1` 表示这组 server 同时补跑了客户端 `max_concurrency=1` 的 `sat1` 工况，不表示 server
只能并发 1 个请求。

| 配置 | 物理部署 | server 有效 active cap | MTP accept length |
| --- | --- | ---: | ---: |
| TP=4 baseline | 4 卡组成一个副本 | 343 | — |
| TP=4 + MTP4 | 4 卡组成一个副本 | 48（spec 默认值） | 3.01–3.05 |
| TP=4 + MTP8 | 4 卡组成一个副本 | 48（spec 默认值） | 3.90–4.04 |
| DP=4 + MTP4 | 4 个单卡副本 | 16/rank，合计 64 | 2.94–3.10 |

TP=4 baseline 的 cap 由 1718 个 mamba slot 限制为 343；DP=4+MTP4 每个 rank 只有
82 个 mamba slot，因此限制为 16。MTP4/8 的 TP=4 复测没有显式抬高 spec 默认 cap 48，
高负载比较同时包含了 cap 差异，应该按“实际部署表现”理解，不能单独归因于 kernel 速度。

### 2.2 单并发与低并发

| 配置 | `sat1` req/s | `sat1` output tok/s | `sat1` TPOT p50 | rate=4 output tok/s | rate=4 peak outstanding | rate=4 TPOT p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=4 baseline | 1.03 | 146.16 | 6.56 ms | 530.66 | 14 | 9.09 ms |
| TP=4 + MTP4 | **2.02** | **286.65** | **2.97 ms** | 547.79 | 13 | **4.07 ms** |
| TP=4 + MTP8 | 1.94 | 275.22 | 3.40 ms | **550.60** | 12 | 4.36 ms |
| DP=4 + MTP4 | 0.83 | 116.95 | 7.74 ms | 530.64 | 15 | 8.49 ms |

单并发下，TP=4+MTP4 相比 TP=4 baseline 的输出速度为 **1.96×**，TPOT 为
**2.21×**；MTP8 的 accept length 更长，但 verify 更重，速度反而低于 MTP4。rate=4
受发送速率限制，四种配置都只有约 531–551 output tok/s，此时 MTP 的价值主要体现为延迟：
TP=4+MTP4 的 E2E p50 为 443 ms，而 baseline 为 821 ms。

### 2.3 中并发（rate=16）

| 配置 | req/s | output tok/s | peak outstanding | TTFT p50/p99 | TPOT p50/p99 | E2E p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=4 baseline | 13.23 | 1871.60 | 114 | 94.57/232.17 ms | 27.50/112.84 ms | 3.854 s |
| TP=4 + MTP4 | 12.50 | 1768.72 | 91 | 168.41/2513.92 ms | 21.91/73.38 ms | 3.678 s |
| TP=4 + MTP8 | 11.26 | 1593.40 | 102 | 786.58/3776.33 ms | 24.22/63.97 ms | 4.798 s |
| DP=4 + MTP4 | **13.33** | **1886.83** | 87 | 240.67/1205.48 ms | **19.43/39.45 ms** | **3.035 s** |

rate=16 时，TP=4 baseline 和 DP=4+MTP4 的吞吐基本相同；DP=4+MTP4 的中位
TPOT 更低，但 TTFT 更高。TP=4+MTP4 吞吐比 baseline 低 5.5%，MTP8 低 14.9%。

### 2.4 高并发（rate=64）

| 配置 | req/s | output tok/s | peak outstanding | TTFT p50/p99 | TPOT p50/p99 | E2E p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP=4 baseline** | **22.31** | **2921.09** | 391 | 330.73/4428.40 ms | 80.49/421.50 ms | 11.313 s |
| TP=4 + MTP4 | 15.59 | 2041.49 | 319 | 6255.76/17513.54 ms | 21.69/38.13 ms | 9.267 s |
| TP=4 + MTP8 | 13.60 | 1781.05 | 326 | 7703.67/20518.08 ms | 25.25/52.35 ms | 11.281 s |
| DP=4 + MTP4 | 16.87 | 2208.37 | 339 | 6538.41/14863.48 ms | 24.59/44.90 ms | 9.767 s |

吞吐优先时，TP=4 baseline 仍是这组数据的最优配置：比 TP=4+MTP4 高 43.1%，比
MTP8 高 64.0%，比 DP=4+MTP4 高 32.3%。MTP 配置的 token 间延迟较低，但默认 cap 48
或 DP rank cap 16 造成长队列，TTFT p50 已达到 6.3–7.7 秒。这个负载下不能只看 TPOT；
用户可见首 token 和端到端时延已经由排队主导。

结论与已有的 [Qwen3.8-27B MTP 对比](../reports/h20_qwen38_27b_mtp_comparison.md) 一致：
MTP4 适合小并发低延迟，MTP8 不划算；饱和吞吐优先则使用 TP=4 baseline。

## 3. Qwen3-8B decoupled drafter：当前实现和并发形态

### 3.1 当前四卡拓扑

当前可复现配置使用：

- GPU0–2：3 个独立 Qwen3-8B TP=1 target；
- GPU3：1 个共享 D3D-1.7B OneStep drafter；
- 每个 target `max_running_requests=3`，总 active cap 为 **9**；
- shared drafter `max_sessions=9`、每个 peer 最多 3 个 session；
- `speculation_batch_size=8`，drafter 最大 batch 64；
- block=8，accept count 1–8，1 anchor/位点，最多 8 branches；
- target local prefix reuse、packed causal pending、draft CUDA Graph 和 deterministic
  inference 均开启；poll interval 为 0.1 ms。

local-only 对照同样使用 GPU0–2 的 3 个 target；区别是没有 GPU3 shared optional
drafter。因此下面的 A/B 对比 target 数和 active cap 相同。

### 3.2 当前端到端 A/B（random512/output256，n=50，rate=4）

| 模式/重复 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 | accept length | 严格匹配 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid numpymask | 50/0 | 445.03 | 1.738 | 7.697/17.879/18.839 s | 3.659 | 50/50 |
| hybrid repeat 1 | 50/0 | 445.02 | 1.738 | 7.451/17.898/18.845 s | 3.652 | 50/50 |
| hybrid repeat 2 | 50/0 | **459.28** | **1.794** | 7.307/17.106/18.090 s | 3.658 | 50/50 |
| hybrid later session | 50/0 | 430.14 | 1.680 | 7.644/18.784/19.688 s | 3.655 | 50/50 |
| local-only repeat 1 | 50/0 | 450.87 | 1.761 | 7.521/17.605/18.587 s | 3.668 | reference |
| local-only repeat 2 | 50/0 | **471.95** | **1.844** | 7.060/16.495/17.497 s | 3.668 | reference |

同一时间段内 hybrid 为 430–459 output tok/s，local-only 为 451–472 output tok/s；
会话间漂移约 ±5%，因此当前只能说两者端到端**尚不可区分**，不能宣称 hybrid 已稳定
提速。六轮都完成 50/50 请求，hybrid 四轮均逐 token 严格匹配 local reference，正确性
阻塞已经解除。

rate=4 明显超过当前系统实际完成能力（1.68–1.84 req/s），所以 E2E p95 已达到
16.5–18.8 秒。当前结果证明的是带排队压力下的正确性和相对吞吐，不是 9-active-cap 下的
无排队延迟。还需要 rate sweep 和 saturation 复测才能给出当前实现的容量曲线。

### 3.3 已完成的性能与正确性里程碑

1. **Target verify 路径已恢复 lossless**：将 Target chain top-k 与 Draft anchor top-k
   解耦后，单 Target 20/20、四卡 50/50 严格匹配。
2. **packed causal/shared-canonical parity 已通过**：22-branch eager/graph 均为 22/22
   exact，多 item graph 为 44/44 exact；说明 grouped branch、graph replay 和 mask rewrite
   在定向输入上等价。
3. **remote model forward 已压到 verify 的量级**：代表性 graph microbench 的模型
   median 约 12.0–12.3 ms，service wall 约 15.1–17.9 ms；早期 grouped remote wall
   约 38.5 ms。
4. **host 路径大幅缩短**：最终 n=50 运行的 651 个 physical forward 中，模型累计
   18.47 s，即 28.38 ms/forward；service 总计 23.85 s，模型外开销约 8.3 ms/forward。
   同一轮优化前的 packed host cost 约 33 ms，优化后的稳定窗口约 6.8–8.3 ms。
5. **远端结果开始被真实消费**：最终三个 target 合计 3585 次 lookup，其中 163 次在
   lookup 时已有 ready outcome，133 次 branch hit。ready 后命中率 81.6%，但总 lookup
   命中率仍只有 **3.71%**；大多数 revision 仍在 remote ready 前走 local fallback。
6. **运行收尾干净**：最终 drafter `error_ct=0`、`stale_completion_ct=0`、
   `active_sessions=0`、queue 清零，没有请求失败。

### 3.4 本批中间结果如何解读

- 早期 Math-500、block=16/46-branch 三 target + shared drafter 在 rate=4 达到
  870–880 output tok/s，饱和达到 1059–1062 output tok/s；block=8/22-branch 对应
  886–887 和 1039–1044 output tok/s。eager/graph 结果接近并通过对应 correctness 检查。
- 这些数字使用 Math-500/不同 prompt 长度、`max_running_requests=8` 和不同 branch
  geometry，不能与当前 random512、cap=3 的 430–459 tok/s 直接比较。
- local prefix reuse 的 microbench/端到端结果仍有交叉：短输出 n=10 时 full replay
  389.18 tok/s、FlashInfer ragged 271.96 tok/s；output=256 时分别为 488.94 和
  427.35 tok/s。它已满足正确性，但尚不能概括为所有长度都更快。
- batched top-k 的 n=50 A/B 为 260.64→261.64 tok/s，端到端几乎不变；其主要价值是
  降低 host 小算子和为后续 shared pending/batching 建立正确路径。
- 本批中 7 份 invalid summary 以及若干 traceback 日志记录了 graph crash、pool ownership、
  retained KV 和早期 causal mask 等失败路径，均应作为诊断证据保留，不能纳入最终性能均值。

## 4. 当前是否达到发布条件

| 验收项 | 当前状态 | 判断 |
| --- | --- | --- |
| 逐 token lossless | n=50 四轮 hybrid 均 50/50 exact | 通过 |
| 服务稳定性 | 0 failed request，drafter error=0，session/queue 清零 | 通过当前规模 |
| remote 真实生效 | 133 次 branch hit，非空跑 | 通过 |
| 相对 local-only 吞吐 | hybrid 430–459，local 451–472 tok/s | 尚无稳定收益 |
| 并发容量 | 3 target × 3 active = 9 | 仅小 cap 验证 |
| 尾延迟 | p95 17–19 s（rate=4 已排队） | 不满足在线低延迟目标 |
| 饱和容量 | 当前最终配置尚无正式 saturation sweep | 未完成 |
| 与 4×AR baseline 比较 | 当前 random512 最终栈尚未同会话复测 | 未完成 |

因此 decoupled 当前是**正确性通过、性能优化仍在进行的 WIP**，不应推送 release。最主要
的剩余瓶颈已经从模型 forward/协议正确性转为 remote outcome 的时效性：ready outcome 中
大多数可命中，但只有 4.5% 的 lookup 在检查时 ready，导致绝大多数请求仍执行约
36–38 ms 的 local fallback。

## 5. 下一步建议

1. 固定当前 commit/config 做 local-only 与 hybrid 的交错多轮复测，至少各 5 轮，消除
   ±5% 会话漂移；同时记录 GPU 利用率和 drafter phase metrics。
2. 做 request-rate 0.5/1/2/4/inf 与 `max_running_requests` 3/8 sweep，给出吞吐、TTFT、
   E2E、实际 outstanding 和内存压力曲线。
3. 优先提高 remote ready 比例，而不是继续只压单次 forward：减少 revision supersede，
   将跨 target 的 pending item 更早聚合，并量化从 publish 到 ready 的完整延迟分布。
4. 用当前最终栈同会话复测 4×TP1 autoregressive baseline；发布门槛应是 lossless 前提下
   saturation throughput 明确提升且 p95 不劣化。
5. 将失败诊断日志继续保留，但后续正式报告只引用 `summary.valid=true` 且 correctness
   明确通过的结果。
