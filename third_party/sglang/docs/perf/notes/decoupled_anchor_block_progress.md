# DECOUPLED SDAR drafter:anchor-conditioned block 改造进展(H20)

> 日期:2026-09-10 · 分支:`feat/async-decoupled-drafter` · 平台:H20 4×96GB
> 目标:draft block 语义对齐参考实现 —— `block = [anchor] + [mask]×(bs-1)`,
> anchor 为 target 刚产出的 bonus token,**不在** draft prefix 内

## 1. 目标流程(用户给定)

```
prefill(target + draft)
  → target forward(生成 anchor)
  → draft 准备 forward([anchor] + [mask]×(block_size-1))
  → draft forward
  → target verify
```

参考实现:`/tmp/async-spec-ref`(`sdar_utils.py::sdar_draft_block`、
`async_spec.py`),同仓库内的等价实现:`python/sglang/srt/speculative/decoupled/sdar_backend.py`
(`block[:, 0] = bonus_token`,其余 mask;每 session 独立 KV,block KV 用完即 `crop`)。

## 2. 本轮关键结论(均附实测证据)

### 2.1 上一会话「sync 1.94 vs packed 0.85」是数据集假象

| 运行 | 数据集 | `spec_num_correct_drafts / spec_verify_ct` |
| --- | --- | --- |
| 旧 sync 基线(2026-09-09 07:47,fe235a7) | **random 输入** | 1.94 |
| 当前代码 sync(本轮复跑) | **Math-500** | **0.948** |
| 当前代码 sync(本轮复跑) | random 输入 | 2.54 |
| packed(2026-09-09 10:35,fixB) | Math-500 | 0.855 |

→ random 垃圾输入天然好猜;真实内容上 sync / packed 只差 0.95 vs 0.86。
**packed 流程本身没有退化**,此前判断的 2.3× 差距主要来自数据集不同。

### 2.2 「把 anchor 复制进 block」的朴素实现被实测证伪

在 `dllm/mixin/req.py` 里把 fill 尾部改成 `[anchor] + [mask]*16`(anchor 同时仍在
prefix 末尾),Math-500 实测:

| 版本 | ratio | 吞吐 |
| --- | --- | --- |
| 基线 | **0.948** | 665 tok/s |
| anchor 复制版 | **0.041** ❌ | 416 tok/s |

原因:参考实现里 anchor **不在** prefix(它是待提交的下一 token);sglang 引擎里
anchor 已是 prefix 末元素,再追加副本 → 模型看到 `X, X, mask...` 连续重复,出分布。

### 2.3 正确语义需要「变长 extend」(本轮实现的核心)

block 必须恰好覆盖 `[anchor] + [mask]×(bs-1)`,即 extend 起点 = anchor 位置
`L-1`;而引擎的 radix 匹配按 `page_size = block_size = 16` 对齐,匹配点 ≤ `L-1`,
中间存在 gap(`L-1-match ≤ 15`)。因此:

```
extend 长度 = gap + block_size = (L-1-match) + bs   ∈ [16, 31]   (bs=16)
```

这不是固定宽度,打破了原有实现的多处硬编码假设(见 §3)。

### 2.4 dLLM 去噪路径对 CUDA 图的隐式依赖(实测 + A/B 确认)

- 去噪步骤必须有 `logits_output.full_logits`(`dllm/algorithm/*.py::step`)。
- `full_logits` 仅在 `LogitsProcessor.forward` 的
  `logits_metadata.forward_mode.is_dllm_extend()` 分支产出;实测插桩确认 eager
  路径会走到该分支并拿到 `hidden=(N, 2048)`。
- 但 **decode CUDA 图**路径是默认产出者;一旦 `--disable-decode-cuda-graph`,
  批次会落到 prefill 分段图,该路径不产出 `full_logits` → `argmax(None)` 崩溃。
  该现象在**未改动的基线代码**上同样复现(A/B:baseline + 禁 decode 图 → 同样
  崩溃),属预存限制,非本轮引入。
- 结论:变长 extend 与 decode 图的「每请求固定 bs」假设不兼容,且 **prefill 分段图
  也不可不产出 full_logits**;因此本轮方案要求该 draft engine **同时禁用
  decode / prefill 图,全程 eager**(`engine_backend.py` 的 engine args)。

### 2.5 变长 extend 已跑通到去噪步(最后一处崩溃已修)

插桩实测(全 eager):

```
[dbg-runner] mode=9 bs=1 tokens=16   → 纯真实 token 的 prefill 分块
[dbg-runner] mode=9 bs=1 tokens=16
[dbg-runner] mode=9 bs=1 tokens=16
[dbg-runner] mode=9 bs=1 tokens=27   ← gap + [anchor] + 15 masks(变长生效)
```

随后崩溃于 `_run_fdfo` 的 `input_ids.view(batch_size, block_size)`
(`shape '[1, 16]' is invalid for input of size 27`)—— 该处本轮已修
(`base.py::_run_fdfo` 按 per-request 段切分;`scheduler.py` FDFO 记账改用实际
token 数)。**端到端验证因中断尚未复跑**。

### 2.6 续实施:补齐 FDFO/调度边界与本地回归(Apple Silicon)

复核 `ea205e3` 后发现,上面的 FDFO 修复仍有一个时序错误:`OneStep.step()` 已经
原地填完所有 mask,代码随后才重新统计 mask 起点,会把新生成的 15 个 token 切成
空列表。现已改为在 `step()` 前保存每个请求的 mask 边界,并按各自变长 segment
切回 draft tail。两请求 `extend_lens=[6,4]`(测试 block size=4)的 CPU 回归确认
分别返回 3 个 draft token,不再被 radix gap 或另一个请求的长度干扰。

同时补齐以下正确性边界:

- native Engine 现在**同时**设置 `disable_decode_cuda_graph=True` 和
  `disable_prefill_cuda_graph=True`,与 §2.4 的全 eager 结论一致;
- native anchor-conditioned backend 默认且仅接受 `OneStep`;通用
  `LowConfidence/JointThreshold` 保留原有“固定宽度、全 mask block”契约;
- anchor block 连同 radix gap 必须一次性得到完整 token 预算;预算不足返回
  `NO_TOKEN` 等待,不再截断 block 并把部分 draft 的末 token 当成新 anchor;
- OneStep 的 radix prefix 匹配上限显式截到 `committed_len-1`,即使缓存中已有
  anchor KV 也不会吞掉 anchor;anchor 必须在本轮 extend 中作为 block 位置 0;
- Engine 请求固定 `max_new_tokens=block_size-1` 且 `ignore_eos=true`;若返回不足
  `block_size-1` 个 token 则直接报错,不再复制末 token 伪造完整 proposal;
- launcher 默认改为 `OneStep`,且 dry-run/建栈前会拒绝不兼容算法。

本地验证边界:Apple Silicon、CPU/协议与 mock Engine,不是 H20 推理。完整选择集:

```
97 passed, 2 warnings, 29 subtests passed
```

覆盖 DECOUPLED entrypoints/runtime/SDAR/Engine/worker/spec-io、anchor block/radix
边界新回归、PrefillAdder 以及 dLLM FDFO KV reuse;packed launcher dry-run 也确认生成
`OneStep + ipc:///dev/shm/... + 3 Target` 拓扑。

### 2.7 H20 端到端复跑:语义与 accept 通过,但 sync 吞吐回退

提交 `4236ae4` 后已在当前 4×H20 环境完成真实 Engine probe 和 Math-500 smoke,
补齐了 §2.6 当时缺少的 GPU 验证。

Anchor probe(`block_size=16`,3 sessions)全部满足:

- proposal 长度恰为 16;
- proposal 位置 0 等于 target 已提交的 bonus token(anchor);
- 无 service error。

Math-500(50 prompts,512 input/256 output,request rate 4,seed 42):

| 模式 | 完成/失败 | `correct_drafts / verify_ct` | output tok/s |
| --- | --- | --- | --- |
| sync run 1 | 50/0 | 6283/6535 = **0.961** | **505.9** |
| sync run 2 | 50/0 | 6305/6519 = **0.967** | **527.5** |
| packed | 50/0 | 6217/6601 = **0.942** | **301.1** |

结论:

- anchor-conditioned block 已在真实 dLLM Engine 上端到端跑通;
- sync accept 较旧 Math-500 基线 0.948 小幅升至 0.961–0.967;
- packed accept 较旧 0.855 恢复到 0.942;
- sync 吞吐相对旧记录约 665 tok/s 下降约 21%–24%,符合全 eager 的预期代价;
  当前结果证明正确性,**尚未证明净加速收益**。

证据目录:

- `docs/perf/results/decoupled-anchor-probe-20260910-064359-956492`
- `docs/perf/results/decoupled-sync-20260910-065224-598416`
- `docs/perf/results/decoupled-packed-20260910-065824-394798`

### 2.8 完整 Math-500:accept length = 2.052

在后续提交 `9f2c036`(补齐 block-size 启动校验和真实 KV 预算门控)上完成完整
Math-500 500 prompts 运行。benchmark 直接读取 `docs/scripts/test.jsonl` 的 `problem`
字段;配置为 3 个 target worker、512 input/256 output、request rate 4、seed 42。

| 指标 | 结果 |
| --- | --- |
| 完成/失败 | **500/0**(`valid=true`) |
| 输出 token | 128,000 |
| elapsed | 213.550 s |
| output throughput | **599.392 tok/s** |
| request throughput | 2.341 req/s |
| E2E p50/p95/p99 | 38.969 / 82.623 / 87.290 s |
| `correct_drafts / verify_ct` | **65716/62489 = 1.051641** |
| accept length(含 target token) | **2.051641** |

这里的 accept length 按 `1 + spec_num_correct_drafts / spec_verify_ct` 计算,即每次
verify 至少包含的 1 个 target token 也计入。完整运行再次确认 anchor-conditioned
proposal 不仅能通过 smoke,还可在 500 条真实 prompt 上持续工作;但 599.4 tok/s 仍
低于旧的全图 Math-500 记录约 665 tok/s,因此尚未收回全 eager 的性能损失。

`drafter.log` 在高并发爬升期间有一次 CUDACachingAllocator 分配 484,442,112 bytes
失败的 warning(当时 free 399,179,776 bytes),但随后 drafter 持续服务到
`active_sessions=0`,`error_ct=0`,`batch_error_ct=0`;500 请求全部成功且没有 Python
traceback。故该 warning 不使本次正确性/accept 数据失效,但说明显存已接近上限,
吞吐数字应视为当前内存压力配置下的实测值。

证据目录:
`docs/perf/results/decoupled-sync-20260910-082104-850511`(`run.json`、完整结果 JSON、
benchmark 摘要及 drafter/target 服务日志)。

### 2.9 输入契约修复:reference-style prompt 将 accept length 恢复到约 7.4

对完整运行 accept length 仅 2.052 的调用链复核后发现,benchmark 虽读取了
Math-500 的 `problem` 字段,但没有使用 Qwen3 chat template,也没有附加参考评测的
逐步推理/`\\boxed{}` 指令。改为
`apply_chat_template(tokenize=True, add_generation_prompt=True, enable_thinking=False)`
并追加 `Please reason step by step, and put your final answer within \\boxed{}.` 后,
50-prompt smoke 的 accept length(含 target token)从约 2 恢复到 **7.2–7.4**。

这证明此前约 2 的主要原因之一是输入分布不匹配,不能直接归因于 native Engine
attention。但约 7.4 仍低于预期约 10,因此继续按 attention/KV/scheduler 维度审计。

### 2.10 native eager SDAR attention 修复与最终 H20 smoke

参考 D3DV2 和 Transformers oracle 的 causal 模式后,native Engine 现显式区分每个
request 的两种 phase:

- prefix staging:只暴露 committed token 到最终 anchor 之前,使用 token-causal mask;
- denoising staging:一次 extend 完整 `[anchor] + [mask]×15`,所有 block query
  可见已缓存 causal prefix 和当前完整 block;
- anchor 在 denoising block 内重新计算,不复用 radix cache 中可能受 rejected suffix
  影响的 transient KV;
- FlashInfer 使用 request-packed paged custom mask;该路径明确为 eager-only,若进入
  dLLM prefill CUDA graph 立即 fail-loud。

最终实现保留 `disable_radix_cache=True`,并把 phase 边界放在 request fill 构造层:
缓存未到 `committed_len-1` 时 fill 仅为 `committed_ids[:-1]`;达到边界后 fill 才扩展为
`committed_ids + 15 masks`。这样通用 scheduler 无需再次承担 anchor 特例,去噪 extend
天然恰为一个完整 block。

同一最终语义在 4×H20 上的 reference-style prompt 结果:

| 运行 | 完成/失败 | `correct_drafts / verify_ct` | accept length(含 target) | output tok/s |
| --- | --- | --- | --- | --- |
| n=10 | 10/0 | 2255/352 | **7.406** | 78.748 |
| n=50 | 50/0 | 11277/1756 | **7.422** | **266.618** |

两次均 `valid=true`,服务日志无 traceback/OOM/error。另一次等价 A1 n=10 为
2256/353,accept length **7.391**,说明小样本结果可复现。这里的约 7.4 与此前包含
prompt/cache/scheduler/旧 attention helper 的组合 smoke 一致:新的 paged custom-mask
实现消除了 native attention 语义歧义,但**没有单独把 accept 从 7.4 推到约 10**。
当前 D3DV2 direct oracle 在相同 checkpoint 上因参考仓库兼容路径退化为每步只接收
1 token,不能作为数值 parity 依据;因此不宣称已达到 Transformers oracle parity。

CPU/静态验证:

- anchor/backend/entrypoint 三组定向测试:**27 tests passed**;
- `compileall`、`git diff --check`:通过;
- 全 `test/registered/unit/spec` discover 受环境缺少 `ninja` 阻塞于既有 cpp-ngram
  JIT 测试,与本次改动无关。

最终证据目录:
`docs/perf/results/decoupled-sync-20260910-102055-769602`。初次等价 A1 复现位于
`docs/perf/results/decoupled-sync-20260910-100748-778272`。

### 2.11 packed 模式复测:accept 持平,吞吐仅约 161 tok/s

在最终 custom-mask/anchor/prompt 实现上,使用与 §2.10 sync 完全相同的 Math-500
配置(50 prompts、512 input/256 output、request rate 4、seed 42)连续运行 packed 两次:

| 模式 | 运行 | 完成/失败 | `correct_drafts / verify_ct` | accept length | output tok/s |
| --- | --- | --- | --- | --- | --- |
| sync | final n=50 | 50/0 | 11277/1756 | **7.422** | **266.618** |
| packed | run 1 | 50/0 | 11268/1742 | **7.468** | 163.595 |
| packed | run 2 | 50/0 | 11249/1758 | **7.399** | 158.451 |
| packed | 两次均值 | — | — | **7.434** | **161.023** |

结论:

- packed accept length 在 7.40–7.47 间波动,均值 7.434,与 sync 7.422
  **基本持平**,没有稳定且有意义的提升;packed 本来主要用于隐藏 draft latency,
  并不应天然改变同一 checkpoint/proposal 的接受质量。
- packed 吞吐约 **158–164 output tok/s**,均值约 **161 tok/s**,仅为 sync 的
  **60.4%**(慢约 39.6%)。
- 两次累计 drafter metrics 为 `packed_hit_ct=175 / packed_lookup_ct=3589`,
  hit rate 仅 **4.88%**;同时执行 1129 次 optimistic speculation、6774 个 branch。
  大量低命中 branch 计算占用唯一 Draft GPU,是当前 packed 比 sync 更慢的直接原因。
- 两次均 50/0、`valid=true`,日志无 traceback/OOM/error,所以该退化不是失败请求或
  fallback 造成。当前实现不满足“packed 应明显优于 sync”的性能目标,不值得在此配置
  上直接扩大到 500 prompts;应先提升 outcome hit rate 或降低 fan-out/speculation 成本。

证据目录:`docs/perf/results/decoupled-packed-20260910-104701-332639`。

### 2.12 packed 与 D3DV2 流程对齐、真实 overlap 和 forward 时延

进一步逐段对照 `/tmp/D3D/D3DV2/eval_async_packed.py` 并在 H20 加 CUDA event/
wall-clock 临时探针（探针代码在运行后已恢复，不进入产品代码），得到以下结论。

**语义与时序:** 当前实现和 D3DV2 的核心状态机一致：先用当前 proposal 启动 Target
verify，同时对 `(accepted length, correction token)` outcomes 乐观生成下一 block；真实
`accepted tail + next bonus` 返回后按 outcome key 选择，命中时用真实 bonus 替换下一
proposal 的首 token，未命中才 required redraft。两者都只乐观前推一个 block，accept
length/correction anchor 的定义没有发现 off-by-one 偏差。

当前 `packed` **确实发生 Target/Draft 并行**，不是仅有 outcome cache 的串行实现。n=6、
bs=1 profile 中：

- 41/41 个 `speculate_begin` 区间内都观察到至少一个 Target verify launch；
- optional speculation 开始到首个 Target verify launch 平均 **0.615 ms**（中位
  **0.644 ms**，最大 **0.819 ms**）；
- 81 个 bs=1、block-size=16 Target verify 的 CUDA event 均报告 `graph=True`，GPU
  时间均值 **6.618 ms**、中位 **6.590 ms**、5%–95% 为 **6.476–6.777 ms**。

因此并行关系成立；问题是两边工作量严重失配。41 个 optional speculation 调用的 wall
时间平均 **188.6 ms**（中位 **123.8 ms**），远长于约 6.6 ms 的 Target verify。
Target 很快完成 verify 后，下一次 required RPC 又会被单线程 DraftService 正在执行的、
不可抢占的 optional `speculate_many()` 阻塞。

**与 D3DV2 的关键性能偏差:** D3DV2 保留/裁剪每 session 的 draft KV，把 pending block
和所有 outcome branch 拼成一个 query，用 branch-isolated attention mask 在**一次模型
forward**中计算；当前 native backend 则把每个 branch 变成独立 Engine request。虽然六个
branch 可进入同一个 `Engine.generate()`，但 `disable_radix_cache=True` 会为每条 branch
重算 committed prefix，也没有 D3DV2 的单-query branch packing。该差异不改变 proposal
语义，却使 optimistic fan-out 的成本完全不同。

单 worker 的 n=1（benchmark 另含同配置 warmup）eager Draft profile 进一步显示：

- native Draft 明确禁用 decode/prefill CUDA Graph；单次实际模型 `extend` forward 的
  CUDA event 中位 **10.211 ms**、均值 **10.428 ms**（去除启动期 JIT）；
- 两个请求共 14 个 required proposal step 和 14 个六分支 optional step，记录到 212
  个 eager model-forward event，说明一个 Engine proposal/fan-out 调用并不等于一次模型
  forward；
- service wall time 增量折算后，required proposal 平均约 **93.3 ms/step**，六分支
  optional fan-out 平均约 **111.1 ms/step**。

所以“单次 forward”对比是 Draft eager 约 10.2 ms vs Target graph verify 约 6.6 ms；但
真正决定调度的端到端 Draft block/fan-out 是约 93/111 ms，而不是 10.2 ms。后续优化应
优先实现 D3DV2 式持久 KV + 单-query branch packing，或至少给 optional work 设置可取消/
required-aware 边界；仅调 accept-count 分布不能消除这一级结构性成本。

profile 证据目录：

- overlap/Target CUDA event：`docs/perf/results/decoupled-packed-20260910-140114-393926`；
- Draft eager CUDA event：`docs/perf/results/decoupled-packed-20260910-141208-178522`。

### 2.13 D3DV2 pending-KV crop/reuse：审查纠错与真实 page-size=1 结果

§2.12 定位到旧 native backend 用 `disable_radix_cache=True` 把每个 outcome branch
变成全 prefix 重算；这与 D3DV2 保留 pending block KV、按 Target accept length crop、再在
branch-isolated forward 中计算 outcomes 的算法不同。`a481a35` 首次尝试启用 radix 并向
Engine 传入 `page_size=1`，但审查发现 `ServerArgs._handle_dllm_inference()` 又把它静默改回
block size 16。日志 `decoupled-packed-20260910-145738-065438/drafter.log` 的
`Setting page size to 16` 和最终 `ServerArgs(page_size=16)` 直接证明该次运行**没有**实现
此前报告所称的 token-level crop；原 765.617 tok/s/accept 6.088 只能作为 page-size=16
中间结果，不能标记为 D3DV2-aligned。

纠错包括：

- anchor-conditioned OneStep + radix cache 强制最终 `page_size=1`；legacy dLLM 仍保留原有
  block-aligned page 规则；
- `ReqDllmMixin._update_block_offset_for_dllm()` 允许 OneStep 的 radix prefix 停在任意 token，
  legacy 算法继续检查 block 对齐；
- 测试不仅检查 Fake Engine 原始 kwargs，还经过真实 `ServerArgs._handle_dllm_inference()`
  post-process，防止 page size 再次被覆盖；
- 每个 anchored branch 的 Engine prefix 仍为
  `committed + proposal[:num_accept_tokens] + bonus_token`。page-size=1 radix 命中到 bonus 前，
  随后 anchor-conditioned extend 计算 `[bonus_token] + [mask]×15`；request-packed custom mask
  隔离 sibling branches；
- 服务时序保持先发布 proposal/启动 Target verify，再在独立 Draft GPU 计算 optimistic
  outcomes，miss 才 required redraft。

`accept_counts=1 2 3`、`num_anchors=2` 下所有 optimistic branches 都带明确
`bonus_token`，没有把 full-accept/unanchored outcome 混入本次对比。native 路径使用多个
request segments 的同一 scheduled batch，而不是 D3DV2 Transformers 实现手工拼接的单一
physical tensor；这里的“对齐”限定为 KV crop、branch attention 可见性和 verify/draft
时序，不宣称尚未修通的 direct oracle 数值 parity。

H20 冷启动 stack、固定 Math-500 n=50（512/256、rate 4、seed 42）结果：

| 实现 | 最终 Draft page size | 完成/失败 | accept length | output tok/s | packed hit rate |
| --- | ---: | --- | ---: | ---: | ---: |
| 旧 packed（全 prefix 重算，两次均值） | — | 50/0 | 7.434 | 161.023 | 4.88% |
| 被误标为 token crop 的中间运行 | **16** | 50/0 | 6.088 | 765.617 | 4.77% |
| 修复后的 token-level KV crop/reuse | **1** | 50/0 | **3.090** | **345.090** | 6.76% |
| sync reference | — | 50/0 | 7.422 | 266.618 | — |

真实 page-size=1 运行 `8660/4144`，即
`1 + correct_drafts/verify_ct = 3.089768`；50/0、`valid=true`，`error_ct=0`、
`batch_error_ct=0`，无 traceback/OOM。吞吐是旧 packed 均值的 **2.14×**、sync 的
**1.29×**，但远低于先前 page-size=16 中间结果。accept length 的大幅变化进一步说明
page-size=16 结果不能替代 token-level D3DV2 语义验证；在 direct oracle 修通前，不把
accept 高低解释为 checkpoint 质量或完整数值 parity。

最终日志明确记录 `ServerArgs(page_size=1, disable_radix_cache=False)`，且不再出现
`Setting page size to 16`。证据目录：
`docs/perf/results/decoupled-packed-20260911-030453-197953`。修正 prompt/output 边界后的 GPU
anchor probe 也归档于该目录，3/3 session 均为 16-token proposal 且首 token 匹配真实
split 出的 anchor。

### 2.14 固定 16-token 去噪 CUDA Graph（实验 opt-in）

在不放宽 custom-mask 语义的前提下新增窄化图路径：prefix staging、mixed phase、非
16-token request 继续 eager；只有 homogeneous denoising batch 且每请求恰为
`[anchor]+[MASK]×15` 时进入 decode CUDA Graph。capture 时 FlashInfer paged wrapper 即以
custom-mask mode 创建；replay 重新构造每请求 packed mask、qo/qk indptr 和 KV indices 并复制
到 graph-stable buffer，执行时保持 `causal=False`。prefill graph 仍禁用。该路径默认关闭，
仅 `--engine-enable-cuda-graph` / launcher 的 `--draft-enable-cuda-graph` 显式启用。

通用 dLLM 默认 capture 到 bs=256 会浪费多个 model-sized graph；Draft 服务实际 required
batch≤64、optional batch≤8×6，因此 opt-in capture buckets 限制为
`1,2,4,8,16,24,32,48,64`。H20 上 mem fraction 0.85 的首次实验还暴露出 graph 启动内存
压力；正式实验将 Draft 降到 0.65，仍足以覆盖本次 50-request 工作集。启动失败和修复过程
日志均保留，最终运行无 traceback/OOM。

固定 Math-500 n=50（其余配置与 page-size=1 eager 基线相同）结果：

| Draft 路径 | 完成/失败 | accept length | output tok/s | required transaction | optional 6-branch transaction |
| --- | --- | ---: | ---: | ---: | ---: |
| eager page-size=1 | 50/0 | 3.090 | 345.090 | 70.06 ms/batch | 75.33 ms/batch |
| CUDA Graph run 1 | 50/0 | 3.025 | 403.865 | 64.64 ms/batch | 68.49 ms/batch |
| CUDA Graph run 2 | 50/0 | 3.050 | 456.722 | 59.58 ms/batch | 62.05 ms/batch |
| CUDA Graph 两次均值 | 50/0 | **3.038** | **430.293** | **62.11 ms/batch** | **65.27 ms/batch** |

两次 Graph 分别为 `8601/4248` 和 `8624/4206`，按统一公式 accept length 为
3.024718/3.050404；均为 50/0、`valid=true`、`error_ct=0`、`batch_error_ct=0`。
aggregate throughput 均值比 eager page-size=1 提高 **24.7%**；required/optional
transaction 均值分别下降 **11.3%/13.4%**。两次 tok/s 的离散度也说明仍需更大样本正式定标。
临时 CUDA-event 探针（运行后已恢复）测得 warm bs=1 graph execute 约 **3.00–3.06 ms**，
而旧 eager model forward 中位 10.211 ms、均值 10.428 ms；bs=2 约 3.19–3.23 ms，
bs=48 约 23.9–24.2 ms。graph execute 指 replay 加图前 metadata GPU work，不与旧探针边界
完全相同，因此只作为方向性 kernel-launch 证据，正式结论以 transaction 和 aggregate
Math-500 为准。

anchor probe 20/20 均返回 16-token proposal、首 token 匹配 anchor，且与历史 eager probe
仅出现一个近 tie token 的跨冷启动差异；request-packed mask 可见性仍由 CPU exact-mask 测试
覆盖。两次 n=50 accept 平均比 eager 低 1.7%，尚不能区分运行方差和低位数值 tie；因此
可以确认协议正确、无请求失败，但暂不宣称逐 token 数值完全等价。

性能证据：`docs/perf/results/decoupled-packed-20260915-072153-379869` 和
`docs/perf/results/decoupled-packed-20260915-073420-034881`；临时 GPU event 证据：
`docs/perf/results/decoupled-packed-20260915-072926-818266`。

### 2.15 预投机 KV 复审与安全修复：accept length 恢复到 7.51

逐行对照 D3DV2 `sdar_draft_all_outcomes_packed()` 后确认，原 `page_size=1`
方案并不等价于参考实现：OneStep forward 的真实输入是
`[anchor]+[MASK]×15`，模型先写入这批输入的 KV，之后才把 MASK 位置改写为预测
Token ID；finished-request radix insertion 却使用改写后的真实 Token ID 作为 key。
因此 radix 中形成了“真实 Token key → MASK-derived KV”的错误映射。D3DV2 则会先用
真实 pending proposal token 重新 forward，再保留/裁剪对应的 session-owned KV。

此外，D3DV2 的 block-causal pending KV 依赖整个 pending block，不能安全放入只按
Token prefix 全局共享的 radix tree。不同 pending suffix 可能为相同短前缀产生不同 KV；
`page_size=1` 只提高了错误状态的命中粒度，并没有实现 session/revision 隔离。

当前安全修复在 native Engine backend 中恢复 `disable_radix_cache=True`，禁止
MASK-derived denoising KV 污染后续 required draft 和其他 optimistic branch。同时不再发布
unanchored outcome：当前 `_generate()` 总把 `prefix[-1]` 当 anchor，而 D3DV2 unanchored
branch 的首位置应是 mask；直接事后替换 bonus 会让其余 token 整体错位一位，因此这类分支
现在安全 miss 并走 required redraft。固定 Math-500
n=50（512/256、rate 4、seed 42、Draft CUDA Graph、mem fraction 0.65）实测：

| 实现 | 完成/失败 | correct drafts / verify | accept length | output tok/s |
| --- | --- | ---: | ---: | ---: |
| 修复前 page-size=1 Graph 两次均值 | 50/0 | — | **3.038** | 430.293 |
| 禁止错误 radix reuse | 50/0 | 11274/1733 | **7.506** | 161.933 |

本次运行 `valid=true`，drafter 最终 `error_ct=0`、`batch_error_ct=0`，服务日志无
traceback/OOM。accept 已恢复到预期 7+，同时也确认现阶段吞吐收益依赖了错误 KV reuse；
该安全修复是 correctness baseline，不是最终性能方案。正式实现仍需 D3DV2 式
session-owned pending KV、真实 proposal-token replay、单次 branch-packed forward、按实际
accept length crop，并禁止 discarded candidate KV 进入全局 radix。

证据目录：`docs/perf/results/decoupled-packed-20260915-115223-390031`。

### 2.16 D3DV2 transient KV 基础：显式 revision commit 与统一 packed layout

下一阶段不再尝试用普通 `Engine.generate()`/radix key 模拟 D3DV2。协议升级到 v3：
每个 reply 返回完整 `Proposal` 的稳定 SHA-256 identity，下一次 `advance` 必须回传该
identity；service 在 outcome lookup 或 required redraft 前先调用 backend `commit_many()`。
因此 stale proposal、跨 revision optional completion 和复用 session token 值均不能裁剪
或推进 backend-owned KV。

Transformers `SDARBackend` 作为语义 oracle 增加了 revision/proposal identity 状态与显式
commit crop。commit 只发布实际 Target committed prefix（不含新的 bonus）；如果 optional
失败已主动丢弃 backend state，则后续 required draft 从 service 的完整 committed prefix
安全重建。

新增纯 CPU `packed_layout.py`，严格生成 D3DV2 的：真实 pending-token replay、每 branch
逻辑 position restart、accepted pending 可见性、sibling branch 隔离、anchored/unanchored
首位置以及 `retained_len=base_kv_len+pending_len`。SDAR oracle 已改为消费该统一 planner；
后续 native `DLLM_PACKED_REPLAY` 将使用同一 layout，而不是另写一套 mask/position 逻辑。

本阶段核心回归共 43 项通过（runtime 15、SDAR/layout 6、Engine 8、anchor block 14）；
再加 spec-io 23、worker 3、entrypoints 8，decoupled 定向测试合计 **77 项**通过，另通过
`compileall` 与 `git diff --check`。这一步建立了 native session-owned allocator 的
事务边界，但尚未宣称 native Engine 已做到单次 physical packed forward。

### 2.17 Scheduler transient KV ownership manager

新增 `transient_state.py`，实现不经过 radix 的 native KV ownership 状态机。每个 slot
绑定 `(session, revision, proposal_id)`、实际 KV provenance token、pending base、req-pool
slot 与逻辑 allocated length。optional forward 必须先 `begin(transaction)`，只有完整成功后
才能 `publish_pending()`；发布时仅保留 committed+pending replay，candidate branch scratch
立即归还 allocator。forward 失败走 `abort()`，只释放本 transaction 新增的 allocation，
不改变上一份已发布状态。

`commit()` 同时校验 revision 和 proposal identity，并只保留 Target committed prefix 与
published KV 的 LCP。若 correction 导致 pending suffix 分叉，缺失的 accepted token 不会
伪装成已有 KV，而由下一次 required draft replay；若在 immutable committed KV 内分叉则
fail-loud。`release()` 幂等归还 token KV 与 req-pool slot。page size >1 时 tail free 与现有
streaming-session 逻辑一致，从 `ceil_align(target, page_size)` 开始，避免释放仍含保留 token
的部分页。

新增 6 项 allocator 生命周期测试，覆盖 publish/candidate discard、actual-accept crop、
stale revision/proposal、forward rollback、idempotent close 及 page-aligned partial page。
当前 decoupled 定向累计 **83 项**通过。下一步是将该 manager 接入 Scheduler 私有 typed
operation，并让 `DLLM_PACKED_REPLAY` ForwardBatch 消费 §2.16 的统一 layout。

### 2.18 packed replay 的 ModelRunner/FlashInfer 执行 seam

`ForwardBatch` 新增私有 `dllm_self_attention_custom_mask`。FlashInfer 的 eager dLLM extend
现在优先消费该显式 flattened request-packed mask，否则保持原 phase-derived causal/block
mask。显式 mask 在 plan 前校验：必须与 input 同 device、必须是一维，元素数必须精确等于
每个 request 的 `extend_len × (prefix_len + extend_len)` 之和；因此 branch 数、KV 长度或
request rectangle 拼接错误会在 attention kernel 前 fail-loud。现有 dLLM CUDA Graph 固定
假设每 request 一个同宽 denoising block，检测到 packed mask 会明确拒绝，避免错误 graph
复用。

同时扩展统一 layout 支持 `causal_prefix_len`：当 backend state 因 required miss/optional
rollback 落后时，可在同一 query 中先 causal replay 缺失 committed suffix，再 block/causal
replay proposal；branch 的可见 pending 长度和 position restart 会自动加上这段 committed
replay。该改动也修正了 SDAR oracle 在 `cached < len(committed)` 时 branch position 未计入
missing committed suffix 的潜在偏移。

新增 3 项回归（mixed committed+proposal layout、显式 mask 优先级/形状、device guard）；
当前定向累计 **86 项**。下一步只需由 Scheduler typed operation 分配 req/token slots、组装
这些已验证的 positions/masks，并直接调用一次 ModelRunner forward。

### 2.19 native typed packed operation 与首轮 H20 正式测试

新增 `DllmPackedReqInput/Output` typed IPC，Engine 通过私有 scheduler RPC 提交多个 session、
proposal identity、committed token、proposal token 和选中的 correction branches。Scheduler
将各 session 的 unified layout 合并为一个 `ScheduleBatch`，显式写入 non-monotonic position
和 packed custom mask，然后调用 `TpModelWorker.forward_dllm_packed()`。该入口绕过
`OneStep.run()`，直接执行且只执行一次 `ModelRunner.forward()`；返回值带
`physical_forward_count=1`。native 路径现在也能正确发布 full-accept unanchored branch，
不再受旧 `_generate()` anchor-shift 限制。当前第一阶段仍是 **stateless full replay**：forward
结束立即释放全部 req/KV，下一阶段才接入 §2.17 manager 保留 committed+pending KV。

新增 backend typed-RPC/unit regression 后，decoupled 定向共 **87 项通过**，同时通过
`compileall` 和 `git diff --check`。

H20 正式测试已于本节实现完成后立即开始：

1. 单 session、6 branches native model smoke 成功，proposal/outcome 均为完整 16 token，
   `physical_forward_count=1`。有效日志：
   `docs/perf/logs/decoupled_native_packed_smoke_20260915_v7.log`。
2. 三 Target + 一 Draft 的 Math-500 n=10 端到端通过：10/10 完成、0 失败、2560 output
   tokens，`spec_verify_ct=357`、`spec_num_correct_drafts=2245`，故 accept length 为
   `1 + 2245/357 = 7.2885`，保持预期 7+。证据目录：
   `docs/perf/results/decoupled-packed-20260915-162859-003675`。
3. 当前 stateless replay 吞吐仅 87.86 tok/s，不是性能验收结果；它验证的是 packed branch
   token/KV、mask、position 和单 physical forward 正确性。正式 n=50/n=500 性能验收安排在
   transient manager 接入并通过 retained-KV/crop/rollback GPU 测试之后，避免为已知会被替换
   的 full-prefix replay 消耗长 benchmark 时间。

早期 smoke 的 v2 失败于 PATH 中缺 `ninja`，v3 失败于 synthetic request 未设置
`extend_range`，v4-v6 用于定位 deferred input materialization；这些均已 fail-loud 修复。
`run_stack.sh` 现显式把 `.venv/bin` 加入 PATH，typed operation 也在 forward 前调用
`resolve_forward_inputs()`。buffer registry 增加 slot name/shape 诊断，后续 metadata 偏移会
报告具体字段而不再只显示不透明的 foreach-copy shape mismatch。

### 2.20 retained KV、commit crop 与 H20 n=500 验收

Scheduler packed operation 已接入 `TransientDraftStateManager`。第一次 optional packed forward
从空 base replay committed+proposal；成功后保留这段 KV 及 req-pool slot，只释放 branch
scratch。后续 revision 直接把 session-owned KV locations 作为 prefix，新增 query 仅分配
missing committed replay、proposal 和 branches。typed `DllmKvControlReqInput/Output` 将 Target
commit/close 送到 Scheduler：control batch 先全量 validate，再按 proposal/revision crop；close
幂等释放 KV 和 req slot。transient owners 已纳入 idle/busy pool invariant 和 page ownership
检查，因此合法 retained KV 不再被误报为 leak。

GPU bring-up 暴露并修复了两处生命周期边界：

1. req-pool prefix locations 必须保持 device tensor，不能转成 `array.array`，否则
   `write_cache_indices()` 无法取得 `data_ptr()`。
2. 当某个 required/outcome proposal 尚未执行 optional packed forward 时，backend state 只有
   较短的安全 committed base，proposal identity 为空。其 commit 只推进 revision、不假装
   proposal KV 已存在；下一次 packed operation 通过 `causal_prefix_len` replay 缺失 committed
   suffix。高并发首次 n=500 因错误要求 retained tokens 与 committed 完全相等出现 8 次
   `packed committed prefix does not match retained KV`；改为严格 prefix 校验并 replay suffix
   后，错误归零。

最终 H20 验收：

| 工况 | 完成/失败 | accept length | output tok/s | packed physical forwards | backend errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Math-500 n=50 | 50/0 | `1 + 11238/1768 = 7.3563` | 246.659 | 133 | 0 |
| Math-500 n=500 | 500/0 | `1 + 112714/17430 = 7.4655` | 390.831 | 124 | 0 |

n=50 证据：`docs/perf/results/decoupled-packed-20260916-022050-724930`；n=500 证据：
`docs/perf/results/decoupled-packed-20260916-022658-700168`。n=500 最终 metrics 为
`speculate_batch_ct=124`、`packed_physical_forward_count=124`，即每个跨 session packed batch
严格一次 physical forward；`error_ct=0`、`batch_error_ct=0`、结束时
`active_sessions=0`。accept length 与 radix-disabled correctness baseline 7.506 相符。n=50
吞吐较该 baseline 的 161.933 tok/s 提升约 **52.3%**。

n=500 drafter 有两条 CUDACachingAllocator 非致命分配重试 warning（约 0.60 GB 与 2.34 GB）；
无 Python traceback、500/500 完成且最终 error counter 为 0，按本项目判据有效，但说明
Draft@mem-fraction 0.85 在高并发 packed scratch 峰值下接近显存上限。最终 packed outcome hit
rate 仅 0.17%，因此 390.8 tok/s 仍主要来自 required batch efficiency；后续性能工作应优先
改善 optional scheduling freshness/命中率并降低 branch scratch 峰值，而不是重新启用 radix。

当前 decoupled 定向 **89 项**全部通过，并通过 `compileall`、`git diff --check`。失败的
retained smoke 初版和首次高并发 n=500 目录只作 bring-up 证据，不作为最终性能数据。

### 2.21 block-size=8、7 位点、每位点 3 anchors 实验

按以下配置运行了完整 H20 对照：

```text
block_size=8
accept_counts=1,2,3,4,5,6,7
num_anchors=3
max_branches=21
```

即 7 个预投机位点，每个位点 3 个 bonus anchors，共 21 branches。`run_stack.sh` 已支持
通过 `BLOCK_SIZE`、`ACCEPT_COUNTS`、`NUM_ANCHORS`、`MAX_BRANCHES` 和
`DLLM_ALGORITHM_CONFIG` 环境变量复现实验，block-8 OneStep 配置归档在
`scripts/decoupled/sdar_block8.yaml`。

| 配置/工况 | 完成/失败 | accept length | output tok/s | packed hit rate | physical forwards | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b8/a1-7/k3, n=50 | 50/0 | `1 + 10614/2296 = 5.6228` | 122.017 | 1.916% | 166 | 0 |
| b8/a1-7/k3, n=500 | 500/0 | `1 + 106158/22812 = 5.6536` | 197.219 | 0.396% | 186 | 0 |
| b16/a1-3/k2, n=50 | 50/0 | 7.3563 | 246.659 | 1.317% | 133 | 0 |
| b16/a1-3/k2, n=500 | 500/0 | 7.4655 | 390.831 | 0.170% | 124 | 0 |

有效 b8 n=50 证据：`docs/perf/results/decoupled-packed-20260916-025957-016274`；有效 b8
n=500 证据：`docs/perf/results/decoupled-packed-20260916-031419-350965`。两轮均无 traceback、
OOM、invariant failure 或 backend error，结束时 active sessions 为 0；n=500 的
`speculate_batch_ct=186` 与 `packed_physical_forward_count=186` 一致。

该配置的 outcome hit rate 略高，但**吞吐约为 b16 对照的一半**：n=50 下降 50.5%，n=500
下降 49.5%。b8 每轮最多只能接受 8 tokens，导致 verify 数比 b16 增加约 30%；同时 n=500
累计 draft model 时间从约 340 秒增至 681 秒。21 branches 的 packed speculation 时间只从
约 5.5 秒增至 8.4 秒，不是主要瓶颈。因此在当前 b16 draft checkpoint 上，缩短 proposal
block 带来的 required draft/verify 频率增加远大于 7 位点覆盖和第三 anchor 的收益，不建议
替换 b16/a1-3/k2 默认配置。

第一次 b8 n=500 bring-up 在 retained sessions 达到约 130 时触发 req-pool slot exhaustion：
144 个 session slots 加 64 个 required batch scratch slots 可达 208，而 launcher 原来只给
engine 192。此运行仅完成 6/500，已判无效：
`docs/perf/results/decoupled-packed-20260916-030602-363564`。launcher 现按
`max_sessions + max(draft_batch_size, speculation_batch_size)` 配置 208 slots，重跑 500/500
通过；这是 req-pool 元数据容量修复，不改变 KV mem fraction。

### 2.22 Target correction token 在 Draft 分布中的 top-4 rank

为区分“anchor 候选不覆盖正确 token”和“optional outcome 没有及时产出”，Draft service
新增条件统计：仅在 Target 真正拒绝某个 draft token 时，取 Target correction/bonus token，
查询同一位置 Draft 返回的去重 top-4 列表。full accept、非连续 commit 和 correction 等于
原 proposal token 的情况不进入分母；top-4 之外只能判为 `>4`，不声称知道其精确全词表
rank。所有请求与 Draft generation 都显式使用 `temperature=0`。

Math-500 n=50、seed 42 结果：

| runtime 配置 | rejection 样本 | rank 1 | rank 2 | rank 3 | rank 4 | >4 | top-4 coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b16/a1-3/k3 diagnostic | 1529 | 18 | 676 | 208 | 100 | 527 | 65.53% |
| b8/a1-7/k3 | 1319 | 17 | 552 | 147 | 97 | 506 | 61.64% |

b16 top-4 coverage 的近似 95% 区间为 63.15%–67.92%，b8 为 59.01%–64.26%。因此若
“多数”只表示超过一半，则同事结论成立；但约 34.5%（b16）到 38.4%（b8）的正确
correction 在 Draft top-4 外，不能解释成“几乎都在 top-4”。该覆盖率只是 anchors 完备且
outcome 已就绪时的条件上限，并不等于 packed hit rate。

同轮 b16 的 packed hit rate 仅 0.82%（15/1825），虽然 correction top-4 coverage 为
65.53%；仅调度了 177 个 speculate items，并且 b16 只为 accept counts 1–3 建 branches。
b8 覆盖所有 1–7 rejection positions 后 packed hit rate 仍只有 1.45%（35/2415），而只调度
164 个 speculate items。故当前低 hit rate 的主因是 optional scheduling freshness/coverage，
其次才是约三分之一 correction 不在 top-4。证据：
`docs/perf/results/decoupled-packed-20260916-070558-534780`（b16）和
`docs/perf/results/decoupled-packed-20260916-065728-771757`（b8）。

温度说明：正温度缩放本身不改变同一组 Draft logits 的 rank 顺序，但会让 Target 从分布中
采样而不是固定取 argmax，从而改变“实际 correction token”的 rank 分布，通常会增加尾部
候选。当前 DECOUPLED v1 明确只支持 greedy `temperature=0`，不能把本结果外推到采样模式；
非零温度请求会在 model forward 前 fail closed。

### 2.23 packed miss 时序与 branch 原因分解

Draft service 新增互斥的 commit 时序计数：`optional_not_scheduled_ct`、
`optional_inflight_at_commit_ct`、`optional_ready_at_commit_ct`、
`optional_missing_at_commit_ct`；并增加 `optional_stale_completion_ct`。outcome ready 时再分解为
`outcome_ready_branch_hit_ct`、`accept_count_uncovered_ct`、`anchor_rank_miss_ct` 和
`outcome_ready_other_miss_ct`。定向测试覆盖 queued、ready hit、accept-count miss、anchor miss
和无效 delta。

Math-500 n=50、temperature 0、seed 42：

| 指标 | b16/a1-3/k2 | b8/a1-7/k3 |
| --- | ---: | ---: |
| packed lookups | 1818 | 2405 |
| optional 尚未调度 | 1708 (93.95%) | 2244 (93.31%) |
| commit 时 outcome ready | 110 (6.05%) | 161 (6.69%) |
| commit 时 in-flight | 0 | 0 |
| stale completion | 0 | 0 |
| ready 后 branch hit | 7 (6.36%) | 36 (22.36%) |
| ready 后 accept count 未覆盖 | 75 (68.18%) | 70 (43.48%) |
| ready 后 anchor miss | 28 (25.45%) | 55 (34.16%) |
| overall packed hit rate | 0.385% | 1.497% |

百分比的第一组以 lookups 为分母，ready 后三类以 ready outcomes 为分母，且三类精确相加。
b8 已覆盖 rejection counts 1–7，其 70 个 uncovered 全部对应 full accept count=8；当前配置
没有 full-accept unanchored branch。b16 的 uncovered 则包括 counts 4–16。

本次数据纠正了此前对第二类时序的推测：当前 Draft service 是单 owner、同步执行 packed
forward；forward 期间不会处理新的 advance，因此“已开始但完成前 revision 已推进”在当前
实现中没有发生，in-flight-at-commit 和 stale-completion 都是 0。真正的主因是 optional
entry 仍在低优先级队列中、尚未开始，就被下一次 required revision 覆盖，约占 93%–94%。
在少数 ready outcomes 中，accept-count coverage 是第二大限制；anchors 是第三项。

有效证据：`docs/perf/results/decoupled-packed-20260916-091326-370693`（默认 b16）和
`docs/perf/results/decoupled-packed-20260916-091856-038305`（b8 全 rejection 位点）。两轮均
50/50、error counter 0、无 traceback/OOM/invariant failure。

### 2.24 Target-local required + Draft-GPU optional 异步拆分

required draft 已从共享 Draft service 移到每个 Target 进程内：Target GPU 加载独立 SDAR
worker，所有 remote optional miss 在本地按 batch 合并为一次 eager physical forward；GPU3
只负责异步 optional packed outcomes。remote endpoint 不存在、超时或 outcome 尚未 ready 时，
Target 不等待远端，立即走本地 required。Target 与本地 SDAR 的 KV pool 由同一显存预算联合
定容，但请求槽、token pool 和 radix 所有权完全独立。

本地 required retained-KV 不能作为 correctness 路径。定向诊断已逐项确认旧实现的缓存 K/V、
req-to-token indices 和 flattened custom mask 与规划完全一致；切换真实 `torch_native` attention
后仍只有 `6/25` correct/verify，两阶段 prompt prefill 后也只有 `7/24`。因此问题不是
FlashInfer mask，而是当前 checkpoint/SGLang paged cached-prefix execution 的数值/语义差异。
本实现删除本地 retained state，required 使用 causal full replay + 完整双向 denoising block，
forward 成功或失败均归还全部临时 KV 与 req slot；remote optional 仍保留 Scheduler-owned
transient KV、revision commit/crop 和 branch scratch rollback。

H20 最终验收（3× Target replica；512 input / 256 output、rate 4、seed 42）：

| 模式 | 完成/失败 | correct drafts / verify | accept length | output tok/s | remote errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| local required only, n=10 | 10/0 | 2255/339 | **7.6519** | 699.998 | — |
| hybrid async, n=10 | 10/0 | 2254/340 | **7.6294** | 728.608 | 0 |

hybrid 的 remote optional 确实执行而非空跑：442 次 publish、191 个 packed batch、191 次
physical forward、2160 branches；最终 `stale_completion=0`、`active_sessions=0`，三个 Target
日志均无 traceback。单请求 full-replay probe 为 `38/5`，accept length 8.6。当前收益只作
n=10 bring-up 结论，不据此宣称正式吞吐提升；required miss 仍随 prefix 长度线性增长。

有效证据：local `docs/perf/results/decoupled-local-20260917-093332-838785`，hybrid
`docs/perf/results/decoupled-hybrid-20260917-093828-919397`。retained/mask/backend A/B 位于
`docs/perf/results/decoupled-local-20260917-084519-662711` 至
`docs/perf/results/decoupled-local-20260917-092112-413905`，只作失败诊断证据。当前 decoupled
定向 **88 项**通过，并通过 `compileall` 与 `git diff --check`。

### 2.25 当前异步流程与 DP=4 吞吐对比

在 4×H20 上按既有 Math-500 口径比较以下部署：

- DP=4 基线：4 个独立 TP=1 Target 副本（launcher `ar4`）；
- local-only：3 个 TP=1 Target，每卡 colocate 一个 stateless full-replay SDAR；
- hybrid async：local-only 的 3 个 Target，加 GPU3 shared optional packed drafter。

模型为 Qwen3-8B + `d3d-1.7b-b16_for_qwen3-8b`，输出固定 256 token，seed 42；
hybrid 沿用 block=16、accept-count=1/2/3、num-anchors=2、max-branches=6、每个
Target `max_running_requests=8`。测试中发现 optional service 的默认 per-peer quota 仍按
4 peer 切分，`max_sessions=24` 只给每个 Target 6 个 session；已改为 launcher 显式传
`max_sessions_per_peer=8`。修复后 Draft `error_ct=0`。

固定 request-rate=4、n=50（Poisson 到达，吞吐主要受 offered load/最后到达时刻影响）：

| 部署 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length |
| --- | ---: | ---: | ---: | ---: |
| DP=4 run 1 | 1091.133 | 4.262 | 1.588 / 1.812 / 1.822 | — |
| DP=4 run 2 | 1091.411 | 4.263 | 1.588 / 1.722 / 1.737 | — |
| local-only | 1111.259 | 4.341 | 1.689 / 3.954 / 4.422 | 7.4423 |
| hybrid async | 1107.835 | 4.327 | 1.817 / 4.402 / 4.951 | 7.4453 |

三种部署均跟得上 rate=4，故这一列不能证明 speculative 容量收益；但 local/hybrid 的
尾延迟已经明显差于 DP=4。为测容量，补充 Math-500 n=100、request-rate=inf 的一次性饱和
提交：

| 部署 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length |
| --- | ---: | ---: | ---: | ---: | ---: |
| DP=4 | 100/0 | **3953.114** | **15.442** | 3.278 / 4.984 / 6.181 | — |
| local-only | 100/0 | 1640.493 | 6.408 | 8.826 / 13.857 / 15.412 | 7.3800 |
| hybrid async | 100/0 | **1604.782** | **6.269** | 9.118 / 14.300 / 15.627 | 7.3684 |

hybrid 只有 DP=4 的 **40.60%** 吞吐（低 **59.40%**），p99 是 **2.53×**；相对
local-only 也低 **2.18%**。shared Draft 确实工作：最终 3631 publish、1842 supersede、
338 packed batches / physical forwards、1709 batch items（mean 5.056）、10254 branches，
`error_ct=0`、`active_sessions=0`。因此当前瓶颈不是“远端没有执行”，而是 remote outcome
没有消除足够多的本地 full replay，且 transport/packing/commit 增加了额外开销；在少一张
Target 副本的前提下，accept length≈7.37 不足以抵消容量损失。

逐 token 检查还暴露两个有效性问题。benchmark 原先将 live tuple 与 JSON list 直接比较，
会把所有请求误报为 mismatch；已统一按 token tuple 比较。修复后，DP=4 两次 n=50 自身为
45/50 完全一致，说明 BF16/批形状存在少量非 bitwise deterministic；但 local/hybrid 对首轮
DP=4 reference 只有 22/50、23/50 完全一致，饱和 n=100 分别为 47/100、44/100。hybrid
分叉集合基本被 local-only 覆盖，故 remote optional 不是主要分叉源，问题位于两者共用的
Target speculative verify/不同 kernel 形状。按严格 lossless 判据，本轮 local/hybrid 的
性能数字只能作为容量诊断，**不能作为可发布的正确性通过结果**。

所有运行均 100% 完成且无 Python traceback；Target 日志存在后续成功重试的 allocator OOM
warning（每实例 1–2 次），按项目既定规则不判为进程失败。首个 hybrid 冷启动还因子进程
PATH 缺 `.venv/bin`、FlashInfer JIT 找不到已安装的 `ninja` 而失败；后续均显式传递正确 PATH，
该失败目录不参与性能比较。

有效/诊断证据：DP=4 `docs/perf/results/decoupled-ar4-20260917-113558-076513` 与
`docs/perf/results/decoupled-ar4-20260917-115536-922225`；修复 quota 后 hybrid rate=4
`docs/perf/results/decoupled-hybrid-20260917-114833-402360`、饱和
`docs/perf/results/decoupled-hybrid-20260917-115912-341566`；local rate=4
`docs/perf/results/decoupled-local-20260917-115214-367672`、饱和
`docs/perf/results/decoupled-local-20260917-120310-338839`。

### 2.26 目标配置：7 个位点、每个位点 3 个备选

按目标把 proposal 改为 block=8（1 anchor + 7 denoise 位点），并将每个拒绝位点的
备选 anchor 开满 3 个：`num_anchors=3`、`accept_counts=1..8`、
`max_branches=7*3+1=22`（额外 1 条为 full-accept 分支）。其余部署和 §2.25 一致：
3×Target + GPU3 shared Draft、每个 Target `max_running_requests=8`、Draft
`speculation_batch_size=8`、Math-500、512 input / 256 output、seed 42。

Draft 累计 metrics 证实 22 条分支均实际生成：1780 个 batch item 对应 39160 branches，
恰为 `1780*22`；最终 7115 publish、5191 supersede、493 次 physical forward，batch mean
3.611、max 8，`error_ct=0`、`stale_completion_ct=0`、`active_sessions=0`。因此本轮不是
`max_branches` 截断后的伪“开满”。

固定 request-rate=4、n=50：

| 配置 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length | 严格 token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP=4 | 50/0 | 1091.133 | 4.262 | 1.588 / 1.812 / 1.822 | — | reference |
| 旧 hybrid：b16 / 2 anchors / 6 branches | 50/0 | 1107.835 | 4.327 | 1.817 / 4.402 / 4.951 | 7.4453 | 23/50 |
| **目标 hybrid：b8 / 3 anchors / 22 branches** | **50/0** | **1096.900** | **4.285** | **2.868 / 4.940 / 5.389** | **5.6436** | **23/50** |

目标配置吞吐相对旧 hybrid 低 0.99%，相对 DP=4 高 0.53%；但 rate=4 受 offered load
限制，不能视为容量提升。其 p95 相对 DP=4 高 172.65%，相对旧 hybrid 高 12.22%。

一次性饱和提交 request-rate=inf、n=100：

| 配置 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length | 严格 token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP=4 | 100/0 | **3953.114** | **15.442** | 3.278 / 4.984 / 6.181 | — | reference |
| 旧 hybrid：b16 / 2 anchors / 6 branches | 100/0 | 1604.782 | 6.269 | 9.118 / 14.300 / 15.627 | 7.3684 | 44/100 |
| **目标 hybrid：b8 / 3 anchors / 22 branches** | **100/0** | **1563.951** | **6.109** | **9.872 / 15.237 / 16.301** | **5.6549** | **40/100** |

目标配置饱和吞吐相对旧 hybrid 再低 2.54%，只有 DP=4 的 **39.56%**（低 60.44%）；
p95 是 DP=4 的 3.06 倍。与旧配置相比，本轮同时缩短 block 并扩大分支数，不能只凭这一组
数据把回退单独归因于其中一项；可以确定的是，目标组合使每个请求需要更多 verify（rate=4
为 2284 次，旧配置 1745 次），而共享 Draft 计算量扩大后没有抵消 Target 侧 full replay。
5191/7115（72.96%）的 publish 在消费前被新 revision 覆盖，也说明大量预投机工作没有转成
最终可复用结果。

两组运行均全部完成、无 Python traceback/请求超时；Target 和 Draft 日志有后续成功继续的
allocator OOM retry warning，按项目判据不属于致命 OOM，但仅约 0.1--1.6 GiB 的瞬时空闲显存
说明该 22 分支配置的内存余量很薄。严格 token 对齐分别只有 23/50、40/100，仍未满足
lossless 验收，所以本节数据只用于性能/容量诊断，不能作为可发布的正确性通过结果。

证据目录：`docs/perf/results/decoupled-hybrid-20260917-121838-903608`，其中 `run.json` 保存完整
生效参数，两个 benchmark JSON 保存逐请求结果，drafter/target 日志保存服务侧 metrics 与
allocator 证据。

### 2.27 增大 block 能否用更长 verify 覆盖异步 Draft

为验证“Target verify 变长后，shared Draft 更容易在下一轮 lookup 前完成”，补充两组 b16：

- **b16/22 受控组**：仍只预计算前 7 个拒绝位点、每点 3 个备选，并增加 full-accept，
  即 `accept_counts=1..7,16`、`max_branches=22`；与 §2.26 b8/22 的分支数一致。
- **b16/46 全开组**：15 个拒绝位点均保留 3 个备选，再加 full-accept，即
  `accept_counts=1..16`、`max_branches=46`。

固定 rate=4、n=50：

| 配置 | output tok/s | E2E p50/p95/p99 (s) | verify 次数 | accept length | 严格 token match |
| --- | ---: | ---: | ---: | ---: | ---: |
| b8 / 7 位点 / 22 分支 | 1096.900 | 2.868 / 4.940 / 5.389 | 2284 | 5.6436 | 23/50 |
| **b16 / 7 位点 / 22 分支** | **1108.136** | **1.648 / 3.788 / 4.338** | 1760 | 7.3898 | 23/50 |
| b16 / 15 位点 / 46 分支 | 1107.121 | 1.934 / 4.202 / 4.605 | 1747 | 7.4476 | 24/50 |

饱和 request-rate=inf、n=100：

| 配置 | output tok/s | req/s | E2E p50/p95/p99 (s) | verify 次数 | accept length | 严格 token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP=4 | **3953.114** | **15.442** | **3.278 / 4.984 / 6.181** | — | — | reference |
| b8 / 7 位点 / 22 分支 | 1563.951 | 6.109 | 9.872 / 15.237 / 16.301 | 4561 | 5.6549 | 40/100 |
| **b16 / 7 位点 / 22 分支** | **1696.943** | **6.629** | **8.578 / 13.681 / 14.935** | 3524 | 7.3828 | 41/100 |
| b16 / 15 位点 / 46 分支 | 1717.374 | 6.708 | 8.475 / 13.574 / 14.758 | 3554 | 7.3303 | 44/100 |

b16/22 相对 b8/22 的饱和吞吐提高 **8.50%**、p95 降低 **10.21%**；verify 次数减少
22.74%，accept length 从 5.65 提高到 7.38。因此增大 block 对端到端性能有效。但 Draft
侧 metrics 不支持“异步更容易被完全 cover”这一解释：

| 配置 | publish / supersede | supersede 比例 | physical forward | Draft 总秒数 | 平均 ms/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| b8/22 | 7115 / 5191 | 72.96% | 493 | 33.391 | 67.73 |
| b16/22 | 5476 / 4418 | 80.68% | 273 | 33.432 | 122.46 |
| b16/46 | 5493 / 4616 | 84.03% | 167 | 69.143 | 414.03 |

b16/22 的分支数未变，但每条分支变长，单次 Draft forward 耗时增至 1.81 倍，supersede
比例反而更高；b16/46 则进一步把单次 forward 推到 414 ms，请求结束后 Draft 队列仍需继续
排空。由此可见，本轮的主要收益是 **b16 提高接受长度、减少 Target required full replay 和
verify 轮数**，而不是 remote optional 已经被 verify 时长完全隐藏。

15 位点全开相对受控 b16/22 只多 1.20% 饱和吞吐，属于单次运行可能出现的波动量级，却使
Draft 总计算秒数增加 2.07 倍、supersede 升至 84.03%。在现有实现下，b16/22 是三组中更合理
的折中，但仍只有 DP=4 的 **42.93%** 饱和吞吐，且严格 token match 只有 41/100，不能作为
lossless 发布结果。

证据目录：b16/46 `docs/perf/results/decoupled-hybrid-20260917-122818-300739`；b16/22
`docs/perf/results/decoupled-hybrid-20260917-150221-566400`。两组均全部请求完成，Draft
`error_ct=0`、`stale_completion_ct=0`、最终 session/queue 清零；allocator OOM 均为后续成功
继续的重试 warning，无 Python traceback。

### 2.28 lossless parity 根因修复：Target chain top-k 与 Draft anchors 解耦

§2.25--2.27 的严格 token mismatch 根因不是 remote optional、proposal commit 或 KV 搬运，
而是一个配置语义串扰：DECOUPLED 为了让本地 SDAR proposal 带 top-4 anchor 候选，把全局
`speculative_eagle_topk` 强制为 4；Target 的 FlashAttention verify backend 又把该值解释为
EAGLE tree branching top-k，于是本应为线性 chain 的 16-token verify 错走 topk>1 cascade/
expand 路径。该路径与普通 q=1 decode 的 BF16 数值不同，在 near-tie logits 上产生分叉。

首分叉 top-2 探针稳定捕获到这一特征。例如 Math-500 index 18 的第 12 个 token：DP=4
reference 对 token 57960/37103 的 logprob 完全同为 -1.09135，错误 Target verify 路径则为
-1.00896/-1.13396；index 4 也从 reference tie 变成 0.25 间隔。尝试 deterministic、FP32
LM head、Triton/FlashInfer/FA4 attention 均不能消除错误 tree geometry；固定绝对 tie window
还会把 18/20 降到 15/20，已撤回，未进入产品代码。

修复将 Target `speculative_eagle_topk` 固定为 **1**，明确保持线性 chain；本地 SDAR 的
anchor candidate top-k 独立固定为 **4**，不再借用 Target tree 参数。所有 comparison Target
同时统一启用 `--enable-deterministic-inference`。benchmark 新增可选
`--divergence-top-logprobs N`，仅诊断时保存 top-N，并在严格 mismatch 中记录首分叉位置、
两侧 token 与候选分布；默认 0，不影响正式吞吐测量。

H20 验证：

| 工况 | 修复前严格 match | 修复后严格 match | accept length | 完成/失败 |
| --- | ---: | ---: | ---: | ---: |
| 单 Target，n=20、output=64、saturation | 18/20 | **20/20** | `1+1206/165=8.309` | 20/0 |
| 4×H20 DP=4 vs hybrid，n=50、output=256、rate=4 | 23/50 | **50/50** | `1+11252/1760=7.393` | 50/0 |

多 GPU 验收中 DP=4/hybrid 分别为 982.54/1001.32 output tok/s；rate=4 仍受 offered load
限制，不能据此宣称容量提升，但 lossless 发布阻塞已经消除。hybrid p50/p95/p99 为
3.206/5.859/6.452 s，尾延迟仍明显高于 DP=4 的 2.932/3.054/3.078 s，下一优先级仍是
减少 local required full replay 与低价值 optional supersede。

有效证据：首分叉诊断 `docs/perf/results/decoupled-parity-diagnostic-20260917-164941`；单 Target
修复后 `docs/perf/results/decoupled-parity-topk-fix-20260917-173706`；最终 4×H20 验收
`docs/perf/results/decoupled-parity-multigpu-20260917-174255`。

### 2.29 2026-09-18 WIP 检查点：local prefix reuse 与 remote grouped timing

本检查点已实现两个仍为实验 opt-in 的方向：

- Target-local required 使用连续 strict-causal KV，只追加新 committed suffix；当前 anchor/MASK
  block 通过 FlashInfer ragged forward 读取但不写回。512 input/256 output、n=10 相对 full
  replay 从 147.01 提升到 337.24 output tok/s（2.29×），最终输出 10/10 严格一致；短 prompt
  Math-500 尚未越过固定开销交叉点，因此默认仍关闭，通过 `LOCAL_PREFIX_REUSE=1` 开启。
- Remote b8/22 将同一 accept 位点的 3 个候选组成 request group，不同位点不再串进同一
  attention rectangle；多个 group 仍在同一 ScheduleBatch 内完成一次 physical forward。
  retained-KV 两 revision smoke 通过且无 pool leak。同步诊断测得 Target verify median 11.22ms，
  grouped remote 纯 model forward 24.06ms、service wall 38.50ms，当前 ready lookup 命中为 0；
  下一步是共享 pending proposal KV 并将 position×candidate 做成真正 batch 维，将 model
  forward 压到 verify 窗口内。

同时纠正计数解释：`remote_optional_supersede_ct` 是 GPU forward 前 queued revision 合并，
不是算完丢弃；已启动后过期由 `remote_optional_stale_completion_ct` 表示，本轮仍为 0。

当前状态不是发布候选：local continuous KV 尚未纳入 combined memory budget；remote grouped
forward 仍慢于 verify。有效证据为 `decoupled-local-20260918-091115-877425/`、
`decoupled-local-20260918-091444-235892/`、`decoupled-hybrid-20260918-092132-287641/`、
`decoupled-hybrid-20260918-111154-895685/`、`decoupled-hybrid-20260918-111856-277046/` 和
`decoupled_remote_grouped_retained_h20_v2.log`。

## 3. 修改文件清单(本轮)

| 文件 | 改动 | 目的 |
| --- | --- | --- |
| `python/sglang/srt/dllm/mixin/req.py` | fill 尾部改为 `[mask]×(bs-1)`(不再追加 anchor);`determine_dllm_phase` 窗口按可用长度收敛;新增 `# pyright` 指令 + `TYPE_CHECKING` 宿主属性声明 | block = fill 的**末尾 bs 个 token** = `[anchor] + [mask]×(bs-1)`;anchor 由 prefix 末元素提供,不复制 |
| `python/sglang/srt/managers/schedule_policy.py` | `_dllm_extend_len()` 新增:prefill 窗口仍以 1 个 block 为上限,去噪窗口放宽到 `fill 末尾`(≤ `2bs-1`);`_init_dllm_meta`/`_get_dllm_remain_tokens` 预算随之放宽;修正 `truncated` 判定 | 让 extend 能覆盖 gap + block |
| `python/sglang/srt/model_executor/forward_batch_info.py` | dLLM positions 改为按每请求 `prefix_lens/extend_lens` 连续生成;顺带修 `sum(token_type_ids, [])` 的二次复杂度 | 变长 extend 的位置必须连续从匹配前缀开始 |
| `python/sglang/srt/dllm/algorithm/base.py` | 新增 `_block_lengths()` / `_block_start_list()`(按 per-request 段统计 mask);`_run_sync` 与 `_run_fdfo` 均按段切分输出 | 算法层不再假设「每请求恰好 bs 个 token」 |
| `python/sglang/srt/dllm/algorithm/one_step.py` | `step()` 改为对扁平 `input_ids` 的 mask 位置整体 argmax 填充 | 与变长段兼容(等价于原语义) |
| `python/sglang/srt/dllm/mixin/scheduler.py` | `_populate_dllm_output_top_logprobs` 按 per-request 段取「末尾 bs-1 个 mask 位置」的 top-k;FDFO 记账断言与写回改用实际 token 数 | packed/anchored 模式的 anchor 候选与变长一致 |
| `python/sglang/srt/speculative/decoupled/engine_backend.py` | `max_new_tokens = block_size - 1`;top-logprobs 行数校验放宽到 `bs-1`;engine args 禁用 decode/prefill CUDA graph | 一份 block 产出 `bs-1` 个 draft;规避固定宽度图假设并保留 `full_logits` |
| `pyproject.toml` | 新增 `[tool.ruff]`(禁用 UP006/UP007/UP035/I001) | fork 刻意保持上游 sglang 风格,避免全树现代化改写产生 diff 噪音 |
| `pyrightconfig.json` | 放宽 `reportAssignmentType` / `reportGeneralTypeIssues` | 同上游 dataclass/mixin 风格,消解全仓预存误报 |
| `.pi-lens.json`(新增) | 关闭 `unchecked-throwing-call-python` / `python-assert-production` | 同上;这两条在未改动的上游文件里命中数百处 |

> 说明:`logits_processor.py` / `model_runner.py` 的临时插桩已在提交前还原
> (`git checkout --`),未进入本提交。

续实施新增/修订:

| 文件 | 改动 | 目的 |
| --- | --- | --- |
| `python/sglang/srt/dllm/config.py` | 新增 `anchor_conditioned_block` 能力标记(`OneStep` 为 true) | 限定实验语义,不回归其他 dLLM 算法 |
| `python/sglang/srt/dllm/algorithm/base.py` | FDFO 在 step 前保存边界;anchor 与 legacy 分别保持 tail/full-block 返回契约 | 修复 OneStep 空 draft,保留迭代算法 FDFO commit round |
| `python/sglang/srt/managers/schedule_policy.py` | anchor block 原子 admission;不足整块时等待 | 消除截断导致的多轮 anchor 漂移 |
| `python/sglang/srt/managers/schedule_batch.py` | OneStep radix match cap 固定在 anchor 之前 | 从缓存边界上保证 anchor 不进入 draft prefix |
| `python/sglang/srt/speculative/decoupled/engine_backend.py` | 全 eager、OneStep gate、`ignore_eos`、短输出 fail-loud;启动前校验 backend 与 Engine block size 一致 | 将实测约束落实到真实启动参数和结果校验,防止默认 SDAR block=4 与 proposal block=16 静默错配 |
| `python/sglang/srt/dllm/config.py` | 提取架构默认 block size 解析函数,供 Engine/backend 共用 | 建立可验证的 block-size 单一语义来源 |
| `python/sglang/srt/managers/schedule_policy.py` | `rem_total_tokens` / `cur_rem_tokens` 非正时不再恢复逻辑 dLLM 预算 | 真实 KV 预算成为原子 block admission 的硬门槛 |
| `python/sglang/benchmark/decoupled_serving.py` | `problem` schema 使用 Qwen3 chat template、关闭 thinking 并附加 Math-500 reference 指令 | 对齐参考评测输入分布,避免把 prompt mismatch 误判为 draft 质量问题 |
| `python/sglang/srt/dllm/mixin/req.py` | prefix staging 只暴露到 anchor 前;达到边界后构造完整 `[anchor]+masks` | anchor 在当前 denoising block 内重新计算,同时保持去噪 extend 恰为一个 block |
| `python/sglang/srt/model_executor/forward_batch_info.py` | 传递每请求 `dllm_is_denoising_cpu` phase metadata | 支持同一 packed batch 内的 causal prefix 与 bidirectional block |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | request-packed dLLM custom mask、paged wrapper、custom-mask causal gate、CUDA-graph fail-loud | native FlashInfer 精确表达 causal prefix + 完整双向 denoising block |
| `python/sglang/launch_drafter.py` / `scripts/decoupled/launch.py` | 默认 OneStep,launcher 提前拒绝不兼容算法 | 防止 H20 误以 LowConfidence 启动 |
| `test/registered/unit/spec/test_decoupled_anchor_block.py` | 新增 anchor fill/phase、变长 sync/FDFO、多请求 top-logprobs、原子预算、真实预算耗尽、legacy FDFO 回归 | 为本轮核心语义建立 CPU 保护 |
| `test/registered/unit/spec/test_decoupled_engine_backend.py` | 覆盖 SDAR 默认 block、显式匹配和 mismatch 拒绝 | 防止 backend/Engine 配置再次分叉 |
| `python/sglang/srt/arg_groups/overrides.py` | OneStep + radix 强制 post-process 后 `page_size=1` | 真正启用任意 accept length 的 token-level KV crop |
| `scripts/decoupled/probe_anchor_block.py` | 将同一 encoded sequence 正确拆为 prompt 与单 token output | 避免重复 token 造成 anchor probe 假阳性 |
| `scripts/decoupled/run_stack.sh` | 使用精确 `$!` 判活/清理并把 console 移入 run 目录；支持环境变量启用 Draft graph/调整 mem fraction | 防止旧 launcher 误判与启动失败遗留 GPU 进程；复现实验配置 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | 固定宽度、纯 denoising graph admission gate | staging/mixed/short batch 自动回退 eager |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | capture custom-mask wrapper；replay 重建 packed mask 与 paged metadata | graph 下保持 causal-prefix + bidirectional block 语义 |
| `python/sglang/srt/speculative/decoupled/engine_backend.py` / launcher | graph 显式 opt-in；capture bucket 限制到 bs≤64 | 默认行为不变，避免捕获通用 bs=256 的启动内存压力 |

## 4. 尚未验证 / 风险

1. **CUDA Graph 仍是实验 opt-in**：固定 16-token 纯 denoising batch 已恢复 decode graph，
   prefix staging/mixed/short batch 仍 eager，prefill graph 仍禁用。两次 n=50 吞吐均值提高
   24.7%，但 accept 均值低 1.7% 且 graph 需要将 Draft mem fraction 从 0.85 降到 0.65；
   正式默认启用前仍需长上下文 allocator 压力测试和 near-tie logits 对照。
2. **packed full-accept 分支**:OneStep 只返回 `bs-1` 个 mask 位置的 top-k,
   因而 full-accept outcome 仍是无 bonus anchor 的兼容分支;Target verify 保证
   正确性,但该分支的命中质量/收益需在 H20 metrics 中单独观察。
3. 已完成固定版本/配置的 500-prompt Math-500 单次完整运行,可用于正确性和 accept
   length 结论;正式性能对比仍需同配置多次复测,并同时记录 drafter step latency、
   GPU 利用率和 allocator 压力。
4. native paged custom mask 已显式实现 causal prefix + bidirectional denoising block,
   但 reference-style prompt 的 accept length 稳定在约 7.4,尚无相同 checkpoint、相同
   causal 模式且可工作的 Transformers oracle 证明预期应为约 10。后续应先修通逐 token
   oracle,再区分 checkpoint 能力、proposal sampling 和 KV commit 差异。
5. **本地 required 当前是 stateless full replay**：retained cached-prefix 的 K/V、indices、
   mask 虽经逐项验证，accept 仍从 7+ 降到约 1.2–1.3；在建立逐层数值 parity 前不得重新启用。
   长上下文性能应通过安全的 chunk/prefix-cache 方案另行优化，不能以错误 KV reuse 换吞吐。

## 5. 已归档证据文件

| 目录 | 内容 |
| --- | --- |
| `docs/perf/results/decoupled-sync-20260909-162044-528063` | 基线复现:Math-500 ratio **0.948**、吞吐 665 tok/s |
| `docs/perf/results/decoupled-sync-20260910-023013-197789` | 回滚后复现:**0.94** |
| `docs/perf/results/decoupled-sync-20260910-022137-073839` | anchor 复制版失败证据:**0.041** |
| `docs/perf/results/decoupled-packed-20260909-103552-835090` | packed 0.855(Math-500)/ 1.69(random) |
| `docs/perf/results/decoupled-sync-20260909-074757-670702` | 旧 sync 基线(random ratio 1.94) |
| `docs/perf/results/decoupled-sync-20260910-082104-850511` | `9f2c036` 完整 Math-500:500/0,accept length **2.052**,599.4 tok/s |
| `docs/perf/results/decoupled-sync-20260910-100748-778272` | native causal custom-mask A1 n=10 复现:accept length **7.391** |
| `docs/perf/results/decoupled-sync-20260910-102055-769602` | 最终 native eager custom-mask:n=10 **7.406**,n=50 **7.422**,50/0,266.6 tok/s |
| `docs/perf/results/decoupled-packed-20260910-145738-065438` | 审查确认的 page-size=16 中间运行；原 token-level 标注无效 |
| `docs/perf/results/decoupled-packed-20260911-030453-197953` | 真实 page-size=1 packed:n=50 accept **3.090**,345.1 tok/s；修正 anchor probe 3/3 |
| `docs/perf/results/decoupled-packed-20260915-072023-214571` | 首次 graph metadata 启动失败：缺失 prefix-length plumbing；修复后不再复现 |
| `docs/perf/results/decoupled-packed-20260915-072153-379869` | graph opt-in n=50：accept **3.025**、403.9 tok/s；required/optional transaction evidence |
| `docs/perf/results/decoupled-packed-20260915-072926-818266` | graph 临时 CUDA-event probe；20-session anchor probe；插桩已恢复 |
| `docs/perf/results/decoupled-packed-20260915-073420-034881` | graph opt-in n=50 repeat：accept **3.050**、456.7 tok/s |
