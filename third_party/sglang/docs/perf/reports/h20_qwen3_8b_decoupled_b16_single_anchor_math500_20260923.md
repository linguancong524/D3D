# H20 Qwen3-8B decoupled：block 16、单备选、原始 Math-500（2026-09-23）

## 配置与数据

- 源码 commit：`ca378efe5352f4b6ae71e710b8255258a558981f`；4×H20，所有启动前与收尾均确认 GPU 无残留计算进程。
- 输入文件：`docs/scripts/test.jsonl`，500 条 `problem`，SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。benchmark 对 `problem` 使用 Qwen3 chat template、关闭 thinking，并附加逐步推理与 `\boxed{}` 指令。旧 `docs/perf/datasets/math500-adapted.jsonl` 使用纯文本编码，结果不能直接横比。
- 模型与拓扑：GPU0–2 各跑一个 Qwen3-8B TP=1 Target，`max_running_requests=3`；hybrid 另在 GPU3 跑共享 D3D-1.7B OneStep Draft。local-only 使用相同三个 Target，不启动远端 Draft。
- Proposal：`block_size=16`、`accept_counts=1..16`、`num_anchors=1`、`max_branches=16`，即前 15 个拒绝位点各 1 个备选，另有 full-accept 分支。其他设置与 [同日 random512 测试](h20_qwen3_8b_decoupled_b16_single_anchor_20260923.md) 相同：local prefix reuse、deterministic inference、packed causal pending、Draft CUDA Graph max tokens=1024、Draft mem fraction=0.65、speculation batch size=8。
- 使用 `sglang.benchmark.decoupled_serving`；temperature=0、`max_new_tokens=256`、`ignore_eos=True`、seed=42。hybrid 对每组相同输入的 local-only 结果做逐 token 匹配。这是部署速度/一致性测试，不计算 Math-500 答案正确率。

## 结果

| 工况 | 模式 | 完成/失败 | output tok/s | req/s | E2E p50/p95/p99 (s) | accept length | token match |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| n=50，rate=4 | local-only | 50/0 | 996.18 | 3.891 | 2.555 / 3.639 / 3.925 | 7.397 | reference |
| n=50，rate=4 | hybrid | 50/0 | 995.28 | 3.888 | 2.562 / 3.670 / 3.935 | 7.397 | 50/50 |
| n=100，饱和 | local-only | 100/0 | 1114.32 | 4.353 | 11.814 / 21.213 / 22.616 | 7.384 | reference |
| n=100，饱和 | hybrid | 100/0 | 1090.79 | 4.261 | 12.044 / 21.472 / 23.148 | 7.384 | 100/100 |
| n=500，饱和 | local-only | 500/0 | 1144.05 | 4.469 | 53.628 / 102.006 / 107.495 | 7.499 | reference |
| n=500，饱和 | hybrid | 500/0 | 1137.25 | 4.442 | 54.403 / 103.769 / 108.179 | 7.499 | 500/500 |

`accept length = 1 + spec_num_correct_drafts / spec_verify_ct`。hybrid 相对 local-only 的 output tok/s 差值依次为 **-0.09%**、**-2.11%**、**-0.59%**；每组的 Target verify/correct-draft 计数都完全一致。饱和测试一次性提交所有请求，其 E2E p95 主要反映排队，不是无排队在线时延。

远端 Draft 累计 3629 个 batch item、58064 条分支，恰为 `3629×16`；800 次 physical forward，`error_ct=0`，最终 `active_sessions=0` 且队列清零。三张 Target 在三组测试及预热中累计 **0 次** `remote_optional_ready_lookup_ct`、**0 次** `remote_optional_lookup_hit_ct`。因此远端确实做了预投机计算，但没有及时供 Target 消费；目前的速度主要是 local fallback 的表现，单备选配置没有显示出吞吐收益。

六份 benchmark 均 `summary.valid=true`，无失败请求；hybrid 三组逐 token 对照全匹配。服务日志没有未捕获 Python traceback 或致命 CUDA OOM。每个工况只运行一次，约 1–2% 的差异不宜归因于稳定回退；但零远端命中足以说明这轮没有展示出远端备选的加速。

## 证据

- local-only：[`decoupled-local-20260923-102157-191685/`](../results/decoupled-local-20260923-102157-191685/)（`run.json`、三组 benchmark JSON/日志、Target 日志）。
- hybrid：[`decoupled-hybrid-20260923-102802-781106/`](../results/decoupled-hybrid-20260923-102802-781106/)（同上，另有 Draft 日志）。
