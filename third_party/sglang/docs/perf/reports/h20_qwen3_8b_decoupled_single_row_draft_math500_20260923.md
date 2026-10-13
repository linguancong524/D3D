# H20 单行 packed Draft：b8、单 anchor、Math-500（2026-09-23）

## 实现与参考

参考 [D3D_concurrency/engine.py](https://github.com/linguancong524/D3D/blob/4314deac274784fbbedae762a7d20d7cd29d64b5/D3D_concurrency/engine.py) 的每请求一行 packed 分支布局：pending 只计算一次，后续分支通过 attention mask 读取各自接受的 pending 前缀及自身 scratch。SGLang 原有 `build_packed_layout` 已支持这个 mask；本次接通原有但未使用的 `dllm_packed_single_row` 配置，并以 `--packed-single-row` 保持默认行为不变。

`gather_shared_pending_kv` 目前不接入：它只能搬运每层的 K/V，不能消除重复 pending 的 Q/MLP/投影计算；还需改动 FlashInfer 的 KV 输入路径。单行布局直接移除重复 token，并使用现有 KV 池和 custom mask。b8/单 anchor 的最小 query token 数由每请求 `8×(8+8)=128` 降为 `8+8×8=72`。

## 对照口径

4×H20，GPU0–2 为 Qwen3-8B TP1 Target，GPU3 为 D3D-1.7B OneStep Draft。`docs/scripts/test.jsonl` 原始 Math-500、`block_size=8`、`accept_counts=1..8`、`num_anchors=1`、`max_branches=8`、Draft CUDA Graph max tokens=1024、speculation batch=8、deterministic、packed causal pending；与 [上轮 b8 报告](h20_qwen3_8b_decoupled_b8_single_anchor_math500_20260923.md)同口径。每轮启动前检查 GPU 无残留计算进程。输出参照上轮相同 n 值的 local-only JSON 逐 token 对照。

| 工况 | 配置 | output tok/s | Draft GPU ms/forward | Draft 服务 ms/batch | 输出一致性 |
| --- | --- | ---: | ---: | ---: | --- |
| n=100，饱和 | 单行 | 851.57 | 约 11.53（含预热及此前 n50） | 约 15.45（同口径） | 100/100 |
| n=500，饱和 | 单行 | 882.14 | 12.08（全程累计） | 16.25（全程累计） | 500/500 |
| n=100，饱和 | 单行 + 批量 top-k | 838.75 | 约 11.34（含预热） | 约 13.81（同口径） | 100/100 |
| n=500，饱和 | 单行 + 批量 top-k | 876.35 | 12.03（全程累计） | 14.65（全程累计） | 500/500 |

旧版分组 b8 全量 hybrid 为 894.31 tok/s、Draft GPU 平均 33.94 ms/forward、服务 44.78 ms/batch；同配置 local-only 为 912.89 tok/s。单行后全量累计 9764 次 physical forward、30805 个 batch item、246440 个分支，GPU 前向 117.966 s，服务 158.711 s，`error_ct=0`、最终队列及 session 清零。三张 Target 累计 ready lookup 1668 次、hit 1102 次，比旧版的 309 次 hit 多，但仍只占约 30254 次 lookup 的 3.64%。

平均每次物理前向由约 33.94 ms 降至 12.08 ms；不过两轮平均 batch item 数不同（旧版约 6.28，新版约 3.15），不能把该倍率当作同 shape 的加速比。按 batch item 归一的 GPU 时间约由 5.40 降至 3.83 ms/item。Target verify 中位约 11 ms，单行服务仍约 16 ms/批，尚不能稳定在 verify 窗口内交付；端到端吞吐也尚未改善。批量 top-k 虽缩短小样本服务时间，但全量吞吐略低，暂不推荐启用。

## 证据与限制

- 单行默认 top-k：[实验目录](../results/decoupled-hybrid-20260923-113352-875025/)；[启动日志](../logs/decoupled_single_row_b8_launch_20260923.log)。
- 单行批量 top-k：[实验目录](../results/decoupled-hybrid-20260923-114430-870086/)；[启动日志](../logs/decoupled_single_row_batched_topk_b8_launch_20260923.log)。
- 布局 mask/位置等价性单元测试、启动器几何测试通过；真实推理 n50/n100/n500 单行模式分别为 50/50、100/100、500/500 逐 token 匹配。
- `remote_optional_branch_group_ct` 目前仍统计逻辑接受长度组，不代表单行模式的物理 packed 行数；物理行数由每批 item 数决定。
- 批量 top-k 的 n500 逐 token 比较后续复核为 500/500；本轮 b16 测试前后检查 GPU 均无残留计算进程。
