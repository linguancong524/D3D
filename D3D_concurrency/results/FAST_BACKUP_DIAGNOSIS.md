# Fast Backup 高并发诊断

## 结论

实现已跑通，并完成独立 greedy 校验。当前不能认定 Fast 在 SDAR-1.7B ->
Qwen3-8B 上稳定优于 Neural。anchor=3、C=32、128-token 子集测试中，新版
Fast 比配对 Neural 快 6.1%；anchor=1、256-token 测试中，两种新版 Fast
均更慢。保留 Neural 默认值，Fast 保留为可配置实验路径。

这不是对原论文结论的否定，也不是 2048-token 全数据集的最终结论。

## 与 SSD 的对应

参考论文 https://arxiv.org/html/2603.03251 的 4.3 节，以及 SSD commit
`d7eb8fa0edb77a6d0876af1903367b9bb82f54e7`：

- `ssd/engine/draft_runner.py:186` 的 `hit_cache_and_respond` 用随机候选填充 miss。
- `draft_runner.py:119` 及后续树构造支持按 hit/miss 使用不同 fanout。
- `ssd/config.py` 要求 hit/miss fanout 的总预算相同。
- SSD 使用 AR glue-decode 产生后续候选；SDAR 的非因果块预测不能直接替代
  这个 AR kernel。本实现仍需 masked anchor seed，这不是完整 SSD 复现。

## 实现改动

`engine.py` 保留三种策略：

- `neural`：同步补算，默认。
- `fast_legacy`：旧版随机 suffix + masked seed + 全接受位置候选预测。
- `fast`：随机 suffix + masked seed；miss 只预测接受首 token 后的候选，
  hit/miss 按不同查询长度分别组成微批。偶然接受更多随机 token 时，缺少
  预测只会再次 backup，不会绕过 target 校验。

新增 `--anchor-num` 和 `--fast-miss-anchors`。后者为 0 时等于 anchor 数：

| 正常 anchor | Fast miss anchors | 正常查询位置/请求 | miss packed 查询位置/请求 |
|---:|---:|---:|---:|
| 3 | 3 | 752 | 64 |
| 1 | 1 | 272 | 32 |
| 1 | 16 | 272 | 272 |

miss 还需一次 16-position masked seed，上表不包含这次前向。
第三行把正常 16 个分支的预算全部集中在首个接受位置；它是 SDAR 对 miss
fanout 重分配的适配，并不声称使用了论文实验的相同超参数。

## 配对吞吐

Math500 前 32 题，greedy、自然 EOS、两张 H800、draft microbatch=4。
同一组模型常驻，策略顺序交替。TPS 包含 prefill、decode、收尾，排除加载、
预热、分词和写文件。它不是此前讨论的最终 decode-only/AR speedup 表。
每行 Fast/Neural 使用该组自己的 Neural 分母，不跨组比较绝对 TPS。

| Anchor | 输出上限 | C | Fast miss anchors | 重复 | Neural TPS | Fast TPS | Fast/Neural |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 128 | 16 | 3 | 2 | 303.9 | 228.4 | 0.751 |
| 3 | 128 | 32 | 3 | 2 | 294.6 | 312.5 | 1.061 |
| 1 | 256 | 16 | 1 | 3 | 509.4 | 422.8 | 0.830 |
| 1 | 256 | 32 | 1 | 3 | 572.1 | 508.8 | 0.889 |
| 1 | 256 | 32 | 16 | 3 | 662.0 | 541.6 | 0.818 |

anchor=3/C=16 的 Fast 两轮为 294.8、186.4，波动很大，不能用其中一轮下结论。
不同组 Neural 也有明显速度漂移，因此不把组间差异归因于 fanout。
全部原始数据和两轮旧 Fast 对照见 `fast_v2_table.md` 及对应 JSON。

## 原因量化

1. **接受长度损失抵消了省下的耗时。** anchor=1/C=32/256、稀疏 Fast 中，
   有效 token/request-step 从 6.90 降到 5.46（约 -20.8%）；平均
   wall ms/request-step 从 12.05 降到 10.73（约 -11.0%）。两者相除即
   TPS 比 0.889。这是用整次运行总耗时分摊的指标，不是单次 GPU kernel 延迟。
2. **增大 miss fanout 不能免费恢复接受长度。** 集中预算组中，Neural 与
   Fast 都约 10.42 ms/request-step，但有效输出分别为 6.90、5.65，
   所以 Fast 慢 18.2%。Fast 每轮 1428 个 request-steps，Neural 为 1169。
   miss 数下降不等于吞吐上升。
3. **当前主要受 draft 端限制。** 独立同步诊断在 anchor=3/C=32/128 下，
   Neural 总耗时 13.81s，其中候选预测 12.39s（89.7%）、recovery 0.96s
   （含 admission bootstrap）、target 暴露等待 0.14s。旧 Fast 的预测
   15.22s，recovery 0.16s：同步补算减少，但更多预测工作仍压在 draft 上。
4. **更少查询不一定更少耗时。** 同一诊断中，新 Fast 相对 Neural 的
   draft query 数从 468256 降到 448896（-4.1%），但前向调用从 210 增到
   250（+19.0%），调度轮从 36 增到 44。分开 hit/miss 微批与额外 seed
   会增加小批次调用。这里只证明存在额外调用，不能把全部差距归因于某个 kernel。

同步诊断主动加入 CUDA synchronization，且只测一轮，不能当作吞吐排序；
其 `draft_forward_s` 嵌套在 prediction/recovery 中，不能再次相加。
原始记录为 `fast_v2_profile.json`，脚本为 `profile_backup.py`。

## 验证与复现

稀疏版通过独立 HF greedy 和原 SDAR 回归；集中预算版 4/4 短序列与独立
HF greedy 一致，验证中实际触发了 miss。单元测试覆盖 hit/miss 混合批次、
集中 fanout、随机 token 必须按序验证以及首 token 保持 target anchor。
测量审计检查并发上限、token 计数、输入哈希和 target KV 全部释放；draft
页泄漏由引擎内断言检查。这不是完整数据集的准确率评估，BF16 跨路径输出
也不保证逐 token 完全一致。

从 D3D_concurrency 目录启动；每次使用新的 output，避免覆盖：

```bash
bash scripts/compare_backup.sh --datasets math500 --max-samples 32 \
  --concurrency 32 --max-new-tokens 256 --repeats 3 \
  --anchor-num 1 --fast-miss-anchors 16 --validate \
  --output results/fast_wide_new.json

python report_fast_diagnosis.py results/fast_wide_new.json \
  --output results/fast_wide_new.md
```

省略 `--fast-miss-anchors 16` 即为稀疏版。常规启动器支持
`BACKUP=fast ANCHOR_NUM=1 FAST_MISS_ANCHORS=16`。本轮没有修改 D3DV2。
