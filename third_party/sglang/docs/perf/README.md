# 性能实验归档

本目录保存当前 H800 Hybrid 实验，以及 H20 和 H800 的历史基线材料。运行约束、数据有效性判据见
仓库根目录 [CLAUDE.md](../../CLAUDE.md)（[AGENTS.md](../../AGENTS.md) 为其软链接）。
本目录及全部子目录不属于 Mintlify 站点，`docs/AGENTS.md` 的站点指南不适用。

| 目录 | 内容 | 示例 |
|---|---|---|
| [logs/](logs/) | 独立的 benchmark、server、runner、smoke、microbench 日志 | `bench_one_batch_h20_qwen35_9b_baseline.log` |
| [results/](results/) | 结构化输出、SPS / STS 校准产物、整套实验目录 | `dspark_sps_g7.json`、`decoupled-hybrid-<时间>/` |
| [scripts/](scripts/) | 性能专用运行、指标提取、容量模拟脚本 | `run_qwen38_27b_h20.sh`、`extract_bench_metrics.py` |
| [reports/](reports/) | 正式基线、对比、容量及 profiling 分析报告 | `qwen35_2b_4b_9b_baseline_v0.5.18.md` |
| [notes/](notes/) | 阶段进展、开发记录和接入规划 | `current_progress_20260923.md`、`decoupled_anchor_block_progress.md` |
| [configs/](configs/) | 手工维护的实验配置 | `adaptive_mtp_config.json` |
| [datasets/](datasets/) | 性能实验专用输入数据 | `math500-adapted.jsonl` |
| [profiles/](profiles/) | profiler trace，保留各批次的内部目录结构 | `decoupled_draft_b16_fp8_profile_20260924/` |

## 常用入口

- **当前 Hybrid 结论**：[双 H800 Qwen3-8B，本地 Graph、Math-500 / GSM8K（2026-10-13）](reports/h800_qwen3_8b_hybrid_20261013.md)
- **当前 Hybrid 用法和提交范围**：[Hybrid 运行指南](../../scripts/decoupled/README.md)
- **可运行配置**：[B16/K2、full accept ON、本地 Graph 1024/1](configs/h800_hybrid_local_graph_full_20261013.json)
- **原始证据**：[固定 512 token](results/h800-math-graph-20261013/) / [自然 EOS，上限 2048](results/h800-math-natural-eos-20261013/)

以上是本次 Hybrid PR 的性能结论入口；以下保留为历史参考，不合并口径：

- [Qwen3.5 2B / 4B / 9B 历史基线汇总](reports/qwen35_2b_4b_9b_baseline_v0.5.18.md)
- [H20 Qwen3.5-9B 容量估算](reports/h20_qwen35_9b_sizing.md)
- [H20 Qwen3.8-27B MTP 对比](reports/h20_qwen38_27b_mtp_comparison.md)
- [当前进展快照（2026-09-23）](notes/current_progress_20260923.md)
- [H20 hybrid block=4 / top-k=4 简测（2026-10-09）](reports/h20_qwen3_8b_hybrid_b4_topk4_20261009.md)
- [H20 hybrid 单请求 block / top-k 逐点搜索，及时到达率 ≥95%（2026-10-09）](reports/h20_qwen3_8b_hybrid_serial_block_topk_search_20261009.md)
- [H20 Qwen3-8B hybrid 远端 / 全模型 FP8 对照、缓存保真与完整答案小测（2026-10-10）](reports/h20_qwen3_8b_hybrid_serial_fp8_comparison_20261010.md)
- [Decoupled anchor-block 开发记录](notes/decoupled_anchor_block_progress.md)

## 新产物存放规则

从仓库根目录执行命令，先创建所需输出目录。顶层只保留本说明文件，新增产物按上表归类。
独立日志写入 `logs/`，结构化输出写入 `results/`，必须显式指定输出路径：

```bash
mkdir -p docs/perf/logs docs/perf/results
# serving 压测命令添加：
#   --output-file docs/perf/results/<名称>.jsonl
# one_batch 命令添加：
#   --result-filename docs/perf/results/<名称>.jsonl
# 终端输出统一通过：
#   2>&1 | tee docs/perf/logs/<名称>.log

# 从归档日志提取指标（输出到终端）：
python3 docs/perf/scripts/extract_bench_metrics.py docs/perf/logs/
```

`decoupled-*` 等运行目录整体放入 `results/<运行名>/`，其中 `run.json`、worker 日志、
console 日志、结果和诊断文件保持在一起，方便核对同一次运行。通用 launcher 和工具仍在
仓库 `scripts/decoupled/`，不复制到本目录。profiling trace 存放在 `profiles/<运行名>/`，
对应的运行日志在 `logs/`、分析报告在 `reports/`。手工配置和输入数据分别放入
`configs/` 和 `datasets/`；运行生成的 SPS / STS 表属于 `results/`。

沿用原文件名与版本后缀，重跑同一工况递增 `_vN`，H20 日志包含 `h20_` 标识。
失败、被污染和中间运行也保留原始证据，在报告或阶段记录中说明有效性。
项目全局 `.gitignore` 会忽略 `.log` / `.jsonl`；归档新证据时使用
`git add -f <具体文件>`，不要误加缓存或临时文件。

## 目录迁移说明

2026-10-08 按用途整理原顶层文件与运行目录，原文件名及整套运行目录内部结构保持不变。
日志、JSON / JSONL、输入数据和压缩 trace 的原始内容不改写。维护中的文档链接、
脚本路径与默认输出位置已更新；原始 `run.json` 和日志中记录的旧路径代表当时的运行位置，
查找证据时按上表映射到新目录。
