# 当前 Hybrid 运行指南（H800，2026-10-13）

当前提交只整理双卡 Hybrid 本地 packed CUDA Graph、必要修复、回归测试，
以及刚完成的 Math-500/GSM8K 性能对照。权威结论见
[正式报告](../../docs/perf/reports/h800_qwen3_8b_hybrid_20261013.md)。
旧 FP8 扫描、profiling、早期试跑和其他模型不属于本次结果包；没有删除它们。

## 拓扑与实现

- GPU0：Qwen3-8B BF16 Target（TP1）+ SDAR FP16 本地 required draft。
- GPU1：SDAR FP16 远端 optional draft，用 packed Graph 异步预计算后续分支。
- 远端命中（包括 full accept）跳过本地调用；没有有效结果时回退本地 full replay。
- 本地图独立持有输出 buffer，使用分桶/padding 和显式 mask，不改变 Target 图地址或全局 dLLM 配置。
- 保留前缀因果、draft block 双向语义；没有启用本地 KV 前缀复用。
- 本次还包含 retained prefix indices 转为 int64 的 ABI 修复，以及本地调用/远端 full accept 指标。

## 已测配置启动

先按本机模型位置替换两个路径；下列为本次 H800 配置，不是通用默认配置。
环境安装见 [SETUP.md](../../SETUP.md)。启动前须确认两张卡均空闲。

```bash
nvidia-smi
.venv/bin/python3 scripts/decoupled/launch.py \
  --mode hybrid --num-targets 1 --gpus 0 1 \
  --target-model /path/to/Qwen3-8B --draft-model /path/to/SDAR-checkpoint \
  --dllm-algorithm-config scripts/decoupled/sdar_block16.yaml \
  --block-size 16 --candidate-topk 2 --num-anchors 1 --max-branches 16 \
  --accept-counts 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --draft-dtype float16 --draft-attention-backend flashinfer \
  --draft-enable-cuda-graph --packed-single-row --batched-branch-topk \
  --packed-causal-pending --batching-window-ms 0 \
  --lookup-wait-ms 12 --wait-for-covered-result \
  --local-cuda-graph-max-tokens 1024 --local-cuda-graph-max-requests 1 \
  --context-length 8192 --max-running-requests 16 \
  --target-mem-fraction-static 0.68 --target-chunked-prefill-size 2048 \
  --draft-mem-fraction-static 0.85 --disable-prefill-cuda-graph \
  --trust-remote-code --startup-timeout 900
```

可先加 `--dry-run` 检查命令。精确实验 overrides 为
[配置 JSON](../../docs/perf/configs/h800_hybrid_local_graph_full_20261013.json)。
性能脚本使用 `h800_hybrid_grid_defaults.json` + overrides；直接启动时默认值未必相同。

| Launcher 参数 | Target 参数 | 作用 |
|---|---|---|
| `--local-cuda-graph-max-tokens` | `--decoupled-local-cuda-graph-max-tokens` | 0=默认 eager；本次为 1024 |
| `--local-cuda-graph-max-requests` | `--decoupled-local-cuda-graph-max-requests` | 本次为 1；越界回退 |
| `--local-cuda-graph-check` | `--decoupled-local-cuda-graph-check` | 额外 eager 对照，仅诊断，禁止用于测速 |

容量按 packed full replay token 总量判断，不仅是新增 token 数。
自然 EOS Math-500 组有 874 次 eager 回退，已计入速度；GSM8K 没有回退。
本次 1024-token 图额外显存约 1.21 GiB。未验证扩大容量或高并发的收益。

## 结果与复现入口

| 文件/目录 | 用途 |
|---|---|
| `docs/perf/scripts/bench_h800_math_graph.py` | 固定随机抽样，AR/eager/graph 各 3 轮，固定输出 512 |
| `docs/perf/scripts/bench_h800_math_natural_eos.py` | 相同样本，遇 EOS 停止，上限 2048，按实际长度计速 |
| `docs/perf/scripts/analyze_h800_math_graph.py` | 独立核对样本、输出、计数、速度和被测源码哈希 |
| `docs/perf/results/h800-math-graph-20261013/` | 固定长度的原始输入、命令、逐请求结果、日志、telemetry |
| `docs/perf/results/h800-math-natural-eos-20261013/` | 自然 EOS 的对应证据及结束原因 |

复跑需要本机模型/数据路径，先检查 `run_h800_hybrid_vs_ar.py` 中的 `TARGET` / `DRAFT`
及 `bench_h800_math_graph.py` 的 `MATH` 路径。GSM8K 来源和 SHA256 已归档。
所有运行使用**新目录**，不能覆盖已有结果；两套测试不得并行占用同一 GPU。
完整 GPU 命令在正式报告中。

只复验已有证据（无需 GPU 或本机模型，输出名必须不存在）：

```bash
python3 docs/perf/scripts/analyze_h800_math_graph.py \
  docs/perf/results/h800-math-graph-20261013 \
  --output-file docs/perf/results/h800-math-graph-20261013/recheck.json
python3 docs/perf/scripts/analyze_h800_math_graph.py \
  docs/perf/results/h800-math-natural-eos-20261013 --natural-eos \
  --output-file docs/perf/results/h800-math-natural-eos-20261013/recheck.json
```

哈希不匹配会拒绝将后续代码误标为已测版本；应回到本次提交复核，不能改写原始哈希。
运行日志与结果均保留原路径，方便复算，不把终端日志当作唯一证据。
逐请求结果和 AR 参考以 `.json.gz` 无损归档，脚本透明读取；明文 JSON 的原始字节未改动。
EOS 元数据随实验目录归档，压缩前后哈希见
[归档索引](../../docs/perf/results/h800-hybrid-20261013-artifact-index.json)。

## 提交与合并范围

只通过 feature branch 提 PR 到 `main`，不自动合并、不推 `release`。
`README.md` → 本指南 → 正式报告 → 原始证据构成当前入口。
根目录 `AGENTS.md` 是 `CLAUDE.md` 的软链接，保持单一工作规则来源。
