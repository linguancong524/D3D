# H20 decoupled Draft W8A8 INT8 初测（2026-09-24）

## 配置与量化方式

在 4×H20 开发机上，GPU0 运行 Qwen3-8B Target 与原本地兜底 Draft，GPU3 运行远端 D3D-1.7B Draft。只量化远端 Draft 的 28 层、每层 7 个 transformer linear 权重（共 196 个矩阵）：按输出通道对称 INT8，缩放因子为每行 `max(abs(weight))/127`；激活由 SGLang W8A8 INT8 算子动态量化。embedding、norm 和 vocabulary head 不量化。Target 与本地兜底 Draft 均保持原配置。

转换脚本为 [`quantize_sdar_w8a8_int8.py`](../../../scripts/decoupled/quantize_sdar_w8a8_int8.py)，本次命令：

```bash
.venv/bin/python3 scripts/decoupled/quantize_sdar_w8a8_int8.py \
  --source /workspace/models/d3d-1.7b-b16_for_qwen3-8b \
  --output /tmp/sglang-draft-int8-xRvZ0g
```

生成 checkpoint 约 3.63 GiB（权重总字节数 3,901,394,944），留在临时目录、未提交到仓库；重跑时需先重新生成。远端用 `--remote-draft-model /tmp/sglang-draft-int8-xRvZ0g --remote-draft-quantization w8a8_int8`，本地仍用原 `--draft-model`。SGLang trace 确认执行了 `_per_token_quant_int8` 和 CUTLASS INT8 GEMM，而非仅改变配置名。[转换日志](../logs/decoupled_draft_w8a8_int8_conversion_20260924.log)、[启动配置](../results/decoupled-hybrid-20260924-030820-930774/run.json)、[INT8 前向 trace](../profiles/decoupled_draft_b16_w8a8_int8_profile_20260924/)。

其余配置沿用 [1+1 实验](h20_qwen3_8b_decoupled_b16_1plus1_wait_math500_20260923.md)：`block_size=16`、`num_anchors=1`、16 个接受位点、单行 packed + causal pending、零等待、CUDA Graph max tokens=1024、Target max-running-requests=3、Math-500 本地 `docs/scripts/test.jsonl`、输出 256 token、忽略 EOS、temperature=0、seed=42。启动前 GPU 无残留进程，结束后 GPU 进程已清理；远端服务 `error_ct=0`，n100/n500 均无失败或致命 OOM。

## 固定形状前向

与 [bf16](h20_qwen3_8b_decoupled_draft_forward_profile_20260923.md)、[FP8](h20_qwen3_8b_decoupled_b16_fp8_math500_20260924.md) 使用相同约 512 token 前缀和 query 形状，以下为单次 CUDA event，不代表多轮稳定性：

| 合批请求数 | bf16 | FP8 | W8A8 INT8 |
| ---: | ---: | ---: | ---: |
| 1（272 query token） | 18.54 ms | 17.49 ms | 17.65 ms |
| 3（816 query token） | 36.75 ms | 30.06 ms | 29.94 ms |

INT8 与 FP8 基本持平，均快于 bf16，但仍无法被约 11 ms 的单次 Target verify 覆盖。[INT8 profiler 日志](../logs/decoupled_draft_b16_w8a8_int8_profile_20260924.log)。

## Math-500 在线结果

均为 1 Target + 1 远端 Draft 的饱和、零等待模式。前三行来自之前的归档运行，因此小幅差值可能含跨运行波动。

| 模式 | n100 tok/s | n500 tok/s | n500 verify | n500 逐 token 匹配 |
| --- | ---: | ---: | ---: | ---: |
| 本地兜底，无远端 | 398.37 | 405.41 | 17,356 | 参考 |
| 远端 bf16 | 368.99 | 383.05 | 17,665 | 500/500 |
| 远端 FP8 | 380.16 | 385.17 | 17,670 | 500/500 |
| 远端 W8A8 INT8 | 378.94 | 387.32 | 17,686 | 500/500 |

INT8 的 n500 比远端 bf16 高 1.12%，比 FP8 高 0.56%，但比 local-only 低 4.46%；n100 则略低于 FP8。这个差值不足以断言 INT8 优于 FP8。n500 服务计数按 n100 结束与 n500 结束之间的累计差分，约 6,072 次物理前向、2.91 请求/批、GPU 前向 25.23 ms/批、远端服务 33.43 ms/批；FP8 对应为 25.31/33.29 ms，二者也基本一致。远端结果命中约 395 次，占 17,686 次 verify 的 2.2%；整体接受统计 `spec_num_correct_drafts=112,462`，接近 FP8 的 112,495，但因多数轮次使用 bf16 本地兜底，不能据此单独评估 INT8 proposal 质量。

原始结果：[INT8 n100](../results/decoupled-hybrid-20260924-030820-930774/one_plus_one_b16_int8_math500_n100_saturation.json)、[INT8 n500](../results/decoupled-hybrid-20260924-030820-930774/one_plus_one_b16_int8_math500_n500_saturation.json)、[FP8 n500](../results/decoupled-hybrid-20260924-023534-803794/one_plus_one_b16_fp8_math500_n500_saturation.json)、[bf16 n500](../results/decoupled-hybrid-20260923-154457-327156/one_plus_one_b16_math500_n500_saturation.json)、[local-only n500](../results/decoupled-local-20260923-155721-447870/one_target_b16_math500_n500_saturation.json)。本次 target/drafter 原始日志见 [运行目录](../results/decoupled-hybrid-20260924-030820-930774/)。

## 判断

W8A8 INT8 远端 Draft 已可加载、执行且通过 Math-500 逐 token 正确性初测，但相对 FP8 没有可信的额外速度收益，也未解决远端 hit 只有约 2% 与整机慢于 local-only 的问题。保留为可选实验路径，不设为默认部署配置。下一步更值得减少 16 分支预算或提前启动远端流水线，再测量真实 hit 与接受长度。
