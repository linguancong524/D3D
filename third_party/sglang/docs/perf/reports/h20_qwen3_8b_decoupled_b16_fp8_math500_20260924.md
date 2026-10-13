# H20 decoupled Draft 在线 FP8 初测（2026-09-24）

## 口径与有效性

在 4×H20 开发机上，GPU0 为 Qwen3-8B Target + 原 bf16 本地兜底 Draft，GPU3 为 D3D-1.7B 远端可选 Draft。仅远端通过 `--remote-draft-quantization fp8` 启用 SGLang 在线 FP8（动态激活量化）；Target 和本地 Draft 未量化。其余沿用 [1+1 bf16 实验](h20_qwen3_8b_decoupled_b16_1plus1_wait_math500_20260923.md)：`block_size=16`、`num_anchors=1`、16 个接受位点、单行 packed + causal pending、零等待、CUDA Graph max tokens=1024、Target max-running-requests=3、Draft mem fraction=0.65，Math-500 `docs/scripts/test.jsonl`、输出 256 token、忽略 EOS、temperature=0、seed=42。启动前 GPU 无残留进程，结束后进程已清理；n100/n500 各 100/100、500/500 成功，无远端服务错误或致命 OOM。

远端 checkpoint 仍是原 bf16 文件，启动时在线转换为 FP8；trace 中能看到 `per_token_quant_fp8_cta_kernel` 和 `cutlass_3x_gemm_sm90_fp8`，并非仅改变启动参数名称。[启动配置与日志](../results/decoupled-hybrid-20260924-023534-803794/run.json)、[FP8 前向 trace](../profiles/decoupled_draft_b16_fp8_profile_20260924/)。

## 固定形状前向

`profile_packed_forward.py` 固定约 512 token 前缀，单次 CUDA event；与 [bf16 原剖析](h20_qwen3_8b_decoupled_draft_forward_profile_20260923.md)相同形状、同一 H20 平台。它适合定位趋势，不是多次重复的稳定性基准。

| 远端合批 | bf16 前向 | FP8 前向 | 变化 |
| ---: | ---: | ---: | ---: |
| 1 请求（272 query token） | 18.54 ms | 17.49 ms | -5.7% |
| 3 请求（816 query token） | 36.75 ms | 30.06 ms | -18.2% |

FP8 确实加速矩阵乘，但完整 16 分支前向仍长于约 11 ms 的 Target verify；单靠 FP8 不能实现同一 verify 窗口覆盖。原始数据见 [FP8 profiler 日志](../logs/decoupled_draft_b16_fp8_profile_20260924.log)。

## Math-500 在线结果

下表均为 1 Target + 1 远端 Draft 的饱和、零等待模式；bf16 与 local-only 是 2026-09-23 的归档结果，FP8 是本次运行，因此小幅 tok/s 差异只能视为趋势，不能宣称稳定收益。`verify` 为 benchmark 请求内统计，不含服务端预热。

| 模式 | n100 tok/s | n500 tok/s | n500 verify | n500 逐 token 匹配 |
| --- | ---: | ---: | ---: | ---: |
| 本地兜底，无远端 | 398.37 | 405.41 | 17,356 | 参考 |
| 远端 bf16 | 368.99 | 383.05 | 17,665 | 500/500 |
| 远端 FP8 | 380.16 | 385.17 | 17,670 | 500/500 |

FP8 相对 bf16：n100 +3.0%、n500 **+0.55%**；相对 local-only 的 n500 仍 **-5.0%**。n500 服务端计数按两组 benchmark 的累计差分：bf16 约 6060 次物理前向、2.91 请求/批、GPU 前向 31.62 ms/批、远端服务 39.72 ms/批；FP8 约 6068 次、2.91 请求/批、前向 25.31 ms/批、服务 33.29 ms/批。即在线前向约快 20%、服务约快 16%，但远端 hit 仅从约 379 增至约 382，约占 17,670 次 verify 的 2.2%。FP8 对**整体**接受长度几乎没有影响，不过由于大多数轮次仍走 bf16 本地兜底，不能据此推断 FP8 远端 proposal 自身质量完全无损。

完整结果：[FP8 n100](../results/decoupled-hybrid-20260924-023534-803794/one_plus_one_b16_fp8_math500_n100_saturation.json)、[FP8 n500](../results/decoupled-hybrid-20260924-023534-803794/one_plus_one_b16_fp8_math500_n500_saturation.json)、[bf16 n500](../results/decoupled-hybrid-20260923-154457-327156/one_plus_one_b16_math500_n500_saturation.json)、[local-only n500](../results/decoupled-local-20260923-155721-447870/one_target_b16_math500_n500_saturation.json)。

## 判断

FP8 功能与正确性通过初测，保留为可选实验开关，不作为默认部署配置。当前瓶颈是 16 分支的前向仍超过 verify 窗口，且仅约 2% 的 verify 真正用到远端结果。下一轮若继续优化，应把 FP8 与更小分支预算 / 更早启动的流水线组合测试，并单独测量远端 proposal 质量；仅继续压低 GEMM 毫秒数不足以保证整机加速。
