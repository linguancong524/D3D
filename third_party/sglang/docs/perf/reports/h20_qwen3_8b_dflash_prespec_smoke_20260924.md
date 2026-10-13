# H20：Target 隐层驱动 DFlash 第二步预投机烟测（2026-09-24）

## 结论

离线前向已跑通：使用 Qwen3-8B Target 的多层隐层、D3D 第一步产生的 16 个未验证候选 token，加上 16 个 mask token，DFlash 能在独立 H20 上一次前向产出第二步的 16 个候选 token。该实验只证明数据形状与前向路径可连接，**不证明候选可被 Target 接受，也不证明端到端加速**。

## 配置与流程

- Target：`/workspace/models/Qwen3-8B`，GPU 0。
- 第一步 D3D：`/workspace/models/d3d-1.7b-b16_for_qwen3-8b`，GPU 0；使用当前项目的 D3D layout 及 proposal 逻辑。
- 第二步 DFlash：[`z-lab/Qwen3-8B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)，revision `9b41424b7109f9c5413454f481b09a82b85333f4`，本地路径 `/tmp/sglang-qwen3-8b-dflash-b16`，GPU 3。
- Target 隐层取第 1、9、17、25、33 层，拼接为宽度 20480；传入 DFlash 时包含原始 prompt 的全部位置。
- DFlash 输入为 `[D3D 第一步的 16 个候选][16 个 mask]`。这是 32-token noise 输入，**不等同于该 b16 checkpoint 的标准训练/推理分布**。
- 仅做单请求、离线 eager 前向；没有接入 SGLang 调度器、跨进程通信、Target verify 或接受判定。

## 已有运行结果

| 日志 | prompt 长度 | 结果 | 第二步单次前向耗时 |
|---|---:|---|---|
| `dflash_prespec_smoke_h20_20260924_v7.log` | 47 | D3D 第一批候选和 DFlash 第二批候选均为有效 token；替换第一批候选或清零 Target 隐层，第二批 16/16 token 改变 | 4.18–4.40 ms |
| `dflash_prespec_smoke_h20_20260924_v8_ctx512.log` | 512 | 两批候选均为有效 token；两种输入消融分别令 14/16 token 改变 | 4.24–4.45 ms |
| `dflash_prespec_smoke_h20_math500_20260924_v10_fp32_short.log` | 47 | Math-500 样本 0，FP32/数学 SDPA 的 D3D 第一块有效；10 次计时中途漂移 | 中位数 5.24 ms，范围 4.08–6.23 ms |
| `dflash_prespec_smoke_h20_math500_20260924_v12_supplied_ctx512.log` | 512 | Math-500 样本 301；第一块 token 复用先前同题成功运行的结果；20 次计时，第二块有效 | **中位数 4.24 ms**，范围 4.15–4.49 ms |

上表耗时包含 Target 隐层的 GPU 0→GPU 3 拷贝、DFlash eager 前向及 GPU 3 上的输出 head；不包含 Target 隐层生成、D3D 第一步、Target verify，也没有并发重叠。不能直接与线上 verify 时间比较。

## 稳定性与后续验证

旧 D3D checkpoint 通过离线 Hugging Face 路径运行时，曾出现第一步 logits 全 NaN（`dflash_prespec_smoke_h20_20260924_v9_final.log`）；权重检查显示 D3D 与 DFlash 均无缺失权重。改为 FP32 和数学 SDPA 后，47-token 样本成功，但 512-token 样本仍全 NaN（`dflash_prespec_smoke_h20_math500_20260924_v11_fp32_ctx512.log`）。因此 512-token 测速复用了先前同题成功产生的第一块 token，以隔离计量 DFlash 第二步。该离线 D3D 路径不是线上 SGLang D3D 执行路径，不能用它的失败推断线上路径也失败。

下一步应解决离线 D3D 长上下文 NaN，或直接改用线上 SGLang D3D 提供第一块，再接入线上隐层交接与 verify，并统计第二步候选接受率、异步覆盖率、端到端吞吐和延迟。只有这些指标成立，才能判断该方案是否值得用于部署。
