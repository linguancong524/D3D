<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">D3D</h1>

<h3 align="center">Decoupled Diffusion Drafting for Asynchronous Speculative Decoding</h3>

<p align="center">
  <strong>扩散草稿，目标验证，D3D 让两者重叠。</strong><br>
  解耦的 draft-and-verify 架构，草稿模型为 Qwen3 派生的扩散模型。
</p>

<p align="center">
  <!-- <a href="https://openreview.net/forum?id=owCZMqNsqR"><img src="https://img.shields.io/badge/%E8%AE%BA%E6%96%87-OpenReview-6366f1?style=flat-square" alt="论文"></a> -->
  <a href="https://huggingface.co/collections/lingc8/d3d-6acc999de5b699dcf37b3c75"><img src="https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-D3D_Collection-fbbf24?style=flat-square" alt="D3D 模型合集"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B-%E5%BC%80%E5%A7%8B%E4%BD%BF%E7%94%A8-22c55e?style=flat-square" alt="快速开始"></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/%E6%9E%B6%E6%9E%84-%E6%96%87%E6%A1%A3-0ea5e9?style=flat-square" alt="架构文档"></a>
  <a href="#benchmarking"><img src="https://img.shields.io/badge/%E5%9F%BA%E5%87%86-%E5%8D%8F%E8%AE%AE-f97316?style=flat-square" alt="基准协议"></a>
</p>

<a id="overview"></a>

## ✨ 概览

**D3D** 把扩散**草稿模型**与**目标验证模型**解耦到不同的 GPU 上。草稿生成
与目标验证相互重叠，并由验证反馈挑选下一步可复用的候选。

**本项目的贡献是 D3D 的执行架构：** 独立起草、重叠验证，以及由验证反馈
驱动的候选复用。

| 异步执行 | 并行扩散草稿 |
|:---|:---|
| 起草与验证分别运行在不同的 GPU 上。 | 按块生成候选，并打包多种接受结果。 |

<p align="center">
  <img src="docs/figure1.png" alt="D3D 方法总览：草稿训练、渐进式层剪枝与异步推理" width="960">
</p>

**工作方式：** 目标模型选出的 anchor 触发一次并行扩散起草；在草稿 GPU 准备
后续候选的同时，目标模型验证当前块；验证反馈决定复用哪一个候选。执行流程
见[架构文档](docs/architecture.md)。

<a id="model-zoo"></a>

## 🤗 模型仓库

**[浏览 D3D 合集 →](https://huggingface.co/collections/lingc8/d3d-6acc999de5b699dcf37b3c75)**

六个 Qwen3 派生的草稿 checkpoint 收录在 **lingc8 / D3D** 合集中。它们是
**草稿模型**，不是目标模型的替代品。名称中的 `p` 表示剪枝，`24D` 与 `31D`
表示保留的 transformer 层数。

| 目标验证模型 | 草稿系列 | 层数 | D3D checkpoint |
|:---|:---|:---:|:---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1.7B | 28 | [D3D-1.7B-to-8B](https://huggingface.co/lingc8/D3D-1.7B-to-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B | 36 | [D3D-8B-to-8B](https://huggingface.co/lingc8/D3D-8B-to-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 剪枝 8B | 31 | [D3D-31D-p8B-to-8B](https://huggingface.co/lingc8/D3D-31D-p8B-to-8B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 1.7B | 28 | [D3D-1.7B-to-32B](https://huggingface.co/lingc8/D3D-1.7B-to-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 8B | 36 | [D3D-8B-to-32B](https://huggingface.co/lingc8/D3D-8B-to-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 剪枝 1.7B | 24 | [D3D-24D-p1.7B-to-32B](https://huggingface.co/lingc8/D3D-24D-p1.7B-to-32B) |

> checkpoint 保留原始 FP32 张量，推理时使用 BF16。每个 Hub 仓库包含
> tokenizer 资源、推理兼容代码、model card 与发布元数据。名称中的系列规模
> 不是剪枝后的精确参数量。精确的来源 step、张量数量与发布兼容性说明见
> [checkpoint 详情](docs/models.md)。

<a id="installation"></a>

## 📦 安装

测试环境为 **PyTorch 2.6.0**、**Transformers 4.52.4**、
**FlashAttention 2.7.4.post1** 与 **CUDA 12.x**；支持 Python 3.10+，本地
GPU 验证使用 Python 3.11 和两张 NVIDIA H800 80 GB。

```bash
git clone https://github.com/linguancong524/D3D.git
cd D3D
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e . --no-deps
```

先安装 PyTorch，再编译 FlashAttention。也可以使用匹配的预编译 wheel；
源码编译需要 CUDA toolkit、C++ 编译器以及足够的主机内存。并发后端依赖
FlashAttention。

<a id="quick-start"></a>

## 🚀 快速开始

### 1. 选择 target / draft 组合

```bash
export CUDA_VISIBLE_DEVICES=0,1

# 可选：覆盖所有脚本的默认模型
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
```

目标模型运行在 `cuda:0`，草稿模型运行在 `cuda:1`。模型 ID 也接受本地
checkpoint 目录。下载公开模型不需要 Hugging Face token。

### 2. 运行异步解码

每个脚本都会把多余参数透传给对应的 Python 入口，因此优先用命令行参数，
少用环境变量。单请求脚本用 `--verifier` / `--drafter`，并发脚本用
`--target` / `--draft`。

**单请求** —— 一条 MATH-500 样本，最多生成 128 个 token：

```bash
bash scripts/quickstart_single.sh

# 例如换一对模型并放宽生成长度
bash scripts/quickstart_single.sh \
    --verifier Qwen/Qwen3-32B \
    --drafter lingc8/D3D-24D-p1.7B-to-32B \
    --max-new-tokens 512
```

**并发请求** —— 四请求并发 4：

```bash
bash scripts/quickstart_concurrent.sh

# 例如并发 8、每卡 256 个 KV page
CONCURRENCY=8 bash scripts/quickstart_concurrent.sh --pages 256
```

<details>
<summary><strong>使用 32B 目标 + 24 层剪枝草稿</strong></summary>

```bash
bash scripts/quickstart_single.sh \
    --verifier Qwen/Qwen3-32B \
    --drafter lingc8/D3D-24D-p1.7B-to-32B

bash scripts/quickstart_concurrent.sh \
    --target Qwen/Qwen3-32B \
    --draft lingc8/D3D-24D-p1.7B-to-32B \
    --concurrency 4 --pages 128
```

</details>

<details>
<summary><strong>常用配置项</strong></summary>

脚本把默认值当参数传给入口，命令行上追加的参数优先。只剩少数环境变量：

| 变量 | 作用 | 默认值 |
|:---|:---|:---|
| `TARGET_MODEL` | 所有脚本使用的目标验证模型 | `Qwen/Qwen3-8B` |
| `DRAFT_MODEL` | 所有脚本使用的草稿 checkpoint | `lingc8/D3D-1.7B-to-8B` |
| `DATASETS` | 两个 benchmark 脚本的数据集列表 | 七个标准数据集 |
| `CONCURRENCY` | quick start 的请求数，或并发 benchmark 的扫描列表 | `4` / `1 4 8 16` |
| `D3D_DATA_CACHE` | 可选的本地 JSONL 数据集缓存 | `data_cache/` |

其余都是命令行参数，例如 `--max-samples`、`--max-new-tokens`、`--block-size`、
`--anchor-num`、`--pages`、`--no-baseline`。

单请求的完整参数用 `python single/eval_async_packed.py --help` 查看，并发用
`python -m serving.bench --help`。

</details>

<a id="benchmarking"></a>

## 📊 性能基准

在标准数据集上运行单请求评测：

```bash
bash scripts/benchmark.sh
```

在并发 1、4、8、16 下运行并发评测：

```bash
bash scripts/benchmark_concurrent.sh
```

数据集在运行时下载，不随仓库分发。LiveCodeBench v6 使用
`livecodebench/code_generation_lite` 的 `test6.jsonl`，其余数据集使用
`datasets`。来源、split、token 上限、计时边界与上报规则见
[benchmark 协议](benchmarks/README.md)。

## ⚙️ 推理引擎支持

仓库支持原生 D3D 推理，也提供可选的严格双卡 SGLang 后端。请单独安装：

```bash
bash scripts/sglang/install.sh
```

先在 GPU 1 启动 Draft，再在 GPU 0 启动 Target：

```bash
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
export TARGET_GPU=0 DRAFT_GPU=1
export MAX_RUNNING_REQUESTS=1
bash scripts/sglang/launch.sh draft   # 终端 1
bash scripts/sglang/launch.sh remote  # 终端 2
```

然后在第三个终端运行 MATH-500 评测：

```bash
export MAX_RUNNING_REQUESTS=1
export TARGET_MODEL=Qwen/Qwen3-8B
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset math500 \
  --tokenizer "$TARGET_MODEL" \
  --samples "$MAX_RUNNING_REQUESTS" --concurrency "$MAX_RUNNING_REQUESTS" \
  --repetitions 1 --max-new-tokens 128 \
  --output outputs/sglang/example.json
```

数据集会自动下载，并由 Hugging Face Datasets 缓存。

在三个终端中将 `MAX_RUNNING_REQUESTS` 统一设为 `1`、`4`、`8` 或 `16`；
修改后需重启 Draft 和 Target。

## 🗂️ 仓库结构

```text
single/       单请求异步 draft-and-verify 解码
serving/      并发解码、paged KV cache 与 AR baseline
models/d3d/   Qwen3 派生的 D3D 草稿骨干与兼容代码
common/       共享工具：设备、checkpoint、数据集、文本、指标
scripts/      quick start、smoke check 与 benchmark 入口
benchmarks/   数据集来源与评测协议
docs/         架构、模型细节与验证说明
tests/        模型加载、发布与指标核算测试
```

## 引用与许可

<!-- 如果 D3D 对你的研究有帮助，请引用我们的
[论文](https://openreview.net/forum?id=owCZMqNsqR)。 -->

D3D 自有代码以 [MIT 许可证](LICENSE)发布。包含的第三方代码与 Qwen 派生权重
沿用各自的条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 以及各个
model card。
