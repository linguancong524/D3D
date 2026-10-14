<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">D3D</h1>

<h3 align="center">Decoupled Diffusion Drafting for Asynchronous Speculative Decoding</h3>

<p align="center">
  <strong>Diffusion drafts. The target verifies. D3D overlaps both.</strong><br>
  A decoupled draft-and-verify architecture, powered by Qwen3-derived diffusion drafts.
</p>

<p align="center">
  <!-- <a href="https://openreview.net/forum?id=owCZMqNsqR"><img src="https://img.shields.io/badge/Paper-OpenReview-6366f1?style=flat-square" alt="Paper"></a> -->
  <a href="https://huggingface.co/collections/lingc8/d3d-6acc999de5b699dcf37b3c75"><img src="https://img.shields.io/badge/Models-D3D_Collection-fbbf24?style=flat-square" alt="D3D model collection"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Get_started-22c55e?style=flat-square" alt="Quick start"></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/Architecture-Docs-0ea5e9?style=flat-square" alt="Architecture guide"></a>
  <a href="#benchmarking"><img src="https://img.shields.io/badge/Benchmarks-Protocol-f97316?style=flat-square" alt="Benchmark protocol"></a>
</p>

<a id="overview"></a>

## ✨ Overview

**D3D** decouples a diffusion **draft model** from the **target verifier** on
separate GPUs. Draft generation overlaps target verification, with verification
feedback selecting reusable candidates for the next decoding step.

**The contribution is the D3D execution architecture:** independent drafting,
overlapped verification, and feedback-driven candidate reuse.

| Asynchronous execution | Parallel diffusion drafts |
|:---|:---|
| Drafting and verification run on separate GPUs. | Block-wise candidate generation with packed outcomes. |

<p align="center">
  <img src="docs/figure1.png" alt="D3D method overview: drafter training, progressive layer pruning, and asynchronous inference" width="960">
</p>

**How it works:** a target-selected anchor starts a parallel diffusion draft;
the target verifies the pending block while the draft GPU prepares possible
continuations; verification feedback selects the continuation to reuse.
See the [architecture guide](docs/architecture.md) for the execution flow.

<a id="model-zoo"></a>

## 🤗 Model Zoo

**[Browse the D3D collection →](https://huggingface.co/collections/lingc8/d3d-6acc999de5b699dcf37b3c75)**

Six Qwen3-derived draft checkpoints are grouped in **lingc8 / D3D**. These are
**draft models**, not replacements for the target models. `p` denotes pruning;
`24D` and `31D` denote the retained transformer-layer counts.

| Target verifier | Draft family | Layers | D3D checkpoint |
|:---|:---|:---:|:---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1.7B | 28 | [D3D-1.7B-to-8B](https://huggingface.co/lingc8/D3D-1.7B-to-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B | 36 | [D3D-8B-to-8B](https://huggingface.co/lingc8/D3D-8B-to-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Pruned 8B | 31 | [D3D-31D-p8B-to-8B](https://huggingface.co/lingc8/D3D-31D-p8B-to-8B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 1.7B | 28 | [D3D-1.7B-to-32B](https://huggingface.co/lingc8/D3D-1.7B-to-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 8B | 36 | [D3D-8B-to-32B](https://huggingface.co/lingc8/D3D-8B-to-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Pruned 1.7B | 24 | [D3D-24D-p1.7B-to-32B](https://huggingface.co/lingc8/D3D-24D-p1.7B-to-32B) |

> Checkpoints retain their original FP32 tensors; inference uses BF16.
> Each Hub repository includes tokenizer assets, inference compatibility code,
> a model card, and release metadata. The family size in a name is not an exact
> post-pruning parameter count. Exact source steps, tensor counts, and release
> compatibility notes are in [checkpoint details](docs/models.md).

<a id="installation"></a>

## 📦 Installation

The tested environment uses **PyTorch 2.6.0**, **Transformers 4.52.4**,
**FlashAttention 2.7.4.post1**, and **CUDA 12.x**. Python 3.10+ is supported;
local GPU validation used Python 3.11 and two NVIDIA H800 80 GB GPUs.

```bash
git clone https://github.com/linguancong524/D3D.git
cd D3D
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e . --no-deps
```

Install PyTorch before building FlashAttention. A matching prebuilt wheel can
be used instead; source builds need a CUDA toolkit, a C++ compiler, and
sufficient host memory. The concurrent backend requires FlashAttention.

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Select a target / draft pair

```bash
export CUDA_VISIBLE_DEVICES=0,1

# optional: override the defaults for every script
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
```

The target runs on `cuda:0`, and the draft runs on `cuda:1`. Model IDs also
accept local checkpoint directories. Public model downloads do not require
a Hugging Face token.

### 2. Run asynchronous decoding

Every script forwards extra flags to its Python entry point, so prefer command
line options over environment variables. Single-request scripts take
`--verifier` / `--drafter`; concurrent scripts take `--target` / `--draft`.

**Single request** — one MATH-500 sample, up to 128 generated tokens:

```bash
bash scripts/quickstart_single.sh

# e.g. another pair with a longer generation cap
bash scripts/quickstart_single.sh \
    --verifier Qwen/Qwen3-32B \
    --drafter lingc8/D3D-24D-p1.7B-to-32B \
    --max-new-tokens 512
```

**Concurrent requests** — four samples at concurrency 4:

```bash
bash scripts/quickstart_concurrent.sh

# e.g. concurrency 8 with 256 KV pages
CONCURRENCY=8 bash scripts/quickstart_concurrent.sh --pages 256
```

<details>
<summary><strong>Use the 32B target with the 24-layer pruned draft</strong></summary>

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
<summary><strong>Useful configuration options</strong></summary>

Scripts pass their defaults as flags, and anything appended on the command line
wins. Only a few environment variables remain:

| Variable | Purpose | Default |
|:---|:---|:---|
| `TARGET_MODEL` | Target verifier used by every script | `Qwen/Qwen3-8B` |
| `DRAFT_MODEL` | Draft checkpoint used by every script | `lingc8/D3D-1.7B-to-8B` |
| `DATASETS` | Dataset list for the two benchmark scripts | the seven benchmarks |
| `CONCURRENCY` | Quick-start request count, or benchmark sweep list | `4` / `1 4 8 16` |
| `D3D_DATA_CACHE` | Optional existing JSONL dataset cache | `data_cache/` |

Everything else is a flag, for example `--max-samples`, `--max-new-tokens`,
`--block-size`, `--anchor-num`, `--pages` or `--no-baseline`.

Use `python single/eval_async_packed.py --help` or
`python -m serving.bench --help` for the full set of options.

</details>

<a id="benchmarking"></a>

## 📊 Benchmarking

Run single-request evaluation over the standard datasets:

```bash
bash scripts/benchmark.sh
```

Run concurrent evaluation at request concurrency 1, 4, 8, and 16:

```bash
bash scripts/benchmark_concurrent.sh
```

Datasets are downloaded at runtime, not bundled in this repository.
LiveCodeBench v6 uses `test6.jsonl` from
`livecodebench/code_generation_lite`; the other datasets use `datasets`.
See the [benchmark protocol](benchmarks/README.md) for sources, splits,
token caps, timing boundaries, and reporting rules.

## ⚙️ Inference Engine Support

The repository supports native D3D inference and an optional two-GPU
SGLang Hybrid backend. Install SGLang separately:

```bash
bash scripts/sglang/install.sh
```

Start the two-GPU backend (GPU 0: Target + local Draft; GPU 1: optional Draft):

```bash
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
export TARGET_GPU=0 DRAFT_GPU=1
export MAX_RUNNING_REQUESTS=1
bash scripts/sglang/launch.sh hybrid
```

Then run a MATH-500 evaluation from another terminal:

```bash
export MAX_RUNNING_REQUESTS=1
export TARGET_MODEL=Qwen/Qwen3-8B
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset math500 \
  --tokenizer "$TARGET_MODEL" \
  --samples 128 --concurrency "$MAX_RUNNING_REQUESTS" \
  --repetitions 1 --max-new-tokens 512 \
  --output outputs/sglang/example.json
```

The dataset is downloaded automatically and cached by Hugging Face Datasets.

Set `MAX_RUNNING_REQUESTS` to `1`, `4`, `8`, or `16` in both terminals;
restart the backend after changing it.

## 🗂️ Repository Guide

```text
single/       Single-request asynchronous draft-and-verify decoding
serving/      Concurrent decoding, paged KV cache, and AR baselines
models/d3d/   Qwen3-derived D3D draft backbone and compatibility code
common/       Shared helpers: devices, checkpoints, datasets, text, metrics
scripts/      Quick starts, smoke checks, and benchmark entry points
benchmarks/   Dataset sources and evaluation protocol
docs/         Architecture, model details, and validation notes
tests/        Model-loading, release, and metric-accounting checks
```

## Citation & License

<!-- If D3D is useful in your research, please cite the
[paper](https://openreview.net/forum?id=owCZMqNsqR). -->

D3D-authored code is released under the [MIT License](LICENSE). Included
third-party code and Qwen-derived weights retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the individual model cards.
