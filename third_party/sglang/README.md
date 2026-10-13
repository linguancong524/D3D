<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## 本仓库实验：双卡 Hybrid 异步预投机 + 本地 CUDA Graph

当前验证配置是 **2×H800 80GB、1 个 Qwen3-8B BF16 Target + 1 个远端 SDAR FP16 Drafter**。Target 同卡另有本地 SDAR FP16 draft；远端异步预计算下一轮候选，命中时跳过一次本地 draft，未命中时本地 full replay 继续执行。新增本地 packed CUDA Graph，保持 Target 验证、因果前缀 mask 和 draft block 双向 mask，不启用本地 KV 前缀复用。Graph 默认关闭，容量不足自动回退 eager。

**2026-10-13，单并发、自然 EOS（上限 2048）、B16/K2、full accept ON：**

| 数据集（各随机 32 题 × 3 轮） | 单卡 AR | Hybrid 本地 eager | Hybrid 本地 Graph | Graph / AR |
| --- | ---: | ---: | ---: | ---: |
| Math-500 | 126.8 | 368.5 | **549.1 token/s** | 4.33× |
| GSM8K | 127.1 | 395.3 | **591.8 token/s** | 4.66× |

指标是每请求实际输出 token / 端到端耗时的合并中位数，**包含 TTFT**，不是 GPU 前向速度或总批吞吐。384/384 个自然 EOS Hybrid 请求与 AR 的 token IDs、长度和结束原因一致；Math-500 2/32 题截断于 2048，GSM8K 全部正常结束。不是全量测试或数学准确率评测，也不能外推到 H20、高并发或多 Target。

- [当前配置、启动与复现指南](scripts/decoupled/README.md)
- [正式报告：自然 EOS / 固定 512 token、回退与证据](docs/perf/reports/h800_qwen3_8b_hybrid_20261013.md)
- [性能实验索引](docs/perf/README.md) · [Agent 工作约定](CLAUDE.md)（[AGENTS.md](AGENTS.md) 为软链接）

### 一轮解码怎样运行

1. Target 完成 prefill 并得到首个 bonus token。随后 `DecoupledWorker` 对每个请求先查共享 Drafter 的 ready outcome；查找依据包含会话、revision、proposal 身份以及实际新增的输出 token。没有可用结果时，`LocalSDARDrafter.propose_many()` 为本轮生成必需的 draft。
2. Target 将选中的 proposal 交给 `DraftClient.publish()`，同时用 `HostDraftWorker` 的 target-only chain/tree verify 验证候选。共享端据此推测**下一轮**可能出现的输出分支；Target 的采样和 KV 状态始终决定最终输出，远端候选不能绕过验证。
3. `DraftClient` 的后台线程通过本机 Unix-domain ZMQ IPC 发送 stage/publish/commit/close 消息；Target 的调度线程不等待远端 forward（`--decoupled-lookup-wait` 可配置一个有限等待窗口）。会话 nonce、revision、proposal ID 和 service epoch 用于隔离请求并拒绝过期结果。
4. `OptionalDraftService` 在各 Target 间公平选取 pending 项，按 `speculation_batch_size` 聚合；`DllmEngineBackend` 调用 native Engine 的 `dllm_packed_replay()`，对多个 anchor/accept 分支做 packed forward。共享端维护临时 KV，按 Target 后续确认的前缀 commit、裁剪或回滚，再把 outcomes 异步回送。
5. 下一轮查找若命中对应分支就消费远端 proposal；未就绪、分支不匹配、超时或服务故障都走本地 draft。请求结束时释放本地状态并关闭远端会话。

OneStep 的 draft block 形状为 `[anchor] + [mask] * (block_size - 1)`。anchor 属于待解码 block，不能同时计入 radix prefix；相关的前缀长度和整块 admission 规则在 dLLM scheduler 中实现。

### 从哪里读代码

| 入口 | 作用 |
| --- | --- |
| [`scripts/decoupled/launch.py`](scripts/decoupled/launch.py) | 生成并启动 `ar4`、`local`、`hybrid` 拓扑；`--dry-run` 可先查看完整命令。 |
| [`python/sglang/srt/speculative/decoupled/worker.py`](python/sglang/srt/speculative/decoupled/worker.py) | Target 每轮的 lookup、本地 fallback、publish 与生命周期；读主流程从 `_prepare_draft_tokens()` 开始。 |
| [`python/sglang/srt/speculative/host_draft_worker.py`](python/sglang/srt/speculative/host_draft_worker.py) | Target 侧 chain/tree 验证与接受 token 的路径。 |
| [`python/sglang/srt/speculative/decoupled/client.py`](python/sglang/srt/speculative/decoupled/client.py) | 异步发送、ready outcome 查找、会话与过期保护；重点读 `lookup()`、`publish()`、`_pump()`。 |
| [`python/sglang/srt/speculative/decoupled/service.py`](python/sglang/srt/speculative/decoupled/service.py) | 共享端的会话、跨 Target batching 和结果回送；重点读 `_publish()`、`speculate_batch()`。 |
| [`python/sglang/srt/speculative/decoupled/engine_backend.py`](python/sglang/srt/speculative/decoupled/engine_backend.py)、[`protocol.py`](python/sglang/srt/speculative/decoupled/protocol.py) | packed Engine 调用、分支选择和消息契约。 |
| [`python/sglang/srt/managers/scheduler.py`](python/sglang/srt/managers/scheduler.py)、[`transient_state.py`](python/sglang/srt/speculative/decoupled/transient_state.py) | `handle_dllm_packed_request()` 及临时 KV 的分配、提交与释放。 |
| [`python/sglang/benchmark/decoupled_serving.py`](python/sglang/benchmark/decoupled_serving.py) | 四卡统一负载、总输出吞吐和逐 token reference 对比。 |

运行前检查 `nvidia-smi`，不能占用或终止其他任务。当前用 `--mode hybrid --num-targets 1 --gpus 0 1` 明确选择双卡；launcher 仍支持历史 3+1 拓扑，但那不是上表的已测配置。H800 使用 cu129 预编译 wheel，Target 带 `--disable-prefill-cuda-graph`，本地 Graph 使用 `--local-cuda-graph-max-tokens 1024 --local-cuda-graph-max-requests 1`。完整开关及模型路径见运行指南。

**证据边界：**上表来自同一对 H800 上的新跑匹配对照，而非旧 H20 3+1 饱和吞吐实验。历史 FP8/INT8、其他模型和旧扫描结果保留在性能索引中，不混入本次结论。本次 draft 使用 FP16，不是 FP8；也未单独测量关闭 full accept 的因果增益。Graph 草稿浮点结果不保证与 eager 逐 bit 一致，最终输出仍由 Target 验证。所有新增结果按 `docs/perf/results/`、独立日志按 `docs/perf/logs/` 归档。

## News
- [2026/07] 🔥 SGLang and Miles add day-0 support for Kimi K3 ([blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)).
- [2026/07] RadixArk and Google bring full SGLang features to TPUs ([blog](https://lmsys.org/blog/2026-07-30-sglang-google-tpu/)).
- [2026/07] Serving GLM5.2 NVFP4 agentic workloads with SGLang: Reaching 500 TPS in two weeks ([blog](https://lmsys.org/blog/2026-07-13-glm52-optimization/)).
- [2026/06] 🔥 The next generation of speculative decoding: DFlash and Spec V2 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).
- [2026/06] SGLang provides day-0 support for latest open models ([Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/), [Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/), [Higgs Audio v3 TTS](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)).
- [2026/04] 🔥 DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles ([blog](https://lmsys.org/blog/2026-04-25-deepseek-v4/)).
- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).

<details>
<summary>More</summary>

- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [GB300 long context](https://lmsys.org/blog/2026-02-19-gb300-longctx/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, NVIDIA, AMD, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, RunPod, InnoMatrix, Modal, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Baseten, Baidu, AntGroup, Alibaba, Tencent, and other major technology organizations.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [sglang@lmsys.org](mailto:sglang@lmsys.org).

Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex. Email [sglang@lmsys.org](mailto:sglang@lmsys.org) with your most important commits or pull requests.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
