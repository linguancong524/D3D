# H800 环境编译与运行指南（CUDA 12.9）

> H20 节点 60c（驱动 ≥580）用 CUDA 13 默认流程，见 `SETUP-H20.md`；本文件仅适用 H800 基线机（驱动 575.x，必须 cu129）。

适用：NVIDIA H800 (sm90) + CUDA 12.9 (nvcc 12.9.86) + 本仓库 v0.5.18 源码。
依据：官方文档 cu12 安装节 + 本仓库 `scripts/ci/cuda/ci_install_dependency.sh` 的 cu129 分支
（h20 CI 同款流程），所有 wheel URL 均已验证存在。
注意：自 v0.5.18 起上游已提交 `rust/Cargo.lock`，Rust 依赖集合冻结
（v0.5.17 时代 dynamo-parsers 版本漂移导致的 E0063 编译失败已在上游解决）。

## 0) 系统依赖（一次性）

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential git pkg-config python3-dev \
    libnuma-dev libssl-dev \
    libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
export CUDA_HOME=/usr/local/cuda-12.9   # 按实际路径（nvcc 所在工具链根目录）
```

## 1) Python 环境

```bash
conda create -n sglang python=3.12 -y && conda activate sglang
pip install --upgrade pip && pip install uv
```

## 2) 预装 cu129 二进制包（与本地 CUDA 12.9 匹配）

```bash
uv pip install torch==2.13.0 torchaudio==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu129
uv pip install "sglang-kernel==0.4.6.post1" \
    --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --no-deps "sgl-deep-gemm==0.1.5.post3" \
    --index-url https://docs.sglang.ai/whl/cu129/
```

说明：PyPI 上 torch 2.13.0 / sgl-deep-gemm 的默认 wheel 是 CUDA 13 构建，
在 CUDA 12.9 机器上必须先装 cu129 版本；预装后第 3 步 pip 不会重复下载。

## 3) 可编辑安装本仓库

```bash
git clone git@github.com:wintem0531/sglang-speedup-nvidia.git
cd sglang-speedup-nvidia && git checkout main
SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python" --index-strategy unsafe-best-match
```

- 重型的 sgl-kernel（CUDA 算子库）直接取第 2 步的预编译 wheel，无需本地编译；
- `SGLANG_BUILD_RUST_EXTS=none` 跳过实验性 Rust 扩展（`SGLANG_RUST_SERVER` 默认关闭，
  标准 serving/压测用不到）。若日后要用 Rust server：安装 rustup（仓库 pinned Rust 1.92）
  和 protoc，去掉该环境变量重装即可，建议同时设 `RUSTUP_TOOLCHAIN=1.92`。
  v0.5.18 起 `rust/Cargo.lock` 已随仓库提交，全新 clone 也能确定性构建。

## 4) flashinfer JIT 缓存（可选，省掉首启 10~30 分钟内核编译）

```bash
mkdir -p ~/.cache/flashinfer-wheels
pip download "flashinfer-jit-cache==0.6.17" \
    --index-url "https://flashinfer.ai/whl/cu129" -d ~/.cache/flashinfer-wheels
uv pip install ~/.cache/flashinfer-wheels/flashinfer_jit_cache-0.6.17+cu129*.whl
```

## 5) 验证

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.13.0+cu129  12.9  True  NVIDIA H800
python -c "import sglang; print(sglang.__version__)"   # 0.5.18
python -c "import sgl_kernel; print(sgl_kernel.__version__)"   # 0.4.6.post1
```

## 6) 跑模型（Qwen 小模型）

```bash
# 下载模型（国内网络慢可先: export HF_ENDPOINT=https://hf-mirror.com）
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507')"

# 启动 server（--cuda-graph-backend-prefill=disabled 为当前驱动所需，见备注）
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 30000 --mem-fraction-static 0.85 \
    --cuda-graph-backend-prefill=disabled

# 测试生成（/generate）
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"你好","sampling_params":{"max_new_tokens":128,"temperature":0.7}}'

# OpenAI 兼容接口（/v1/chat/completions）
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

2B/4B/9B 对应的现成模型：`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`、`Qwen/Qwen3-8B`
（或 Qwen2.5-1.5B/3B/7B），改 `--model-path` 即可。

## 7) 性能基准（prefill/decode 效率）

```bash
# 在线压测：TTFT / TPOT / 吞吐（server 需已启动）
python3 -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts 200 --request-rate 8

# 离线单批 prefill/decode 延迟（无需 server）
python3 -m sglang.bench_one_batch --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --batch-size 1,8,32 --input-len 128,1024,4096 --output-len 128
```

## 备注

- 官方在线文档默认 CUDA 13；本仓库在 CUDA 12 机器上必须走 cu129 wheel 路径（本文档）。
- v0.5.18 的 prefill 默认 CUDA graph 后端为 breakable（BCG），经 cuda-python 调用
  `cudaStreamGetCaptureInfo`；pyproject 固定 `cuda-python>=13.0`（CUDA 13 运行时）。
  本机驱动只支持到 CUDA 12.9（575.x 代），CUDA 13 运行时无法初始化，prefill 图捕获会报
  `CUDA error 35 (CUDA driver version is insufficient for CUDA runtime version)`。
  驱动升级到 580.x（CUDA 13 代）之前，启动命令必须带
  `--cuda-graph-backend-prefill=disabled`（prefill 走 eager；decode 的 full 图不受影响）。
  升级后可去掉该参数。基线与后续实验（dflash/dspark 等）请保持同一参数以保证可比。
- 后续若改 `sgl-kernel/` 里的 CUDA 算子（dspark 等实验），需要本地编译 sgl-kernel wheel
  （`python/sglang/kernels/aot`），流程到时单独整理；现阶段一律用预编译 wheel 保证基线干净。
- Docker 备选：`lmsysorg/sglang:dev-cu12` 镜像 + 挂载本仓库，但镜像内 sglang 版本不一定是
  v0.5.18，做性能基线对比时建议用本文的原生安装。
