#!/usr/bin/env bash
# dev-pod 自包含环境一键构建：venv（CUDA 13 默认栈）+ 模型下载
#
# 用法（Pod 内 code-server 终端，仓库已 kubectl cp 进 /workspace/ 后）:
#   cd /workspace/sglang-speedup-nvidia && bash k8s/pod-bootstrap.sh
#
# 幂等：.venv 与模型已存在则跳过对应步骤，可重复执行。
# 模型来源：优先用环境变量，也可临时命令行覆盖，例如
#   MODEL_HF_ID=Qwen/Qwen3.5-9B bash k8s/pod-bootstrap.sh
#   MODEL_URL=https://.../qwen3.5-9b.tar.gz bash k8s/pod-bootstrap.sh
set -euxo pipefail

cd "$(dirname "$0")/.."   # 仓库根目录
WORKSPACE=/workspace

# ---- 模型来源（二选一；都不设则报错。确定固定来源后填为默认值）----
MODEL_HF_ID="${MODEL_HF_ID:-}"        # HF 仓库 ID（经 HF_ENDPOINT 镜像下载）
MODEL_URL="${MODEL_URL:-}"            # 或 tar.gz 直链（解压到 MODEL_DIR）
MODEL_DIR="${WORKSPACE}/models/Qwen3.5-9B"
MODEL_TAG="${MODEL_DIR}/config.json"  # 存在即视为模型就绪

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- 1) python 3.12 venv（python-build-standalone）----
#    国内慢时先启用: export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
if [ ! -x .venv/bin/python3 ]; then
  pip3 install -q uv
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 2) CUDA 13 默认栈（与 SETUP-H20.md 一致；已装则秒过）----
uv pip install torch==2.13.0 torchaudio==2.11.0 torchvision torchcodec==0.15.0 \
    --index-url https://download.pytorch.org/whl/cu130
SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python"
# flashinfer 预编译缓存（省首跑 10~30 分钟 JIT）
uv pip install "flashinfer-cubin==0.6.17" --index-url https://flashinfer.ai/whl
uv pip install "flashinfer-jit-cache==0.6.17" --index-url https://flashinfer.ai/whl/cu130

# ---- 3) 模型 ----
if [ ! -f "${MODEL_TAG}" ]; then
  mkdir -p "${MODEL_DIR}"
  if [ -n "${MODEL_HF_ID}" ]; then
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_HF_ID}', local_dir='${MODEL_DIR}')"
  elif [ -n "${MODEL_URL}" ]; then
    # tar 包若带顶层目录，视实际结构加 --strip-components=1
    curl -fSL --retry 3 -o /tmp/model.tar.gz "${MODEL_URL}"
    tar -xzf /tmp/model.tar.gz -C "${MODEL_DIR}"
    rm -f /tmp/model.tar.gz
  else
    echo "!! 未指定模型来源：请设 MODEL_HF_ID 或 MODEL_URL（见脚本头部说明）"
    exit 1
  fi
fi

# ---- 4) 验证 ----
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import sglang; print(sglang.__version__)"
python3 -c "import sgl_kernel; print(sgl_kernel.__version__)"
python3 -c "import flashinfer; print(flashinfer.__version__)"
echo "=== pod-bootstrap 完成：可运行基准（见 SETUP-H20.md）==="
