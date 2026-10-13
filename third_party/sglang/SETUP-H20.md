# H20（60c）环境搭建指南（CUDA 13 默认流程）

适用：K8s H20 节点 e01-cn-c6k4blkz60c（2026-08-26 管理员已升级驱动，`nvidia-smi`
确认 ≥580.x）+ 本仓库 v0.5.18 源码。H800 基线机继续走 cu129 流程（见 `SETUP.md`），
两环境互相独立；H800 数据与 H20 数据不做跨栈横比（见第 6 节）。

依据：官方安装文档默认即 CUDA 13（`docs/docs/get-started/install.mdx`："The major
version of Cuda is 13 by default"）；本仓库 CI 默认 `CU_VERSION=cu130`
（`scripts/ci/cuda/ci_install_dependency.sh`）；`docker/Dockerfile` 的 CUDA 13.0.3 分支
（flashinfer jit-cache 用 `cu130` index）。torch 2.13.0 / sglang-kernel 0.4.6.post1 /
sgl-deep-gemm 0.1.5.post3 的 PyPI 默认 wheel 均为 cu130 构建（2026-08-29 已验证存在）。

## 1) 前置

- 驱动 ≥580.x：`kubectl apply -f k8s/h20-gpu-test.yaml`，`kubectl logs gpu-test`
  看 driver_version。若 <580：BCG 图捕获与 cu130 wheel 都不可用，先找管理员
- **节点无需预置任何东西**：dev-pod 无 hostPath，不再需要节点预建目录 /
  clone 仓库 / 放 code-server tar 包。模型和实验归档使用 `sglang-dev-oss` PVC 的
  Pod 专属 subPath 持久化；仓库、venv 和 JIT cache 仍在容器本地可写层
- 旧 60c 节点目录 `/home/songtao/workspaces/sglang-speedup-nvidia`（2026-08-29 前的
  hostPath 数据，含 venv/模型约 30GB）已弃用；节点磁盘紧张时请管理员删除
- dev Pod（在 iZ 管理机上执行）：先 `kubectl apply -f k8s/sglang-dev-oss-pvc.yaml`，
  等 `kubectl get pvc -n bio-prod sglang-dev-oss` 为 `Bound`；再
  `kubectl apply -f k8s/dev-pod.yaml`，等 `kubectl logs -f -n bio-prod dev-pod` 出现
  "code-server 就绪"（约 2~5 分钟）

## 2) OSS subPath 与仓库进容器（iZ 管理机）

`sais-staging-bio-prod` 挂载共享 `fuse.ossfs` 根目录。Pod 的 initContainer 只用于创建和
校验固定专属目录：`dev-pod` 使用 `h20-dev/dev-pod1/`，`dev-pod-2` 使用
`h20-dev/dev-pod2/`。code-server 主容器只挂载其中的 `models` 到
`/workspace/models`，以及 `persist` 到
`/workspace/persist`，看不到共享 OSS 根目录。两个目录互相隔离，并且不会因 Pod 重建而改变。

仓库 / venv / Torch、Triton、FlashInfer JIT cache 仍在容器本地可写层，避免在 OSS
FUSE 上运行 Git、Python 大量小文件与 JIT 编译。Pod 重建后需重拷仓库、重建 venv，
但模型与 `/workspace/persist` 中的归档保留。

```bash
# 在 iZ 的仓库父目录执行（仓库含 .git，约 400MB，几分钟）
kubectl cp -n bio-prod sglang-speedup-nvidia dev-pod:/workspace/sglang-speedup-nvidia
# 若拷完出现 /workspace/sglang-speedup-nvidia/sglang-speedup-nvidia 嵌套，mv 一层
```

- 更新仓库：iZ 上 `git pull` 后重跑上面的 kubectl cp（容器内无 GitHub 凭据，
  不能 `git pull`）
- 日志归档：先按项目约束 `tee` 进 `docs/perf/logs/`，完成后可备份到
  `/workspace/persist/perf/`；并继续执行
  `kubectl cp -n bio-prod dev-pod:/workspace/sglang-speedup-nvidia/docs/perf/logs/<log> \
  <iZ 仓库>/docs/perf/logs/` → 在 iZ 上 commit + push

## 3) 一键构建 venv + 模型（容器终端）

```bash
cd /workspace/sglang-speedup-nvidia
MODEL_HF_ID=<HF 仓库 ID> bash k8s/pod-bootstrap.sh
# 或 tar.gz 直链: MODEL_URL=<url> bash k8s/pod-bootstrap.sh
# 模型落到 /workspace/models/Qwen3.5-9B（~18GB，耗时视带宽）
```

脚本内容（幂等，可重复执行；已装的依赖、已下载的模型自动跳过）：

1. `uv venv --python 3.12 .venv`（python-build-standalone；国内慢时先
   `export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download`）
2. 安装 CUDA 13 默认栈：torch 2.13.0 cu130 → 可编辑安装本仓库
   （`SGLANG_BUILD_RUST_EXTS=none`）→ flashinfer cubin + jit-cache
3. 下载模型（HF 镜像 `HF_ENDPOINT=https://hf-mirror.com` 或直链）
4. 版本验证（同第 4 节）

说明：

- `SGLANG_BUILD_RUST_EXTS=none` 跳过实验性 Rust 扩展（同 H800，标准 serving/压测用不到）
- 与 H800 流程的差异：无 `--index-url .../cu129/` 特判、无 `--index-strategy
  unsafe-best-match`、无需 sed 改 pyproject（cu12 构建才需要把 `flashinfer_python[cu13]`
  改成 `[cu12]`，见 `docker/Dockerfile`）
- dev-pod 镜像固定用 `13.0.0-devel` 变体（`k8s/dev-pod.yaml`，2026-08-29 起；
  与 torch cu130 运行时对齐）：
  v0.5.18 的 sgl-kernel JIT 路径（silu_and_mul 等 elementwise 算子首调时 nvcc 编译）
  需要 nvcc，base 镜像无 CUDA toolkit，首跑报 "Failed to build JIT module
  sgl_kernel_jit_activation_bf16_t_true ... nvcc: not found"（首调发生在 prefill 图捕获
  warmup；换/禁 prefill 图都解决不了，eager prefill 同样会触发 JIT 编译）。
  tag 已验证 SWR 存在；镜像与 pip 栈无耦合（torch wheel 自带 runtime）

## 4) 验证

```bash
.venv/bin/python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.13.0+cu130  13.0  True  NVIDIA H20
.venv/bin/python3 -c "import sglang; print(sglang.__version__)"        # 0.5.18
.venv/bin/python3 -c "import sgl_kernel; print(sgl_kernel.__version__)"  # 0.4.6.post1
.venv/bin/python3 -c "import flashinfer; print(flashinfer.__version__)"  # 0.6.17
```

## 5) 起 server（不带 prefill 图禁用参数）

```bash
# 跑前 nvidia-smi 清残留；mem-fraction/ssm-dtype 参考
# docs/perf/reports/h20_qwen35_9b_sizing.md §8.4（0.85 + bfloat16 ssm）
.venv/bin/python3 -m sglang.launch_server \
    --model-path /workspace/models/Qwen3.5-9B \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 --mamba-ssm-dtype bfloat16
# 日志应出现 prefill 图（BCG）捕获成功；不应再报
# "CUDA driver version is insufficient"（error 35）
# 若仍报 error 35：驱动实际 <580，回到第 1 节前置检查
```

## 6) 跑基准

- 命令同 H800 流程（`sglang.benchmark.serving` / `one_batch`），但**不带**
  `--cuda-graph-backend-prefill=disabled`（60c 驱动 ≥580，prefill 走 BCG 图捕获；
  ≤8192 token 的 prefill 走图，超长上下文（如 131k 单发）超图范围走 eager）
- 日志 `tee` 进 `docs/perf/logs/`，命名沿用惯例（建议带 `h20_` 前缀区分）；
  跑完按第 2 节流程 kubectl cp 回 iZ 并 commit + push
- H20 bf16 峰值仅 148 TFLOPS（H800 的 ~1/6.7），短工况 prefill 是瓶颈；
  262k 上下文场景与卡数结论见 `docs/perf/reports/h20_qwen35_9b_sizing.md`
- 注意：H20（cu13 栈）与 H800（cu129 栈）不是同一栈，绝对值不做跨栈横比；
  H20 实测只与 sizing 文档的 H20 估算对照
- **/dev/shm 坑（2026-08-31 实测）**：K8s pod 默认 /dev/shm 仅 64Mi，而 NCCL 2.29 的
  P2P/IPC 传输要求每 rank（每进程）的 proxy 线程建一个 ~32.6 MiB 共享内存段
  （`/dev/shm/nccl-*`），tp=4 合计 ~131 MiB → tp≥2 直接 ENOSPC 崩在通信组初始化：
  `NCCL WARN ... failed to extend /dev/shm/nccl-... No space left on device (28)`
  → `RuntimeError: NCCL error: unhandled system error`（H20 9B one_batch tp4 首遇，
  首版日志作废）。dev-pod.yaml / dev-pod-2.yaml（2026-08-31 起）已挂 8Gi
  emptyDir(Memory) 的 /dev/shm，新 pod 免疫；旧 pod 若再遇：`df -h /dev/shm` 确认
  → `rm -f /dev/shm/nccl-*`（先确认无运行任务）→ 试
  `mount -o remount,size=8G /dev/shm`（无 CAP_SYS_ADMIN 时 EPERM）；不行只能按第
  2/3 节重建 pod 套新 yaml（可先把旧 pod 的 /workspace/models kubectl cp 回 iZ，
  省 18GB 重下载）
