#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
# Keep the same eager kernels as the recorded thread/process comparison.
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PWD/vendor:$PWD/..:${PYTHONPATH:-}"
PY="${D3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}"
exec "$PY" -u bench.py "$@"
