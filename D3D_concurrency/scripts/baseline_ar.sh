#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${D3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PWD/vendor:$PWD/..:${PYTHONPATH:-}"
exec "$PY" -u baseline_ar.py "$@"
