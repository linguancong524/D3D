#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON_BIN" -m compileall -q single serving models common
"$PYTHON_BIN" - <<'PY'
import sys

import torch
import transformers

import single.eval_sync
import single.eval_async_packed
import serving.engine
import serving.backend
import models.d3d.modeling_d3d

assert torch.__version__.startswith("2.6"), torch.__version__
assert transformers.__version__.startswith("4.52"), transformers.__version__
assert torch.cuda.is_available(), "CUDA is required for the D3D smoke test"
assert torch.cuda.device_count() >= 2, "D3D requires two CUDA devices"
print("D3D import and hardware smoke test passed")
PY
