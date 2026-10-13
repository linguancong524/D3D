#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT="${SGLANG_ENV:-$ROOT/.venv-sglang}"
uv venv --python "${SGLANG_PYTHON_VERSION:-3.12}" "$ENVIRONMENT"
uv pip install --python "$ENVIRONMENT/bin/python" --no-deps \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://docs.sglang.ai/whl/cu129/ \
  -r "$ROOT/scripts/sglang/runtime.lock"
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18 \
  uv pip install --python "$ENVIRONMENT/bin/python" --no-deps \
  -e "$ROOT/third_party/sglang/python" -e "$ROOT"
"$ENVIRONMENT/bin/python" -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
