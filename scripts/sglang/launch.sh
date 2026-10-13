#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SGLANG_PYTHON:-$ROOT/.venv-sglang/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
MODE="${1:-remote}"
shift || true
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-lingc8/D3D-1.7B-to-8B}"
ENDPOINT="${DRAFT_ENDPOINT:-tcp://127.0.0.1:17891}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="$NO_PROXY"
if [[ "$MODE" == "draft" ]]; then
  export CUDA_VISIBLE_DEVICES="${DRAFT_GPU:-1}"
  DRAFT_EXTRA=()
  if [[ "${DRAFT_CUDA_GRAPH:-1}" == "1" ]]; then
    DRAFT_EXTRA+=(--cuda-graph)
  fi
  if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
    DRAFT_EXTRA+=(--deterministic)
  fi
  exec "$PYTHON_BIN" -m d3d_sglang.service \
    --model "$DRAFT_MODEL" --tokenizer "$TARGET_MODEL" --endpoint "$ENDPOINT" \
    --block-size "${BLOCK_SIZE:-16}" --anchors "${ANCHORS:-1}" \
    --max-branches "${MAX_BRANCHES:-$(( ${BLOCK_SIZE:-16} * ${ANCHORS:-1} ))}" \
    --packed-graph-max-requests "${MAX_RUNNING_REQUESTS:-1}" "${DRAFT_EXTRA[@]}" "$@"
fi
if [[ "$MODE" != "remote" && "$MODE" != "ar" ]]; then
  echo "Usage: $0 {draft|remote|ar} [arguments]" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${TARGET_GPU:-0}"
EXTRA=()
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
  EXTRA+=(--enable-deterministic-inference)
fi
if [[ "$MODE" == "remote" ]]; then
  EXTRA+=(--speculative-algorithm D3D_REMOTE \
    --speculative-num-draft-tokens "${BLOCK_SIZE:-16}" \
    --decoupled-draft-endpoint "$ENDPOINT" --decoupled-draft-timeout 120)
fi
exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --host 127.0.0.1 --port "${TARGET_PORT:-30000}" \
  --dtype bfloat16 --tp-size 1 --context-length 8192 \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-1}" \
  --mem-fraction-static 0.7 --attention-backend fa3 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE:-${MAX_RUNNING_REQUESTS:-1}}" \
  "${EXTRA[@]}" "$@"
