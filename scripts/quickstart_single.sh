#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p outputs/single
exec "$PYTHON_BIN" single/eval_async_packed.py \
  --drafter "$DRAFT_MODEL" \
  --verifier "$TARGET_MODEL" \
  --dataset math500 \
  --max-samples 1 \
  --max-new-tokens 128 \
  --verifier-device cuda:0 \
  --draft-device cuda:1 \
  --dtype bf16 \
  --temperature 0 \
  --block-size 16 \
  --denoising-steps 1 \
  --attention-mask-mode causal \
  --anchor-num 3 \
  --packed-accepted-counts all \
  --output-json outputs/single/result.json \
  --summary-json outputs/single/summary.json \
  "$@"
