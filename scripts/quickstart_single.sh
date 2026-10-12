#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET="${D3D_DATASET:-math500}"
MAX_SAMPLES="${D3D_MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${D3D_MAX_NEW_TOKENS:-128}"
OUTPUT_DIR="${D3D_OUTPUT_DIR:-outputs/single}"
BASELINE_FLAG=""
[ "${D3D_BASELINE:-1}" = "0" ] && BASELINE_FLAG="--no-baseline"

mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" single/eval_async_packed.py \
  --drafter "$D3D_DRAFT_MODEL" \
  --verifier "$D3D_TARGET_MODEL" \
  --dataset "$DATASET" \
  --max-samples "$MAX_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --verifier-device cuda:0 \
  --draft-device cuda:1 \
  --dtype bf16 \
  --temperature 0 \
  --block-size "${D3D_BLOCK_SIZE:-16}" \
  --denoising-steps "${D3D_DENOISING_STEPS:-1}" \
  --attention-mask-mode causal \
  --anchor-num "${D3D_ANCHOR_NUM:-3}" \
  --packed-accepted-counts all \
  --output-json "${OUTPUT_DIR}/result.json" \
  --summary-json "${OUTPUT_DIR}/summary.json" \
  $BASELINE_FLAG
