#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASETS="${D3D_DATASETS:-gsm8k math500 humaneval mbpp mt-bench aime25 livecodebench-v6}"
OUTPUT_DIR="${D3D_OUTPUT_DIR:-outputs/benchmark}"
BASELINE_FLAG=""
[ "${D3D_BASELINE:-1}" = "0" ] && BASELINE_FLAG="--no-baseline"
mkdir -p "$OUTPUT_DIR"

echo "Running D3D benchmark with target=$D3D_TARGET_MODEL draft=$D3D_DRAFT_MODEL"
echo "Datasets: $DATASETS"

for dataset in $DATASETS; do
  "$PYTHON_BIN" single/eval_async_packed.py \
    --drafter "$D3D_DRAFT_MODEL" \
    --verifier "$D3D_TARGET_MODEL" \
    --dataset "$dataset" \
    --max-samples "${D3D_MAX_SAMPLES:-32}" \
    --max-new-tokens "${D3D_MAX_NEW_TOKENS:-2048}" \
    --verifier-device cuda:0 \
    --draft-device cuda:1 \
    --dtype bf16 \
    --temperature 0 \
    --block-size "${D3D_BLOCK_SIZE:-16}" \
    --denoising-steps "${D3D_DENOISING_STEPS:-1}" \
    --attention-mask-mode causal \
    --anchor-num "${D3D_ANCHOR_NUM:-3}" \
    --packed-accepted-counts all \
    --output-json "${OUTPUT_DIR}/${dataset}.json" \
    --summary-json "${OUTPUT_DIR}/${dataset}.summary.json" \
    $BASELINE_FLAG
done
