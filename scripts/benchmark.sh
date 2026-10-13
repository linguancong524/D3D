#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASETS="${DATASETS:-gsm8k math500 humaneval mbpp mt-bench aime25 livecodebench-v6}"
mkdir -p outputs/benchmark

echo "Running D3D benchmark with target=$TARGET_MODEL draft=$DRAFT_MODEL"
echo "Datasets: $DATASETS"

for dataset in $DATASETS; do
  "$PYTHON_BIN" single/eval_async_packed.py \
    --drafter "$DRAFT_MODEL" \
    --verifier "$TARGET_MODEL" \
    --dataset "$dataset" \
    --max-samples 32 \
    --max-new-tokens 2048 \
    --verifier-device cuda:0 \
    --draft-device cuda:1 \
    --dtype bf16 \
    --temperature 0 \
    --block-size 16 \
    --denoising-steps 1 \
    --attention-mask-mode causal \
    --anchor-num 3 \
    --packed-accepted-counts all \
    --output-json "outputs/benchmark/${dataset}.json" \
    --summary-json "outputs/benchmark/${dataset}.summary.json" \
    "$@"
done
