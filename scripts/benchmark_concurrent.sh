#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASETS="${D3D_DATASETS:-gsm8k math500 humaneval mbpp mt-bench aime25 livecodebench-v6}"
OUTPUT="${D3D_CONCURRENT_OUTPUT:-outputs/concurrent/benchmark.json}"
BASELINE_FLAG=""
[ "${D3D_BASELINE:-1}" = "0" ] && BASELINE_FLAG="--no-baseline"
OUTPUT_BASE="${OUTPUT%.json}"

echo "Running concurrent D3D benchmark with target=$D3D_TARGET_MODEL draft=$D3D_DRAFT_MODEL"
echo "Datasets: $DATASETS"

for CONCURRENCY in ${D3D_CONCURRENCY:-1 4 8 16}; do
  if [ "$CONCURRENCY" -gt 1 ]; then
    ANCHOR_NUM="${D3D_CONCURRENT_ANCHOR_NUM:-1}"
  else
    ANCHOR_NUM="${D3D_ANCHOR_NUM:-3}"
  fi
  echo "Concurrency $CONCURRENCY with anchor_num=$ANCHOR_NUM"
  "$PYTHON_BIN" serving/bench.py \
    --target "$D3D_TARGET_MODEL" \
    --draft "$D3D_DRAFT_MODEL" \
    --datasets $DATASETS \
    --concurrency "$CONCURRENCY" \
    --anchor-num "$ANCHOR_NUM" \
    --max-samples "${D3D_MAX_SAMPLES:-32}" \
    --max-new-tokens "${D3D_MAX_NEW_TOKENS:-2048}" \
    --pages "${D3D_PAGES:-512}" \
    --output "${OUTPUT_BASE}_c${CONCURRENCY}.json" \
    $BASELINE_FLAG
done
