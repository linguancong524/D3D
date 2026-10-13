#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASETS="${DATASETS:-gsm8k math500 humaneval mbpp mt-bench aime25 livecodebench-v6}"

echo "Running concurrent D3D benchmark with target=$TARGET_MODEL draft=$DRAFT_MODEL"
echo "Datasets: $DATASETS"

for CONCURRENCY in ${CONCURRENCY:-1 4 8 16}; do
  if [ "$CONCURRENCY" -gt 1 ]; then
    ANCHOR_NUM=1
  else
    ANCHOR_NUM=3
  fi
  echo "Concurrency $CONCURRENCY with anchor_num=$ANCHOR_NUM"
  "$PYTHON_BIN" serving/bench.py \
    --target "$TARGET_MODEL" \
    --draft "$DRAFT_MODEL" \
    --datasets $DATASETS \
    --concurrency "$CONCURRENCY" \
    --anchor-num "$ANCHOR_NUM" \
    --max-samples 128 \
    --max-new-tokens 2048 \
    --pages 512 \
    --output "outputs/concurrent/benchmark_c${CONCURRENCY}.json" \
    "$@"
done
