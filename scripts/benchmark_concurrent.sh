#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASETS="${D3D_DATASETS:-gsm8k math500 humaneval mbpp mt-bench aime25 livecodebench-v6}"
OUTPUT="${D3D_CONCURRENT_OUTPUT:-outputs/concurrent/benchmark.json}"
BASELINE_FLAG=""
[ "${D3D_BASELINE:-1}" = "0" ] && BASELINE_FLAG="--no-baseline"

echo "Running concurrent D3D benchmark with target=$D3D_TARGET_MODEL draft=$D3D_DRAFT_MODEL"
echo "Datasets: $DATASETS"

"$PYTHON_BIN" serving/bench.py \
  --target "$D3D_TARGET_MODEL" \
  --draft "$D3D_DRAFT_MODEL" \
  --datasets $DATASETS \
  --concurrency ${D3D_CONCURRENCY:-1 4 8 16} \
  --anchor-num "${D3D_ANCHOR_NUM:-1}" \
  --max-samples "${D3D_MAX_SAMPLES:-32}" \
  --max-new-tokens "${D3D_MAX_NEW_TOKENS:-2048}" \
  --pages "${D3D_PAGES:-512}" \
  --output "$OUTPUT" \
  $BASELINE_FLAG
