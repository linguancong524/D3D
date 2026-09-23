#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_ID=${1:-$(date -u +%Y%m%d_%H%M%S)}
MAX_SAMPLES=${MAX_SAMPLES:-32}
MODEL_SUBDIR=${MODEL_SUBDIR:-8B}
TARGET=${TARGET:-$PWD/../models/Qwen3-8B}
PAGES=${PAGES:-512}
REFERENCE_DEVICE=${REFERENCE_DEVICE:-cuda:0}
for dataset in math500 humaneval mt-bench; do
  mkdir -p "results_baseline/$MODEL_SUBDIR/$dataset"
  bash scripts/baseline_concurrency.sh --datasets "$dataset" \
    --concurrency 1 4 8 16 --max-samples "$MAX_SAMPLES" --max-new-tokens 512 \
    --warmup-new-tokens 256 --repeats 2 \
    --max-retests 1 --slow-fraction 0.85 --run-id "$RUN_ID" \
    --target "$TARGET" --pages "$PAGES" --reference-device "$REFERENCE_DEVICE" \
    --output-root "results_baseline/$MODEL_SUBDIR" \
    >"results_baseline/$MODEL_SUBDIR/$dataset/${RUN_ID}_driver.log" 2>&1
done
