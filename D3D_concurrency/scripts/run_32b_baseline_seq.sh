#!/usr/bin/env bash
# Single-card variant of the documented Qwen3-32B AR baseline run.
# The three datasets load the 62 GB verifier one at a time instead of in
# parallel, and the independent HF reference check is skipped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_32b_n32_t512_r2}"
TARGET="${TARGET:-/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(math500 humaneval mt-bench)
fi

echo "DRIVER start run_id=$RUN_ID cuda_visible_devices=$CUDA_VISIBLE_DEVICES target=$TARGET max_samples=$MAX_SAMPLES datasets=${DATASETS[*]}"
for ds in "${DATASETS[@]}"; do
  mkdir -p "results_baseline/32B/$ds"
  log="results_baseline/32B/$ds/${RUN_ID}_driver.log"
  echo "DRIVER begin dataset=$ds $(date -u +%FT%TZ)"
  bash scripts/baseline_concurrency.sh \
    --target "$TARGET" \
    --datasets "$ds" --concurrency 1 4 8 16 --max-samples "$MAX_SAMPLES" \
    --max-new-tokens 512 --warmup-new-tokens 256 --repeats 2 \
    --max-retests 1 --slow-fraction 0.85 --pages 96 \
    --skip-reference-validation \
    --output-root results_baseline/32B --run-id "$RUN_ID" \
    >"$log" 2>&1
  rc=$?
  echo "DRIVER end dataset=$ds rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "DRIVER aborting: dataset=$ds failed with rc=$rc"
    exit "$rc"
  fi
done
echo "DRIVER all datasets complete $(date -u +%FT%TZ)"
