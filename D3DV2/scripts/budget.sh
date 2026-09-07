#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
DATASETS="math500"
RESULT_ROOT="${OUT_DIR:-/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_budget}"
EVAL_SCRIPT="/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh"

QWEN3_8B="/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B"
QWEN3_32B="/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B"

DRAFTER_1_7B="/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-840000"
DRAFTER_8B_B16="/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-250000"
DRAFTER_8B_B32="/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b32_a64_ar1.0/step-250000"

accepted_counts_for_n() {
  local n="$1"
  local block_size="$2"
  local counts=()
  local position

  for ((position = 1; position < n; position++)); do
    counts+=("$position")
  done
  counts+=("$block_size")
  local IFS=,
  echo "${counts[*]}"
}

run_grid() {
  local name="$1"
  local drafter="$2"
  local verifier="$3"
  local block_size="$4"
  local anchor_num n accepted_counts out_dir

  for anchor_num in 1 2 3 4 5; do
    for n in 2 4 8 12 16; do
      accepted_counts="$(accepted_counts_for_n "$n" "$block_size")"
      out_dir="${RESULT_ROOT}/${name}/k${anchor_num}_n${n}"

      echo "[budget] model=${name} anchor_num=${anchor_num} accepted_counts=${accepted_counts} out=${out_dir}"
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      VERIFIER_DEVICE="$VERIFIER_DEVICE" \
      DRAFT_DEVICE="$DRAFT_DEVICE" \
      MAX_SAMPLES="$MAX_SAMPLES" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      DATASETS="$DATASETS" \
      BLOCK_SIZE="$block_size" \
      DRAFTER="$drafter" \
      VERIFIER="$verifier" \
      OUT_DIR="$out_dir" \
      bash "$EVAL_SCRIPT" \
        --anchor-num "$anchor_num" \
        --packed-accepted-counts "$accepted_counts"
    done
  done
}

run_grid "1.7bq8bb16" "$DRAFTER_1_7B" "$QWEN3_8B" 16
run_grid "8bq32bb16" "$DRAFTER_8B_B16" "$QWEN3_32B" 16
run_grid "8bq8bb16" "$DRAFTER_8B_B16" "$QWEN3_8B" 16
run_grid "8bq8bb32" "$DRAFTER_8B_B32" "$QWEN3_8B" 32
