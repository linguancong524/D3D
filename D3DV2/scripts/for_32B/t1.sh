#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2
EVAL_SYNC="$ROOT/scripts/eval_sync.sh"
EVAL_ASYNC="$ROOT/scripts/eval_async_packed.sh"
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B
DRAFTER_1_7B=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0_from255044_ce_3epoch_4gpu/step-1041183
DRAFTER_8B=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16_a128_ar1.0_from255576_ce_3epoch_4gpu/resume_from_step170000/resume_from_step140000/step-731183

common=(
  CUDA_VISIBLE_DEVICES=0,1
  MAX_SAMPLES=all
  MAX_NEW_TOKENS=2048
  DATASETS=7bench
  BLOCK_SIZE=16
  TEMPERATURE=1.0
  PROPOSAL_TOP_K=64
  PROPOSAL_TOP_P=1.0
  TARGET_TOP_K=64
  TARGET_TOP_P=1.0
  VERIFIER_DEVICE=cuda:0
  DRAFT_DEVICE=cuda:1
  VERIFIER="$VERIFIER"
)

env "${common[@]}" DRAFTER="$DRAFTER_1_7B" OUT_DIR="$ROOT/results_full_temperature_1/1.7bq32bb16" bash "$EVAL_SYNC"
env "${common[@]}" DRAFTER="$DRAFTER_1_7B" OUT_DIR="$ROOT/results_full_temperature_1/1.7bq32bb16" bash "$EVAL_ASYNC" --anchor-num 7 --packed-accepted-counts all

env "${common[@]}" DRAFTER="$DRAFTER_8B" OUT_DIR="$ROOT/results_full_temperature_1/8bq32bb16" bash "$EVAL_SYNC"
env "${common[@]}" DRAFTER="$DRAFTER_8B" OUT_DIR="$ROOT/results_full_temperature_1/8bq32bb16" bash "$EVAL_ASYNC" --anchor-num 3 --packed-accepted-counts all
