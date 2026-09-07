#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2
EVAL_ASYNC="$ROOT/scripts/eval_async_packed.sh"
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B
DRAFTER_1_7B=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908
DRAFTER_8B=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227
DRAFTER_31D=/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/APLF/results/progressive/b16_progressive_seed42/d31/train/step-16000

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

env "${common[@]}" DRAFTER="$DRAFTER_1_7B" OUT_DIR="$ROOT/results_full_temperature_1/1.7bq8bb16" bash "$EVAL_ASYNC" --anchor-num 3 --packed-accepted-counts all

env "${common[@]}" DRAFTER="$DRAFTER_8B" OUT_DIR="$ROOT/results_full_temperature_1/8bq8bb16" bash "$EVAL_ASYNC" --anchor-num 1 --packed-accepted-counts all

env "${common[@]}" DRAFTER="$DRAFTER_31D" OUT_DIR="$ROOT/results_full_temperature_1/31dq8bb16" bash "$EVAL_ASYNC" --anchor-num 2 --packed-accepted-counts all
