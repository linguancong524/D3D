#!/usr/bin/env bash
set -euo pipefail

# # Qwen3-8B target with the 1.7B SDAR drafter.
# CUDA_VISIBLE_DEVICES=0 \
# VERIFIER_DEVICE=cuda:0 \
# DRAFT_DEVICE=cuda:0 \
# MAX_NEW_TOKENS=2048 \
# DATASETS=7bench \
# BLOCK_SIZE=16 \
# TEMPERATURE=1.0 \
# PROPOSAL_TOP_K=64 \
# PROPOSAL_TOP_P=1.0 \
# TARGET_TOP_K=64 \
# TARGET_TOP_P=1.0 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_full_temperature_1/1.7bq8bb16 \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh

# # Qwen3-8B target with the 8B SDAR drafter.
# CUDA_VISIBLE_DEVICES=0 \
# VERIFIER_DEVICE=cuda:0 \
# DRAFT_DEVICE=cuda:0 \
# MAX_NEW_TOKENS=2048 \
# DATASETS=7bench \
# BLOCK_SIZE=16 \
# TEMPERATURE=1.0 \
# PROPOSAL_TOP_K=64 \
# PROPOSAL_TOP_P=1.0 \
# TARGET_TOP_K=64 \
# TARGET_TOP_P=1.0 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_full_temperature_1/8bq8bb16 \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh

# Qwen3-8B target with the 8B drafter whose fused core uses 31 layers.
CUDA_VISIBLE_DEVICES=0 \
VERIFIER_DEVICE=cuda:0 \
DRAFT_DEVICE=cuda:0 \
MAX_NEW_TOKENS=2048 \
DATASETS="aime25 math500" \
BLOCK_SIZE=16 \
TEMPERATURE=1.0 \
PROPOSAL_TOP_K=64 \
PROPOSAL_TOP_P=1.0 \
TARGET_TOP_K=64 \
TARGET_TOP_P=1.0 \
DRAFTER=/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/APLF/results/progressive/b16_progressive_seed42/d31/train/step-16000 \
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_full_temperature_1/8b_cut31q8bb16 \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh

bash /inspire/ssd/project/sais-bio/public/linguancong/gpu_tool/occupy_gpu.sh
