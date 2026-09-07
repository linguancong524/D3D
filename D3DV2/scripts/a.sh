#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 5 --packed-accepted-counts 1,2,3,4,5,6,7,16

CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 6 --packed-accepted-counts 1,2,3,4,5,6,7,16

CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 7 --packed-accepted-counts 1,2,3,4,5,6,7,16

CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 1 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 0 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0/step-250000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 4 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0/step-250000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 5 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0/step-250000    \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 6 --packed-accepted-counts all


# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0/step-250000    \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 7 --packed-accepted-counts all





CUDA_VISIBLE_DEVICES=0,1 \
MAX_NEW_TOKENS=2048 \
DATASETS=7bench \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16_a128_ar1.0_from255576_ce_3epoch_4gpu/step-170000 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_smoke/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 3 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 5 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 6 --packed-accepted-counts all


# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16-a128-ar1.0/step-255576 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq32bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 7 --packed-accepted-counts all


