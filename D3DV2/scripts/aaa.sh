# source /inspire/ssd/project/sais-bio/public/chenbaoyou/miniconda3/bin/activate nemo
# tmux new -s run


CUDA_VISIBLE_DEVICES=0 \
MAX_NEW_TOKENS=2048 \
DATASETS=gsm8k \
MAX_SAMPLES=100 \
BLOCK_SIZE=16 \
TEMPERATURE=0.5 \
PROPOSAL_TOP_K=64 \
PROPOSAL_TOP_P=1.0 \
TARGET_TOP_K=64 \
TARGET_TOP_P=1.0 \
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq8bb16_tem0.5  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh 

CUDA_VISIBLE_DEVICES=0 \
MAX_NEW_TOKENS=2048 \
DATASETS=gsm8k \
MAX_SAMPLES=100 \
BLOCK_SIZE=16 \
TEMPERATURE=0.8 \
PROPOSAL_TOP_K=64 \
PROPOSAL_TOP_P=1.0 \
TARGET_TOP_K=64 \
TARGET_TOP_P=1.0 \
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq8bb16_tem0.8  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh 

CUDA_VISIBLE_DEVICES=0 \
MAX_NEW_TOKENS=2048 \
DATASETS=gsm8k \
MAX_SAMPLES=100 \
BLOCK_SIZE=16 \
TEMPERATURE=1.0 \
PROPOSAL_TOP_K=64 \
PROPOSAL_TOP_P=1.0 \
TARGET_TOP_K=64 \
TARGET_TOP_P=1.0 \
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq8bb16_tem1.0  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh 


# CUDA_VISIBLE_DEVICES=0 \
# MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# MAX_SAMPLES=30 \
# BLOCK_SIZE=16 \
# TEMPERATURE=1.0 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sampling/sdar4b_b16_a128_ar1.0/step-350000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/4bq8bb16/temperature_1.0  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh 

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_NEW_TOKENS=2048 \
# DATASETS=7bench \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/APLF/results/progressive/b16_progressive_seed42/d31/train/step-15000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_full/8bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 1 --packed-accepted-counts all


# CUDA_VISIBLE_DEVICES=0,1 \
# DATASETS=7bench \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_full/8bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 1 --packed-accepted-counts all



bash /inspire/ssd/project/sais-bio/public/linguancong/gpu_tool/occupy_gpu.sh