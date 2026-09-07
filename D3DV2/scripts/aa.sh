
# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/8bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 2 --packed-accepted-counts all



# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/layer_fusion/results/recovery/b16_d24_function_fused_acceptance_from_step2000_2gpu/step-12000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 4 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/layer_fusion/results/recovery/b16_d24_function_fused_acceptance_from_step2000_2gpu/step-12000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results/1.7bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh


# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-840000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_smoke/1.7bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 4 --packed-accepted-counts all

# CUDA_VISIBLE_DEVICES=0,1 \
# MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
# DATASETS=math500 \
# BLOCK_SIZE=16 \
# VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
# DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-840000 \
# OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_smoke/1.7bq8bb16  \
# bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_sync.sh








