#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python
exec "$PY" -u D3DV2/eval_async_packed.py \
 --drafter /inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908 \
 --verifier /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
 --dataset aime25 --max-new-tokens 2048 --block-size 16 --denoising-steps 1 \
 --temperature 0 --proposal-top-k 64 --target-top-k 64 --use-kvcache \
 --attention-mask-mode causal --target-attn-implementation sdpa --draft-attn-implementation sdpa \
 --verifier-device cuda:0 --draft-device cuda:1 --anchor-num 3 --packed-accepted-counts all \
 --output-json D3D_concurrency/results/aime_diagnosis/original_rerun.json \
 --summary-json D3D_concurrency/results/aime_diagnosis/original_rerun.summary.json
