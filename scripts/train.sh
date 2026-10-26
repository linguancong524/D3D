#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Edit paths and settings here. One option per line; command-line options override defaults.
# Use `wandb login` once, or export WANDB_MODE=offline. Never put API keys in this file.
"${PY:-python}" -m torch.distributed.run \
    --nproc_per_node "${NUM_GPUS:-4}" \
    --master_port "${MASTER_PORT:-29674}" \
    --module train.train \
    --draft-model "${DRAFT_MODEL:-checkpoints/Qwen3-1.7B}" \
    --draft-code-model "$PWD/models/d3d_training" \
    --tokenizer-model "${TARGET_MODEL:-checkpoints/Qwen3-8B}" \
    --data-path "${DATA_PATH:-data/train.jsonl}" \
    --output-dir "${OUTPUT_DIR:-outputs/d3d_b16_a128}" \
    --objective ce \
    --ce-weight 1.0 \
    --ar-loss-weight 1.0 \
    --block-size 16 \
    --parallel-anchors-per-example 128 \
    --loss-decay-gamma 7.75 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --ar-logit-chunk-size 512 \
    --epochs 1 \
    --lr 1e-5 \
    --min-lr-ratio 0.5 \
    --warmup-steps 200 \
    --weight-decay 0.0 \
    --grad-clip 1.0 \
    --max-prompt-len 4096 \
    --max-seq-len 8192 \
    --save-every 5000 \
    --log-every 1 \
    --workers 8 \
    --prefetch 2 \
    --dtype bf16 \
    --attn-implementation flash_attention_2 \
    --disable-thinking-template \
    --fsdp \
    --fsdp-reshard-after-forward \
    --wandb-project d3d \
    --wandb-name d3d_b16_a128_ce_ar \
    "$@"
