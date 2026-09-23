#!/usr/bin/env bash
# Full-sample speculative concurrency suite for both 1.7B draft pairs.
# Runs sequentially on cuda:0 (verifier) + cuda:1 (draft); the 32B pair needs
# pages=96 because 62 GB of weights plus a 512-page KV pool does not fit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${D3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}
MODELS=${MODELS:-/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models}
DRAFT_1_7B_8B=${DRAFT_1_7B_8B:-/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908}
DRAFT_1_7B_32B=${DRAFT_1_7B_32B:-/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0_from255044_ce_3epoch_4gpu/step-1041183}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PWD/vendor:$PWD/..:${PYTHONPATH:-}"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}

run_pair () {
  local pair="$1" target="$2" draft="$3" pages="$4" anchors="$5"
  local root="results_full/$pair"
  mkdir -p "$root"
  echo "[$(date -u +%FT%TZ)] $pair start target=$target pages=$pages anchors=$anchors" |
    tee -a "$root/driver_${RUN_ID}.log"
  "$PY" -u sweep_full.py \
    --pair "$pair" --output-root "$root" \
    --target "$target" --draft "$draft" --pages "$pages" \
    --datasets math500 humaneval mt-bench \
    --concurrency 1 4 8 16 --anchor-map "$anchors" \
    --max-samples 0 --max-new-tokens 512 --warmup-new-tokens 256 \
    --repeats 2 --max-retests 1 --slow-fraction 0.85 \
    --run-id "${RUN_ID}_nfull_t512_r2" --skip-existing \
    >>"$root/driver_${RUN_ID}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] $pair complete" | tee -a "$root/driver_${RUN_ID}.log"
}

# C=1 prefers a longer anchor (3 for 8B, 7 for 32B); multi-concurrency uses 1.
run_pair 1.7b_8b  "$MODELS/Qwen3-8B"  "$DRAFT_1_7B_8B"  512 "1:3,4:1,8:1,16:1"
run_pair 1.7b_32b "$MODELS/Qwen3-32B" "$DRAFT_1_7B_32B"  96 "1:7,4:1,8:1,16:1"

echo "[$(date -u +%FT%TZ)] suite complete" | tee -a "results_full/driver_${RUN_ID}.log"
