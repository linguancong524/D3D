#!/usr/bin/env bash
set -eo pipefail

# Unified speculative accept-length benchmark:
# eval_sync.sh only resolves runtime/paths; eval_sync.py owns Bard/SDAR draft-verify.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
T3D_ROOT="$(cd "${ROOT}/.." && pwd)"
FUN_SPACE="$(cd "${T3D_ROOT}/.." && pwd)"
PUBLIC_ROOT="$(cd "${FUN_SPACE}/.." && pwd)"

cd "$ROOT"

setup_runtime() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export BARD_REPO_ROOT="${BARD_REPO_ROOT:-${T3D_ROOT}/models/bard_vl_b32_mask_8b}"
  export PYTHONPATH="${T3D_ROOT}:${FUN_SPACE}/dflash:${BARD_REPO_ROOT}:${PYTHONPATH:-}"
  export T3D_PYTHON="${T3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}"
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

set_defaults() {
  PY="${PY:-${PYTHON:-$T3D_PYTHON}}"

  DATASET="${DATASET:-7bench}"
  # humaneval mbpp mt-bench math500 gsm8k aime25 livecodebench-v6 5bench
  DATASETS="${DATASETS:-$DATASET}"
  MAX_SAMPLES="${MAX_SAMPLES:-all}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DENOISING_STEP="${DENOISING_STEP:-${DENOISING_STEPS:-1}}"
  BLOCK_SIZE="${BLOCK_SIZE:-${BARD_BLOCK_SIZE:-8}}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  DRAFTER_TAG="${DRAFTER_TAG:-${DRAFT_NAME:-}}"
  DRAFTER="${DRAFTER:-${DRAFT_MODEL:-${PUBLIC_ROOT}/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-8-a128-v2/step-80000}}"
  # Override VERIFIER for Qwen3.6-27B, for example:
  # VERIFIER=${FUN_SPACE}/models/Qwen3.6-27B bash scripts/eval_sync.sh
  VERIFIER="${VERIFIER:-${TARGET_MODEL:-/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B}}"
  TARGET_ATTN_IMPLEMENTATION="${TARGET_ATTN_IMPLEMENTATION:-sdpa}"
  DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"

  TEMPERATURE="${TEMPERATURE:-0.0}"
  PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-64}"
  PROPOSAL_TOP_P="${PROPOSAL_TOP_P:-1.0}"
  TARGET_TOP_K="${TARGET_TOP_K:-64}"
  TARGET_TOP_P="${TARGET_TOP_P:-1.0}"
  CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.9}"
  DTYPE="${DTYPE:-bf16}"
  DEVICE="${DEVICE:-cuda:0}"
  VERIFIER_DEVICE="${VERIFIER_DEVICE:-$DEVICE}"
  DRAFT_DEVICE="${DRAFT_DEVICE:-$DEVICE}"
  USE_KVCACHE="${USE_KVCACHE:-1}"
  COLLECT_STAGE_TIMINGS="${COLLECT_STAGE_TIMINGS:-false}"
  REMASKING_STRATEGY="${REMASKING_STRATEGY:-low_confidence}"
  ATTENTION_MASK_MODE="${ATTENTION_MASK_MODE:-causal}"

  TREE_MODE="${TREE_MODE:-none}"
  TREE_BUDGET="${TREE_BUDGET:-128}"
  TREE_TOPK="${TREE_TOPK:-8}"
  TREE_BRANCH_START_DEPTH="${TREE_BRANCH_START_DEPTH:-0}"
  TREE_RUN_TAG="${TREE_RUN_TAG:-${TREE_MODE}_budget${TREE_BUDGET}_topk${TREE_TOPK}_branch${TREE_BRANCH_START_DEPTH}}"

  DRAFT_BACKEND="${DRAFT_BACKEND:-direct}"
  TOKENIZER="${TOKENIZER:-$VERIFIER}"
  # OUT_DIR=


  if [ "$TREE_MODE" = "ddtree" ]; then
    export TARGET_ATTN_IMPLEMENTATION="${TARGET_ATTN_IMPLEMENTATION:-sdpa}"
  fi
}

log() {
  echo "[sdar-t3d] $*"
}

die() {
  echo "[sdar-t3d] $*" >&2
  exit 2
}

parse_datasets() {
  NORMALIZED_DATASETS=()
  local item

  DATASETS="${DATASETS//,/ }"
  for item in $DATASETS; do
    if [ "$item" = "5bench" ]; then
      NORMALIZED_DATASETS+=(humaneval mbpp mt-bench math500 gsm8k)
    elif [ "$item" = "7bench" ]; then
      NORMALIZED_DATASETS+=(aime25 humaneval mbpp mt-bench math500 livecodebench-v6 gsm8k )
    else
      NORMALIZED_DATASETS+=("$item")
    fi
  done

  [ "${#NORMALIZED_DATASETS[@]}" -gt 0 ] || die "no datasets requested"
}

is_supported_dataset() {
  case "$1" in
    gsm8k|math500|aime25|humaneval|mbpp|mt-bench|livecodebench-v6) return 0 ;;
    *) return 1 ;;
  esac
}

validate_datasets() {
  local dataset
  for dataset in "${NORMALIZED_DATASETS[@]}"; do
    if ! is_supported_dataset "$dataset"; then
      die "unsupported dataset: $dataset; supported: gsm8k math500 aime25 humaneval mbpp mt-bench livecodebench-v6"
    fi
  done
}

resolve_metadata() {
  IFS='|' read -r MODEL_TYPE DRAFT_TAG VERIFIER_TAG < <(
    "$PY" "${ROOT}/scripts/resolve_run_metadata.py" \
      --drafter "$DRAFTER" \
      --drafter-tag "$DRAFTER_TAG" \
      --verifier "$VERIFIER"
  )

  MODEL_TYPE="${MODEL_TYPE:-}"
  DRAFT_TAG="${DRAFT_TAG:-drafter}"
  VERIFIER_TAG="${VERIFIER_TAG:-model}"
}

choose_result_root() {
  if [ "$TREE_MODE" = "none" ]; then
    RESULT_ROOT="${OUT_DIR:-results_D3D/temperature_${TEMPERATURE}/${DRAFT_TAG}}"
  else
    RESULT_ROOT="${OUT_DIR:-results_tree_probe/temperature_${TEMPERATURE}/${DRAFT_TAG}/${TREE_RUN_TAG}}"
  fi
}

print_run_header() {
  log "drafter=$DRAFTER"
  log "verifier=$VERIFIER"
  log "model_type=${MODEL_TYPE:-sdar-or-other}"
  log "datasets=${NORMALIZED_DATASETS[*]}"
  log "block_size=$BLOCK_SIZE denoising_step=$DENOISING_STEP max_new_tokens=$MAX_NEW_TOKENS"
  log "temperature=$TEMPERATURE proposal_top_k=$PROPOSAL_TOP_K proposal_top_p=$PROPOSAL_TOP_P target_top_k=$TARGET_TOP_K target_top_p=$TARGET_TOP_P confidence_threshold=$CONFIDENCE_THRESHOLD"
  log "attention_mask_mode=$ATTENTION_MASK_MODE"
  log "verifier_device=$VERIFIER_DEVICE draft_device=$DRAFT_DEVICE"
  log "out_root=$RESULT_ROOT"
  log "tree_mode=$TREE_MODE draft_backend=$DRAFT_BACKEND use_kvcache=$USE_KVCACHE"
  log "collect_stage_timings=$COLLECT_STAGE_TIMINGS"
}

append_common_limit_args() {
  local -n args_ref=$1
  if [ "$MAX_SAMPLES" != "all" ] && [ -n "$MAX_SAMPLES" ]; then
    args_ref+=(--max-samples "$MAX_SAMPLES")
  fi
  if [ -n "${SAMPLE_INDEX:-}" ]; then
    args_ref+=(--sample-index "$SAMPLE_INDEX")
  fi
}

build_acceptance_args() {
  local dataset=$1
  local result_json=$2
  local summary_json=$3

  ACCEPT_ARGS=(
    eval_sync.py
    --drafter "$DRAFTER"
    --verifier "$VERIFIER"
    --tokenizer "$TOKENIZER"
    --model-type "$MODEL_TYPE"
    --drafter-tag "$DRAFT_TAG"
    --dataset "$dataset"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --proposal-top-k "$PROPOSAL_TOP_K"
    --proposal-top-p "$PROPOSAL_TOP_P"
    --target-top-k "$TARGET_TOP_K"
    --target-top-p "$TARGET_TOP_P"
    --verifier-device "$VERIFIER_DEVICE"
    --draft-device "$DRAFT_DEVICE"
    --dtype "$DTYPE"
    --block-size "$BLOCK_SIZE"
    --denoising-steps "$DENOISING_STEP"
    --remasking-strategy "$REMASKING_STRATEGY"
    --confidence-threshold "$CONFIDENCE_THRESHOLD"
    --attention-mask-mode "$ATTENTION_MASK_MODE"
    --draft-backend "$DRAFT_BACKEND"
    --target-attn-implementation "$TARGET_ATTN_IMPLEMENTATION"
    --draft-attn-implementation "$DRAFT_ATTN_IMPLEMENTATION"
    --eos-token-id 151643
    --eos-token-id 151645
    --tree-mode "$TREE_MODE"
    --tree-budget "$TREE_BUDGET"
    --tree-topk "$TREE_TOPK"
    --tree-branch-start-depth "$TREE_BRANCH_START_DEPTH"
    --output-json "$result_json"
    --summary-json "$summary_json"
  )

  append_common_limit_args ACCEPT_ARGS

  if [ -n "${MASK_TOKEN_ID:-}" ]; then
    ACCEPT_ARGS+=(--mask-token-id "$MASK_TOKEN_ID")
  fi
  if [ "${VERBOSE:-0}" = "1" ] || [ "${VERBOSE:-0}" = "true" ]; then
    ACCEPT_ARGS+=(--verbose)
  fi
  if [ "${NO_TARGET_FIRST_TOKEN:-0}" = "1" ] || [ "${NO_TARGET_FIRST_TOKEN:-0}" = "true" ]; then
    ACCEPT_ARGS+=(--no-target-first-token)
  fi
  if [ "$USE_KVCACHE" = "1" ] || [ "$USE_KVCACHE" = "true" ]; then
    ACCEPT_ARGS+=(--use-kvcache)
  fi
  if [ "$COLLECT_STAGE_TIMINGS" = "1" ] || [ "$COLLECT_STAGE_TIMINGS" = "true" ]; then
    ACCEPT_ARGS+=(--collect-stage-timings)
  fi
}

run_dataset() {
  local dataset=$1
  local timestamp_out_dir="${RESULT_ROOT}/${RUN_TIMESTAMP}/${dataset}"
  local mask_tag=""
  if [ "$ATTENTION_MASK_MODE" != "block_causal" ]; then
    mask_tag="_attn${ATTENTION_MASK_MODE}"
  fi
  local out_prefix="${timestamp_out_dir}/${dataset}_${DRAFT_TAG}_with_${VERIFIER_TAG}_b${BLOCK_SIZE}_s${DENOISING_STEP}_usecache${USE_KVCACHE}${mask_tag}"
  local result_json="${out_prefix}.json"
  local summary_json="${out_prefix}.summary.json"

  mkdir -p "$timestamp_out_dir"
  build_acceptance_args "$dataset" "$result_json" "$summary_json"

  log "START dataset=$dataset json=$result_json"
  "$PY" "${ACCEPT_ARGS[@]}"
  log "DONE dataset=$dataset summary=$summary_json"
}

run_all() {
  local dataset

  validate_datasets
  choose_result_root
  mkdir -p "$RESULT_ROOT"
  print_run_header

  for dataset in "${NORMALIZED_DATASETS[@]}"; do
    run_dataset "$dataset"
  done
}

main() {
  setup_runtime
  RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ=Asia/Shanghai date '+%Y_%m_%d_%H_%M')}"
  set_defaults
  parse_datasets
  resolve_metadata
  run_all
}

main "$@"
