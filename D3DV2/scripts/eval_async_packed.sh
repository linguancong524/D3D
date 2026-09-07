#!/usr/bin/env bash
set -eo pipefail

# Two-GPU all-acceptance-position packed speculative benchmark.
# eval_async_packed.py owns the packed outcome draft/verify pipeline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
T3D_ROOT="$(cd "${ROOT}/.." && pwd)"
FUN_SPACE="$(cd "${ROOT}/../.." && pwd)"
PUBLIC_ROOT="$(cd "${FUN_SPACE}/../.." && pwd)"

cd "$ROOT"

setup_runtime() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export BARD_REPO_ROOT="${BARD_REPO_ROOT:-${T3D_ROOT}/models}"
  export PYTHONPATH="${T3D_ROOT}:${FUN_SPACE}/dflash:${PYTHONPATH:-}"
  export T3D_PYTHON="${T3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}"
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

set_defaults() {
  PY="${PY:-${PYTHON:-$T3D_PYTHON}}"

    # CUDA_VISIBLE_DEVICES=0 \
    # VERIFIER_DEVICE=cuda:0 \
    # DRAFT_DEVICE=cuda:0 \

  DATASET="${DATASET:-5bench}"
  MAX_SAMPLES="${MAX_SAMPLES:-all}"
  DATASETS="${DATASETS:-$DATASET}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  DENOISING_STEP="${DENOISING_STEP:-${DENOISING_STEPS:-1}}"
  BLOCK_SIZE="${BLOCK_SIZE:-${BARD_BLOCK_SIZE:-8}}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
  TEMPERATURE="${TEMPERATURE:-0.0}"
  PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-64}"
  PROPOSAL_TOP_P="${PROPOSAL_TOP_P:-1.0}"
  TARGET_TOP_K="${TARGET_TOP_K:-64}"
  TARGET_TOP_P="${TARGET_TOP_P:-1.0}"
  DRAFTER_TAG="${DRAFTER_TAG:-${DRAFT_NAME:-}}"
  DRAFTER="${DRAFTER:-${PUBLIC_ROOT}/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-8-a128-v2/step-80000}"
  TARGET_ATTN_IMPLEMENTATION="${TARGET_ATTN_IMPLEMENTATION:-sdpa}"
  DRAFT_ATTN_IMPLEMENTATION="${DRAFT_ATTN_IMPLEMENTATION:-sdpa}"
  # Override VERIFIER for Qwen3.6-27B, for example:
  # VERIFIER=${FUN_SPACE}/models/Qwen3.6-27B bash scripts/eval_async.sh
  VERIFIER="${VERIFIER:-${FUN_SPACE}/models/Qwen3-32B}"

  CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.9}"
  DTYPE="${DTYPE:-bf16}"
  VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda:0}"
  DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:1}"
  DEVICE="${DEVICE:-$VERIFIER_DEVICE}"
  USE_KVCACHE="${USE_KVCACHE:-1}"
  REMASKING_STRATEGY="${REMASKING_STRATEGY:-low_confidence}"
  ATTENTION_MASK_MODE="${ATTENTION_MASK_MODE:-causal}"
  ASYNC_BLOCKS="${ASYNC_BLOCKS:-1}"
  ANCHOR_NUM="${ANCHOR_NUM:-0}"
  PACKED_ACCEPTED_COUNTS="${PACKED_ACCEPTED_COUNTS:-all}"
  VERIFIER_CPU_AFFINITY="${VERIFIER_CPU_AFFINITY:-}"
  DRAFT_CPU_AFFINITY="${DRAFT_CPU_AFFINITY:-}"
  ASYNC_PREFIX_MODE="${ASYNC_PREFIX_MODE:-none}"
  ASYNC_ENTROPY_THRESHOLD="${ASYNC_ENTROPY_THRESHOLD:-0.35}"
  ASYNC_MIN_PREFIX_LEN="${ASYNC_MIN_PREFIX_LEN:-4}"

  DRAFT_BACKEND="${DRAFT_BACKEND:-direct}"
  TOKENIZER="${TOKENIZER:-$VERIFIER}"
}

parse_cli_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --anchor-num)
        [ "$#" -ge 2 ] || die "$1 requires a value"
        ANCHOR_NUM="$2"
        shift 2
        ;;
      --packed-accepted-counts)
        [ "$#" -ge 2 ] || die "$1 requires a value"
        PACKED_ACCEPTED_COUNTS="$2"
        shift 2
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done
}

log() {
  echo "[sdar-async-packed] $*"
}

die() {
  echo "[sdar-async-packed] $*" >&2
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
  RESULT_ROOT="${OUT_DIR:-results_test/${DRAFT_TAG}}"
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
  log "async_execution_mode=process async_blocks=$ASYNC_BLOCKS"
  log "packed_outcomes=$PACKED_ACCEPTED_COUNTS branch_tokens=$((ASYNC_BLOCKS * BLOCK_SIZE))"
  log "anchor_num=$ANCHOR_NUM"
  log "verifier_cpu_affinity=${VERIFIER_CPU_AFFINITY:-auto} draft_cpu_affinity=${DRAFT_CPU_AFFINITY:-auto}"
  log "async_prefix_mode=$ASYNC_PREFIX_MODE entropy_threshold=$ASYNC_ENTROPY_THRESHOLD min_prefix_len=$ASYNC_MIN_PREFIX_LEN"
  log "out_root=$RESULT_ROOT"
}

append_common_limit_args() {
  local -n args_ref=$1
  if [ "$MAX_SAMPLES" != "all" ] && [ -n "$MAX_SAMPLES" ]; then
    args_ref+=(--max-samples "$MAX_SAMPLES")
  fi
}

build_async_args() {
  local dataset=$1
  local result_json=$2
  local summary_json=$3

  ACCEPT_ARGS=(
    eval_async_packed.py
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
    --async-blocks "$ASYNC_BLOCKS"
    --anchor-num "$ANCHOR_NUM"
    --packed-accepted-counts "$PACKED_ACCEPTED_COUNTS"
    --async-prefix-mode "$ASYNC_PREFIX_MODE"
    --async-entropy-threshold "$ASYNC_ENTROPY_THRESHOLD"
    --async-min-prefix-len "$ASYNC_MIN_PREFIX_LEN"
    --eos-token-id 151643
    --eos-token-id 151645
    --output-json "$result_json"
    --summary-json "$summary_json"
  )

  append_common_limit_args ACCEPT_ARGS
  if [ -n "$VERIFIER_CPU_AFFINITY" ]; then
    ACCEPT_ARGS+=(--verifier-cpu-affinity "$VERIFIER_CPU_AFFINITY")
  fi
  if [ -n "$DRAFT_CPU_AFFINITY" ]; then
    ACCEPT_ARGS+=(--draft-cpu-affinity "$DRAFT_CPU_AFFINITY")
  fi
  if [ -n "${MASK_TOKEN_ID:-}" ]; then
    ACCEPT_ARGS+=(--mask-token-id "$MASK_TOKEN_ID")
  fi
  if [ "${VERBOSE:-0}" = "1" ] || [ "${VERBOSE:-0}" = "true" ]; then
    ACCEPT_ARGS+=(--verbose)
  fi
  if [ "${NO_TARGET_FIRST_TOKEN:-0}" = "1" ] || [ "${NO_TARGET_FIRST_TOKEN:-0}" = "true" ]; then
    ACCEPT_ARGS+=(--no-target-first-token)
  fi
}

run_dataset() {
  local dataset=$1
  local timestamp_out_dir="${RESULT_ROOT}/${RUN_TIMESTAMP}/${dataset}"
  local mask_tag=""
  if [ "$ATTENTION_MASK_MODE" != "block_causal" ]; then
    mask_tag="_attn${ATTENTION_MASK_MODE}"
  fi
  local prefix_tag=""
  if [ "$ASYNC_PREFIX_MODE" != "none" ]; then
    prefix_tag="_${ASYNC_PREFIX_MODE}_h${ASYNC_ENTROPY_THRESHOLD}_m${ASYNC_MIN_PREFIX_LEN}"
  fi
  local accepted_tag="${PACKED_ACCEPTED_COUNTS//,/+}"
  local out_prefix="${timestamp_out_dir}/${dataset}_${DRAFT_TAG}_with_${VERIFIER_TAG}_b${BLOCK_SIZE}_s${DENOISING_STEP}_async${ASYNC_BLOCKS}_packed${accepted_tag}_anchor${ANCHOR_NUM}${mask_tag}${prefix_tag}"
  local result_json="${out_prefix}.json"
  local summary_json="${out_prefix}.summary.json"

  mkdir -p "$timestamp_out_dir"
  build_async_args "$dataset" "$result_json" "$summary_json"

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
  parse_cli_args "$@"
  parse_datasets
  resolve_metadata
  run_all
}

main "$@"
