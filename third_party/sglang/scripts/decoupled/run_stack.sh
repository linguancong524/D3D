#!/usr/bin/env bash
# Launch one decoupled comparison run in the background and wait for READY.
#
#   scripts/decoupled/run_stack.sh <mode> <tag>
# mode selects Target-local required drafting with or without the shared
# remote optional packed accelerator (local or hybrid).
set -euo pipefail

MODE="${1:?mode: local|hybrid}"
if [[ "$MODE" != "local" && "$MODE" != "hybrid" ]]; then
  echo "mode must be local or hybrid" >&2
  exit 2
fi
TAG="${2:?tag: e.g. anchorblock}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
DLLM_ALGORITHM_CONFIG="${DLLM_ALGORITHM_CONFIG:-/tmp/sdar_dllm.yaml}"
cd "$ROOT"
mkdir -p docs/perf/logs

# The native dLLM Engine JIT-builds flashinfer kernels; the venv ships ninja in
# -/.venv/bin, so it must be on PATH or the build fails with FileNotFoundError.
export PATH="$ROOT/.venv/bin:$PATH"

common=(
  --mode "$MODE"
  --target-model /workspace/models/Qwen3-8B
  --draft-model /workspace/models/d3d-1.7b-b16_for_qwen3-8b
  --dllm-algorithm OneStep
  --draft-tokenizer /workspace/models/Qwen3-8B
  --block-size "$BLOCK_SIZE"
  --dllm-algorithm-config "$DLLM_ALGORITHM_CONFIG"
  --trust-remote-code
  --max-running-requests "${MAX_RUNNING_REQUESTS:-48}"
  --lookup-wait-ms "${LOOKUP_WAIT_MS:-0}"
  --candidate-topk "${CANDIDATE_TOPK:-4}"
  --branch-group-size "${BRANCH_GROUP_SIZE:-3}"
  --poll-interval-ms "${POLL_INTERVAL_MS:-1}"
)
if [[ "${SHADOW_LOCAL:-0}" == "1" ]]; then
  common+=(--shadow-local)
fi
if [[ "${LOCAL_PREFIX_REUSE:-0}" == "1" ]]; then
  common+=(--local-prefix-reuse)
fi
if [[ "${DECOUPLED_TIMING_SYNC:-0}" == "1" ]]; then
  common+=(--timing-sync)
fi

if [[ "$MODE" == "hybrid" ]]; then
  NUM_ANCHORS="${NUM_ANCHORS:-2}"
  MAX_BRANCHES="${MAX_BRANCHES:-6}"
  read -r -a ACCEPT_COUNTS_ARRAY <<<"${ACCEPT_COUNTS:-1 2 3}"
  extra=(--draft-mem-fraction-static "${DRAFT_MEM_FRACTION_STATIC:-0.85}"
    --speculation-batch-size "${SPECULATION_BATCH_SIZE:-8}"
    --batching-window-ms "${BATCHING_WINDOW_MS:-1}"
    --num-anchors "$NUM_ANCHORS" --max-branches "$MAX_BRANCHES"
    --accept-counts "${ACCEPT_COUNTS_ARRAY[@]}")
else
  extra=(--num-anchors 0)
fi
if [[ "${DRAFT_CUDA_GRAPH:-0}" == "1" ]]; then
  extra+=(--draft-enable-cuda-graph)
  if [[ -n "${DRAFT_CUDA_GRAPH_MAX_TOKENS:-}" ]]; then
    extra+=(--draft-cuda-graph-max-tokens "$DRAFT_CUDA_GRAPH_MAX_TOKENS")
  fi
fi
if [[ "${BATCHED_BRANCH_TOPK:-0}" == "1" ]]; then
  extra+=(--batched-branch-topk)
fi
if [[ "${PACKED_CAUSAL_PENDING:-0}" == "1" ]]; then
  extra+=(--packed-causal-pending)
fi
if [[ "${ENGINE_DETERMINISTIC:-0}" == "1" ]]; then
  extra+=(--engine-deterministic)
fi

stamp="$(date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/docs/perf/logs/decoupled-${MODE}-${TAG}-${stamp}.console.log"
rm -f /dev/shm/sglang-decoupled-32000.sock
nohup .venv/bin/python3 scripts/decoupled/launch.py "${common[@]}" "${extra[@]}" \
  >"$LOG" 2>&1 &
launcher_pid=$!
ready=0
run_dir=""
echo "launcher_pid=$launcher_pid log=$LOG"

archive_console() {
  if [[ -z "$run_dir" ]]; then
    run_dir="$(awk -F'Logs: ' '/^Logs: / { print $2; exit }' "$LOG" 2>/dev/null || true)"
  fi
  if [[ -n "$run_dir" && -d "$run_dir" && -f "$LOG" ]]; then
    target_log="$run_dir/console.log"
    if [[ "$LOG" != "$target_log" ]]; then
      # rename(2) keeps the launcher's open descriptor attached to the archived
      # file, so later worker-exit diagnostics are preserved there as well.
      mv "$LOG" "$target_log"
      LOG="$target_log"
    fi
  fi
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  archive_console
  if [[ "$ready" -eq 0 ]]; then
    if kill -0 "$launcher_pid" 2>/dev/null; then
      kill -TERM "$launcher_pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$launcher_pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$launcher_pid" 2>/dev/null; then
        kill -KILL "$launcher_pid" 2>/dev/null || true
      fi
    fi
    wait "$launcher_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for _ in $(seq 1 240); do
  if [[ -z "$run_dir" ]]; then
    run_dir="$(awk -F'Logs: ' '/^Logs: / { print $2; exit }' "$LOG" 2>/dev/null || true)"
  fi
  if grep -qE "^READY:" "$LOG" 2>/dev/null; then
    ready=1
    grep -E "^READY:|^Logs:" "$LOG"
    archive_console
    exit 0
  fi
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "launcher $launcher_pid exited early; tail:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "READY not reached within 1200s for launcher $launcher_pid; tail:" >&2
tail -30 "$LOG" >&2
exit 1
