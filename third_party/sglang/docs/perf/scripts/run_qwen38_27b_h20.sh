#!/usr/bin/env bash
# H20 (4x96GB) serving sweep for Qwen3.8-27B (BF16, in-checkpoint MTP head).
#
# Modes (mirrors the arg style of run_mtp_adaptive.sh):
#   tp4       TP=4, no speculation           -> baseline "current sglang" launch
#   tp4_mtp4  TP=4 + EAGLE/MTP (steps=3, topk=1, draft=4)
#   tp4_mtp8  TP=4 + EAGLE/MTP (steps=7, topk=1, draft=8)
#   dp4       TP=1 x DP=4, no speculation
#   dp4_mtp4  TP=1 x DP=4 + EAGLE/MTP (3/1/4)
#   tp4_dspark TP=4 + DSPARK draft (/workspace/models/Qwen3.8-27B-DSpark, gamma=7)
#
# Standard project workload: random 512 in / 256 out.
#   rate 4 / 16 / 64  with n = 100 / 200 / 400
# Override with RATES / NMAP-free env: RATES="4 16" bash run_qwen38_27b_h20.sh tp4
#
# All logs go to docs/perf/logs/ (project rule: no /tmp).
set -uo pipefail

REPO=/workspace/projects/sglang-speedup-nvidia
cd "$REPO"
mkdir -p docs/perf/logs docs/perf/results
export PATH="$PWD/.venv/bin:$PATH"

MODEL=/workspace/models/Qwen3.8-27B
PORT="${PORT:-30000}"
MODE="${1:?mode: tp4|tp4_mtp4|tp4_mtp8|tp4_mtp4_rss|tp4_dspark|dp4|dp4_mtp4}"
RATES="${RATES:-4 16 64}"

TP=4
DP=1
SPEC=()
case "$MODE" in
  tp4)      TAG=tp4 ;;
  tp4_mtp4) TAG=tp4_mtp4
            SPEC=(--speculative-algorithm EAGLE
                  --speculative-draft-model-path "$MODEL"
                  --speculative-num-steps 3 --speculative-eagle-topk 1
                  --speculative-num-draft-tokens 4) ;;
  tp4_mtp8) TAG=tp4_mtp8
            SPEC=(--speculative-algorithm EAGLE
                  --speculative-draft-model-path "$MODEL"
                  --speculative-num-steps 7 --speculative-eagle-topk 1
                  --speculative-num-draft-tokens 8) ;;
  # DSpark draft: separate 1.86B/5-layer checkpoint, verify window = gamma+1 = 8.
  # gamma (block_size 7) comes from the draft config; passed explicitly anyway.
  tp4_dspark) TAG=tp4_dspark
            SPEC=(--speculative-algorithm DSPARK
                  --speculative-draft-model-path "${DRAFT_MODEL:-/workspace/models/Qwen3.8-27B-DSpark}"
                  --speculative-draft-attention-backend flashinfer
                  --speculative-dspark-block-size "${DSPARK_GAMMA:-7}")
            # Confidence-budget scheduler inputs (compact ragged-verify only):
            #   DSPARK_SPS = profiled SPS cost table (dspark_sps_profiler)
            #   DSPARK_STS = STS calibration (dspark_sts_fit)
            if [ -n "${DSPARK_SPS:-}" ]; then
              SPEC+=(--speculative-dspark-sps-table-path "$DSPARK_SPS")
            fi
            if [ -n "${DSPARK_STS:-}" ]; then
              SPEC+=(--speculative-dspark-confidence-sts-path "$DSPARK_STS")
            fi ;;
  dp4)      TAG=dp4;       TP=1; DP=4 ;;
  dp4_mtp4) TAG=dp4_mtp4;  TP=1; DP=4
            SPEC=(--speculative-algorithm EAGLE
                  --speculative-draft-model-path "$MODEL"
                  --speculative-num-steps 3 --speculative-eagle-topk 1
                  --speculative-num-draft-tokens 4) ;;
  # MTP4 + ReplaySSM spec-verify: draft verify SSM states move onto a fixed ring
  # (cookbook: D=0 in the mamba-ratio formula), i.e. the per-request verify
  # state cost that cuts --max-running-requests should disappear.
  tp4_mtp4_rss) TAG=tp4_mtp4_rss
            SPEC=(--speculative-algorithm EAGLE
                  --speculative-draft-model-path "$MODEL"
                  --speculative-num-steps 3 --speculative-eagle-topk 1
                  --speculative-num-draft-tokens 4
                  --enable-linear-replayssm-spec) ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac

# Optional explicit concurrency pin (cap alignment between MTP and non-MTP runs).
CAP_ARGS=()
if [ -n "${CAP:-}" ]; then
  CAP_ARGS=(--max-running-requests "$CAP")
  TAG="${TAG}_cap${CAP}"
fi
# Optional decode-graph capture ceiling (draft graph capture can OOM at high cap).
GRAPH_ARGS=()
if [ -n "${GRAPH_BS:-}" ]; then GRAPH_ARGS=(--cuda-graph-max-bs-decode "$GRAPH_BS"); fi
# Optional free-form extra server args (space separated), e.g.
#   EXTRA="--disable-overlap-schedule"
EXTRA_ARGS=()
if [ -n "${EXTRA:-}" ]; then read -r -a EXTRA_ARGS <<< "$EXTRA"; fi
# Optional tag suffix, to keep variants of the same mode from overwriting logs.
TAG="${TAG}${TAGSUF:-}"

SLOG="docs/perf/logs/server_h20_qwen38_27b_serving_${TAG}.log"
RLOG="docs/perf/logs/runner_h20_qwen38_27b_${TAG}.log"
rm -f "$SLOG"

read -r -a RATE_ARR <<< "$RATES"
echo "=== $(date -Is) launching ${TAG} (tp=${TP} dp=${DP}) -> $SLOG ==="

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size "$TP" --dp-size "$DP" \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  --mamba-ssm-dtype bfloat16 \
  "${CAP_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${SPEC[@]}" \
  > "$SLOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID"

ready=0
for i in $(seq 1 900); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "SERVER DIED during startup"; tail -80 "$SLOG"; exit 2
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ready=1; break; fi
  if [ $((i % 15)) -eq 0 ]; then
    echo "  ...waiting (${i}x4s) $(grep -c . "$SLOG" 2>/dev/null) log lines"
  fi
  sleep 4
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY in 3600s"; tail -80 "$SLOG"; kill "$SRV_PID" 2>/dev/null; exit 3
fi
echo "=== server ready ==="
grep -E "max_running_requests|Load weight|Memory pool end|KV Cache is allocated|max_total_num_tokens|Mamba" "$SLOG" | tail -20

# Warm-up (trigger JIT / graph capture paths outside the measured window)
echo "=== warmup ==="

# Optional: profile the DSpark SPS cost table from this (static-mode) server and
# feed it back at the next launch via DSPARK_SPS=<json> + SGLANG_RAGGED_VERIFY_MODE=compact.
if [ -n "${PROFILE_SPS:-}" ]; then
  mkdir -p "$(dirname "$PROFILE_SPS")"
  PLOG="docs/perf/logs/$(basename "${PROFILE_SPS%.json}").log"
  echo "=== $(date -Is) profiling SPS table -> $PROFILE_SPS (log $PLOG) ==="
  python3 -m sglang.benchmark.dspark_sps_profiler all \
    --base-url "http://127.0.0.1:$PORT" \
    --out "$PROFILE_SPS" \
    --batch-size ${SPS_BS:-8 16 32 48 64 96 128} \
    --min-steady-seconds "${SPS_STEADY:-5}" \
    > "$PLOG" 2>&1 \
    && echo "  SPS table OK" || echo "  SPS profiling FAILED (exit=$?)"
fi
curl -s -o /dev/null "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d "{\"text\":\"The capital of France is\",\"max_new_tokens\":32}" || true

for RATE in "${RATE_ARR[@]}"; do
  case "$RATE" in
    4)  N="${N4:-100}" ;;
    16) N="${N16:-200}" ;;
    64) N="${N64:-400}" ;;
    *)  N="${N_DEFAULT:-100}" ;;
  esac
  BLOG="docs/perf/logs/bench_serving_h20_qwen38_27b_${TAG}_rate${RATE}.log"
  echo "=== $(date -Is) bench rate=${RATE} n=${N} -> $BLOG ==="
  python3 -m sglang.benchmark.serving --backend sglang \
    --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts "$N" --request-rate "$RATE" \
    --output-file "docs/perf/results/$(basename "${BLOG%.log}").jsonl" \
    2>&1 | tee "$BLOG"
  echo "  rate=${RATE} exit=$?  $(grep -E '^Successful requests|^Request throughput' "$BLOG" | tr '\n' ' ')"
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "SERVER DIED during rate=${RATE}"; tail -40 "$SLOG"; exit 4
  fi
done

# Optional saturation run: all prompts at once, capped concurrency
if [ -n "${SAT:-}" ]; then
  BLOG="docs/perf/logs/bench_serving_h20_qwen38_27b_${TAG}_sat${SAT}.log"
  echo "=== $(date -Is) saturation max-concurrency=${SAT} -> $BLOG ==="
  python3 -m sglang.benchmark.serving --backend sglang \
    --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts "${SATN:-256}" --request-rate inf --max-concurrency "$SAT" \
    --output-file "docs/perf/results/$(basename "${BLOG%.log}").jsonl" \
    2>&1 | tee "$BLOG"
  echo "  sat=${SAT} exit=$?  $(grep -E '^Successful requests|^Request throughput' "$BLOG" | tr '\n' ' ')"
fi

kill "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null
echo "=== $(date -Is) ${TAG} done (runner log: $RLOG) ==="
