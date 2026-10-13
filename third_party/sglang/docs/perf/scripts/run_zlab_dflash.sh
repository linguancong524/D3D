#!/usr/bin/env bash
# Runner for z-lab/Qwen3.5-9B-DFlash (official) DFlash serving benchmarks on H20.
# Mirrors the unieai DFlash matrix: TP=1/TP=4/DP=4 x rate 4/16/64, bs16 + bs8, cap48.
# Usage: run_zlab_dflash.sh <tp|tp4|dp4> <bs16|bs8>
set -uo pipefail

REPO=/workspace/projects/sglang-speedup-nvidia
cd "$REPO"
mkdir -p docs/perf/logs docs/perf/results
export PATH="$PWD/.venv/bin:$PATH"

TARGET=/workspace/models/Qwen3.5-9B
DRAFT=/workspace/models/Qwen3.5-9B-DFlash          # z-lab official
PORT=30000

MODE="${1:?mode: tp|tp4|dp4}"
BS="${2:?bs: bs16|bs8}"

case "$MODE" in
  tp)  TP=1; DP=1; TAG=tp1 ;;
  tp4) TP=4; DP=1; TAG=tp4 ;;
  dp4) TP=1; DP=4; TAG=dp4 ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac

case "$BS" in
  bs16) BLOCK_ARGS=(--speculative-num-draft-tokens 16) ;;
  bs8)  BLOCK_ARGS=(--speculative-dflash-block-size 8) ;;
  *) echo "bad bs $BS"; exit 1 ;;
esac

SLOG="docs/perf/logs/server_h20_qwen35_9b_serving_${TAG}_dflash_zlab_${BS}.log"
rm -f "$SLOG"

echo "=== launching ${TAG} ${BS} (z-lab) -> $SLOG ==="
python3 -m sglang.launch_server \
  --model-path "$TARGET" \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size "$TP" --dp-size "$DP" \
  --mem-fraction-static 0.85 --mamba-ssm-dtype bfloat16 \
  --max-running-requests 48 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  "${BLOCK_ARGS[@]}" \
  --mamba-scheduler-strategy extra_buffer \
  > "$SLOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID"

# Wait for readiness (up to 600s)
ready=0
for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "SERVER DIED during startup"; tail -40 "$SLOG"; exit 2
  fi
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
    ready=1; break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY in 600s"; tail -40 "$SLOG"; kill "$SRV_PID" 2>/dev/null; exit 3
fi
echo "=== server ready ==="

# Run the 3 bench rates
declare -A NMAP=( [4]=100 [16]=200 [64]=400 )
for RATE in 4 16 64; do
  N=${NMAP[$RATE]}
  BLOG="docs/perf/logs/bench_serving_h20_qwen35_9b_${TAG}_dflash_zlab_${BS}_rate${RATE}.log"
  echo "=== bench rate=${RATE} n=${N} -> $BLOG ==="
  python3 -m sglang.benchmark.serving --backend sglang \
    --model "$TARGET" \
    --dataset-name random --random-input-len 512 --random-output-len 256 \
    --num-prompts "$N" --request-rate "$RATE" \
    --output-file "docs/perf/results/$(basename "${BLOG%.log}").jsonl" \
    2>&1 | tee "$BLOG"
  echo "  rate=${RATE} done (exit $?)"
done

# Cleanup
kill "$SRV_PID" 2>/dev/null
for i in $(seq 1 60); do
  kill -0 "$SRV_PID" 2>/dev/null || break
  sleep 2
done
kill -9 "$SRV_PID" 2>/dev/null
# Wait for GPU memory to release
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [ "${USED:-99999}" -lt 1000 ] && break
  sleep 2
done
echo "=== ${TAG} ${BS} complete, GPU mem used=${USED} MiB ==="
