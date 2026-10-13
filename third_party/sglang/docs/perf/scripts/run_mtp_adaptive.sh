#!/usr/bin/env bash
# Runner for MTP + --speculative-adaptive serving benchmarks on H20.
# Mirrors the static MTP matrix (TP=1/TP=4/DP=4 x rate 4/16/64, cap48) but adds
#   --speculative-adaptive --speculative-adaptive-config <cfg>
# so num_steps adapts at runtime per batch-size slot (default cfg:
#   bs1->[0,1,3,7], bs8->[0,1,3], bs32->[0,1], bs64->[0]).
# Static MTP baseline (steps=3) numbers already exist in
#   docs/perf/reports/h20_qwen35_9b_mtp_comparison.md  -> this is the comparison target.
# Usage: run_mtp_adaptive.sh <tp|tp4|dp4>
set -uo pipefail

REPO=/workspace/projects/sglang-speedup-nvidia
cd "$REPO"
mkdir -p docs/perf/logs docs/perf/results
export PATH="$PWD/.venv/bin:$PATH"

TARGET=/workspace/models/Qwen3.5-9B
PORT=30000
ADAPTIVE_CFG="$REPO/docs/perf/configs/adaptive_mtp_config.json"

MODE="${1:?mode: tp|tp4|dp4}"
case "$MODE" in
  tp)  TP=1; DP=1; TAG=tp1 ;;
  tp4) TP=4; DP=1; TAG=tp4 ;;
  dp4) TP=1; DP=4; TAG=dp4 ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac

SLOG="docs/perf/logs/server_h20_qwen35_9b_serving_${TAG}_mtp_adaptive.log"
rm -f "$SLOG"

echo "=== launching ${TAG} MTP+adaptive -> $SLOG ==="
python3 -m sglang.launch_server \
  --model-path "$TARGET" \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size "$TP" --dp-size "$DP" \
  --mem-fraction-static 0.85 --mamba-ssm-dtype bfloat16 \
  --max-running-requests 48 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$TARGET" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-adaptive \
  --speculative-adaptive-config "$ADAPTIVE_CFG" \
  > "$SLOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID"

# Wait for readiness (up to 900s; adaptive captures graphs for all candidate steps)
ready=0
for i in $(seq 1 450); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "SERVER DIED during startup"; tail -60 "$SLOG"; exit 2
  fi
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
    ready=1; break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY in 900s"; tail -60 "$SLOG"; kill "$SRV_PID" 2>/dev/null; exit 3
fi
echo "=== server ready ==="

# Run the 3 bench rates (same n as static MTP baseline)
declare -A NMAP=( [4]=100 [16]=200 [64]=400 )
for RATE in 4 16 64; do
  N=${NMAP[$RATE]}
  BLOG="docs/perf/logs/bench_serving_h20_qwen35_9b_${TAG}_mtp_adaptive_rate${RATE}.log"
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
wait "$SRV_PID" 2>/dev/null
echo "=== ${TAG} MTP+adaptive all rates done ==="
