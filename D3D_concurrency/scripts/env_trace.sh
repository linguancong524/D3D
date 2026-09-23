#!/usr/bin/env bash
# Samples host/cgroup load so per-cell timings can be attributed to the
# environment. Writes one CSV row every 30 seconds.
set -uo pipefail
OUT="${1:?usage: env_trace.sh <out.csv> [seconds]}"
INTERVAL="${2:-30}"
echo "utc,load1,load5,load15,cpu_throttled_usec,nr_throttled,gpu_util,gpu_temp_c,gpu_power_w" > "$OUT"
while true; do
  read -r l1 l5 l15 < <(awk '{print $1, $2, $3}' /proc/loadavg)
  read -r thrt nrt < <(awk '/^throttled_usec/{t=$2} /^nr_throttled/{n=$2} END{print t, n}' /sys/fs/cgroup/cpu.stat)
  IFS=, read -r util temp pwr < <(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date -u +%FT%TZ),$l1,$l5,$l15,$thrt,$nrt,$util,$temp,$pwr" >> "$OUT"
  sleep "$INTERVAL"
done
