#!/usr/bin/env bash
# Run the remaining 5 z-lab DFlash configs sequentially (tp1 bs16 already done).
set -uo pipefail
cd /workspace/projects/sglang-speedup-nvidia
for cfg in "tp bs8" "tp4 bs16" "tp4 bs8" "dp4 bs16" "dp4 bs8"; do
  echo "########## START $cfg $(date) ##########"
  bash docs/perf/scripts/run_zlab_dflash.sh $cfg
  echo "########## END $cfg (exit $?) $(date) ##########"
done
echo "########## ALL DONE $(date) ##########"
