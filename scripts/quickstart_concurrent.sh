#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASELINE_FLAG=""
[ "${D3D_BASELINE:-1}" = "0" ] && BASELINE_FLAG="--no-baseline"

exec "$PYTHON_BIN" serving/bench.py \
  --target "$D3D_TARGET_MODEL" \
  --draft "$D3D_DRAFT_MODEL" \
  --datasets ${D3D_DATASETS:-math500} \
  --max-samples "${D3D_MAX_SAMPLES:-4}" \
  --max-new-tokens "${D3D_MAX_NEW_TOKENS:-128}" \
  --concurrency ${D3D_CONCURRENCY:-4} \
  --anchor-num "${D3D_ANCHOR_NUM:-1}" \
  --pages "${D3D_PAGES:-512}" \
  --output "${D3D_CONCURRENT_OUTPUT:-outputs/concurrent/quickstart.json}" \
  $BASELINE_FLAG
