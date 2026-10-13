#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONCURRENCY="${CONCURRENCY:-4}"
if [ "$CONCURRENCY" -gt 1 ]; then
  ANCHOR_NUM=1
else
  ANCHOR_NUM=3
fi

exec "$PYTHON_BIN" serving/bench.py \
  --target "$TARGET_MODEL" \
  --draft "$DRAFT_MODEL" \
  --datasets math500 \
  --max-samples 4 \
  --max-new-tokens 128 \
  --concurrency "$CONCURRENCY" \
  --anchor-num "$ANCHOR_NUM" \
  --pages 512 \
  --output outputs/concurrent/quickstart.json \
  "$@"
