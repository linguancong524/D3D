#!/usr/bin/env bash
set -euo pipefail
export DATASETS="${DATASETS:-math500 humaneval mt-bench}" CONCURRENCY="${CONCURRENCY:-1 8 16 32}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export REPEATS="${REPEATS:-3}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/benchmark.sh" "$@"
