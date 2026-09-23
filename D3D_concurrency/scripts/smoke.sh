#!/usr/bin/env bash
set -euo pipefail
export DATASETS="${DATASETS:-math500}" CONCURRENCY="${CONCURRENCY:-1 8 16 32}"
export MAX_SAMPLES="${MAX_SAMPLES:-40}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export REPEATS="${REPEATS:-1}" VALIDATE="${VALIDATE:-1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/benchmark.sh" "$@"
