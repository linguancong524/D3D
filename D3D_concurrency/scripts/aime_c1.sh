#!/usr/bin/env bash
set -euo pipefail
export DATASETS="${DATASETS:-aime25}" CONCURRENCY="${CONCURRENCY:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export REPEATS="${REPEATS:-2}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/benchmark.sh" "$@"
