#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 2
  }
}

check_hardware() {
  require_command "$PYTHON_BIN"
  "$PYTHON_BIN" -c 'from common.runtime import check_hardware; check_hardware()'
}

check_models() {
  : "${TARGET_MODEL:=Qwen/Qwen3-8B}"
  : "${DRAFT_MODEL:=lingc8/D3D-1.7B-to-8B}"
  export TARGET_MODEL DRAFT_MODEL
}

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
check_hardware
check_models
export TOKENIZERS_PARALLELISM=false
