#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false

# Four explicit stages. Examples and data schemas: README.md.
ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$ACTION" in
    select) "${PY:-python}" -m pruning.select_layer "$@" ;;
    prune) "${PY:-python}" -m pruning.prune "$@" ;;
    train)
        if [[ "${1:-}" != "--nproc-per-node" || $# -lt 3 ]]; then
            echo "train requires --nproc-per-node N followed by recovery arguments" >&2
            exit 2
        fi
        NPROC="$2"
        shift 2
        if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
            echo "--nproc-per-node must be a positive integer" >&2
            exit 2
        fi
        "${PY:-python}" -m torch.distributed.run \
            --standalone \
            --nproc_per_node "$NPROC" \
            --module pruning.train_recovery \
            "$@"
        ;;
    choose) "${PY:-python}" -m pruning.select_checkpoint "$@" ;;
    help|-h|--help) echo "Usage: bash scripts/prune.sh {select|prune|train|choose} [arguments]; see docs/training.md" ;;
    *) echo "Unknown pruning action: $ACTION" >&2; exit 2 ;;
esac
