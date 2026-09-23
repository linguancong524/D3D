#!/usr/bin/env bash
# Shared launcher. Environment variables follow D3DV2's scripts convention.
set -euo pipefail
CONCURRENCY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: [environment overrides] bash scripts/benchmark.sh [--dry-run]
Presets: smoke.sh, aime_c1.sh, table.sh (same overrides and flags).
DATASETS="math500 humaneval mt-bench" CONCURRENCY="1 8 16 32"
MAX_SAMPLES=0 MAX_NEW_TOKENS=2048 REPEATS=3
DRAFT_MICROBATCH=4 PAGES=512 VALIDATE=0 IGNORE_EOS=0 BACKUP=neural SEED=42 ANCHOR_NUM=3
CUDA_VISIBLE_DEVICES=0,1  (first GPU=target, second GPU=draft)
DRAFTER=/checkpoint VERIFIER=/checkpoint D3D_PYTHON=/path/to/python
OUT_DIR=/absolute/new/output/directory (must not already exist)
Defaults are SDAR-1.7B → Qwen3-8B. MAX_SAMPLES=0 means all samples.
A normal run writes command.sh, run.log, benchmark.json, table.md/.csv/.tex.
--dry-run prints commands without creating outputs or loading models.
HELP
    exit 0 ;;
  --dry-run|'') ;;
  *) echo "Unsupported argument: $1; use --help" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected at most one flag' >&2; exit 2; }
DRY_RUN=0
[[ "${1:-}" != --dry-run ]] || DRY_RUN=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export D3D_PYTHON="${D3D_PYTHON:-/inspire/ssd/project/sais-bio/public/linguancong/miniconda3/envs/D3D/bin/python}"
DATASETS="${DATASETS:-math500 humaneval mt-bench}"
CONCURRENCY="${CONCURRENCY:-1 8 16 32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
REPEATS="${REPEATS:-3}"
DRAFT_MICROBATCH="${DRAFT_MICROBATCH:-4}"
PAGES="${PAGES:-512}"
VALIDATE="${VALIDATE:-0}"
IGNORE_EOS="${IGNORE_EOS:-0}"
for key in MAX_SAMPLES MAX_NEW_TOKENS REPEATS DRAFT_MICROBATCH PAGES; do
  value="${!key}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$key must be an integer" >&2; exit 2; }
  [[ "$key" == MAX_SAMPLES || "$value" =~ [1-9] ]] || { echo "$key must be positive" >&2; exit 2; }
done
[[ "$VALIDATE" =~ ^[01]$ && "$IGNORE_EOS" =~ ^[01]$ ]] || { echo 'VALIDATE and IGNORE_EOS must be 0 or 1' >&2; exit 2; }
read -r -a datasets <<< "$DATASETS"
read -r -a concurrencies <<< "$CONCURRENCY"
[[ ${#datasets[@]} -gt 0 && ${#concurrencies[@]} -gt 0 ]] || exit 2
for c in "${concurrencies[@]}"; do
  [[ "$c" =~ ^[0-9]+$ && "$c" =~ [1-9] ]] || { echo "Invalid concurrency: $c" >&2; exit 2; }
done
[[ -x "$D3D_PYTHON" ]] || { echo "Python is not executable: $D3D_PYTHON" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-$CONCURRENCY_ROOT/results/runs/$(date -u +%Y%m%d_%H%M%S)_$$}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
command=(bash "$CONCURRENCY_ROOT/run.sh" --datasets "${datasets[@]}"
  --concurrency "${concurrencies[@]}" --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$REPEATS"
  --draft-microbatch "$DRAFT_MICROBATCH" --pages "$PAGES"
  --backup "${BACKUP:-neural}" --seed "${SEED:-42}"
  --anchor-num "${ANCHOR_NUM:-3}"
  --fast-miss-anchors "${FAST_MISS_ANCHORS:-0}"
  --output "$OUT_DIR/benchmark.json")
[[ -z "${DRAFTER:-}" ]] || command+=(--draft "$DRAFTER")
[[ -z "${VERIFIER:-}" ]] || command+=(--target "$VERIFIER")
[[ "$VALIDATE" != 1 ]] || command+=(--validate)
[[ "$IGNORE_EOS" != 1 ]] || command+=(--ignore-eos)
print_commands() {
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'export CUDA_VISIBLE_DEVICES=%q D3D_PYTHON=%q\n' "$CUDA_VISIBLE_DEVICES" "$D3D_PYTHON"
  printf '%q ' "${command[@]}"; printf '\n'
  printf '%q ' "$D3D_PYTHON" "$CONCURRENCY_ROOT/audit_results.py" "$OUT_DIR/benchmark.json"; printf '\n'
  printf '%q ' "$D3D_PYTHON" "$CONCURRENCY_ROOT/report.py" "$OUT_DIR/benchmark.json" --output "$OUT_DIR/table.md"; printf '\n'
}
if [[ "$DRY_RUN" == 1 ]]; then print_commands; exit 0; fi
# Refuse an existing run directory to preserve earlier measurements.
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR"
print_commands > "$OUT_DIR/command.sh"
printf 'Output directory: %s\n' "$OUT_DIR"
{
  "${command[@]}"
  "$D3D_PYTHON" "$CONCURRENCY_ROOT/audit_results.py" "$OUT_DIR/benchmark.json"
  "$D3D_PYTHON" "$CONCURRENCY_ROOT/report.py" "$OUT_DIR/benchmark.json" --output "$OUT_DIR/table.md"
} 2>&1 | tee "$OUT_DIR/run.log"
