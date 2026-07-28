#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=""
OUTPUT_DIR="reports/generated"
ITERATIONS=5000
SOAK_SECONDS=1
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --soak-seconds) SOAK_SECONDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; exit 2; }
PYTHON="$ROOT/.venv/bin/python"
BENCHMARK_OUT="$ROOT/$OUTPUT_DIR/cpu_benchmark.json"
SOAK_OUT="$ROOT/$OUTPUT_DIR/cpu_soak.json"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "$PYTHON" scripts/benchmark_cpu.py --checkpoint "$CHECKPOINT" --iterations "$ITERATIONS" --output "$BENCHMARK_OUT"; echo
  printf '%q ' "$PYTHON" scripts/soak_cpu.py --checkpoint "$CHECKPOINT" --seconds "$SOAK_SECONDS" --max-iterations "$ITERATIONS" --output "$SOAK_OUT"; echo
  exit 0
fi
"$PYTHON" scripts/benchmark_cpu.py --checkpoint "$CHECKPOINT" --iterations "$ITERATIONS" --output "$BENCHMARK_OUT"
"$PYTHON" scripts/soak_cpu.py --checkpoint "$CHECKPOINT" --seconds "$SOAK_SECONDS" --max-iterations "$ITERATIONS" --output "$SOAK_OUT"
