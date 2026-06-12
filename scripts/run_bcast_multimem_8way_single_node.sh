#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/bcast-multimem-8way-test}"
MASTER_ADDR="${BCAST_MM_MASTER_ADDR:-2605:340:cd51:4900:476c:99af:d4df:a32b}"
MASTER_PORT="${BCAST_MM_MASTER_PORT:-22430}"

export BCAST_MM_MASTER_ADDR="${MASTER_ADDR}"
export BCAST_MM_MASTER_PORT="${MASTER_PORT}"

mpirun --allow-run-as-root -np "${NP:-8}" --map-by "${MAP_BY:-ppr:8:node}" \
  -x LD_LIBRARY_PATH \
  -x BCAST_MM_MASTER_ADDR -x BCAST_MM_MASTER_PORT \
  "${BIN}" \
    --min-bytes "${MIN_BYTES:-64}" \
    --max-bytes "${MAX_BYTES:-512M}" \
    --factor "${FACTOR:-2}" \
    --warmup "${WARMUP:-100}" \
    --iters "${ITERS:-1000}" \
    --threads "${THREADS:-512}" \
    --num-blocks "${NUM_BLOCKS:-1}" \
    --stats-mode collective
