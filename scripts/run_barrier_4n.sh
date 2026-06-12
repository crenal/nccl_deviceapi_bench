#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/barrier-test}"
HOSTFILE="${HOSTFILE:-/opt/tiger/github-latest-20260611/hostfile.4n}"
MASTER_ADDR="${BARRIER_TEST_MASTER_ADDR:-2605:340:cd51:4900:476c:99af:d4df:a32b}"
MASTER_PORT="${BARRIER_TEST_MASTER_PORT:-22400}"

export BARRIER_TEST_MASTER_ADDR="${MASTER_ADDR}"
export BARRIER_TEST_MASTER_PORT="${MASTER_PORT}"

mpirun --allow-run-as-root -np "${NP:-32}" --hostfile "${HOSTFILE}" \
  --map-by "${MAP_BY:-ppr:8:node}" --mca routed direct \
  -x LD_LIBRARY_PATH -x NCCL_IB_HCA -x NCCL_GIN_TYPE \
  -x BARRIER_TEST_MASTER_ADDR -x BARRIER_TEST_MASTER_PORT \
  "${BIN}" \
    --warmup "${WARMUP:-100}" \
    --iters "${ITERS:-1000}" \
    --threads "${THREADS:-512}" \
    --connection "${CONNECTION:-rail}"
