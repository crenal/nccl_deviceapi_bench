#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/allgather-gin-deviceapi-perf}"
MASTER_ADDR="${AG_GIN_MASTER_ADDR:-2605:340:cd51:4900:476c:99af:d4df:a32b}"
MASTER_PORT="${AG_GIN_MASTER_PORT:-22540}"
HOSTFILE="${HOSTFILE:-/opt/tiger/github-latest-20260611/hostfile.4n}"

export AG_GIN_MASTER_ADDR="${MASTER_ADDR}"
export AG_GIN_MASTER_PORT="${MASTER_PORT}"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"

ARGS=(
  --kernel "${KERNEL:-oneshot-rail}"
  --min-bytes "${MIN_BYTES:-32K}"
  --max-bytes "${MAX_BYTES:-8M}"
  --factor "${FACTOR:-2}"
  --warmup "${WARMUP:-100}"
  --iters "${ITERS:-1000}"
  --threads "${THREADS:-512}"
  --num-blocks "${NUM_BLOCKS:-1}"
  --split-blocks "${SPLIT_BLOCKS:-4}"
  --split-threshold "${SPLIT_THRESHOLD:-4M}"
  --gin-contexts "${GIN_CONTEXTS:-4}"
  --gin-flush "${GIN_FLUSH:-0}"
  --stats-mode "${STATS_MODE:-collective}"
)

if [[ "${CHECK:-0}" == "1" ]]; then
  ARGS+=(--check)
fi

mpirun --allow-run-as-root -np "${NP:-32}" \
  --hostfile "${HOSTFILE}" \
  --map-by "${MAP_BY:-ppr:8:node}" \
  --mca routed direct \
  -x LD_LIBRARY_PATH \
  -x NCCL_IB_HCA \
  -x NCCL_GIN_TYPE \
  -x CUDA_DEVICE_MAX_CONNECTIONS \
  -x AG_GIN_MASTER_ADDR -x AG_GIN_MASTER_PORT \
  "${BIN}" \
    "${ARGS[@]}"
