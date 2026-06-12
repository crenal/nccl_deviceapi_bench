#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/waitsignal_perf}"
OUT="${OUT:-${ROOT_DIR}/results/waitsignal_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT}"

mpirun --allow-run-as-root -np "${NP:-8}" --map-by "${MAP_BY:-ppr:8:node}" \
  -x PATH -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}" \
  "${BIN}" \
    --warmup-iters "${WARMUP_ITERS:-100}" \
    --iters "${ITERS:-1000}" \
    --csv "${OUT}/waitsignal_rank0_trace.csv"
