#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/nccl_gin_pingpong_perf}"
OUT="${OUT:-${ROOT_DIR}/results/nccl_gin_pingpong_$(date +%Y%m%d_%H%M%S)}"
HOSTFILE="${HOSTFILE:-${ROOT_DIR}/hostfile}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"
ITERS="${ITERS:-1000}"
GIN_TYPE="${NCCL_GIN_TYPE:-3}"

LABELS=(64B 128B 256B 512B 1KB 2KB 4KB 8KB 16KB 32KB 64KB 128KB 256KB 512KB 1MB 2MB 4MB 8MB 16MB 32MB 64MB 128MB 256MB 512MB)
BYTES=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912)

if [[ ! -x "${BIN}" ]]; then
  echo "build first: cmake -S ${ROOT_DIR} -B ${ROOT_DIR}/build -DNCCL_HOME=/path/to/nccl && cmake --build ${ROOT_DIR}/build -j" >&2
  exit 1
fi

mkdir -p "${OUT}"
cp "${HOSTFILE}" "${OUT}/hostfile"

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  bytes="${BYTES[$i]}"
  echo "running ${label} (${bytes})"
  mpirun --allow-run-as-root -np 2 --hostfile "${OUT}/hostfile" --mca routed direct --map-by slot \
    -x PATH -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES \
    -x NCCL_IB_HCA \
    -x NCCL_GIN_TYPE="${GIN_TYPE}" \
    -x CUDA_DEVICE_MAX_CONNECTIONS \
    -x NCCL_DEBUG \
    "${BIN}" \
      --bytes "${bytes}" \
      --warmup-iters "${WARMUP_ITERS}" \
      --iters "${ITERS}" \
      --check \
      --csv "${OUT}/${label}.csv" \
    > "${OUT}/${label}.full.log" 2>&1
done

python3 - "${OUT}" <<'PY'
import csv
import glob
import math
import os
import sys

out = sys.argv[1]

def pct(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)

rows = []
for path in glob.glob(os.path.join(out, "*.rank0.csv")):
    vals = []
    size_b = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            size_b = int(row["bytes"])
            vals.append(float(row["one_way_us"]))
    if vals:
        p50 = pct(vals, 0.50)
        rows.append({
            "size_label": os.path.basename(path).replace(".rank0.csv", ""),
            "size_B": size_b,
            "min_us": min(vals),
            "p50_us": p50,
            "p99_us": pct(vals, 0.99),
            "max_us": max(vals),
            "bw_p50_gib_s": size_b / 1024 / 1024 / 1024 / p50 * 1000000,
        })

rows.sort(key=lambda x: x["size_B"])
summary = os.path.join(out, "rank0_one_way_summary.csv")
with open(summary, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "size_label", "size_B", "min_us", "p50_us", "p99_us", "max_us", "bw_p50_gib_s"])
    writer.writeheader()
    writer.writerows(rows)

print(summary)
print(out)
PY
