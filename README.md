# NCCL Device API Benchmarks

This repository collects the standalone NCCL device API microbenchmarks used for GIN/LSA experiments:

- `nccl_gin_pingpong_perf`: two-rank GIN pingpong latency.
- `waitsignal_perf`: same-GPU GIN signal wait latency trace.
- `barrier-test`: NCCL device API barrier latency.
- `bcast-multimem-test`: `bcastMultimem<char, false>()` LSA/multimem broadcast benchmark.

The benchmarks are intentionally small and do not modify NCCL. `barrier-test` and `bcast-multimem-test` include NCCL private symmetric headers, so they require an NCCL source tree in addition to an NCCL build/install tree.

## Build

Example on the current test machines:

```bash
cmake -S . -B build \
  -DNCCL_HOME=/opt/tiger/github-latest-20260611/nccl/build_noprofile \
  -DNCCL_SOURCE_DIR=/opt/tiger/github-latest-20260611/nccl \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

Runtime library path:

```bash
export LD_LIBRARY_PATH=/opt/tiger/github-latest-20260611/nccl/build_noprofile/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Run Examples

### GIN pingpong

Two ranks, one GPU per rank:

```bash
mpirun --allow-run-as-root -np 2 --hostfile hostfile \
  --mca routed direct --map-by slot \
  -x PATH -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x NCCL_IB_HCA=mlx5_1 \
  -x NCCL_GIN_TYPE=3 \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  ./build/nccl_gin_pingpong_perf \
    --bytes 1KB \
    --warmup-iters 100 \
    --iters 1000 \
    --check \
    --csv ./pingpong_1KB.csv
```

The program writes one CSV per rank, for example `pingpong_1KB.rank0.csv`.

### waitSignal trace

This tests local GIN signal update plus `gin.waitSignal` readiness on each rank. Rank 0 prints the summary and writes the CSV.

```bash
mpirun --allow-run-as-root -np 8 --map-by ppr:8:node \
  -x PATH -x LD_LIBRARY_PATH \
  -x NCCL_GIN_TYPE=3 \
  ./build/waitsignal_perf \
    --warmup-iters 100 \
    --iters 1000 \
    --csv ./waitsignal_rank0_trace.csv
```

Reported metrics:

- `local_atomic_add`: signal atomic increment cost on the signaling CTA.
- `signal_to_wait_done`: `t_wait_done - t_signal_before`, matching the previous waitsignal test.
- `signal_after_to_wait_done`: `t_wait_done - t_signal_after`, useful for isolating post-update readiness latency.

### barrier-test

Example 4-node/32-rank run:

```bash
export BARRIER_TEST_MASTER_ADDR=2605:340:cd51:4900:476c:99af:d4df:a32b
export BARRIER_TEST_MASTER_PORT=22400

mpirun --allow-run-as-root -np 32 \
  --hostfile /opt/tiger/github-latest-20260611/hostfile.4n \
  --map-by ppr:8:node --mca routed direct \
  -x LD_LIBRARY_PATH -x NCCL_IB_HCA -x NCCL_GIN_TYPE \
  -x BARRIER_TEST_MASTER_ADDR -x BARRIER_TEST_MASTER_PORT \
  ./build/barrier-test \
    --warmup 100 \
    --iters 1000 \
    --threads 512 \
    --connection rail
```

The measured loop uses the two-barrier-middle timing scheme: one barrier aligns, the second barrier is timed, and the third barrier realigns the next iteration.

### bcastMultimem test

Single-node 8-rank run:

```bash
export BCAST_MM_MASTER_ADDR=2605:340:cd51:4900:476c:99af:d4df:a32b
export BCAST_MM_MASTER_PORT=22420

mpirun --allow-run-as-root -np 8 --map-by ppr:8:node \
  -x LD_LIBRARY_PATH \
  -x BCAST_MM_MASTER_ADDR -x BCAST_MM_MASTER_PORT \
  ./build/bcast-multimem-test \
    --min-bytes 64 \
    --max-bytes 512M \
    --factor 2 \
    --warmup 100 \
    --iters 1000 \
    --threads 512 \
    --num-blocks 1
```

Use `--num-blocks N` to control how many CTAs split and execute one bcast. For `N > 1`, each CTA records its local elapsed cycles and the host reports the per-iteration max across CTAs.

## Notes

- Use `NCCL_GIN_TYPE=3` for GDAKI and `NCCL_GIN_TYPE=2` for proxy where applicable.
- `bcast-multimem-test` uses `ncclMemAlloc`; ordinary `cudaMalloc` cannot be registered by the current NCCL symmetric window path.
- `bcast-multimem-test` requests `lsaBarrierCount`, not `barrierCount`; `barrierCount` is a GIN/world barrier resource in this NCCL version.
