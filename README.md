# NCCL Device API Benchmarks

This repository collects the standalone NCCL device API microbenchmarks used for GIN/LSA experiments:

- `nccl_gin_pingpong_perf`: two-rank GIN pingpong latency.
- `gin_put_signal_visibility`: two-rank GIN put-with-signal visibility check.
- `waitsignal_perf`: same-GPU GIN signal wait latency trace.
- `barrier-test`: NCCL device API barrier latency.
- `bcast-multimem-test`: `bcastMultimem<char, false>()` LSA/multimem broadcast benchmark.
- `bcast-multimem-8way-test`: 8-rank simultaneous `bcastMultimem` benchmark with collective-level timing.
- `allgather-gin-deviceapi-perf`: standalone benchmark for the ported `AllGather_RailRing_LsaSTMC`
  kernel and the experimental `oneshot-rail` GIN/LSA AllGather variants.

The benchmarks are intentionally small and do not modify NCCL. `barrier-test` and `bcast-multimem-test` include NCCL private symmetric headers, so they require an NCCL source tree in addition to an NCCL build/install tree.

## Layout

```text
src/common/                      shared parsing, checks, MPI, socket OOB, stats, and timer helpers
src/coll/all_gather_gin.cuh       ported NCCL AllGather_RailRing_LsaSTMC kernel
test/nccl_gin_pingpong_perf.cu    two-rank GIN pingpong benchmark
test/gin_put_signal_visibility.cu two-rank GIN put-with-signal visibility check
test/waitsignal_perf.cu           GIN waitSignal trace benchmark
test/barrier_test.cu              GIN/world barrier benchmark
test/bcast_multimem_test.cu       LSA bcastMultimem benchmark
test/allgather_gin_deviceapi_perf.cu  standalone AllGather GIN performance test
scripts/                          example launch scripts
```

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

### GIN put-with-signal visibility

This mirrors the NVSHMEM `putmem_signal_nbi_block` visibility repro: rank 0 writes a round-specific payload into
`src`, issues one GIN `put(..., SignalInc)` into rank 1's `dst`, and rank 1 immediately reads `dst` after
`waitSignal` returns. The sender only calls `gin.flush()` after the receiver-side check, so a failure means the
signal became visible before the payload was safely readable.

```bash
mpirun --allow-run-as-root -np 2 --hostfile hostfile \
  --mca routed direct --map-by slot \
  -x PATH -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x NCCL_IB_HCA=mlx5_1 \
  -x NCCL_GIN_TYPE=3 \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  ./build/gin_put_signal_visibility \
    --bytes 128K \
    --iters 10000
```

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

The default `bcast-multimem-test` reports the distribution of per-rank samples. To measure the single-node 8-rank simultaneous operation as one collective, use `bcast-multimem-8way-test` or pass `--stats-mode collective`:

```bash
mpirun --allow-run-as-root -np 8 --map-by ppr:8:node \
  -x LD_LIBRARY_PATH \
  -x BCAST_MM_MASTER_ADDR -x BCAST_MM_MASTER_PORT \
  ./build/bcast-multimem-8way-test \
    --min-bytes 64 \
    --max-bytes 512M \
    --factor 2 \
    --warmup 100 \
    --iters 1000 \
    --threads 512 \
    --num-blocks 1
```

The 8-way output uses the max rank time for each iteration and prints:

- `per_rank_inj_bw_GBps = size_B / p50_us`
- `aggregate_inj_bw_GBps = per_rank_inj_bw_GBps * lsa_ranks`
- `aggregate_delivered_bw_GBps = aggregate_inj_bw_GBps * lsa_ranks`

### AllGather GIN device API test

This benchmark launches AllGather kernels directly from NCCL device API building blocks. It allocates symmetric send/recv windows, creates a rail GIN device communicator, and reports CUDA event time using one measured kernel per size. The measured kernel loops over `--iters` AllGather operations internally, and the reported time is `event_elapsed / iters`.

Use `--kernel railring` for the ported `AllGather_RailRing_LsaSTMC` implementation in `src/coll/all_gather_gin.cuh`.

Use `--kernel oneshot-rail` for the optimized experimental implementation in `src/coll/all_gather_gin_oneshot_rail.cuh`. The default script runs this path. Its default policy is:

- `--num-blocks 1`: small messages use the original one-CTA logic.
- `--split-threshold 4M`: threshold is based on `recv_B`, not `send_B`.
- `--split-blocks 4`: when `recv_B >= 4M`, launch split CTAs. The value must be a multiple of 4, because blocks are divided evenly across the four rail/data peers.
- `--gin-flush 0`: by default the GIN put warp does not call `gin.flush()` after the put loop. Set `--gin-flush 1` to restore explicit flush after each put loop.
- In the split path, each data-peer group owns `split_blocks / 4` CTAs. For a remote peer group, the first CTA's first warp issues the GIN put to that peer. After the put loop, all warps in the CTA, including the put warp, participate in the LSA `bcastMultimem` stage over that peer group's chunk range.

```bash
export AG_GIN_MASTER_ADDR=2605:340:cd51:4900:476c:99af:d4df:a32b
export AG_GIN_MASTER_PORT=22540
export NCCL_GIN_TYPE=3

mpirun --allow-run-as-root -np 32 \
  --hostfile /opt/tiger/github-latest-20260611/hostfile.4n \
  --map-by ppr:8:node --mca routed direct \
  -x LD_LIBRARY_PATH -x NCCL_IB_HCA -x NCCL_GIN_TYPE \
  -x AG_GIN_MASTER_ADDR -x AG_GIN_MASTER_PORT \
  ./build/allgather-gin-deviceapi-perf \
    --kernel oneshot-rail \
    --min-bytes 32K \
    --max-bytes 8M \
    --factor 2 \
    --warmup 100 \
    --iters 1000 \
    --threads 512 \
    --num-blocks 1 \
    --split-blocks 4 \
    --split-threshold 4M \
    --gin-flush 0 \
    --stats-mode collective
```

`send_B` is the per-rank input size. `recv_B` is the per-rank allgather receive size, `send_B * ranks`. The output row `kernel_loop_event` is the single measured kernel event time divided by `--iters`. In `collective` stats mode the printed value is the mean of the per-rank `event_elapsed / iters` values; in `rank` mode it summarizes the per-rank values.

The same run can be launched through the script:

```bash
NCCL_GIN_TYPE=3 CHECK=1 scripts/run_allgather_gin_deviceapi_4n.sh
```

Useful overrides:

```bash
MIN_BYTES=1K MAX_BYTES=256K WARMUP=20 ITERS=1000 CHECK=1 scripts/run_allgather_gin_deviceapi_4n.sh
KERNEL=railring scripts/run_allgather_gin_deviceapi_4n.sh
SPLIT_BLOCKS=1 scripts/run_allgather_gin_deviceapi_4n.sh   # force one-CTA baseline
SPLIT_BLOCKS=8 scripts/run_allgather_gin_deviceapi_4n.sh   # split blocks must be 4,8,12,...
SPLIT_THRESHOLD=8M scripts/run_allgather_gin_deviceapi_4n.sh
GIN_FLUSH=1 scripts/run_allgather_gin_deviceapi_4n.sh      # call gin.flush() after put loops
```

## Notes

- Use `NCCL_GIN_TYPE=3` for GDAKI and `NCCL_GIN_TYPE=2` for proxy where applicable.
- `bcast-multimem-test` uses `ncclMemAlloc`; ordinary `cudaMalloc` cannot be registered by the current NCCL symmetric window path.
- `bcast-multimem-test` requests `lsaBarrierCount`, not `barrierCount`; `barrierCount` is a GIN/world barrier resource in this NCCL version.
