# High-throughput ML-DSA (FIPS 204) signing and key generation on GPUs

This repository contains the code for the paper *Profiling-Guided High-Throughput ML-DSA Signing and Key Generation on GPUs*.

- `GPU-MLDSA-Sign/` covers signing, verification and the benchmarks. It is a fork of the released code of Shen et al. (cuDilithium), ported to FIPS 204.
- `GPU-MLDSA-Keygen/` covers coarse-grained key generation, with one key pair per thread.
- `ref/` is the FIPS 204 reference implementation (pq-crystals), used for testing.

Signing is deterministic (`rnd` = 0, empty context string). In this mode, every configuration produces key pairs and signatures that are byte-for-byte identical to the reference implementation.

## Requirements

- NVIDIA GPU with compute capability 8.0 or higher. We tested the A100 (sm_80), RTX 4050 Laptop (sm_89), H100 (sm_90) and B200 (sm_100).
- CUDA 12.2 or newer; the B200 needs 12.8 or newer.
- GCC 11 or newer, CMake 3.20 or newer, OpenMP.

## Signing: build and test

```bash
cd GPU-MLDSA-Sign
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
cd build && ctest --output-on-failure -E test_cuDilithium   # see below
./bin/bench_cuDilithium2 10000 32768        # ML-DSA-44; 3 = ML-DSA-65, 5 = ML-DSA-87
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (80, 89, 90 or 100). Without further options, this builds the final design of the paper: the fused kernel with the warp NTT with whole butterflies, mask pre-generation for the first 8 attempts, staged challenge hashing, and the single-key cache.

`ctest` compares key pairs and signatures with the FIPS 204 reference implementation, cross-verifies the signatures, and checks that modified signatures and messages are rejected. With `USE_CACHED_SK_SINGLE`, all test messages are signed with one key pair.

### Configurations of the paper (Table "signing throughput")

The final design is the default. For the other rows, configure a fresh build directory with `-DFINAL_DESIGN=OFF` and the options of the row:

| Row | CMake options |
|---|---|
| Released code of Shen et al. (FIPS 204 port), tuned | `-DFINAL_DESIGN=OFF -DUSE_128T_SIGN=ON`, with `BENCH_STREAMS=10 BENCH_EXEC_THRESHOLD=2048\|5120` |
| Fused kernel, shared-memory NTT | `-DFINAL_DESIGN=OFF -DUSE_PERSISTENT=ON` |
| Fused kernel, warp NTT, half butterflies | `-DFINAL_DESIGN=OFF -DUSE_PERSISTENT_WARPNTT=ON` |
| Fused kernel, warp NTT, whole butterflies | `-DFINAL_DESIGN=OFF -DUSE_PERSISTENT_WARPNTT=ON -DFUSED_WARPX=ON` |
| + mask pre-generation | add `-DFUSED_PREGEN_Y=ON` |
| + staged challenge hashing | add `-DFUSED_STAGED=ON` |
| + key caching (final design, default) | add `-DUSE_CACHED_SK_SINGLE=ON` |

`FINAL_DESIGN=ON` (the default) is the same as `-DUSE_PERSISTENT_WARPNTT=ON -DFUSED_WARPX=ON -DFUSED_PREGEN_Y=ON -DFUSED_STAGED=ON -DUSE_CACHED_SK_SINGLE=ON`. `FUSED_PREGEN_ATTEMPTS` defaults to 8, the value used in the paper. `FINAL_DESIGN` only sets the defaults of these options when a build directory is first configured, so use a new build directory for each configuration.

The `test_cuDilithium*` tests compare with stored hashes of test vectors that use a different key for every signature, so they do not apply to `USE_CACHED_SK_SINGLE`, which is why the default build excludes them with `ctest -E test_cuDilithium`. Run them in configurations without `USE_CACHED_SK_SINGLE`.

### Other options

| Option | Effect |
|---|---|
| `USE_CACHED_SK` | instead of `USE_CACHED_SK_SINGLE` (set `-DUSE_CACHED_SK_SINGLE=OFF`): cache a separately expanded key in every signature slot (a different key per signature) |
| `USE_CP_NTT` | multi-kernel code: compute the NTT of the challenge in the challenge kernel (requires `USE_128T_SIGN`) |
| `USE_128T_KEYPAIR` | key generation of the signing library with 128 threads and the warp NTT |
| `FUSED_PREGEN_ATTEMPTS` | number of attempts with pre-generated masks (default 8) |
| `FUSED_STAGED_ROUNDS` | number of staged rounds (default: `FUSED_PREGEN_ATTEMPTS`) |
| `FUSED_OVERLAP_Y` | fused kernel: generate the next mask while warp 0 computes the challenge (studied in the paper, not used in the final design; cannot be combined with `FUSED_STAGED`) |
| `FUSED_TRIM_WARPS`, `FUSED_MIN_BLOCKS` | fused kernel: launch only the warps that do work; minimum resident blocks per SM for `__launch_bounds__` |
| `SIGN_PHASE_PROFILE` | per-phase cycle counters of the fused kernels, printed by the benchmark (profiling only) |
| `TEST_SINGLE_KEY`, `TEST_NVECTORS` | test only: one key for all test vectors; number of test vectors |

### Benchmark settings

The benchmark takes the batch sizes as arguments and reads these environment variables:
- `BENCH_TRIALS` (default 50) and `BENCH_STREAMS` (default 10);
- `BENCH_EXEC_THRESHOLD`: concurrent signing attempts of the multi-kernel code (default 2048).

A measurement covers the whole signing call, including the transfers of the messages and signatures. Set `OMP_NUM_THREADS` to the number of streams on machines whose CPU quota is smaller than the number of visible cores, such as containers.

### Isolated NTT comparison

```bash
./bin/bench_ntt_compare 262144 > ntt_compare.csv
```

It checks all variants against each other and prints the time per transform:
- `smem`: the shared-memory NTT;
- `warp`: warp NTT, half butterflies, twiddles in constant memory;
- `warp_s`: warp NTT, half butterflies, twiddles in shared memory;
- `warp_g`: warp NTT, half butterflies, twiddles through the read-only cache;
- `warp_x`: warp NTT, whole butterflies.

## Key generation

```bash
cd GPU-MLDSA-Keygen
make MODE=2 NTT_OPT=1 ARCH="-arch=sm_80" all
./verify 2000      # compare 2,000 key pairs with the reference implementation
./benchmark        # kernel and end-to-end throughput (10 streams) over batch sizes
```

`NTT_OPT` selects the NTT: `0` is radix-2, `1` is radix-4 (the paper's result) and `2` is radix-4 SDFS. `MODE` is 2, 3 or 5 (ML-DSA-44, -65, -87). Set `ARCH` to your GPU. Run `make clean` between builds with different options.

## License

- **`GPU-MLDSA-Sign/` and `GPU-MLDSA-Keygen/`**: GNU General Public License v3.0 or later, as cuDilithium, from which the signing code is forked (`LICENSE`, `GPU-MLDSA-Sign/COPYING`). Files changed from cuDilithium carry a modification notice. New files carry `SPDX-License-Identifier: GPL-3.0-or-later`. Some files of cuDilithium, and the parts of the key generation code adapted from the reference implementation, also note their origins. These are the CRYSTALS-Dilithium reference (CC0 / Apache-2.0; `GPU-MLDSA-Sign/include/APACHE_LICENSE`) and the FIPS 202 code in `fips202/` (MIT and CC0; see `MIT_LICENSE` and `CC0_LICENSE` there).
- **`ref/`** and its copy **`GPU-MLDSA-Sign/ref/`**: the pq-crystals reference implementation, unchanged, under CC0 or Apache-2.0 (`ref/LICENSE`).
