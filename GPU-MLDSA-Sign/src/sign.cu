/*
 * Copyright 2022-2023 [Anonymous].
 *
 * This file is part of cuDilithium.
 *
 * cuDilithium is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * cuDilithium is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * cuDilithium. If not, see <https://www.gnu.org/licenses/>.
 *
 * Modified in 2026: ported to FIPS 204 (ML-DSA); fused and staged signing,
 * warp NTT and key caching.
 */

#include "sign.cuh"

#include "fips202/fips202.cuh"
#include "ntt.cuh"
#include "params.h"
#include "poly.cuh"
#include "reduce.cuh"
#include "rounding.cuh"

#ifdef USE_CACHED_SK_SINGLE
#define KEY_OFFSET(ptr, offset) (ptr)
#else
#define KEY_OFFSET(ptr, offset) ((ptr) + (offset) / sizeof(int32_t))
#endif

#ifdef USE_128T_SIGN
#define PREFIX_SUM_HINT_COUNT                                                  \
  if (threadIdx.x < DILITHIUM_WARPS) {                                         \
    unsigned int val = s_hint_count[threadIdx.x];                              \
    unsigned int sum = val;                                                    \
    unsigned int tmp = __shfl_up_sync((1 << DILITHIUM_WARPS) - 1, sum, 1);     \
    if (threadIdx.x >= 1) sum += tmp;                                          \
    tmp = __shfl_up_sync((1 << DILITHIUM_WARPS) - 1, sum, 2);                  \
    if (threadIdx.x >= 2) sum += tmp;                                          \
    tmp = __shfl_up_sync((1 << DILITHIUM_WARPS) - 1, sum, 4);                  \
    if (threadIdx.x >= 4) sum += tmp;                                          \
    s_hint_count[threadIdx.x] = sum - val;                                     \
    if (threadIdx.x == DILITHIUM_WARPS - 1) s_hint_count_sum = sum;            \
  }
#else
#define PREFIX_SUM_HINT_COUNT                                                  \
  if (threadIdx.x == 0) {                                                      \
    unsigned int count = 0;                                                    \
    for (size_t i = 0; i < DILITHIUM_WARPS; ++i) {                             \
      unsigned int val = s_hint_count[i];                                      \
      s_hint_count[i] = count;                                                 \
      count += val;                                                            \
    }                                                                          \
    s_hint_count_sum = count;                                                  \
  }
#endif

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

#define POLY_UNIFORM_GAMMA1_NBLOCKS                                            \
  ((POLYZ_PACKEDBYTES + SHAKE256_RATE - 1) / SHAKE256_RATE)

#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#endif

__global__ void polyvec_matrix_expand_kernel(int32_t *g_mat,
                                             const uint8_t *g_rho,
                                             size_t mem_pool_pitch) {
  __shared__ uint8_t s_rho[SEEDBYTES];
  __shared__ uint8_t s_buf[POLY_UNIFORM_NBLOCKS * SHAKE128_RATE];

  s_rho[threadIdx.x] = g_rho[blockIdx.x * mem_pool_pitch + threadIdx.x];

  for (unsigned int i = 0; i < DILITHIUM_K; ++i) {
    for (unsigned int j = 0; j < DILITHIUM_L; ++j) {
      s_buf[threadIdx.x] = s_rho[threadIdx.x];
      if (threadIdx.x == 0) {
        s_buf[SEEDBYTES] = (i << 8) + j;
        s_buf[SEEDBYTES + 1] = i;
      }
      __syncwarp();
      shake<SHAKE128_RATE, 0x1f>(s_buf, POLY_UNIFORM_NBLOCKS, s_buf,
                                 SEEDBYTES + 2);
      __syncwarp();

      size_t ctr = 0;
      auto g_poly = g_mat + blockIdx.x * mem_pool_pitch / sizeof(int32_t) +
                    i * DILITHIUM_L * DILITHIUM_N + j * DILITHIUM_N;
      for (size_t round = 0; round < 8; round++) {
        size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
        uint8_t t0 = s_buf[s_buf_offset + 0];
        uint8_t t1 = s_buf[s_buf_offset + 1];
        uint8_t t2 = s_buf[s_buf_offset + 2] & 0x7F;
        uint32_t t = t0 | (t1 << 8) | (t2 << 16);

        int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;

        unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
        if (good_mask == 0xFFFFFFFF) {
          g_poly[ctr + threadIdx.x] = t;
          ctr += 32;
        } else {
          good_mask <<= 31 - threadIdx.x;
          unsigned int ctr_offset = __popc(good_mask);
          if (ctr + ctr_offset <= DILITHIUM_N && good) {
            g_poly[ctr + ctr_offset - 1] = t;
          }
          ctr += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
        }
      }
      if (ctr < DILITHIUM_N) {
        size_t s_buf_offset = 8 * 32 * 3 + threadIdx.x * 3;
        uint8_t t0 = s_buf[s_buf_offset + 0];
        uint8_t t1 = s_buf[s_buf_offset + 1];
        uint8_t t2 = s_buf[s_buf_offset + 2] & 0x7F;
        uint32_t t = t0 | (t1 << 8) | (t2 << 16);

        int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;
        unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
        good_mask <<= 31 - threadIdx.x;
        unsigned int ctr_offset = __popc(good_mask);
        if (ctr + ctr_offset <= DILITHIUM_N && good) {
          g_poly[ctr + ctr_offset - 1] = t;
        }
      }
    }
  }
}

__global__ void polyvec_matrix_expand_opt_kernel(int32_t *g_mat,
                                                 const uint8_t *g_rho,
                                                 size_t mem_pool_pitch,
                                                 size_t n_inputs) {
  __shared__ uint8_t s_rho[4][SEEDBYTES];
  __shared__ uint8_t s_buf[4][POLY_UNIFORM_NBLOCKS * SHAKE128_RATE + 32];

  const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

  if (input_id < n_inputs) {

    s_rho[threadIdx.y][threadIdx.x] =
        g_rho[input_id * mem_pool_pitch + threadIdx.x];

    uint8_t *s_buf_y = s_buf[threadIdx.y];

    for (unsigned int i = 0; i < DILITHIUM_K; ++i) {
      for (unsigned int j = 0; j < DILITHIUM_L; ++j) {
        s_buf_y[threadIdx.x] = s_rho[threadIdx.y][threadIdx.x];
        if (threadIdx.x == 0) {
          s_buf_y[SEEDBYTES] = (i << 8) + j;
          s_buf_y[SEEDBYTES + 1] = i;
        }
        __syncwarp();
        shake<SHAKE128_RATE, 0x1f>(s_buf_y, POLY_UNIFORM_NBLOCKS, s_buf_y,
                                   SEEDBYTES + 2);
        __syncwarp();

        size_t ctr = 0;
        auto g_poly = g_mat + input_id * mem_pool_pitch / sizeof(int32_t) +
                      i * DILITHIUM_L * DILITHIUM_N + j * DILITHIUM_N;
        for (size_t round = 0; round < 8; round++) {
          size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
          uint8_t t0 = s_buf_y[s_buf_offset + 0];
          uint8_t t1 = s_buf_y[s_buf_offset + 1];
          uint8_t t2 = s_buf_y[s_buf_offset + 2] & 0x7F;
          uint32_t t = t0 | (t1 << 8) | (t2 << 16);

          int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;

          unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
          if (good_mask == 0xFFFFFFFF) {
            g_poly[ctr + threadIdx.x] = t;
            ctr += 32;
          } else {
            good_mask <<= 31 - threadIdx.x;
            unsigned int ctr_offset = __popc(good_mask);
            if (ctr + ctr_offset <= DILITHIUM_N && good) {
              g_poly[ctr + ctr_offset - 1] = t;
            }
            ctr += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
          }
        }
        if (ctr < DILITHIUM_N) {
          size_t s_buf_offset = 8 * 32 * 3 + threadIdx.x * 3;
          uint8_t t0 = s_buf_y[s_buf_offset + 0];
          uint8_t t1 = s_buf_y[s_buf_offset + 1];
          uint8_t t2 = s_buf_y[s_buf_offset + 2] & 0x7F;
          uint32_t t = t0 | (t1 << 8) | (t2 << 16);

          int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;
          unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
          good_mask <<= 31 - threadIdx.x;
          unsigned int ctr_offset = __popc(good_mask);
          if (ctr + ctr_offset <= DILITHIUM_N && good) {
            g_poly[ctr + ctr_offset - 1] = t;
          }
        }
      }
    }
  }
}

template <unsigned int VEC_SIZE, unsigned int PACKED_BYTES>
__global__ void unpack_fuse_ntt_kernel(int32_t *g_polyvec,
                                       const uint8_t *g_polyvec_packed,
                                       size_t mem_pool_pitch) {
  __shared__ int32_t s_poly[DILITHIUM_N + 32];
  int32_t regs[8];

  for (int vec_i = 0; vec_i < VEC_SIZE; vec_i++) {
    const uint8_t *g_poly_packed =
        g_polyvec_packed + blockIdx.x * mem_pool_pitch + vec_i * PACKED_BYTES;
    for (size_t i = 0; i < 8; i++) {
      if (PACKED_BYTES == POLYETA_PACKEDBYTES) {
#if ETA == 2
        uint32_t t = (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 0]) |
                     (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 1] << 8) |
                     (g_poly_packed[i * 12 + (threadIdx.x / 8) * 3 + 2] << 16);
        t >>= (threadIdx.x & 7) * 3;
        t &= 0x7;
        regs[i] = ETA - (int32_t)t;
#elif ETA == 4
        uint32_t t = g_poly_packed[i * 16 + threadIdx.x / 2];
        t >>= (threadIdx.x & 1) * 4;
        t &= 0xF;
        regs[i] = ETA - (int32_t)t;
#endif // ETA
      } else if (PACKED_BYTES == POLYT0_PACKEDBYTES) {
        uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                    ((threadIdx.x & 7) / 2) * 3 + 0]) |
                     (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                    ((threadIdx.x & 7) / 2) * 3 + 1]
                      << 8) |
                     (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                    ((threadIdx.x & 7) / 2) * 3 + 2]
                      << 16) |
                     (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                    ((threadIdx.x & 7) / 2) * 3 + 3]
                      << 24);
        t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
        t &= 0x1FFF;
        regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
      }
    }

    int32_t *g_poly = g_polyvec +
                      blockIdx.x * mem_pool_pitch / sizeof(int32_t) +
                      vec_i * DILITHIUM_N;
    ntt_inner(regs, s_poly);
    for (size_t i = 0; i < 8; i++)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

template __global__ void
unpack_fuse_ntt_kernel<DILITHIUM_L, POLYETA_PACKEDBYTES>(
    int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);
#if DILITHIUM_L != DILITHIUM_K
template __global__ void
unpack_fuse_ntt_kernel<DILITHIUM_K, POLYETA_PACKEDBYTES>(
    int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);
#endif
template __global__ void
unpack_fuse_ntt_kernel<DILITHIUM_K, POLYT0_PACKEDBYTES>(
    int32_t *g_polyvec, const uint8_t *g_polyvec_packed, size_t mem_pool_pitch);

#ifdef USE_128T_SIGN
// k6-style: 128 threads per block (4 warps), same block = same vector.
// Each warp processes a different poly of the vector in parallel.
// Uses warp-shuffle NTT with zero shared memory and coalesced writes.
// Launch as <<<batch_size, 128>>>.
__global__ void unpack_fuse_ntt_shuffle_k6_kernel(int32_t *g_s1, int32_t *g_s2,
                                                  int32_t *g_t0,
                                                  const uint8_t *g_s1_packed,
                                                  const uint8_t *g_s2_packed,
                                                  const uint8_t *g_t0_packed,
                                                  size_t mem_pool_pitch) {
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x & 0x1f;
  int32_t regs[8];

  for (int l = warpId; l < DILITHIUM_L; l += 4) {
#if ETA == 2
    const uint8_t *a =
        g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
    for (int i = 0; i < 8; i++) {
      uint32_t t = (a[i * 12 + (laneId / 8) * 3 + 0]) |
                   (a[i * 12 + (laneId / 8) * 3 + 1] << 8) |
                   (a[i * 12 + (laneId / 8) * 3 + 2] << 16);
      t >>= (laneId & 7) * 3;
      t &= 0x7;
      regs[i] = ETA - (int32_t)t;
    }
#elif ETA == 4
    const uint8_t *a =
        g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
    for (int i = 0; i < 8; i++) {
      uint32_t t = a[i * 16 + laneId / 2];
      t >>= (laneId & 1) * 4;
      t &= 0xF;
      regs[i] = ETA - (int32_t)t;
    }
#endif
    ntt_inner_shuffle_notranspose(regs);
    int32_t *g_s1_l =
        g_s1 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
    for (int i = 0; i < 8; i++)
      g_s1_l[i * 32 + laneId] = regs[i];
  }

  for (int k = warpId; k < DILITHIUM_K; k += 4) {
#if ETA == 2
    const uint8_t *a =
        g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
    for (int i = 0; i < 8; i++) {
      uint32_t t = (a[i * 12 + (laneId / 8) * 3 + 0]) |
                   (a[i * 12 + (laneId / 8) * 3 + 1] << 8) |
                   (a[i * 12 + (laneId / 8) * 3 + 2] << 16);
      t >>= (laneId & 7) * 3;
      t &= 0x7;
      regs[i] = ETA - (int32_t)t;
    }
#elif ETA == 4
    const uint8_t *a =
        g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
    for (int i = 0; i < 8; i++) {
      uint32_t t = a[i * 16 + laneId / 2];
      t >>= (laneId & 1) * 4;
      t &= 0xF;
      regs[i] = ETA - (int32_t)t;
    }
#endif
    ntt_inner_shuffle_notranspose(regs);
    int32_t *g_s2_k =
        g_s2 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    for (int i = 0; i < 8; i++)
      g_s2_k[i * 32 + laneId] = regs[i];
  }

  for (int k = warpId; k < DILITHIUM_K; k += 4) {
    const uint8_t *a =
        g_t0_packed + blockIdx.x * mem_pool_pitch + k * POLYT0_PACKEDBYTES;
    for (int i = 0; i < 8; i++) {
      uint32_t t =
          (a[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 0]) |
          (a[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 1] << 8) |
          (a[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 2] << 16) |
          (a[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 3] << 24);
      t >>= (laneId & 7) * 13 - ((laneId & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    ntt_inner_shuffle_notranspose(regs);
    int32_t *g_t0_k =
        g_t0 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    for (int i = 0; i < 8; i++)
      g_t0_k[i * 32 + laneId] = regs[i];
  }
}
#endif // USE_128T_SIGN

__global__ void unpack_fuse_ntt_radix2_kernel(int32_t *g_s1, int32_t *g_s2,
                                              int32_t *g_t0,
                                              const uint8_t *g_s1_packed,
                                              const uint8_t *g_s2_packed,
                                              const uint8_t *g_t0_packed,
                                              size_t mem_pool_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  __shared__ int32_t s_zetas[DILITHIUM_N];

  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();

  for (size_t l = 0; l < DILITHIUM_L; l++) {
    const uint8_t *a =
        g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
#if ETA == 2
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0] = a[3 * i + 0] >> 0;
      s_ntt[8 * i + 1] = a[3 * i + 0] >> 3;
      s_ntt[8 * i + 2] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
      s_ntt[8 * i + 3] = a[3 * i + 1] >> 1;
      s_ntt[8 * i + 4] = a[3 * i + 1] >> 4;
      s_ntt[8 * i + 5] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
      s_ntt[8 * i + 6] = a[3 * i + 2] >> 2;
      s_ntt[8 * i + 7] = a[3 * i + 2] >> 5;
    }
    __syncthreads();
    int32_t reg0 = ETA - (s_ntt[threadIdx.x] & 7);
    int32_t reg1 = ETA - (s_ntt[128 + threadIdx.x] & 7);
#elif ETA == 4
    uint8_t t = a[threadIdx.x];
    s_ntt[2 * threadIdx.x + 0] = t & 0x0F;
    s_ntt[2 * threadIdx.x + 1] = t >> 4;
    __syncthreads();
    int32_t reg0 = ETA - s_ntt[threadIdx.x];
    int32_t reg1 = ETA - s_ntt[128 + threadIdx.x];
#endif
    ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_s1_l =
        g_s1 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
    g_s1_l[threadIdx.x] = reg0;
    g_s1_l[128 + threadIdx.x] = reg1;
    __syncthreads();
  }

  for (size_t k = 0; k < DILITHIUM_K; k++) {
    const uint8_t *a =
        g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
#if ETA == 2
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0] = a[3 * i + 0] >> 0;
      s_ntt[8 * i + 1] = a[3 * i + 0] >> 3;
      s_ntt[8 * i + 2] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
      s_ntt[8 * i + 3] = a[3 * i + 1] >> 1;
      s_ntt[8 * i + 4] = a[3 * i + 1] >> 4;
      s_ntt[8 * i + 5] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
      s_ntt[8 * i + 6] = a[3 * i + 2] >> 2;
      s_ntt[8 * i + 7] = a[3 * i + 2] >> 5;
    }
    __syncthreads();
    int32_t reg0 = ETA - (s_ntt[threadIdx.x] & 7);
    int32_t reg1 = ETA - (s_ntt[128 + threadIdx.x] & 7);
#elif ETA == 4
    s_ntt[2 * threadIdx.x + 0] = a[threadIdx.x] & 0x0F;
    s_ntt[2 * threadIdx.x + 1] = a[threadIdx.x] >> 4;
    __syncthreads();
    int32_t reg0 = ETA - s_ntt[threadIdx.x];
    int32_t reg1 = ETA - s_ntt[128 + threadIdx.x];
#endif
    ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_s2_k =
        g_s2 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    g_s2_k[threadIdx.x] = reg0;
    g_s2_k[128 + threadIdx.x] = reg1;
    __syncthreads();
  }

  for (size_t k = 0; k < DILITHIUM_K; k++) {
    const uint8_t *a =
        g_t0_packed + blockIdx.x * mem_pool_pitch + k * POLYT0_PACKEDBYTES;
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0] = a[13 * i + 0] | (a[13 * i + 1] << 8);
      s_ntt[8 * i + 1] =
          (a[13 * i + 1] >> 5) | (a[13 * i + 2] << 3) | (a[13 * i + 3] << 11);
      s_ntt[8 * i + 2] = (a[13 * i + 3] >> 2) | (a[13 * i + 4] << 6);
      s_ntt[8 * i + 3] =
          (a[13 * i + 4] >> 7) | (a[13 * i + 5] << 1) | (a[13 * i + 6] << 9);
      s_ntt[8 * i + 4] =
          (a[13 * i + 6] >> 4) | (a[13 * i + 7] << 4) | (a[13 * i + 8] << 12);
      s_ntt[8 * i + 5] = (a[13 * i + 8] >> 1) | (a[13 * i + 9] << 7);
      s_ntt[8 * i + 6] =
          (a[13 * i + 9] >> 6) | (a[13 * i + 10] << 2) | (a[13 * i + 11] << 10);
      s_ntt[8 * i + 7] = (a[13 * i + 11] >> 3) | (a[13 * i + 12] << 5);
    }
    __syncthreads();
    int32_t reg0 = (1 << (DILITHIUM_D - 1)) - (s_ntt[threadIdx.x] & 0x1FFF);
    int32_t reg1 =
        (1 << (DILITHIUM_D - 1)) - (s_ntt[128 + threadIdx.x] & 0x1FFF);
    ntt_radix2_inner(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_t0_k =
        g_t0 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    g_t0_k[threadIdx.x] = reg0;
    g_t0_k[128 + threadIdx.x] = reg1;
    __syncthreads();
  }
}

__global__ void unpack_fuse_ntt_radix2_opt_kernel(int32_t *g_s1, int32_t *g_s2,
                                                  int32_t *g_t0,
                                                  const uint8_t *g_s1_packed,
                                                  const uint8_t *g_s2_packed,
                                                  const uint8_t *g_t0_packed,
                                                  size_t mem_pool_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  __shared__ int32_t s_zetas[DILITHIUM_N];

  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();

  for (size_t l = 0; l < DILITHIUM_L; l++) {
#if ETA == 2
    const uint8_t *a =
        g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0 + (i >> 2)] = a[3 * i + 0] >> 0;
      s_ntt[8 * i + 1 + (i >> 2)] = a[3 * i + 0] >> 3;
      s_ntt[8 * i + 2 + (i >> 2)] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
      s_ntt[8 * i + 3 + (i >> 2)] = a[3 * i + 1] >> 1;
      s_ntt[8 * i + 4 + (i >> 2)] = a[3 * i + 1] >> 4;
      s_ntt[8 * i + 5 + (i >> 2)] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
      s_ntt[8 * i + 6 + (i >> 2)] = a[3 * i + 2] >> 2;
      s_ntt[8 * i + 7 + (i >> 2)] = a[3 * i + 2] >> 5;
    }
    __syncthreads();
    int32_t reg0 = ETA - (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 7);
    int32_t reg1 =
        ETA - (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 7);
#elif ETA == 4
    const uint8_t *g_s1_packed_l =
        g_s1_packed + blockIdx.x * mem_pool_pitch + l * POLYETA_PACKEDBYTES;
    uint8_t t = g_s1_packed_l[threadIdx.x];
    s_ntt[2 * threadIdx.x + 0 + (threadIdx.x >> 4)] = t & 0x0F;
    s_ntt[2 * threadIdx.x + 1 + (threadIdx.x >> 4)] = t >> 4;
    __syncthreads();
    int32_t reg0 = ETA - s_ntt[threadIdx.x + (threadIdx.x >> 5)];
    int32_t reg1 = ETA - s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
#endif
    __syncthreads();
    ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_s1_l =
        g_s1 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + l * DILITHIUM_N;
    g_s1_l[threadIdx.x] = reg0;
    g_s1_l[128 + threadIdx.x] = reg1;
    __syncthreads();
  }

  for (size_t k = 0; k < DILITHIUM_K; k++) {
#if ETA == 2
    const uint8_t *a =
        g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0 + (i >> 2)] = a[3 * i + 0] >> 0;
      s_ntt[8 * i + 1 + (i >> 2)] = a[3 * i + 0] >> 3;
      s_ntt[8 * i + 2 + (i >> 2)] = (a[3 * i + 0] >> 6) | (a[3 * i + 1] << 2);
      s_ntt[8 * i + 3 + (i >> 2)] = a[3 * i + 1] >> 1;
      s_ntt[8 * i + 4 + (i >> 2)] = a[3 * i + 1] >> 4;
      s_ntt[8 * i + 5 + (i >> 2)] = (a[3 * i + 1] >> 7) | (a[3 * i + 2] << 1);
      s_ntt[8 * i + 6 + (i >> 2)] = a[3 * i + 2] >> 2;
      s_ntt[8 * i + 7 + (i >> 2)] = a[3 * i + 2] >> 5;
    }
    __syncthreads();
    int32_t reg0 = ETA - (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 7);
    int32_t reg1 =
        ETA - (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 7);
#elif ETA == 4
    const uint8_t *g_s2_packed_k =
        g_s2_packed + blockIdx.x * mem_pool_pitch + k * POLYETA_PACKEDBYTES;
    uint8_t t = g_s2_packed_k[threadIdx.x];
    s_ntt[2 * threadIdx.x + 0 + (threadIdx.x >> 4)] = t & 0x0F;
    s_ntt[2 * threadIdx.x + 1 + (threadIdx.x >> 4)] = t >> 4;
    __syncthreads();
    int32_t reg0 = ETA - s_ntt[threadIdx.x + (threadIdx.x >> 5)];
    int32_t reg1 = ETA - s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)];
#endif
    ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_s2_k =
        g_s2 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    g_s2_k[threadIdx.x] = reg0;
    g_s2_k[128 + threadIdx.x] = reg1;
    __syncthreads();
  }

  for (size_t k = 0; k < DILITHIUM_K; k++) {
    const uint8_t *a =
        g_t0_packed + blockIdx.x * mem_pool_pitch + k * POLYT0_PACKEDBYTES;
    for (size_t i = threadIdx.x; i < DILITHIUM_N / 8; i += blockDim.x) {
      s_ntt[8 * i + 0 + (i >> 2)] = a[13 * i + 0] | (a[13 * i + 1] << 8);
      s_ntt[8 * i + 1 + (i >> 2)] =
          (a[13 * i + 1] >> 5) | (a[13 * i + 2] << 3) | (a[13 * i + 3] << 11);
      s_ntt[8 * i + 2 + (i >> 2)] = (a[13 * i + 3] >> 2) | (a[13 * i + 4] << 6);
      s_ntt[8 * i + 3 + (i >> 2)] =
          (a[13 * i + 4] >> 7) | (a[13 * i + 5] << 1) | (a[13 * i + 6] << 9);
      s_ntt[8 * i + 4 + (i >> 2)] =
          (a[13 * i + 6] >> 4) | (a[13 * i + 7] << 4) | (a[13 * i + 8] << 12);
      s_ntt[8 * i + 5 + (i >> 2)] = (a[13 * i + 8] >> 1) | (a[13 * i + 9] << 7);
      s_ntt[8 * i + 6 + (i >> 2)] =
          (a[13 * i + 9] >> 6) | (a[13 * i + 10] << 2) | (a[13 * i + 11] << 10);
      s_ntt[8 * i + 7 + (i >> 2)] =
          (a[13 * i + 11] >> 3) | (a[13 * i + 12] << 5);
    }
    __syncthreads();
    int32_t reg0 = (1 << (DILITHIUM_D - 1)) -
                   (s_ntt[threadIdx.x + (threadIdx.x >> 5)] & 0x1FFF);
    int32_t reg1 = (1 << (DILITHIUM_D - 1)) -
                   (s_ntt[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] & 0x1FFF);
    ntt_radix2_inner_pad1(reg0, reg1, s_ntt, s_zetas);
    int32_t *g_t0_k =
        g_t0 + blockIdx.x * mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
    g_t0_k[threadIdx.x] = reg0;
    g_t0_k[128 + threadIdx.x] = reg1;
    __syncthreads();
  }
}

// shake, only fuse with unpack
__global__ void compute_y_kernel(int32_t *g_y, const uint8_t *g_rhoprime,
                                 const uint32_t *g_exec_lut,
                                 size_t sign_mem_pool_pitch,
                                 size_t temp_mem_pool_pitch) {
  __shared__ uint8_t s_in_buf[CRHBYTES + 2 + 6]; // pad 6 to align 8 bytes
  __shared__ uint8_t s_buf[POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];

  uint32_t exec_info = g_exec_lut[blockIdx.x];
  uint16_t nonce = exec_info & 0xFFFF;

  size_t sign_idx_offset = (exec_info >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  // copy rhoprime to shared memory for expand mask
  s_in_buf[threadIdx.x] = g_rhoprime[sign_idx_offset + threadIdx.x];
  s_in_buf[threadIdx.x + 32] = g_rhoprime[sign_idx_offset + threadIdx.x + 32];

  for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
    // padding two bytes
    if (threadIdx.x == 0) {
      // write kappa to shared memory for expand mask
      uint16_t nonce_l = DILITHIUM_L * nonce + l;
      s_in_buf[CRHBYTES + 0] = nonce_l & 0xFF;
      s_in_buf[CRHBYTES + 1] = nonce_l >> 8;
    }
    __syncwarp();
    // y = expand_mask(rhoprime || kappa)
    shake<SHAKE256_RATE, 0x1f>(s_buf, POLY_UNIFORM_GAMMA1_NBLOCKS, s_in_buf,
                               CRHBYTES + 2);
    __syncwarp();

    // unpack y
    int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    // 18 bit per coefficient
#if GAMMA1 == (1 << 17)
    for (size_t i = 0; i < 8; i++) {
      uint32_t t =
          (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 0]) |
          (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 1]
           << 8) |
          (s_buf[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 2]
           << 16);
      t >>= (threadIdx.x & 3) * 2;
      t &= 0x3FFFF;
      g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t)t;
    }
    // 20 bit per coefficient
#elif GAMMA1 == (1 << 19)
    for (size_t i = 0; i < 8; i++) {
      uint32_t t =
          (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 0]) |
          (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 1]
           << 8) |
          (s_buf[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 2]
           << 16);
      t >>= (threadIdx.x & 1) * 4;
      t &= 0xFFFFF;
      g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t)t;
    }
#endif
  }
}

__global__ void compute_y_opt_kernel(int32_t *g_y, const uint8_t *g_rhoprime,
                                     const uint32_t *g_exec_lut,
                                     size_t sign_mem_pool_pitch,
                                     size_t temp_mem_pool_pitch,
                                     size_t n_inputs) {
  __shared__ uint8_t s_in_buf[4][CRHBYTES + 2 + 6]; // pad 6 to align 8 bytes
  __shared__ uint8_t s_buf[4][POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];

  const unsigned global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const unsigned input_id = global_warp_id / DILITHIUM_L;
  const unsigned l = global_warp_id % DILITHIUM_L;

  if (input_id < n_inputs) {

    uint8_t *s_in_buf_y = s_in_buf[threadIdx.y];
    uint8_t *s_buf_y = s_buf[threadIdx.y];

    uint32_t exec_info = g_exec_lut[input_id];
    uint16_t nonce = exec_info & 0xFFFF;

    size_t sign_idx_offset = (exec_info >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                 ? sign_idx_offset
                                 : input_id * temp_mem_pool_pitch;

    // copy rhoprime to shared memory for expand mask
    s_in_buf_y[threadIdx.x] = g_rhoprime[sign_idx_offset + threadIdx.x];
    s_in_buf_y[threadIdx.x + 32] =
        g_rhoprime[sign_idx_offset + threadIdx.x + 32];

    // padding two bytes
    if (threadIdx.x == 0) {
      // write kappa to shared memory for expand mask
      uint16_t nonce_l = DILITHIUM_L * nonce + l;
      s_in_buf_y[CRHBYTES + 0] = nonce_l & 0xFF;
      s_in_buf_y[CRHBYTES + 1] = nonce_l >> 8;
    }
    __syncwarp();
    // y = expand_mask(rhoprime || kappa)
    shake<SHAKE256_RATE, 0x1f>(s_buf_y, POLY_UNIFORM_GAMMA1_NBLOCKS, s_in_buf_y,
                               CRHBYTES + 2);
    __syncwarp();

    // unpack y
    int32_t *g_y_l = g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    // 18 bit per coefficient
#if GAMMA1 == (1 << 17)
    for (size_t i = 0; i < 8; i++) {
      uint32_t t =
          (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 +
                   0]) |
          (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 1]
           << 8) |
          (s_buf_y[i * 72 + (threadIdx.x / 4) * 9 + (threadIdx.x & 3) * 2 + 2]
           << 16);
      t >>= (threadIdx.x & 3) * 2;
      t &= 0x3FFFF;
      g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t)t;
    }
    // 20 bit per coefficient
#elif GAMMA1 == (1 << 19)
    for (size_t i = 0; i < 8; i++) {
      uint32_t t =
          (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 +
                   0]) |
          (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 1]
           << 8) |
          (s_buf_y[i * 80 + (threadIdx.x / 2) * 5 + (threadIdx.x & 1) * 2 + 2]
           << 16);
      t >>= (threadIdx.x & 1) * 4;
      t &= 0xFFFFF;
      g_y_l[32 * i + threadIdx.x] = GAMMA1 - (int32_t)t;
    }
#endif
  }
}

__global__ void compute_w_32t_kernel(int32_t *g_z, int32_t *g_w0, int32_t *g_w1,
                                     uint8_t *g_w1_packed, const int32_t *g_y,
                                     const int32_t *g_mat,
                                     const uint32_t *g_exec_lut,
                                     size_t sign_mem_pool_pitch,
                                     size_t temp_mem_pool_pitch) {
  __shared__ int32_t s_tmp[DILITHIUM_N + 32];

  int32_t regs[8];

  size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
    // make a copy of y, stored in z, perform ntt(z), for later use to calculate
    // w = Ay
    const int32_t *g_y_l =
        g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    for (size_t i = 0; i < 8; i++)
      regs[i] = g_y_l[i * 32 + threadIdx.x];
    ntt_inner(regs, s_tmp);
    for (size_t i = 0; i < 8; i++)
      g_z_l[threadIdx.x * 8 + i] = regs[i];
  }

  for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
    const int32_t *g_mat_k = KEY_OFFSET(g_mat, sign_idx_offset) +
                             k * DILITHIUM_L * DILITHIUM_N;
    uint8_t *g_w1_packed_k =
        g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
    int32_t *g_w0_k =
        g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
    int32_t *g_w1_k =
        g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

    for (unsigned int i = 0; i < 8; ++i)
      regs[i] = 0;
    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
      const int32_t *g_mat_k_l = g_mat_k + l * DILITHIUM_N;
      int32_t *g_z_l =
          g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
      for (unsigned int i = 0; i < 8; ++i)
        regs[i] += gpu_montgomery_multiply(g_mat_k_l[i * 32 + threadIdx.x],
                                           g_z_l[i * 32 + threadIdx.x]);
    }

    for (unsigned int i = 0; i < 8; ++i)
      regs[i] = reduce32(regs[i]);

    for (unsigned int i = 0; i < 8; ++i)
      s_tmp[i * 32 + threadIdx.x] = regs[i];

    for (unsigned int i = 0; i < 8; ++i)
      regs[i] = s_tmp[threadIdx.x * 8 + i];

    invntt_inner(regs, s_tmp);

    for (unsigned int i = 0; i < 8; ++i) {
      regs[i] = caddq(regs[i]);
      regs[i] = decompose(&g_w0_k[32 * i + threadIdx.x], regs[i]);
    }

    for (unsigned int i = 0; i < 8; ++i) {
      s_tmp[32 * i + threadIdx.x] = regs[i];
      g_w1_k[32 * i + threadIdx.x] = regs[i];
    }
    __syncwarp();

#if GAMMA2 == (DILITHIUM_Q - 1) / 88
    for (unsigned int i = 0; i < 2; ++i) {
      g_w1_packed_k[6 * threadIdx.x + 3 * i + 0] =
          (s_tmp[8 * threadIdx.x + 4 * i + 0]) |
          (s_tmp[8 * threadIdx.x + 4 * i + 1] << 6);
      g_w1_packed_k[6 * threadIdx.x + 3 * i + 1] =
          (s_tmp[8 * threadIdx.x + 4 * i + 1] >> 2) |
          (s_tmp[8 * threadIdx.x + 4 * i + 2] << 4);
      g_w1_packed_k[6 * threadIdx.x + 3 * i + 2] =
          (s_tmp[8 * threadIdx.x + 4 * i + 2] >> 4) |
          (s_tmp[8 * threadIdx.x + 4 * i + 3] << 2);
    }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
    for (unsigned int i = 0; i < 4; ++i)
      g_w1_packed_k[4 * threadIdx.x + i] =
          (s_tmp[8 * threadIdx.x + 2 * i + 0]) |
          (s_tmp[8 * threadIdx.x + 2 * i + 1] << 4);
#endif
    __syncthreads();
  }
}

__global__ void compute_w_128t_kernel(int32_t *g_w0, int32_t *g_w1,
                                      uint8_t *g_w1_packed, const int32_t *g_y,
                                      const int32_t *g_mat,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch) {
  __shared__ int32_t s_tmp[DILITHIUM_N + 128];
  __shared__ int32_t s_zetas[DILITHIUM_N];

  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();

  size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  int32_t acc[DILITHIUM_K][2];

  for (auto &acc_k : acc) {
    acc_k[0] = 0;
    acc_k[1] = 0;
  }

  for (size_t l = 0; l < DILITHIUM_L; ++l) {
    const int32_t *g_y_l =
        g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    int32_t reg0 = g_y_l[threadIdx.x];
    int32_t reg1 = g_y_l[128 + threadIdx.x];

    ntt_radix2_inner(reg0, reg1, s_tmp, s_zetas);

    for (size_t k = 0; k < DILITHIUM_K; ++k) {
      const int32_t *g_mat_k_l = KEY_OFFSET(g_mat, sign_idx_offset) +
                                 k * DILITHIUM_L * DILITHIUM_N +
                                 l * DILITHIUM_N;
      acc[k][0] += gpu_montgomery_multiply(g_mat_k_l[threadIdx.x], reg0);
      acc[k][1] += gpu_montgomery_multiply(g_mat_k_l[128 + threadIdx.x], reg1);
    }
    __syncthreads();
  }

  for (size_t k = 0; k < DILITHIUM_K; ++k) {

    acc[k][0] = reduce32(acc[k][0]);
    acc[k][1] = reduce32(acc[k][1]);

    intt_radix2_inner(acc[k][0], acc[k][1], s_tmp, s_zetas);

    acc[k][0] = caddq(acc[k][0]);
    acc[k][1] = caddq(acc[k][1]);

    // decompose w0, w1, then write to w0
    int32_t *g_w0_k =
        g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
    acc[k][0] = decompose(&g_w0_k[threadIdx.x], acc[k][0]);
    acc[k][1] = decompose(&g_w0_k[128 + threadIdx.x], acc[k][1]);

    // write to w1
    int32_t *g_w1_k =
        g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
    g_w1_k[threadIdx.x] = acc[k][0];
    g_w1_k[128 + threadIdx.x] = acc[k][1];

    // pack w1 into sig
    __syncthreads();
    s_tmp[threadIdx.x + (threadIdx.x >> 5)] = acc[k][0];
    s_tmp[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] = acc[k][1];
    __syncthreads();
    uint8_t *g_w1_packed_k =
        g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
#if GAMMA2 == (DILITHIUM_Q - 1) / 88
    if (threadIdx.x < 64) {
      size_t offset = 4 * threadIdx.x + (threadIdx.x >> 3);
      g_w1_packed_k[3 * threadIdx.x + 0] =
          (s_tmp[offset + 0]) | (s_tmp[offset + 1] << 6);
      g_w1_packed_k[3 * threadIdx.x + 1] =
          (s_tmp[offset + 1] >> 2) | (s_tmp[offset + 2] << 4);
      g_w1_packed_k[3 * threadIdx.x + 2] =
          (s_tmp[offset + 2] >> 4) | (s_tmp[offset + 3] << 2);
    }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
    size_t offset = 2 * threadIdx.x + (threadIdx.x >> 4);
    g_w1_packed_k[threadIdx.x] = (s_tmp[offset + 0]) | (s_tmp[offset + 1] << 4);
#endif
    __syncthreads();
  }
}




__global__ void compute_cp_ntt_kernel(int32_t *g_cp, uint8_t *g_mu,
                                      uint8_t *g_seed, const uint8_t *g_ori_mu,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch,
                                      size_t n_inputs) {
  __shared__ uint8_t s_buf2[4][SHAKE256_RATE];
  __shared__ int32_t s_tmp[4][DILITHIUM_N + 32];

  const int lane = threadIdx.x;
  const int inst = threadIdx.y;
  const unsigned int block_idx = blockIdx.x * 4 + inst;

  if (block_idx >= n_inputs)
    return;

  size_t sign_idx_offset = (g_exec_lut[block_idx] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : block_idx * temp_mem_pool_pitch;

  g_mu[temp_idx_offset + lane] = g_ori_mu[sign_idx_offset + lane];
  g_mu[temp_idx_offset + 32 + lane] = g_ori_mu[sign_idx_offset + 32 + lane];
  __syncwarp();
  shake<SHAKE256_RATE, 0x1f>(s_buf2[inst], 1, g_mu + temp_idx_offset,
                             CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
  __syncwarp();

  for (int i = lane; i < CTILDEBYTES; i += 32)
    g_seed[temp_idx_offset + i] = s_buf2[inst][i];
  shake<SHAKE256_RATE, 0x1f>(s_buf2[inst], 1, s_buf2[inst], CTILDEBYTES);
  __syncwarp();

  int32_t regs[8];
  poly_challenge_shake256_opt(regs, s_buf2[inst], lane, s_tmp[inst]);
  ntt_inner(regs, s_tmp[inst]);
  for (int i = 0; i < 8; i++)
    g_cp[temp_idx_offset / sizeof(int32_t) + lane * 8 + i] = regs[i];
}

__global__ void compute_cp_kernel(int32_t *g_cp, uint8_t *g_mu, uint8_t *g_seed,
                                  const uint8_t *g_ori_mu,
                                  const uint32_t *g_exec_lut,
                                  size_t sign_mem_pool_pitch,
                                  size_t temp_mem_pool_pitch) {
  __shared__ uint8_t s_buf2[SHAKE256_RATE];
  __shared__ int32_t s_tmp[DILITHIUM_N + 32];

  size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  g_mu[temp_idx_offset + threadIdx.x] = g_ori_mu[sign_idx_offset + threadIdx.x];
  g_mu[temp_idx_offset + 32 + threadIdx.x] =
      g_ori_mu[sign_idx_offset + 32 + threadIdx.x];
  __syncwarp();
  shake<SHAKE256_RATE, 0x1f>(s_buf2, 1, g_mu + temp_idx_offset,
                             CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
  __syncwarp();

  // write to seed in sig
  for (size_t i = threadIdx.x; i < CTILDEBYTES; i += 32)
    g_seed[temp_idx_offset + i] = s_buf2[i];

  // poly_challenge(&cp, sig);
  shake<SHAKE256_RATE, 0x1f>(s_buf2, 1, s_buf2, CTILDEBYTES);
  __syncwarp();
  for (size_t i = 0; i < 8; i++) {
    s_tmp[i * 32 + threadIdx.x] = 0;
  }
  __syncwarp();
  if (threadIdx.x == 0) {
    uint64_t signs = reinterpret_cast<uint64_t *>(s_buf2)[0];
    size_t b;
    size_t pos = 8;
    for (size_t i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
      do {
        b = s_buf2[pos++];
      } while (b > i);

      s_tmp[i] = s_tmp[b];
      s_tmp[b] = 1 - 2 * (signs & 1);
      signs >>= 1;
    }
  }
  __syncwarp();
  for (size_t i = 0; i < 8; i++) {
    g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x] =
        s_tmp[i * 32 + threadIdx.x];
  }
}

__global__ void compute_cp_opt_kernel(int32_t *g_cp, uint8_t *g_mu,
                                      uint8_t *g_seed, const uint8_t *g_ori_mu,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch,
                                      size_t n_inputs) {
  __shared__ uint8_t s_buf2[4][SHAKE256_RATE];
  __shared__ int32_t s_tmp[4][DILITHIUM_N + 32];

  const unsigned input_id = blockIdx.x * blockDim.y + threadIdx.y;

  if (input_id < n_inputs) {

    uint8_t *s_buf2_y = s_buf2[threadIdx.y];
    int32_t *s_tmp_y = s_tmp[threadIdx.y];

    size_t sign_idx_offset = (g_exec_lut[input_id] >> 16) * sign_mem_pool_pitch;
    size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                                 ? sign_idx_offset
                                 : input_id * temp_mem_pool_pitch;

    g_mu[temp_idx_offset + threadIdx.x] =
        g_ori_mu[sign_idx_offset + threadIdx.x];
    g_mu[temp_idx_offset + 32 + threadIdx.x] =
        g_ori_mu[sign_idx_offset + 32 + threadIdx.x];
    __syncwarp();
    shake<SHAKE256_RATE, 0x1f>(s_buf2_y, 1, g_mu + temp_idx_offset,
                               CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
    __syncwarp();

    // write to seed in sig
    for (size_t i = threadIdx.x; i < CTILDEBYTES; i += 32)
      g_seed[temp_idx_offset + i] = s_buf2_y[i];

    // poly_challenge(&cp, sig);
    shake<SHAKE256_RATE, 0x1f>(s_buf2_y, 1, s_buf2_y, CTILDEBYTES);
    __syncwarp();
    for (size_t i = 0; i < 8; i++) {
      s_tmp_y[i * 32 + threadIdx.x] = 0;
    }
    __syncwarp();
    if (threadIdx.x == 0) {
      uint64_t signs = reinterpret_cast<uint64_t *>(s_buf2_y)[0];
      size_t pos = 8;
      size_t b;
      for (size_t i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
        do {
          b = s_buf2_y[pos++];
        } while (b > i);

        s_tmp_y[i] = s_tmp_y[b];
        s_tmp_y[b] = 1 - 2 * (signs & 1);
        signs >>= 1;
      }
    }
    __syncwarp();
    for (size_t i = 0; i < 8; i++) {
      g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x] =
          s_tmp_y[i * 32 + threadIdx.x];
    }
  }
}

__global__ void
rej_loop_32t_kernel(const int32_t *g_y, int32_t *g_z, const int32_t *g_w0,
                    const int32_t *g_w1, const int32_t *g_cp,
                    uint8_t *g_z_packed, uint8_t *g_hint, uint8_t *g_done_lut,
                    const int32_t *g_s1, const int32_t *g_s2,
                    const int32_t *g_t0, const uint32_t *g_exec_lut,
                    size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
  __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];

  __shared__ int32_t s_tmp[DILITHIUM_N + 32];
  __shared__ int32_t s_tmp2[DILITHIUM_N + 32];

  __shared__ int32_t s_cp[DILITHIUM_N + 32];

  int32_t regs[8];

  size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  // poly_ntt(&cp);
  for (size_t i = 0; i < 8; i++)
    regs[i] = g_cp[temp_idx_offset / sizeof(int32_t) + i * 32 + threadIdx.x];
#ifndef USE_CP_NTT
  ntt_inner(regs, s_tmp);
#endif
  for (size_t i = 0; i < 8; i++)
    s_cp[threadIdx.x * 8 + i] = regs[i];
  __syncwarp();

  // validation

  for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += 32) {
    s_hint[i] = 0;
  }
  __syncwarp();

  for (size_t l = 0; l < DILITHIUM_L; ++l) {
    const int32_t *g_s1_l =
        KEY_OFFSET(g_s1, sign_idx_offset) + l * DILITHIUM_N;
    const int32_t *g_y_l =
        g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;

    for (unsigned int i = 0; i < 8; ++i)
      // pointwise_montgomery
      s_tmp[32 * i + threadIdx.x] = gpu_montgomery_multiply(
          s_cp[32 * i + threadIdx.x], g_s1_l[32 * i + threadIdx.x]);

    for (size_t i = 0; i < 8; i++)
      regs[i] = s_tmp[threadIdx.x * 8 + i];
    // invntt_tomont
    invntt_inner(regs, s_tmp);

    for (size_t i = 0; i < 8; i++) {
      // add
      regs[i] += g_y_l[i * 32 + threadIdx.x];
      // reduce
      regs[i] = reduce32(regs[i]);

      // chknorm
      unsigned int rej_mask =
          __ballot_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(regs[i]));
      if (rej_mask)
        return;
      // write to tmp z
      g_z_l[i * 32 + threadIdx.x] = regs[i];
    }
  }

  unsigned int pos = 0;

  for (size_t k = 0; k < DILITHIUM_K; ++k) {
    const int32_t *g_s2_k =
        KEY_OFFSET(g_s2, sign_idx_offset) + k * DILITHIUM_N;
    const int32_t *g_t0_k =
        KEY_OFFSET(g_t0, sign_idx_offset) + k * DILITHIUM_N;
    const int32_t *g_w0_k =
        g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
    const int32_t *g_w1_k =
        g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

    // tmp = cp * s2
    for (size_t i = 0; i < 8; ++i)
      // poly_pointwise_montgomery(&tmp, &cp, &s2.vec[i]);
      s_tmp[32 * i + threadIdx.x] = gpu_montgomery_multiply(
          s_cp[32 * i + threadIdx.x], g_s2_k[32 * i + threadIdx.x]);

    // regs = invntt(tmp)
    for (size_t i = 0; i < 8; i++)
      regs[i] = s_tmp[threadIdx.x * 8 + i];
    // poly_invntt_tomont(&tmp);
    invntt_inner(regs, s_tmp);

    for (size_t i = 0; i < 8; i++) {
      // poly_sub(&tmp, &w0.vec[i], &tmp);
      regs[i] = g_w0_k[32 * i + threadIdx.x] - regs[i];
      // poly_reduce(&tmp);
      regs[i] = reduce32(regs[i]);
      // poly_chknorm(&tmp, GAMMA2 - BETA)
      unsigned int rej_mask =
          __ballot_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(regs[i]));
      if (rej_mask)
        return;

      // write to tmp
      s_tmp[i * 32 + threadIdx.x] = regs[i];
    }

    // tmp2 = cp * t0; regs = invntt(tmp2)
    for (size_t i = 0; i < 8; i++)
      // poly_pointwise_montgomery(&tmp2, &cp, &t0.vec[i]);
      s_tmp2[32 * i + threadIdx.x] = gpu_montgomery_multiply(
          s_cp[32 * i + threadIdx.x], g_t0_k[32 * i + threadIdx.x]);

    for (size_t i = 0; i < 8; i++)
      // load to regs for invntt
      regs[i] = s_tmp2[threadIdx.x * 8 + i];

    // poly_invntt_tomont(&tmp2);
    invntt_inner(regs, s_tmp2);

    for (size_t i = 0; i < 8; i++) {
      // poly_reduce(&tmp2);
      regs[i] = reduce32(regs[i]);
      // poly_chknorm(&tmp2, GAMMA2)
      unsigned int rej_mask =
          __ballot_sync(0xFFFFFFFF, chknorm<GAMMA2>(regs[i]));
      if (rej_mask)
        return;

      // poly_add(&tmp, &tmp, &tmp2);
      regs[i] = regs[i] + s_tmp[32 * i + threadIdx.x];
    }

    // make hint
    for (size_t i = 0; i < 8; ++i) {
      // unsigned int n = poly_make_hint(&h.vec[i], &tmp, &w1.vec[i]);
      unsigned int hint_bit = make_hint(regs[i], g_w1_k[32 * i + threadIdx.x]);
      unsigned int hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
      unsigned int inclusive_hint_mask = hint_mask << (31 - threadIdx.x);
      unsigned int n = __popc(hint_mask);
      unsigned int inclusive_n = __popc(inclusive_hint_mask);
      unsigned int rej_mask =
          __ballot_sync(0xFFFFFFFF, pos + inclusive_n > OMEGA);
      if (rej_mask)
        return;

      if (hint_bit)
        s_hint[pos + inclusive_n - 1] = 32 * i + threadIdx.x;
      __syncwarp();
      // for next loop
      pos += n;
    }
    // set last hint byte
    if (threadIdx.x == 0)
      s_hint[OMEGA + k] = pos;
    __syncwarp();
  }

  // write to packed z in signature
  for (size_t l = 0; l < DILITHIUM_L; ++l) {
    uint8_t *g_z_packed_l =
        g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
    int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
    for (size_t i = 0; i < 2; ++i) {
      size_t offset = 32 * i + threadIdx.x;

      uint32_t t0 = GAMMA1 - g_z_l[4 * offset + 0];
      uint32_t t1 = GAMMA1 - g_z_l[4 * offset + 1];
      uint32_t t2 = GAMMA1 - g_z_l[4 * offset + 2];
      uint32_t t3 = GAMMA1 - g_z_l[4 * offset + 3];

      g_z_packed_l[9 * offset + 0] = t0;
      g_z_packed_l[9 * offset + 1] = t0 >> 8;
      g_z_packed_l[9 * offset + 2] = t0 >> 16;
      g_z_packed_l[9 * offset + 2] |= t1 << 2;
      g_z_packed_l[9 * offset + 3] = t1 >> 6;
      g_z_packed_l[9 * offset + 4] = t1 >> 14;
      g_z_packed_l[9 * offset + 4] |= t2 << 4;
      g_z_packed_l[9 * offset + 5] = t2 >> 4;
      g_z_packed_l[9 * offset + 6] = t2 >> 12;
      g_z_packed_l[9 * offset + 6] |= t3 << 6;
      g_z_packed_l[9 * offset + 7] = t3 >> 2;
      g_z_packed_l[9 * offset + 8] = t3 >> 10;
    }
#elif GAMMA1 == (1 << 19)
    for (size_t i = 0; i < 4; ++i) {
      size_t offset = 32 * i + threadIdx.x;

      uint32_t t0 = GAMMA1 - g_z_l[2 * offset + 0];
      uint32_t t1 = GAMMA1 - g_z_l[2 * offset + 1];

      g_z_packed_l[5 * offset + 0] = t0;
      g_z_packed_l[5 * offset + 1] = t0 >> 8;
      g_z_packed_l[5 * offset + 2] = t0 >> 16;
      g_z_packed_l[5 * offset + 2] |= t1 << 4;
      g_z_packed_l[5 * offset + 3] = t1 >> 4;
      g_z_packed_l[5 * offset + 4] = t1 >> 12;
    }
#endif
  }

  // write to hint in signature
  for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += 32) {
    g_hint[temp_idx_offset + i] = s_hint[i];
  }

  // set done flag
  if (threadIdx.x == 0)
    g_done_lut[blockIdx.x] = 1;
}

__global__ void
rej_loop_128t_kernel(const int32_t *g_y, int32_t *g_z, const int32_t *g_w0,
                     const int32_t *g_w1, const int32_t *g_cp,
                     uint8_t *g_z_packed, uint8_t *g_hint, uint8_t *g_done_lut,
                     const int32_t *g_s1, const int32_t *g_s2,
                     const int32_t *g_t0, const uint32_t *g_exec_lut,
                     size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch) {
  __shared__ int32_t s_zetas[DILITHIUM_N];
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];

  __shared__ int s_rej_flag;
  __shared__ size_t s_hint_count[4];
  __shared__ size_t s_hint_count_sum;

  // init rej flag with 0
  if (threadIdx.x == 0)
    s_rej_flag = 0;
  // cache zetas table to SMEM
  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[threadIdx.x + 128] = c_zetas[threadIdx.x + 128];
  // init hint with all 0
  for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x) {
    s_hint[i] = 0;
  }
  __syncthreads();

  size_t lane_id = threadIdx.x & 0x1f;
  size_t warp_id = threadIdx.x >> 5;

  size_t sign_idx_offset = (g_exec_lut[blockIdx.x] >> 16) * sign_mem_pool_pitch;
  size_t temp_idx_offset = (sign_mem_pool_pitch == temp_mem_pool_pitch)
                               ? sign_idx_offset
                               : blockIdx.x * temp_mem_pool_pitch;

  // poly_ntt(&cp);
  int32_t cp_regs0 = g_cp[temp_idx_offset / sizeof(int32_t) + threadIdx.x];
  int32_t cp_regs1 =
      g_cp[temp_idx_offset / sizeof(int32_t) + 128 + threadIdx.x];
#ifndef USE_CP_NTT
  ntt_radix2_inner(cp_regs0, cp_regs1, s_ntt, s_zetas);
#endif

  // validation

  for (size_t l = 0; l < DILITHIUM_L; ++l) {
    const int32_t *g_s1_l =
        KEY_OFFSET(g_s1, sign_idx_offset) + l * DILITHIUM_N;
    const int32_t *g_y_l =
        g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
    int32_t *g_z_l = g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;

    int32_t cps1_reg0 = gpu_montgomery_multiply(cp_regs0, g_s1_l[threadIdx.x]);
    int32_t cps1_reg1 =
        gpu_montgomery_multiply(cp_regs1, g_s1_l[128 + threadIdx.x]);
    intt_radix2_inner(cps1_reg0, cps1_reg1, s_ntt, s_zetas);
    cps1_reg0 += g_y_l[threadIdx.x];
    cps1_reg1 += g_y_l[128 + threadIdx.x];
    cps1_reg0 = reduce32(cps1_reg0);
    cps1_reg1 = reduce32(cps1_reg1);
    // chknorm
    int rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_reg0));
    rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_reg1));
    if (lane_id == 0 && rej_flag)
      s_rej_flag = 1;
    __syncthreads();
    if (s_rej_flag)
      return;
    // write to tmp z
    g_z_l[threadIdx.x] = cps1_reg0;
    g_z_l[128 + threadIdx.x] = cps1_reg1;
  }

  unsigned int pos = 0;
  for (size_t k = 0; k < DILITHIUM_K; ++k) {
    const int32_t *g_s2_k =
        KEY_OFFSET(g_s2, sign_idx_offset) + k * DILITHIUM_N;
    const int32_t *g_t0_k =
        KEY_OFFSET(g_t0, sign_idx_offset) + k * DILITHIUM_N;
    const int32_t *g_w0_k =
        g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
    const int32_t *g_w1_k =
        g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

    // tmp = cp * s2
    int32_t cps2_reg0 = gpu_montgomery_multiply(cp_regs0, g_s2_k[threadIdx.x]);
    int32_t cps2_reg1 =
        gpu_montgomery_multiply(cp_regs1, g_s2_k[128 + threadIdx.x]);
    intt_radix2_inner(cps2_reg0, cps2_reg1, s_ntt, s_zetas);

    // poly_sub(&tmp, &w0.vec[i], &tmp);
    cps2_reg0 = g_w0_k[threadIdx.x] - cps2_reg0;
    cps2_reg1 = g_w0_k[128 + threadIdx.x] - cps2_reg1;
    // poly_reduce(&tmp);
    cps2_reg0 = reduce32(cps2_reg0);
    cps2_reg1 = reduce32(cps2_reg1);
    // poly_chknorm(&tmp, GAMMA2 - BETA)
    int rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_reg0));
    rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_reg1));
    if (lane_id == 0 && rej_flag)
      s_rej_flag = 1;
    __syncthreads();
    if (s_rej_flag)
      return;

    // tmp2 = cp * t0; regs = invntt(tmp2)
    int32_t cpt0_reg0 = gpu_montgomery_multiply(cp_regs0, g_t0_k[threadIdx.x]);
    int32_t cpt0_reg1 =
        gpu_montgomery_multiply(cp_regs1, g_t0_k[128 + threadIdx.x]);
    intt_radix2_inner(cpt0_reg0, cpt0_reg1, s_ntt, s_zetas);

    // poly_reduce(&tmp2);
    cpt0_reg0 = reduce32(cpt0_reg0);
    cpt0_reg1 = reduce32(cpt0_reg1);

    // poly_chknorm(&tmp2, GAMMA2)
    rej_flag = __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_reg0));
    rej_flag += __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_reg1));
    if (lane_id == 0 && rej_flag)
      s_rej_flag = 1;
    __syncthreads();
    if (s_rej_flag)
      return;

    // poly_add(&tmp, &tmp, &tmp2);
    cpt0_reg0 += cps2_reg0;
    cpt0_reg1 += cps2_reg1;

    // make hint
    unsigned int hint_bit = make_hint(cpt0_reg0, g_w1_k[threadIdx.x]);
    unsigned int hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
    if (lane_id == 0)
      s_hint_count[warp_id] = __popc(hint_mask);
    __syncthreads();
    PREFIX_SUM_HINT_COUNT
    __syncthreads();
    size_t total_count_within_block = s_hint_count_sum;
    if (pos + total_count_within_block > OMEGA)
      return;
    unsigned int inclusive_count_within_block =
        __popc(hint_mask << (31 - lane_id)) + s_hint_count[warp_id];
    if (hint_bit)
      s_hint[pos + inclusive_count_within_block - 1] = threadIdx.x;
    __syncthreads();
    pos += total_count_within_block;

    hint_bit = make_hint(cpt0_reg1, g_w1_k[128 + threadIdx.x]);
    hint_mask = __ballot_sync(0xFFFFFFFF, hint_bit);
    if (lane_id == 0)
      s_hint_count[warp_id] = __popc(hint_mask);
    __syncthreads();
    PREFIX_SUM_HINT_COUNT
    __syncthreads();
    total_count_within_block = s_hint_count_sum;
    if (pos + total_count_within_block > OMEGA)
      return;
    inclusive_count_within_block =
        __popc(hint_mask << (31 - lane_id)) + s_hint_count[warp_id];
    if (hint_bit)
      s_hint[pos + inclusive_count_within_block - 1] = 128 + threadIdx.x;
    __syncthreads();
    pos += total_count_within_block;

    // set last hint byte
    if (threadIdx.x == 0)
      s_hint[OMEGA + k] = pos;
    __syncthreads();
  }

  // write to hint in signature
  for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x) {
    g_hint[temp_idx_offset + i] = s_hint[i];
  }

  // write to packed z in signature
  for (size_t l = 0; l < DILITHIUM_L; ++l) {
    uint8_t *g_z_packed_l =
        g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
    const int32_t *g_z_l =
        g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
    if (threadIdx.x < 64) {
      size_t i = threadIdx.x;
      uint32_t t[4];

      t[0] = GAMMA1 - g_z_l[4 * i + 0];
      t[1] = GAMMA1 - g_z_l[4 * i + 1];
      t[2] = GAMMA1 - g_z_l[4 * i + 2];
      t[3] = GAMMA1 - g_z_l[4 * i + 3];

      g_z_packed_l[9 * i + 0] = t[0];
      g_z_packed_l[9 * i + 1] = t[0] >> 8;
      g_z_packed_l[9 * i + 2] = t[0] >> 16;
      g_z_packed_l[9 * i + 2] |= t[1] << 2;
      g_z_packed_l[9 * i + 3] = t[1] >> 6;
      g_z_packed_l[9 * i + 4] = t[1] >> 14;
      g_z_packed_l[9 * i + 4] |= t[2] << 4;
      g_z_packed_l[9 * i + 5] = t[2] >> 4;
      g_z_packed_l[9 * i + 6] = t[2] >> 12;
      g_z_packed_l[9 * i + 6] |= t[3] << 6;
      g_z_packed_l[9 * i + 7] = t[3] >> 2;
      g_z_packed_l[9 * i + 8] = t[3] >> 10;
    }
#elif GAMMA1 == (1 << 19)
    size_t i = threadIdx.x;
    uint32_t t[2];

    t[0] = GAMMA1 - g_z_l[2 * i + 0];
    t[1] = GAMMA1 - g_z_l[2 * i + 1];

    g_z_packed_l[5 * i + 0] = t[0];
    g_z_packed_l[5 * i + 1] = t[0] >> 8;
    g_z_packed_l[5 * i + 2] = t[0] >> 16;
    g_z_packed_l[5 * i + 2] |= t[1] << 4;
    g_z_packed_l[5 * i + 3] = t[1] >> 4;
    g_z_packed_l[5 * i + 4] = t[1] >> 12;
#endif
  }

  // set done flag
  if (threadIdx.x == 0)
    g_done_lut[blockIdx.x] = 1;
}

__global__ void sig_copy_kernel(uint8_t *d_sig, size_t d_sign_mem_pool_pitch,
                                const uint8_t *d_temp_sig,
                                size_t d_temp_mem_pool_pitch,
                                const copy_lut_element *d_copy_lut) {
  copy_lut_element e = d_copy_lut[blockIdx.x];
  size_t exec_idx_offset = e.exec_idx * d_temp_mem_pool_pitch;
  size_t sign_idx_offset = e.sign_idx * d_sign_mem_pool_pitch;
  for (size_t i = threadIdx.x; i < CRYPTO_BYTES; i += blockDim.x) {
    d_sig[sign_idx_offset + i] = d_temp_sig[exec_idx_offset + i];
  }
}


#if defined(USE_PERSISTENT_WARPNTT) || defined(USE_PERSISTENT)
#ifdef FUSED_PREGEN_Y
__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned n) {
  return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] with the whole state in the registers of one thread
__device__ __forceinline__ void keccakf1600_thread(uint64_t s[25]) {
  const unsigned rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                             27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  const unsigned piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};
#pragma unroll 1
  for (unsigned r = 0; r < keccak_rounds; r++) {
    uint64_t bc[5];
    // theta
#pragma unroll
    for (int i = 0; i < 5; i++)
      bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
#pragma unroll
    for (int i = 0; i < 5; i++) {
      uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5)
        s[j + i] ^= t;
    }
    // rho and pi
    uint64_t t = s[1];
#pragma unroll
    for (int i = 0; i < 24; i++) {
      uint64_t b = s[piln[i]];
      s[piln[i]] = rotl64(t, rotc[i]);
      t = b;
    }
    // chi
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
#pragma unroll
      for (int i = 0; i < 5; i++)
        bc[i] = s[j + i];
#pragma unroll
      for (int i = 0; i < 5; i++)
        s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    // iota
    s[0] ^= __ldg(&keccak_f_round_constants[r]);
  }
}

// Packed y (POLYZ_PACKEDBYTES per polynomial) of the first
// FUSED_PREGEN_ATTEMPTS attempts: one thread per (signature, attempt, l), i.e.
// SHAKE256(rho'' || le16(L * attempt + l)). Row i of g_ypre is at i * pitch.
__global__ void expand_mask_pregen_kernel(uint8_t *g_ypre,
                                          const uint8_t *g_rhoprime,
                                          size_t pitch, size_t batch_size) {
  constexpr unsigned per_sig = FUSED_PREGEN_ATTEMPTS * DILITHIUM_L;
  constexpr unsigned n_words = POLYZ_PACKEDBYTES / 8;
  constexpr unsigned rate_words = SHAKE256_RATE / 8;
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch_size * per_sig)
    return;
  const size_t sig = t / per_sig;
  const unsigned nonce = t % per_sig; // = L * attempt + l

  const uint64_t *rp =
      reinterpret_cast<const uint64_t *>(g_rhoprime + sig * pitch);
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; i++)
    s[i] = 0;
#pragma unroll
  for (int i = 0; i < CRHBYTES / 8; i++)
    s[i] = rp[i];
  s[CRHBYTES / 8] = (uint64_t)nonce | (0x1fULL << 16); // le16 nonce, then pad
  s[rate_words - 1] ^= 0x8000000000000000ULL;

  uint64_t *out = reinterpret_cast<uint64_t *>(g_ypre + sig * pitch +
                                               (size_t)nonce * POLYZ_PACKEDBYTES);
#pragma unroll 1
  for (unsigned w = 0; w < n_words; w += rate_words) {
    keccakf1600_thread(s);
#pragma unroll
    for (unsigned i = 0; i < rate_words; i++)
      if (w + i < n_words)
        out[w + i] = s[i];
  }
}
#endif // FUSED_PREGEN_Y

// FULL: loop until an attempt is accepted.
// COMMIT: one attempt up to w1. RESPOND: rest of that attempt, using the
// challenge from staged_challenge_kernel (FUSED_STAGED only).
enum { SIGN_MODE_FULL = 0, SIGN_MODE_COMMIT = 1, SIGN_MODE_RESPOND = 2 };

#ifdef FUSED_STAGED
// one thread per signature: c~ = H(mu || w1_packed) -> g_seed, plus the first
// SHAKE256(c~) block for SampleInBall (mu and w1_packed are contiguous)
__global__ void staged_challenge_kernel(uint8_t *g_seed, const uint8_t *g_mu,
                                        uint8_t *g_ypre, size_t pitch,
                                        size_t batch_size) {
  constexpr unsigned rate_words = SHAKE256_RATE / 8;
  constexpr unsigned in_bytes = CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES;
  static_assert(in_bytes % 8 == 0 && CTILDEBYTES % 8 == 0, "word-sized input");
  constexpr unsigned in_words = in_bytes / 8;
  constexpr unsigned n_full = in_words / rate_words;
  constexpr unsigned rem = in_words % rate_words;
  constexpr unsigned ct_words = CTILDEBYTES / 8;

  const size_t sig = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (sig >= batch_size)
    return;
  uint32_t *state = reinterpret_cast<uint32_t *>(g_ypre + sig * pitch +
                                                 SIGN_POOL_YPRE_DATA_BYTES);
  if (state[1])
    return;

  const uint64_t *in = reinterpret_cast<const uint64_t *>(g_mu + sig * pitch);
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; i++)
    s[i] = 0;
#pragma unroll 1
  for (unsigned b = 0; b < n_full; b++) {
#pragma unroll
    for (unsigned i = 0; i < rate_words; i++)
      s[i] ^= in[b * rate_words + i];
    keccakf1600_thread(s);
  }
#pragma unroll
  for (unsigned i = 0; i < rem; i++)
    s[i] ^= in[n_full * rate_words + i];
  s[rem] ^= 0x1f;
  s[rate_words - 1] ^= 0x8000000000000000ULL;
  keccakf1600_thread(s);

  uint64_t ct[ct_words];
  uint64_t *g_ct = reinterpret_cast<uint64_t *>(g_seed + sig * pitch);
#pragma unroll
  for (unsigned i = 0; i < ct_words; i++)
    g_ct[i] = ct[i] = s[i];

  // SHAKE256(c~), first block
#pragma unroll
  for (int i = 0; i < 25; i++)
    s[i] = 0;
#pragma unroll
  for (unsigned i = 0; i < ct_words; i++)
    s[i] = ct[i];
  s[ct_words] ^= 0x1f;
  s[rate_words - 1] ^= 0x8000000000000000ULL;
  keccakf1600_thread(s);
  uint64_t *blk = reinterpret_cast<uint64_t *>(state + 2);
#pragma unroll
  for (unsigned i = 0; i < rate_words; i++)
    blk[i] = s[i];
}
#endif // FUSED_STAGED

#define SIGN_FUSED_ARGS                                                        \
  g_w0, g_w1, g_w1_packed, g_y, g_rhoprime, g_mat, g_z, g_cp, g_z_packed,      \
      g_hint, g_s1, g_s2, g_t0, g_mu, g_seed, g_ori_mu, g_sig,                 \
      sign_mem_pool_pitch, batch_size

#ifdef SIGN_PHASE_PROFILE
// cycle counters for the fused kernels (profiling builds only)
// PROF_PHASE: end of a block-wide phase, PROF_SUB: steps of warp 0 inside it
enum {
  PP_P1, PP_SHAKE_Y, PP_NTT_Y,
  PP_P2, PP_AY, PP_INTT_W, PP_DECOMP,
  PP_P3, PP_H_MU_W1, PP_SHAKE_C, PP_SAMPLE_BALL, PP_NTT_C,
  PP_P4A, PP_P4B, PP_P4C, PP_P5,
  PP_TOTAL, PP_ATTEMPTS, PP_BLOCKS, PP_N
};
#define PP_HIST 64
__device__ unsigned long long g_sign_prof[PP_N];
__device__ unsigned long long g_sign_att_hist[PP_HIST];
#define PROF_DECL                                                              \
  __shared__ unsigned long long s_prof[PP_N];                                  \
  unsigned long long prof_t0 = 0, prof_sub = 0, prof_start = 0;                \
  if (threadIdx.x == 0) {                                                      \
    for (int i = 0; i < PP_N; i++)                                             \
      s_prof[i] = 0;                                                           \
    prof_start = prof_t0 = prof_sub = clock64();                               \
  }
#define PROF_PHASE(i)                                                          \
  if (threadIdx.x == 0) {                                                      \
    unsigned long long t = clock64();                                          \
    s_prof[i] += t - prof_t0;                                                  \
    prof_t0 = prof_sub = t;                                                    \
  }
#define PROF_SUB(i)                                                            \
  if (threadIdx.x == 0) {                                                      \
    unsigned long long t = clock64();                                          \
    s_prof[i] += t - prof_sub;                                                 \
    prof_sub = t;                                                              \
  }
#define PROF_RESYNC                                                            \
  if (threadIdx.x == 0)                                                        \
    prof_t0 = prof_sub;
#define PROF_FLUSH(attempts)                                                   \
  if (threadIdx.x == 0) {                                                      \
    s_prof[PP_TOTAL] = clock64() - prof_start;                                 \
    s_prof[PP_ATTEMPTS] = (attempts);                                          \
    s_prof[PP_BLOCKS] = 1;                                                     \
    for (int i = 0; i < PP_N; i++)                                             \
      atomicAdd(&g_sign_prof[i], s_prof[i]);                                   \
    atomicAdd(&g_sign_att_hist[min((int)(attempts), PP_HIST - 1)], 1ULL);      \
  }

extern "C" void sign_phase_profile_reset() {
  unsigned long long zero[PP_N > PP_HIST ? PP_N : PP_HIST] = {0};
  cudaMemcpyToSymbol(g_sign_prof, zero, sizeof(g_sign_prof));
  cudaMemcpyToSymbol(g_sign_att_hist, zero, sizeof(g_sign_att_hist));
}

extern "C" void sign_phase_profile_dump() {
  unsigned long long p[PP_N], h[PP_HIST];
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(p, g_sign_prof, sizeof(p));
  cudaMemcpyFromSymbol(h, g_sign_att_hist, sizeof(h));
  static const char *names[PP_N] = {
      "P1 ExpandMask+NTT(y)", "  warp0 SHAKE256 y", "  warp0 unpack+NTT y",
      "P2 Ay+INTT+Decompose", "  warp0 A*y MAC", "  warp0 INTT w",
      "  warp0 decompose+pack w1", "P3 challenge", "  warp0 H(mu||w1)",
      "  warp0 H(c~) for ball", "  lane0 SampleInBall", "  warp0 NTT(c)",
      "P4a c*s1, z check", "P4b c*s2, c*t0", "P4c hints", "P5 pack sig",
      "TOTAL", "attempts", "blocks"};
  double blocks = (double)p[PP_BLOCKS];
  double total = (double)p[PP_TOTAL];
  printf("PROFILE,blocks,%.0f,attempts_per_sig,%.3f\n", blocks,
         p[PP_ATTEMPTS] / blocks);
  for (int i = 0; i < PP_TOTAL; i++)
    printf("PROFILE,%s,cycles_per_sig,%.0f,pct_of_total,%.1f\n", names[i],
           p[i] / blocks, 100.0 * p[i] / total);
  printf("PROFILE,TOTAL,cycles_per_sig,%.0f\n", total / blocks);
  printf("PROFILE_HIST");
  for (int i = 0; i < PP_HIST; i++)
    printf(",%llu", h[i]);
  printf("\n");
}
#else
#define PROF_DECL
#define PROF_PHASE(i)
#define PROF_SUB(i)
#define PROF_FLUSH(attempts)
#define PROF_RESYNC
#endif
#endif

#ifdef USE_PERSISTENT
// same as sign_persistent_warpntt_kernel_t, but with the shared-memory
// radix-2 NTT of cuDilithium (128 threads per transform)
template <int MODE>
__global__ __launch_bounds__(128, 6) void sign_persistent_kernel_t(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size) {

  const unsigned input_id = blockIdx.x;
  if (input_id >= batch_size)
    return;

  uint16_t nonce = 0;
  const size_t sign_idx_offset = input_id * sign_mem_pool_pitch;
  const size_t temp_idx_offset =
      sign_idx_offset; // Since we write directly to the pool
#ifdef FUSED_STAGED
  // stage region: [0] next attempt, [1] done flag, [2..] SHAKE256(c~) block
  uint32_t *g_state = reinterpret_cast<uint32_t *>(
      reinterpret_cast<uint8_t *>(g_cp + DILITHIUM_N) +
      SIGN_POOL_YPRE_DATA_BYTES + sign_idx_offset);
  if (g_state[1])
    return; // signed in an earlier round
  nonce = g_state[0];
#endif

  const unsigned warp_id = threadIdx.x / 32;
  const unsigned lane_id = threadIdx.x % 32;

  __shared__ int32_t s_zetas[DILITHIUM_N];
  __shared__ int32_t s_tmp[DILITHIUM_N + 128];

  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();

  int32_t cp_reg0, cp_reg1;

  // shake<> reads its input in 8-byte words
  alignas(8) __shared__ uint8_t s_in_buf[4][CRHBYTES + 2 + 6];
  alignas(8) __shared__ uint8_t s_buf_y[4][POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];
  alignas(8) __shared__ uint8_t s_buf_cp[SHAKE256_RATE];
  __shared__ int32_t s_cp_poly[DILITHIUM_N];

  __shared__ int s_rej_flag;
  __shared__ size_t s_hint_count[4];
  __shared__ size_t s_hint_count_sum;
  __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];
  PROF_DECL

  while (true) {
    // Phase 1: Compute Y
    // (the respond pass reuses y and w from the commit pass)
    for (unsigned l = warp_id; MODE != SIGN_MODE_RESPOND && l < DILITHIUM_L;
         l += 4) {
      uint8_t *s_in_l = s_in_buf[warp_id];
      uint8_t *s_buf_l = s_buf_y[warp_id];

#ifdef FUSED_PREGEN_Y
      if (nonce < FUSED_PREGEN_ATTEMPTS) {
        // packed y from expand_mask_pregen_kernel (stored after g_cp)
        const uint64_t *src = reinterpret_cast<const uint64_t *>(
            reinterpret_cast<const uint8_t *>(g_cp + DILITHIUM_N) +
            sign_idx_offset +
            (size_t)(DILITHIUM_L * nonce + l) * POLYZ_PACKEDBYTES);
        for (unsigned i = lane_id; i < POLYZ_PACKEDBYTES / 8; i += 32)
          reinterpret_cast<uint64_t *>(s_buf_l)[i] = src[i];
        __syncwarp();
      } else
#endif
      {
        s_in_l[lane_id] = g_rhoprime[sign_idx_offset + lane_id];
        s_in_l[lane_id + 32] = g_rhoprime[sign_idx_offset + lane_id + 32];
        if (lane_id == 0) {
          uint16_t nonce_l = DILITHIUM_L * nonce + l;
          s_in_l[CRHBYTES + 0] = nonce_l & 0xFF;
          s_in_l[CRHBYTES + 1] = nonce_l >> 8;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_l, POLY_UNIFORM_GAMMA1_NBLOCKS,
                                   s_in_l, CRHBYTES + 2);
        __syncwarp();
      }
      PROF_SUB(PP_SHAKE_Y)

      int32_t *g_y_l =
          g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
      for (size_t i = 0; i < 8; i++) {
        uint32_t t =
            (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 0]) |
            (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 1] << 8) |
            (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 2] << 16);
        t >>= (lane_id & 3) * 2;
        t &= 0x3FFFF;
        g_y_l[32 * i + lane_id] = GAMMA1 - (int32_t)t;
      }
#elif GAMMA1 == (1 << 19)
      for (size_t i = 0; i < 8; i++) {
        uint32_t t =
            (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 0]) |
            (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 1] << 8) |
            (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 2] << 16);
        t >>= (lane_id & 1) * 4;
        t &= 0xFFFFF;
        g_y_l[32 * i + lane_id] = GAMMA1 - (int32_t)t;
      }
#endif
    }
    PROF_SUB(PP_NTT_Y)
    __syncthreads();
    PROF_PHASE(PP_P1)

    // Phase 2: Compute W
    if (MODE != SIGN_MODE_RESPOND) {
      int32_t acc[DILITHIUM_K][2];
      for (int k = 0; k < DILITHIUM_K; k++) {
        acc[k][0] = 0;
        acc[k][1] = 0;
      }

      for (size_t l = 0; l < DILITHIUM_L; ++l) {
        const int32_t *g_y_l =
            g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t r0 = g_y_l[threadIdx.x], r1 = g_y_l[128 + threadIdx.x];
        ntt_radix2_inner(r0, r1, s_tmp, s_zetas);
        PROF_SUB(PP_NTT_Y)
        for (size_t k = 0; k < DILITHIUM_K; ++k) {
          const int32_t *m = KEY_OFFSET(g_mat, sign_idx_offset) +
                             k * DILITHIUM_L * DILITHIUM_N + l * DILITHIUM_N;
          acc[k][0] += gpu_montgomery_multiply(m[threadIdx.x], r0);
          acc[k][1] += gpu_montgomery_multiply(m[128 + threadIdx.x], r1);
        }
        PROF_SUB(PP_AY)
        __syncthreads();
      }

      for (size_t k = 0; k < DILITHIUM_K; ++k) {
        acc[k][0] = reduce32(acc[k][0]);
        acc[k][1] = reduce32(acc[k][1]);
        intt_radix2_inner(acc[k][0], acc[k][1], s_tmp, s_zetas);
        PROF_SUB(PP_INTT_W)
        acc[k][0] = caddq(acc[k][0]);
        acc[k][1] = caddq(acc[k][1]);

        int32_t *g_w0_k =
            g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        acc[k][0] = decompose(&g_w0_k[threadIdx.x], acc[k][0]);
        acc[k][1] = decompose(&g_w0_k[128 + threadIdx.x], acc[k][1]);

        int32_t *g_w1_k =
            g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        g_w1_k[threadIdx.x] = acc[k][0];
        g_w1_k[128 + threadIdx.x] = acc[k][1];

        __syncthreads();
        s_tmp[threadIdx.x + (threadIdx.x >> 5)] = acc[k][0];
        s_tmp[128 + 4 + threadIdx.x + (threadIdx.x >> 5)] = acc[k][1];
        __syncthreads();

        uint8_t *g_w1_packed_k =
            g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
#if GAMMA2 == (DILITHIUM_Q - 1) / 88
        if (threadIdx.x < 64) {
          size_t off = 4 * threadIdx.x + (threadIdx.x >> 3);
          g_w1_packed_k[3 * threadIdx.x + 0] =
              (s_tmp[off + 0]) | (s_tmp[off + 1] << 6);
          g_w1_packed_k[3 * threadIdx.x + 1] =
              (s_tmp[off + 1] >> 2) | (s_tmp[off + 2] << 4);
          g_w1_packed_k[3 * threadIdx.x + 2] =
              (s_tmp[off + 2] >> 4) | (s_tmp[off + 3] << 2);
        }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
        {
          size_t off = 2 * threadIdx.x + (threadIdx.x >> 4);
          g_w1_packed_k[threadIdx.x] = (s_tmp[off + 0]) | (s_tmp[off + 1] << 4);
        }
#endif
        __syncthreads();
        PROF_SUB(PP_DECOMP)
      }
    }
    PROF_PHASE(PP_P2)
    if (MODE == SIGN_MODE_COMMIT)
      return; // w1_packed is in the pool for staged_challenge_kernel

    // Phase 3: Compute cp NTT
    if (warp_id == 0) {
      if (MODE == SIGN_MODE_RESPOND) {
#ifdef FUSED_STAGED
        // c~ is already in g_seed; copy the SHAKE256(c~) block
        if (lane_id < SHAKE256_RATE / 8)
          reinterpret_cast<uint64_t *>(s_buf_cp)[lane_id] =
              reinterpret_cast<const uint64_t *>(g_state + 2)[lane_id];
        __syncwarp();
#endif
      } else {
        g_mu[temp_idx_offset + lane_id] = g_ori_mu[sign_idx_offset + lane_id];
        g_mu[temp_idx_offset + 32 + lane_id] =
            g_ori_mu[sign_idx_offset + 32 + lane_id];
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_cp, 1, g_mu + temp_idx_offset,
                                   CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
        __syncwarp();
        PROF_SUB(PP_H_MU_W1)
        for (unsigned i = lane_id; i < CTILDEBYTES; i += 32)
          g_seed[temp_idx_offset + i] = s_buf_cp[i];
        shake<SHAKE256_RATE, 0x1f>(s_buf_cp, 1, s_buf_cp, CTILDEBYTES);
        __syncwarp();
        PROF_SUB(PP_SHAKE_C)
      }
    }
    __syncthreads();

    s_cp_poly[threadIdx.x] = 0;
    s_cp_poly[128 + threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
      uint64_t signs = ((const uint64_t *)s_buf_cp)[0];
      int pos = 8;
      for (int i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
        int b;
        do {
          b = s_buf_cp[pos++];
        } while (b > i);
        s_cp_poly[i] = s_cp_poly[b];
        s_cp_poly[b] = 1 - 2 * (int32_t)(signs & 1);
        signs >>= 1;
      }
    }
    __syncthreads();
    PROF_SUB(PP_SAMPLE_BALL)

    cp_reg0 = s_cp_poly[threadIdx.x];
    cp_reg1 = s_cp_poly[128 + threadIdx.x];
    ntt_radix2_inner(cp_reg0, cp_reg1, s_tmp, s_zetas);
    PROF_SUB(PP_NTT_C)

    if (threadIdx.x == 0)
      s_rej_flag = 0;
    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x)
      s_hint[i] = 0;
    __syncthreads();
    PROF_PHASE(PP_P3)

    // Phase 4: Rejection Sampling
    // no early exit: all checks and the hint are done, flags combined below
    for (size_t l = 0; l < DILITHIUM_L; ++l) {
      const int32_t *g_s1_l =
          KEY_OFFSET(g_s1, sign_idx_offset) + l * DILITHIUM_N;
      const int32_t *g_y_l =
          g_y + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
      int32_t *g_z_l =
          g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;

      int32_t cps1_0 = gpu_montgomery_multiply(cp_reg0, g_s1_l[threadIdx.x]);
      int32_t cps1_1 =
          gpu_montgomery_multiply(cp_reg1, g_s1_l[128 + threadIdx.x]);
      intt_radix2_inner(cps1_0, cps1_1, s_tmp, s_zetas);
      cps1_0 += g_y_l[threadIdx.x];
      cps1_0 = reduce32(cps1_0);
      cps1_1 += g_y_l[128 + threadIdx.x];
      cps1_1 = reduce32(cps1_1);

      int rf = __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_0));
      rf += __any_sync(0xFFFFFFFF, chknorm<GAMMA1 - BETA>(cps1_1));
      if (lane_id == 0 && rf)
        s_rej_flag = 1;
      g_z_l[threadIdx.x] = cps1_0;
      g_z_l[128 + threadIdx.x] = cps1_1;
      __syncthreads(); // s_tmp is reused by the next transform
    }
    PROF_PHASE(PP_P4A)

    bool rejected = false;
    unsigned int pos = 0;
    for (size_t k = 0; k < DILITHIUM_K; ++k) {
      const int32_t *g_s2_k =
          KEY_OFFSET(g_s2, sign_idx_offset) + k * DILITHIUM_N;
      const int32_t *g_t0_k =
          KEY_OFFSET(g_t0, sign_idx_offset) + k * DILITHIUM_N;
      const int32_t *g_w0_k =
          g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
      const int32_t *g_w1_k =
          g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

      int32_t cps2_0 = gpu_montgomery_multiply(cp_reg0, g_s2_k[threadIdx.x]);
      int32_t cps2_1 =
          gpu_montgomery_multiply(cp_reg1, g_s2_k[128 + threadIdx.x]);
      intt_radix2_inner(cps2_0, cps2_1, s_tmp, s_zetas);
      cps2_0 = g_w0_k[threadIdx.x] - cps2_0;
      cps2_0 = reduce32(cps2_0);
      cps2_1 = g_w0_k[128 + threadIdx.x] - cps2_1;
      cps2_1 = reduce32(cps2_1);

      int rf = __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_0));
      rf += __any_sync(0xFFFFFFFF, chknorm<GAMMA2 - BETA>(cps2_1));
      if (lane_id == 0 && rf)
        s_rej_flag = 1;
      __syncthreads(); // s_tmp is reused by the next transform

      int32_t cpt0_0 = gpu_montgomery_multiply(cp_reg0, g_t0_k[threadIdx.x]);
      int32_t cpt0_1 =
          gpu_montgomery_multiply(cp_reg1, g_t0_k[128 + threadIdx.x]);
      intt_radix2_inner(cpt0_0, cpt0_1, s_tmp, s_zetas);
      cpt0_0 = reduce32(cpt0_0);
      cpt0_1 = reduce32(cpt0_1);

      rf = __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_0));
      rf += __any_sync(0xFFFFFFFF, chknorm<GAMMA2>(cpt0_1));
      if (lane_id == 0 && rf)
        s_rej_flag = 1;
      __syncthreads();
      PROF_SUB(PP_P4B)

      cpt0_0 += cps2_0;
      cpt0_1 += cps2_1;

      // hints of both halves; an overflow rejects the attempt but the
      // remaining polynomials are still processed
      unsigned int hb0 = make_hint(cpt0_0, g_w1_k[threadIdx.x]);
      unsigned int hm0 = __ballot_sync(0xFFFFFFFF, hb0);
      if (lane_id == 0)
        s_hint_count[warp_id] = __popc(hm0);
      __syncthreads();
      PREFIX_SUM_HINT_COUNT
      __syncthreads();
      bool overflow = (pos + s_hint_count_sum > OMEGA);
      rejected |= overflow;
      {
        unsigned inc = __popc(hm0 << (31 - lane_id)) + s_hint_count[warp_id];
        if (hb0 && !overflow)
          s_hint[pos + inc - 1] = threadIdx.x;
      }
      __syncthreads();
      if (!overflow)
        pos += s_hint_count_sum;

      unsigned int hb1 = make_hint(cpt0_1, g_w1_k[128 + threadIdx.x]);
      unsigned int hm1 = __ballot_sync(0xFFFFFFFF, hb1);
      if (lane_id == 0)
        s_hint_count[warp_id] = __popc(hm1);
      __syncthreads();
      PREFIX_SUM_HINT_COUNT
      __syncthreads();
      overflow = (pos + s_hint_count_sum > OMEGA);
      rejected |= overflow;
      {
        unsigned inc = __popc(hm1 << (31 - lane_id)) + s_hint_count[warp_id];
        if (hb1 && !overflow)
          s_hint[pos + inc - 1] = 128 + threadIdx.x;
      }
      __syncthreads();
      if (!overflow)
        pos += s_hint_count_sum;

      if (threadIdx.x == 0)
        s_hint[OMEGA + k] = pos;
      __syncthreads();
      PROF_SUB(PP_P4C)
    }
    PROF_RESYNC // P4b and P4c were timed per polynomial
    rejected = rejected || s_rej_flag;

    if (rejected) {
      if (MODE == SIGN_MODE_RESPOND) {
#ifdef FUSED_STAGED
        if (threadIdx.x == 0)
          g_state[0] = nonce + 1;
#endif
        return;
      }
      nonce++;
      continue;
    }

    // Phase 5: Pack Signature
    for (size_t l = 0; l < DILITHIUM_L; ++l) {
      uint8_t *g_z_packed_l =
          g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
      const int32_t *g_z_l =
          g_z + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
      if (threadIdx.x < 64) {
        size_t i = threadIdx.x;
        uint32_t t[4];
        t[0] = GAMMA1 - g_z_l[4 * i + 0];
        t[1] = GAMMA1 - g_z_l[4 * i + 1];
        t[2] = GAMMA1 - g_z_l[4 * i + 2];
        t[3] = GAMMA1 - g_z_l[4 * i + 3];
        g_z_packed_l[9 * i + 0] = t[0];
        g_z_packed_l[9 * i + 1] = t[0] >> 8;
        g_z_packed_l[9 * i + 2] = t[0] >> 16;
        g_z_packed_l[9 * i + 2] |= t[1] << 2;
        g_z_packed_l[9 * i + 3] = t[1] >> 6;
        g_z_packed_l[9 * i + 4] = t[1] >> 14;
        g_z_packed_l[9 * i + 4] |= t[2] << 4;
        g_z_packed_l[9 * i + 5] = t[2] >> 4;
        g_z_packed_l[9 * i + 6] = t[2] >> 12;
        g_z_packed_l[9 * i + 6] |= t[3] << 6;
        g_z_packed_l[9 * i + 7] = t[3] >> 2;
        g_z_packed_l[9 * i + 8] = t[3] >> 10;
      }
#elif GAMMA1 == (1 << 19)
      {
        size_t i = threadIdx.x;
        uint32_t t[2];
        t[0] = GAMMA1 - g_z_l[2 * i + 0];
        t[1] = GAMMA1 - g_z_l[2 * i + 1];
        g_z_packed_l[5 * i + 0] = t[0];
        g_z_packed_l[5 * i + 1] = t[0] >> 8;
        g_z_packed_l[5 * i + 2] = t[0] >> 16;
        g_z_packed_l[5 * i + 2] |= t[1] << 4;
        g_z_packed_l[5 * i + 3] = t[1] >> 4;
        g_z_packed_l[5 * i + 4] = t[1] >> 12;
      }
#endif
    }

    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x)
      g_hint[temp_idx_offset + i] = s_hint[i];

    PROF_PHASE(PP_P5)
    PROF_FLUSH(nonce + 1)
#ifdef FUSED_STAGED
    if (threadIdx.x == 0)
      g_state[1] = 1;
#endif
    return;
  }
}

void sign_persistent_launch(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size, cudaStream_t stream) {
#ifdef FUSED_PREGEN_Y
  uint8_t *g_ypre = reinterpret_cast<uint8_t *>(g_cp + DILITHIUM_N);
  {
    size_t n_threads = batch_size * FUSED_PREGEN_ATTEMPTS * DILITHIUM_L;
    expand_mask_pregen_kernel<<<(n_threads + 127) / 128, 128, 0, stream>>>(
        g_ypre, g_rhoprime, sign_mem_pool_pitch, batch_size);
  }
#endif
#ifdef FUSED_STAGED
  cudaMemset2DAsync(g_ypre + SIGN_POOL_YPRE_DATA_BYTES, sign_mem_pool_pitch, 0,
                    2 * sizeof(uint32_t), batch_size, stream);
  for (int r = 0; r < FUSED_STAGED_ROUNDS; r++) {
    sign_persistent_kernel_t<SIGN_MODE_COMMIT>
        <<<batch_size, 128, 0, stream>>>(SIGN_FUSED_ARGS);
    staged_challenge_kernel<<<(batch_size + 127) / 128, 128, 0, stream>>>(
        g_seed, g_mu, g_ypre, sign_mem_pool_pitch, batch_size);
    sign_persistent_kernel_t<SIGN_MODE_RESPOND>
        <<<batch_size, 128, 0, stream>>>(SIGN_FUSED_ARGS);
  }
#endif
  sign_persistent_kernel_t<SIGN_MODE_FULL>
      <<<batch_size, 128, 0, stream>>>(SIGN_FUSED_ARGS);
}
#endif

#ifdef USE_PERSISTENT_WARPNTT



// NTT used by the fused kernel. With FUSED_WARPX the NTT-domain registers
// are in layout B (see ntt_warpx), so A, s1, s2, t0 are read via NTTD_IDX.
#ifdef FUSED_WARPX
#define WARP_NTT ntt_warpx
#define WARP_INTT invntt_warpx
#define NTTD_IDX(i, lane) (64 * ((i) >> 1) + 2 * (lane) + ((i) & 1))
#else
#define WARP_NTT ntt_inner_shuffle_notranspose
#define WARP_INTT invntt_inner_shuffle_notranspose
#define NTTD_IDX(i, lane) ((i) * 32 + (lane))
#endif

template <int MODE>
__global__ LAUNCH_BOUNDS_SIGN void sign_persistent_warpntt_kernel_t(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size) {

  const unsigned input_id = blockIdx.x;
  if (input_id >= batch_size)
    return;

  uint16_t nonce = 0;
  const size_t sign_idx_offset = input_id * sign_mem_pool_pitch;
  const size_t temp_idx_offset = sign_idx_offset;
#ifdef FUSED_STAGED
  // stage region: [0] next attempt, [1] done flag, [2..] SHAKE256(c~) block
  uint32_t *g_state = reinterpret_cast<uint32_t *>(
      reinterpret_cast<uint8_t *>(g_cp + DILITHIUM_N) +
      SIGN_POOL_YPRE_DATA_BYTES + sign_idx_offset);
  if (g_state[1])
    return; // signed in an earlier round
  nonce = g_state[0];
#endif

  const unsigned warp_id = threadIdx.x / 32;
  const unsigned lane_id = threadIdx.x % 32;

  alignas(8) __shared__ uint8_t s_in_buf[FUSED_WARPS][CRHBYTES + 2 + 6];
  alignas(8) __shared__ uint8_t
      s_buf_y[FUSED_WARPS][POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE];
  alignas(8) __shared__ uint8_t s_buf_cp[SHAKE256_RATE];

  // Shared memory for Y NTT, CP NTT, and CPT0_CPS2
  __shared__ int32_t s_y_ntt[DILITHIUM_L][DILITHIUM_N];
  __shared__ int32_t s_cp_ntt[DILITHIUM_N];
  __shared__ int32_t s_cpt0_cps2[DILITHIUM_K][DILITHIUM_N];
  __shared__ int32_t s_w1_tmp[DILITHIUM_N];

  __shared__ int s_rej_flag;
  __shared__ size_t s_hint_count[FUSED_WARPS];
  __shared__ size_t s_hint_count_sum;
  __shared__ uint8_t s_hint[POLYVECH_PACKEDBYTES];
  __shared__ unsigned s_y_next; // next y polynomial to generate
  __shared__ int32_t s_zetas[DILITHIUM_N]; // twiddles of the warp NTT
  PROF_DECL

  // y = ExpandMask(rho'', L*attempt) -> g_ybuf, NTT(y) -> s_y_ntt.
  // Warps grab polynomials via s_y_next (reset it + barrier before calling).
  auto expand_mask = [&](uint16_t attempt, int32_t *g_ybuf) {
    while (true) {
      unsigned l = 0;
      if (lane_id == 0)
        l = atomicAdd(&s_y_next, 1u);
      l = __shfl_sync(0xFFFFFFFF, l, 0);
      if (l >= DILITHIUM_L)
        break;
      {
        uint8_t *s_in_l = s_in_buf[warp_id];
        uint8_t *s_buf_l = s_buf_y[warp_id];

#ifdef FUSED_PREGEN_Y
        if (attempt < FUSED_PREGEN_ATTEMPTS) {
          // packed y from expand_mask_pregen_kernel (stored after g_cp)
          const uint64_t *src = reinterpret_cast<const uint64_t *>(
              reinterpret_cast<const uint8_t *>(g_cp + DILITHIUM_N) +
              sign_idx_offset +
              (size_t)(DILITHIUM_L * attempt + l) * POLYZ_PACKEDBYTES);
          for (unsigned i = lane_id; i < POLYZ_PACKEDBYTES / 8; i += 32)
            reinterpret_cast<uint64_t *>(s_buf_l)[i] = src[i];
          __syncwarp();
        } else
#endif
        {
          s_in_l[lane_id] = g_rhoprime[sign_idx_offset + lane_id];
          s_in_l[lane_id + 32] = g_rhoprime[sign_idx_offset + lane_id + 32];
          if (lane_id == 0) {
            uint16_t nonce_l = DILITHIUM_L * attempt + l;
            s_in_l[CRHBYTES + 0] = nonce_l & 0xFF;
            s_in_l[CRHBYTES + 1] = nonce_l >> 8;
          }
          __syncwarp();
          shake<SHAKE256_RATE, 0x1f>(s_buf_l, POLY_UNIFORM_GAMMA1_NBLOCKS,
                                     s_in_l, CRHBYTES + 2);
          __syncwarp();
        }
        PROF_SUB(PP_SHAKE_Y)

        int32_t *g_y_l =
            g_ybuf + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t y_regs[8];
#if GAMMA1 == (1 << 17)
        for (size_t i = 0; i < 8; i++) {
          uint32_t t =
              (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 0]) |
              (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 1] << 8) |
              (s_buf_l[i * 72 + (lane_id / 4) * 9 + (lane_id & 3) * 2 + 2] << 16);
          t >>= (lane_id & 3) * 2;
          t &= 0x3FFFF;
          int32_t y_val = GAMMA1 - (int32_t)t;
          g_y_l[32 * i + lane_id] = y_val;
          y_regs[i] = y_val;
        }
#elif GAMMA1 == (1 << 19)
        for (size_t i = 0; i < 8; i++) {
          uint32_t t =
              (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 0]) |
              (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 1] << 8) |
              (s_buf_l[i * 80 + (lane_id / 2) * 5 + (lane_id & 1) * 2 + 2] << 16);
          t >>= (lane_id & 1) * 4;
          t &= 0xFFFFF;
          int32_t y_val = GAMMA1 - (int32_t)t;
          g_y_l[32 * i + lane_id] = y_val;
          y_regs[i] = y_val;
        }
#endif
        WARP_NTT(y_regs, s_zetas);
        for (size_t i = 0; i < 8; i++) {
          s_y_ntt[l][i * 32 + lane_id] = y_regs[i];
        }
        PROF_SUB(PP_NTT_Y)
      }
    }
  };

#ifdef FUSED_OVERLAP_Y
  // y for attempt n+1 is generated during attempt n's challenge phase, so
  // attempts alternate between the g_y and g_z buffers (z is formed in place).
  constexpr bool overlap_y = true;
#else
  constexpr bool overlap_y = false;
#endif
  bool y_ready = (MODE == SIGN_MODE_RESPOND); // y was made by the commit pass
  for (unsigned i = threadIdx.x; i < DILITHIUM_N; i += blockDim.x)
    s_zetas[i] = c_zetas[i];
  if (threadIdx.x == 0)
    s_y_next = 0;
  __syncthreads();

  while (true) {
    int32_t *g_ycur = (overlap_y && (nonce & 1)) ? g_z : g_y;
    int32_t *g_ynext = (nonce & 1) ? g_y : g_z;
    if (!y_ready) {
      expand_mask(nonce, g_ycur);
      __syncthreads();
      PROF_PHASE(PP_P1)
      if (threadIdx.x == 0)
        s_y_next = 0; // for the overlapped call in phase 3
    }

    if (MODE != SIGN_MODE_RESPOND && warp_id < DILITHIUM_SIGN_WARPS) {
      for (size_t k = warp_id; k < DILITHIUM_K; k += DILITHIUM_SIGN_WARPS) {
        int32_t w_regs[8] = {0};
        for (size_t l = 0; l < DILITHIUM_L; ++l) {
          const int32_t *m = KEY_OFFSET(g_mat, sign_idx_offset) +
                             k * DILITHIUM_L * DILITHIUM_N + l * DILITHIUM_N;
          for (size_t i = 0; i < 8; i++) {
            w_regs[i] += gpu_montgomery_multiply(m[NTTD_IDX(i, lane_id)],
                                                 s_y_ntt[l][i * 32 + lane_id]);
          }
        }
        for (size_t i = 0; i < 8; i++) {
          w_regs[i] = reduce32(w_regs[i]);
        }
        PROF_SUB(PP_AY)
        WARP_INTT(w_regs, s_zetas);
        PROF_SUB(PP_INTT_W)

        int32_t *g_w0_k =
            g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;
        int32_t *g_w1_k =
            g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

        for (size_t i = 0; i < 8; i++) {
          w_regs[i] = caddq(w_regs[i]);
          int32_t w1_val = decompose(&g_w0_k[i * 32 + lane_id], w_regs[i]);
          g_w1_k[i * 32 + lane_id] = w1_val;
        }
        __syncwarp();

        uint8_t *g_w1_packed_k =
            g_w1_packed + temp_idx_offset + k * POLYW1_PACKEDBYTES;
#if GAMMA2 == (DILITHIUM_Q - 1) / 88
        for (int i = 0; i < 2; i++) {
          size_t t_idx = i * 32 + lane_id;
          size_t off = 4 * t_idx;
          g_w1_packed_k[3 * t_idx + 0] =
              (g_w1_k[off + 0]) | (g_w1_k[off + 1] << 6);
          g_w1_packed_k[3 * t_idx + 1] =
              (g_w1_k[off + 1] >> 2) | (g_w1_k[off + 2] << 4);
          g_w1_packed_k[3 * t_idx + 2] =
              (g_w1_k[off + 2] >> 4) | (g_w1_k[off + 3] << 2);
        }
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
        for (int i = 0; i < 4; i++) {
          size_t t_idx = i * 32 + lane_id;
          size_t off = 2 * t_idx;
          g_w1_packed_k[t_idx] = (g_w1_k[off + 0]) | (g_w1_k[off + 1] << 4);
        }
#endif
        __syncwarp();
        PROF_SUB(PP_DECOMP)
      }
    }
    __syncthreads();
    PROF_PHASE(PP_P2)
    if (MODE == SIGN_MODE_COMMIT)
      return; // w1_packed is in the pool for staged_challenge_kernel

    // Phase 3: Compute cp NTT
    if (warp_id == 0) {
      if (MODE == SIGN_MODE_RESPOND) {
#ifdef FUSED_STAGED
        // c~ is already in g_seed; copy the SHAKE256(c~) block
        if (lane_id < SHAKE256_RATE / 8)
          reinterpret_cast<uint64_t *>(s_buf_cp)[lane_id] =
              reinterpret_cast<const uint64_t *>(g_state + 2)[lane_id];
        __syncwarp();
#endif
      } else {
        g_mu[temp_idx_offset + lane_id] = g_ori_mu[sign_idx_offset + lane_id];
        g_mu[temp_idx_offset + 32 + lane_id] =
            g_ori_mu[sign_idx_offset + 32 + lane_id];
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_cp, 1, g_mu + temp_idx_offset,
                                   CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES);
        __syncwarp();
        PROF_SUB(PP_H_MU_W1)
        for (unsigned i = lane_id; i < CTILDEBYTES; i += 32)
          g_seed[temp_idx_offset + i] = s_buf_cp[i];
        shake<SHAKE256_RATE, 0x1f>(s_buf_cp, 1, s_buf_cp, CTILDEBYTES);
        __syncwarp();
        PROF_SUB(PP_SHAKE_C)
      }

      int32_t cp_poly[8] = {0};
      if (lane_id == 0) {
        uint64_t signs = ((const uint64_t *)s_buf_cp)[0];
        int pos = 8;
        for (int i = 0; i < DILITHIUM_N; i++)
          s_w1_tmp[i] = 0;
        for (int i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
          int b;
          do {
            b = s_buf_cp[pos++];
          } while (b > i);
          s_w1_tmp[i] = s_w1_tmp[b];
          s_w1_tmp[b] = 1 - 2 * (int32_t)(signs & 1);
          signs >>= 1;
        }
        PROF_SUB(PP_SAMPLE_BALL)
      }
      __syncwarp();
      for (int i = 0; i < 8; i++) {
        cp_poly[i] = s_w1_tmp[i * 32 + lane_id];
      }
      WARP_NTT(cp_poly, s_zetas);
      for (int i = 0; i < 8; i++) {
        s_cp_ntt[i * 32 + lane_id] = cp_poly[i];
      }
      PROF_SUB(PP_NTT_C)
    }
    if (overlap_y) {
      // Speculative: wasted only if this attempt is accepted.
      expand_mask(nonce + 1, g_ynext);
      y_ready = true;
    }

    if (threadIdx.x == 0)
      s_rej_flag = 0;
    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x)
      s_hint[i] = 0;
    __syncthreads();
    PROF_PHASE(PP_P3)

    // Phase 4: Rejection Sampling
    bool rejected = false;
    if (warp_id < DILITHIUM_SIGN_WARPS) {
      for (size_t l = warp_id; l < DILITHIUM_L; l += DILITHIUM_SIGN_WARPS) {
        const int32_t *g_s1_l =
            KEY_OFFSET(g_s1, sign_idx_offset) + l * DILITHIUM_N;
        // z overwrites y in place (each lane reads then writes its own words)
        int32_t *g_y_l =
            g_ycur + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
        int32_t *g_z_l = g_y_l;

        int32_t cps1_regs[8];
        for (int i = 0; i < 8; i++) {
          cps1_regs[i] = gpu_montgomery_multiply(s_cp_ntt[i * 32 + lane_id],
                                                 g_s1_l[NTTD_IDX(i, lane_id)]);
        }
        WARP_INTT(cps1_regs, s_zetas);

        int rf = 0;
        for (int i = 0; i < 8; i++) {
          cps1_regs[i] += g_y_l[i * 32 + lane_id];
          cps1_regs[i] = reduce32(cps1_regs[i]);
          g_z_l[i * 32 + lane_id] = cps1_regs[i];
          if (chknorm<GAMMA1 - BETA>(cps1_regs[i]))
            rf = 1;
        }

        rf = __any_sync(0xFFFFFFFF, rf);
        if (lane_id == 0 && rf)
          s_rej_flag = 1;
      }
    }
    __syncthreads();
    PROF_PHASE(PP_P4A)
    if (threadIdx.x == 0)
      s_y_next = 0; // next expand_mask call is several barriers away

    if (warp_id < DILITHIUM_SIGN_WARPS) {
      for (size_t k = warp_id; k < DILITHIUM_K; k += DILITHIUM_SIGN_WARPS) {
        const int32_t *g_s2_k =
            KEY_OFFSET(g_s2, sign_idx_offset) + k * DILITHIUM_N;
        const int32_t *g_t0_k =
            KEY_OFFSET(g_t0, sign_idx_offset) + k * DILITHIUM_N;
        const int32_t *g_w0_k =
            g_w0 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

        int32_t cps2_regs[8];
        for (int i = 0; i < 8; i++) {
          cps2_regs[i] = gpu_montgomery_multiply(s_cp_ntt[i * 32 + lane_id],
                                                 g_s2_k[NTTD_IDX(i, lane_id)]);
        }
        WARP_INTT(cps2_regs, s_zetas);

        int rf = 0;
        for (int i = 0; i < 8; i++) {
          cps2_regs[i] = g_w0_k[i * 32 + lane_id] - cps2_regs[i];
          cps2_regs[i] = reduce32(cps2_regs[i]);
          if (chknorm<GAMMA2 - BETA>(cps2_regs[i]))
            rf = 1;
        }

        rf = __any_sync(0xFFFFFFFF, rf);
        if (lane_id == 0 && rf)
          s_rej_flag = 1;

        int32_t cpt0_regs[8];
        for (int i = 0; i < 8; i++) {
          cpt0_regs[i] = gpu_montgomery_multiply(s_cp_ntt[i * 32 + lane_id],
                                                 g_t0_k[NTTD_IDX(i, lane_id)]);
        }
        WARP_INTT(cpt0_regs, s_zetas);

        int rf_t0 = 0;
        for (int i = 0; i < 8; i++) {
          cpt0_regs[i] = reduce32(cpt0_regs[i]);
          if (chknorm<GAMMA2>(cpt0_regs[i]))
            rf_t0 = 1;
          s_cpt0_cps2[k][i * 32 + lane_id] = cpt0_regs[i] + cps2_regs[i];
        }
        rf_t0 = __any_sync(0xFFFFFFFF, rf_t0);
        if (lane_id == 0 && rf_t0)
          s_rej_flag = 1;
      }
    }

    __syncthreads();
    PROF_PHASE(PP_P4B)

    unsigned int pos = 0;
    for (size_t k = 0; k < DILITHIUM_K; ++k) {
      const int32_t *g_w1_k =
          g_w1 + temp_idx_offset / sizeof(int32_t) + k * DILITHIUM_N;

      // Hints for this polynomial, FUSED_BLOCK_SIZE coefficients at a time.
      for (unsigned base = 0; base < DILITHIUM_N; base += FUSED_BLOCK_SIZE) {
        const unsigned idx = base + threadIdx.x;
        unsigned int hb = 0;
        if (idx < DILITHIUM_N) {
          hb = make_hint(s_cpt0_cps2[k][idx], g_w1_k[idx]);
        }
        unsigned int hm = __ballot_sync(0xFFFFFFFF, hb);
        if (lane_id == 0)
          s_hint_count[warp_id] = __popc(hm);
        __syncthreads();
        if (threadIdx.x == 0) {
          unsigned int count = 0;
          for (unsigned w = 0; w < FUSED_WARPS; ++w) {
            unsigned int val = s_hint_count[w];
            s_hint_count[w] = count;
            count += val;
          }
          s_hint_count_sum = count;
        }
        __syncthreads();

        bool overflow = (pos + s_hint_count_sum > OMEGA);
        if (overflow) {
          rejected = true;
        }
        {
          unsigned inc = __popc(hm << (31 - lane_id)) + s_hint_count[warp_id];
          if (hb && !overflow)
            s_hint[pos + inc - 1] = idx;
        }
        __syncthreads();
        if (!overflow) {
          pos += s_hint_count_sum;
        }
      }

      if (threadIdx.x == 0)
        s_hint[OMEGA + k] = pos;
      __syncthreads();
    }

    PROF_PHASE(PP_P4C)
    if (s_rej_flag || rejected) {
      if (MODE == SIGN_MODE_RESPOND) {
#ifdef FUSED_STAGED
        if (threadIdx.x == 0)
          g_state[0] = nonce + 1;
#endif
        return;
      }
      nonce++;
      __syncthreads();
      continue;
    }

    // Phase 5: Pack Signature
    for (size_t l = 0; l < DILITHIUM_L; ++l) {
      uint8_t *g_z_packed_l =
          g_z_packed + temp_idx_offset + l * POLYZ_PACKEDBYTES;
      const int32_t *g_z_l =
          g_ycur + temp_idx_offset / sizeof(int32_t) + l * DILITHIUM_N;
#if GAMMA1 == (1 << 17)
      for (size_t i = threadIdx.x; i < DILITHIUM_N / 4; i += FUSED_BLOCK_SIZE) {
        uint32_t t[4];
        t[0] = GAMMA1 - g_z_l[4 * i + 0];
        t[1] = GAMMA1 - g_z_l[4 * i + 1];
        t[2] = GAMMA1 - g_z_l[4 * i + 2];
        t[3] = GAMMA1 - g_z_l[4 * i + 3];
        g_z_packed_l[9 * i + 0] = t[0];
        g_z_packed_l[9 * i + 1] = t[0] >> 8;
        g_z_packed_l[9 * i + 2] = t[0] >> 16;
        g_z_packed_l[9 * i + 2] |= t[1] << 2;
        g_z_packed_l[9 * i + 3] = t[1] >> 6;
        g_z_packed_l[9 * i + 4] = t[1] >> 14;
        g_z_packed_l[9 * i + 4] |= t[2] << 4;
        g_z_packed_l[9 * i + 5] = t[2] >> 4;
        g_z_packed_l[9 * i + 6] = t[2] >> 12;
        g_z_packed_l[9 * i + 6] |= t[3] << 6;
        g_z_packed_l[9 * i + 7] = t[3] >> 2;
        g_z_packed_l[9 * i + 8] = t[3] >> 10;
      }
#elif GAMMA1 == (1 << 19)
      for (size_t i = threadIdx.x; i < DILITHIUM_N / 2; i += FUSED_BLOCK_SIZE) {
        uint32_t t[2];
        t[0] = GAMMA1 - g_z_l[2 * i + 0];
        t[1] = GAMMA1 - g_z_l[2 * i + 1];
        g_z_packed_l[5 * i + 0] = t[0];
        g_z_packed_l[5 * i + 1] = t[0] >> 8;
        g_z_packed_l[5 * i + 2] = t[0] >> 16;
        g_z_packed_l[5 * i + 2] |= t[1] << 4;
        g_z_packed_l[5 * i + 3] = t[1] >> 4;
        g_z_packed_l[5 * i + 4] = t[1] >> 12;
      }
#endif
    }

    for (size_t i = threadIdx.x; i < POLYVECH_PACKEDBYTES; i += blockDim.x)
      g_hint[temp_idx_offset + i] = s_hint[i];

    PROF_PHASE(PP_P5)
    PROF_FLUSH(nonce + 1)
#ifdef FUSED_STAGED
    if (threadIdx.x == 0)
      g_state[1] = 1;
#endif
    return;
  }
}


void sign_persistent_warpntt_launch(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size, cudaStream_t stream) {
#ifdef FUSED_PREGEN_Y
  // pre-generated y and the stage region follow d_cp in each pool row
  uint8_t *g_ypre = reinterpret_cast<uint8_t *>(g_cp + DILITHIUM_N);
  {
    size_t n_threads = batch_size * FUSED_PREGEN_ATTEMPTS * DILITHIUM_L;
    expand_mask_pregen_kernel<<<(n_threads + 127) / 128, 128, 0, stream>>>(
        g_ypre, g_rhoprime, sign_mem_pool_pitch, batch_size);
  }
#endif
#ifdef FUSED_STAGED
  // staged rounds, then the fused kernel for the signatures still left
  cudaMemset2DAsync(g_ypre + SIGN_POOL_YPRE_DATA_BYTES, sign_mem_pool_pitch, 0,
                    2 * sizeof(uint32_t), batch_size, stream);
  for (int r = 0; r < FUSED_STAGED_ROUNDS; r++) {
    sign_persistent_warpntt_kernel_t<SIGN_MODE_COMMIT>
        <<<batch_size, FUSED_BLOCK_SIZE, 0, stream>>>(SIGN_FUSED_ARGS);
    staged_challenge_kernel<<<(batch_size + 127) / 128, 128, 0, stream>>>(
        g_seed, g_mu, g_ypre, sign_mem_pool_pitch, batch_size);
    sign_persistent_warpntt_kernel_t<SIGN_MODE_RESPOND>
        <<<batch_size, FUSED_BLOCK_SIZE, 0, stream>>>(SIGN_FUSED_ARGS);
  }
#endif
  sign_persistent_warpntt_kernel_t<SIGN_MODE_FULL>
      <<<batch_size, FUSED_BLOCK_SIZE, 0, stream>>>(SIGN_FUSED_ARGS);
}
#endif // USE_PERSISTENT_WARPNTT

