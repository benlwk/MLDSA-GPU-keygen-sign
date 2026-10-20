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

#pragma once

#include <cstddef>
#include <cstdint>

#include "params.h"
#include "util.cuh"
// Switch: define this to fuse compute_y and compute_w kernels
// #define USE_FUSED_Y_W
// Switch: define this to cache SK setups (matrix expand, NTT unpack) across
// calls. This significantly improves throughput for repeated signing with the
// same key.
// #define USE_CACHED_SK

// Switch: define ONE of these to override the default radix2_opt kernel.
// #define USE_128T_SIGN // 128T sign optimizations

// Switch: define this to compute cp and its NTT domain immediately
// #define USE_CP_NTT

#define ALIGN_TO_256_BYTES(x) ((((x) + 255) / 256) * 256)

// Byte region at the start of each sign-pool row (followed by int32 arrays):
// sig | sk | rho | tr||M' | key||rnd||mu||w1_packed | rhoprime.
// Hash inputs tr||M' and key||rnd||mu||w1 must be contiguous for shake<>.
#define SIGN_POOL_BYTE_SIZE(mlen)                                              \
  (ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                                          \
   ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES) +                                 \
   ALIGN_TO_256_BYTES(SEEDBYTES) +                                             \
   ALIGN_TO_256_BYTES(TRBYTES + MSG_PREFIX_BYTES + (mlen)) +                   \
   ALIGN_TO_256_BYTES(SEEDBYTES + RNDBYTES + CRHBYTES +                        \
                      DILITHIUM_K * POLYW1_PACKEDBYTES) +                      \
   ALIGN_TO_256_BYTES(CRHBYTES))

// Byte region at the start of each verify-pool row:
// sig | pk | tr||M' | mu||w1_packed | c_tilde.
#define VERIFY_POOL_BYTE_SIZE(mlen)                                            \
  (ALIGN_TO_256_BYTES(CRYPTO_BYTES) +                                          \
   ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +                                 \
   ALIGN_TO_256_BYTES(TRBYTES + MSG_PREFIX_BYTES + (mlen)) +                   \
   ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +           \
   ALIGN_TO_256_BYTES(CTILDEBYTES))

int crypto_sign_keypair(uint8_t *pk, uint8_t *sk, uint8_t *d_keypair_mem_pool,
                        size_t keypair_mem_pool_pitch, size_t batch_size = 1,
                        cudaStream_t stream = nullptr, size_t rand_index = 0);

struct sign_lut_element {
  int done_flag = 0;
  uint16_t nonce = 0;
};

struct copy_lut_element {
  size_t exec_idx;
  size_t sign_idx;
};

struct task_lut {
  std::vector<sign_lut_element> sign_lut;
  uint32_t *h_exec_lut;
  uint8_t *h_done_lut;
  copy_lut_element *h_copy_lut;
  uint32_t *d_exec_lut;
  uint8_t *d_done_lut;
  copy_lut_element *d_copy_lut;
};

int crypto_sign_signature(uint8_t *sig, size_t sig_pitch, size_t *siglen,
                          const uint8_t *m, size_t m_pitch, size_t mlen,
                          const uint8_t *sk, uint8_t *d_sign_mem_pool,
                          size_t d_sign_mem_pool_pitch,
                          uint8_t *d_temp_mem_pool,
                          size_t d_temp_mem_pool_pitch, task_lut &lut,
                          size_t exec_threshold, size_t batch_size = 1,
                          cudaStream_t stream = nullptr);

void crypto_sign_verify(int *ret, const uint8_t *sig, size_t sig_pitch,
                        size_t siglen, const uint8_t *m, size_t m_pitch,
                        size_t mlen, const uint8_t *pk,
                        uint8_t *d_verify_mem_pool,
                        size_t verify_mem_pool_pitch, size_t batch_size = 1,
                        cudaStream_t stream = nullptr);

#if defined(SIGN_PHASE_PROFILE) &&                                             \
    (defined(USE_PERSISTENT_WARPNTT) || defined(USE_PERSISTENT))
extern "C" void sign_phase_profile_reset();
extern "C" void sign_phase_profile_dump();
#endif

#ifdef USE_CACHED_SK_SINGLE
extern "C" void crypto_sign_invalidate_single_cache();
extern "C" void crypto_sign_destroy_single_cache();
#endif
