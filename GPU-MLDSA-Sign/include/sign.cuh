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

#include <cstdint>

#include "api.cuh"

__global__ void polyvec_matrix_expand_kernel(int32_t *g_mat,
                                             const uint8_t *g_rho,
                                             size_t mem_pool_pitch);

__global__ void polyvec_matrix_expand_opt_kernel(int32_t *g_mat,
                                                 const uint8_t *g_rho,
                                                 size_t mem_pool_pitch,
                                                 size_t n_inputs);



#ifdef USE_128T_SIGN
__global__ void unpack_fuse_ntt_shuffle_k6_kernel(int32_t *g_s1, int32_t *g_s2,
                                                  int32_t *g_t0,
                                                  const uint8_t *g_s1_packed,
                                                  const uint8_t *g_s2_packed,
                                                  const uint8_t *g_t0_packed,
                                                  size_t mem_pool_pitch);
#endif

template <unsigned int VEC_SIZE, unsigned int PACKED_BYTES>
__global__ void unpack_fuse_ntt_kernel(int32_t *g_polyvec,
                                       const uint8_t *g_polyvec_packed,
                                       size_t mem_pool_pitch);

__global__ void unpack_fuse_ntt_radix2_kernel(int32_t *g_s1, int32_t *g_s2,
                                              int32_t *g_t0,
                                              const uint8_t *g_s1_packed,
                                              const uint8_t *g_s2_packed,
                                              const uint8_t *g_t0_packed,
                                              size_t mem_pool_pitch);

__global__ void unpack_fuse_ntt_radix2_opt_kernel(int32_t *g_s1, int32_t *g_s2,
                                                  int32_t *g_t0,
                                                  const uint8_t *g_s1_packed,
                                                  const uint8_t *g_s2_packed,
                                                  const uint8_t *g_t0_packed,
                                                  size_t mem_pool_pitch);

__global__ void compute_y_kernel(int32_t *g_y, const uint8_t *g_rhoprime,
                                 const uint32_t *g_exec_lut,
                                 size_t sign_mem_pool_pitch,
                                 size_t temp_mem_pool_pitch);

__global__ void compute_y_opt_kernel(int32_t *g_y, const uint8_t *g_rhoprime,
                                     const uint32_t *g_exec_lut,
                                     size_t sign_mem_pool_pitch,
                                     size_t temp_mem_pool_pitch,
                                     size_t n_inputs);

__global__ void compute_w_32t_kernel(int32_t *g_z, int32_t *g_w0, int32_t *g_w1,
                                     uint8_t *g_w1_packed, const int32_t *g_y,
                                     const int32_t *g_mat,
                                     const uint32_t *g_exec_lut,
                                     size_t sign_mem_pool_pitch,
                                     size_t temp_mem_pool_pitch);

__global__ void compute_w_128t_kernel(int32_t *g_w0, int32_t *g_w1,
                                      uint8_t *g_w1_packed, const int32_t *g_y,
                                      const int32_t *g_mat,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch);
/**
 * @brief Computes cp via SHAKE and immediately applies NTT(cp), writing
 * NTT-domain cp to g_cp. rej_loop can then skip its NTT(cp) step.
 */
__global__ void compute_cp_ntt_kernel(int32_t *g_cp, uint8_t *g_mu,
                                      uint8_t *g_seed, const uint8_t *g_ori_mu,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch,
                                      size_t n_inputs);

__global__ void compute_cp_kernel(int32_t *g_cp, uint8_t *g_mu, uint8_t *g_seed,
                                  const uint8_t *g_ori_mu,
                                  const uint32_t *g_exec_lut,
                                  size_t sign_mem_pool_pitch,
                                  size_t temp_mem_pool_pitch);

__global__ void compute_cp_opt_kernel(int32_t *g_cp, uint8_t *g_mu,
                                      uint8_t *g_seed, const uint8_t *g_ori_mu,
                                      const uint32_t *g_exec_lut,
                                      size_t sign_mem_pool_pitch,
                                      size_t temp_mem_pool_pitch,
                                      size_t n_inputs);

__global__ void
rej_loop_32t_kernel(const int32_t *g_y, int32_t *g_z, const int32_t *g_w0,
                    const int32_t *g_w1, const int32_t *g_cp,
                    uint8_t *g_z_packed, uint8_t *g_hint, uint8_t *g_done_lut,
                    const int32_t *g_s1, const int32_t *g_s2,
                    const int32_t *g_t0, const uint32_t *g_exec_lut,
                    size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch);

__global__ void
rej_loop_128t_kernel(const int32_t *g_y, int32_t *g_z, const int32_t *g_w0,
                     const int32_t *g_w1, const int32_t *g_cp,
                     uint8_t *g_z_packed, uint8_t *g_hint, uint8_t *g_done_lut,
                     const int32_t *g_s1, const int32_t *g_s2,
                     const int32_t *g_t0, const uint32_t *g_exec_lut,
                     size_t sign_mem_pool_pitch, size_t temp_mem_pool_pitch);

__global__ void sig_copy_kernel(uint8_t *d_sig, size_t d_sign_mem_pool_pitch,
                                const uint8_t *d_temp_sig,
                                size_t d_temp_mem_pool_pitch,
                                const copy_lut_element *d_copy_lut);

#ifdef USE_PERSISTENT
// As sign_persistent_warpntt_launch, with the shared-memory NTT of Shen et al.
void sign_persistent_launch(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size, cudaStream_t stream);
#endif

#ifdef USE_PERSISTENT_WARPNTT
// Launches the fused signing pipeline on stream: optional y pre-generation
// (FUSED_PREGEN_Y), optional staged rounds (FUSED_STAGED), then the fused kernel.
void sign_persistent_warpntt_launch(
    int32_t *g_w0, int32_t *g_w1, uint8_t *g_w1_packed, int32_t *g_y,
    const uint8_t *g_rhoprime, const int32_t *g_mat, int32_t *g_z,
    int32_t *g_cp, uint8_t *g_z_packed, uint8_t *g_hint, const int32_t *g_s1,
    const int32_t *g_s2, const int32_t *g_t0, uint8_t *g_mu, uint8_t *g_seed,
    const uint8_t *g_ori_mu, uint8_t *g_sig, size_t sign_mem_pool_pitch,
    size_t batch_size, cudaStream_t stream);
#endif

