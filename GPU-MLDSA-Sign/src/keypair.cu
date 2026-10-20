/*
 * Copyright 2022-2023 [Anonymous].
 *
 * This file is part of cuDilithium.
 *
 * cuDilithium is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * cuDilithium is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with cuDilithium.
 * If not, see <https://www.gnu.org/licenses/>.
 *
 * Modified in 2026: ported to FIPS 204 (ML-DSA); fused and staged signing,
 * warp NTT and key caching.
 */

#include "keypair.cuh"

#include "fips202/fips202.cuh"
#include "ntt.cuh"
#include "params.h"
#include "reduce.cuh"
#include "rounding.cuh"

#define POLY_UNIFORM_NBLOCKS ((768 + SHAKE128_RATE - 1) / SHAKE128_RATE)

#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + SHAKE256_RATE - 1) / SHAKE256_RATE + 1)
#define POLY_UNIFORM_ETA_BUFLEN (POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE)
#endif

__global__ void gpu_keypair(uint8_t *g_pk_rho, uint8_t *g_pk_t1_packed,
                            uint8_t *g_sk_rho, uint8_t *g_sk_key, uint8_t *g_sk_tr,
                            uint8_t *g_sk_s1_packed, uint8_t *g_sk_s2_packed, uint8_t *g_sk_t0_packed,
                            int32_t *g_t1, size_t keypair_mem_pool_pitch, size_t rand_index) {
    __shared__ uint8_t s_random_buf[8];
    __shared__ uint8_t s_seedbuf[SHAKE256_RATE];// 2 * SEEDBYTES + CRHBYTES
    __shared__ uint8_t s_buf_gena[(POLY_UNIFORM_NBLOCKS + 1) * SHAKE128_RATE];
    __shared__ uint8_t s_buf_gens_in[CRHBYTES + 2 + 6];// pad 6 to align to 8 bytes
    __shared__ uint8_t s_buf_gens_out[POLY_UNIFORM_ETA_BUFLEN];

    __shared__ int32_t s_tmp[DILITHIUM_N + 32];
    __shared__ int32_t s_tmp2[DILITHIUM_N + 32];

    int32_t regs[8];

    // get randombytes to s_seedbuf 32 bytes
    // use pseudo random bytes as a workaround to pass testvectors (slightly decrease performance)
    // in real applications, we must use true random bytes
    uint64_t random_ctr = rand_index + blockIdx.x * 2 + 1;
    if (threadIdx.x < 8)
        s_random_buf[threadIdx.x] = random_ctr >> 8 * threadIdx.x;
    __syncwarp();
    shake<SHAKE128_RATE, 0x1F>(s_buf_gens_out, 1, s_random_buf, 8);
    __syncwarp();
    s_seedbuf[threadIdx.x] = s_buf_gens_out[threadIdx.x];
    // FIPS 204 domain separation: H(xi || k || l)
    if (threadIdx.x == 0) {
        s_seedbuf[SEEDBYTES + 0] = DILITHIUM_K;
        s_seedbuf[SEEDBYTES + 1] = DILITHIUM_L;
    }
    __syncwarp();

    shake<SHAKE256_RATE, 0x1f>(s_seedbuf, 1, s_seedbuf, SEEDBYTES + 2);
    __syncwarp();

    uint8_t *s_rho = s_seedbuf;
    uint8_t *s_rhoprime = s_rho + SEEDBYTES;
    uint8_t *s_key = s_rhoprime + CRHBYTES;

    // copy s_rho to g_pk_rho
    g_pk_rho[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_rho[threadIdx.x];
    // copy s_rho to g_sk_rho
    g_sk_rho[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_rho[threadIdx.x];
    // copy s_key to g_sk_key
    g_sk_key[blockIdx.x * keypair_mem_pool_pitch + threadIdx.x] = s_key[threadIdx.x];
    // copy s_rhoprime to s_buf_gens_in
    s_buf_gens_in[threadIdx.x] = s_rhoprime[threadIdx.x];
    s_buf_gens_in[threadIdx.x + 32] = s_rhoprime[threadIdx.x + 32];
    __syncwarp();

    // expand A
    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        // use rhoprime to sample s1 to s_tmp
        // padding two bytes
        if (threadIdx.x == 0) {
            s_buf_gens_in[CRHBYTES + 0] = l & 0xFF;
            s_buf_gens_in[CRHBYTES + 1] = (l >> 8) & 0xFF;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, POLY_UNIFORM_ETA_NBLOCKS, s_buf_gens_in, CRHBYTES + 2);
        __syncwarp();
        unsigned int ctr_rej_eta = 0;
        unsigned int pos_rej_eta = 0;
        while (ctr_rej_eta < DILITHIUM_N && pos_rej_eta + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_buf_gens_out[pos_rej_eta + (threadIdx.x >> 1)];
            uint32_t t = (buf >> (4 * (threadIdx.x & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - threadIdx.x;
            unsigned int ctr_offset = __popc(good_mask);
            if (ctr_rej_eta + ctr_offset <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 2 - t;
#elif ETA == 4
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 4 - t;
#endif
            }
            ctr_rej_eta += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
            pos_rej_eta += 16;
        }
        __syncwarp();
        // pack s1 into g_sk_s1_packed
        uint8_t *g_sk_s1_packed_l = g_sk_s1_packed + blockIdx.x * keypair_mem_pool_pitch + l * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_tmp[8 * threadIdx.x + i];
        g_sk_s1_packed_l[3 * threadIdx.x + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s1_packed_l[3 * threadIdx.x + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s1_packed_l[3 * threadIdx.x + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_tmp[2 * i + 0];
            regs[1] = ETA - s_tmp[2 * i + 1];
            g_sk_s1_packed_l[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();
        // compute ntt(s1) into s_tmp
        for (size_t i = 0; i < 8; i++)
            regs[i] = s_tmp[i * 32 + threadIdx.x];
        ntt_inner(regs, s_tmp);
        for (size_t i = 0; i < 8; i++)
            s_tmp[8 * threadIdx.x + i] = regs[i];
        __syncwarp();

        for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
            // use g_t1 to accumulate A * s1
            int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
            s_buf_gena[threadIdx.x] = s_rho[threadIdx.x];
            if (threadIdx.x == 0) {
                s_buf_gena[SEEDBYTES] = (k << 8) + l;
                s_buf_gena[SEEDBYTES + 1] = k;
            }
            __syncwarp();
            shake<SHAKE128_RATE, 0x1f>(s_buf_gena, POLY_UNIFORM_NBLOCKS, s_buf_gena, SEEDBYTES + 2);
            __syncwarp();

            size_t ctr = 0;
            for (size_t round = 0; round < 8; round++) {
                size_t s_buf_offset = round * 32 * 3 + threadIdx.x * 3;
                uint8_t t0 = s_buf_gena[s_buf_offset + 0];
                uint8_t t1 = s_buf_gena[s_buf_offset + 1];
                uint8_t t2 = s_buf_gena[s_buf_offset + 2] & 0x7F;
                uint32_t t = t0 | (t1 << 8) | (t2 << 16);

                int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;

                unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                if (good_mask == 0xFFFFFFFF) {
                    s_tmp2[ctr + threadIdx.x] = t;
                    ctr += 32;
                } else {
                    good_mask <<= 31 - threadIdx.x;
                    unsigned int ctr_offset = __popc(good_mask);
                    if (ctr + ctr_offset <= DILITHIUM_N && good) {
                        s_tmp2[ctr + ctr_offset - 1] = t;
                    }
                    ctr += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
                }
            }
            if (ctr < DILITHIUM_N) {
                size_t s_buf_offset = 8 * 32 * 3 + threadIdx.x * 3;
                uint8_t t0 = s_buf_gena[s_buf_offset + 0];
                uint8_t t1 = s_buf_gena[s_buf_offset + 1];
                uint8_t t2 = s_buf_gena[s_buf_offset + 2] & 0x7F;
                uint32_t t = t0 | (t1 << 8) | (t2 << 16);

                int good = (static_cast<int32_t>(t - DILITHIUM_Q) >> 31) & 1;
                unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                good_mask <<= 31 - threadIdx.x;
                unsigned int ctr_offset = __popc(good_mask);
                if (ctr + ctr_offset <= DILITHIUM_N && good) {
                    s_tmp2[ctr + ctr_offset - 1] = t;
                }
            }
            __syncwarp();
            for (size_t i = 0; i < 8; ++i) {
                g_t1_k[32 * i + threadIdx.x] += gpu_montgomery_multiply(s_tmp2[i * 32 + threadIdx.x], s_tmp[i * 32 + threadIdx.x]);
            }
            __syncwarp();
        }
        __syncwarp();
    }

    for (unsigned int k = 0; k < DILITHIUM_K; ++k) {
        // use rhoprime to sample s2 to s_tmp
        // padding two bytes
        if (threadIdx.x == 0) {
            s_buf_gens_in[CRHBYTES + 0] = (DILITHIUM_L + k) & 0xFF;
            s_buf_gens_in[CRHBYTES + 1] = (DILITHIUM_L + k) >> 8;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, POLY_UNIFORM_ETA_NBLOCKS, s_buf_gens_in, CRHBYTES + 2);
        __syncwarp();
        unsigned int ctr_rej_eta = 0;
        unsigned int pos_rej_eta = 0;
        while (ctr_rej_eta < DILITHIUM_N && pos_rej_eta + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_buf_gens_out[pos_rej_eta + (threadIdx.x >> 1)];
            uint32_t t = (buf >> (4 * (threadIdx.x & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - threadIdx.x;
            unsigned int ctr_offset = __popc(good_mask);
            if (ctr_rej_eta + ctr_offset <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 2 - t;
#elif ETA == 4
                s_tmp[ctr_rej_eta + ctr_offset - 1] = 4 - t;
#endif
            }
            ctr_rej_eta += __shfl_sync(0xFFFFFFFF, ctr_offset, 31);
            pos_rej_eta += 16;
        }
        __syncwarp();
        // pack s2 into g_sk_s2_packed
        uint8_t *g_sk_s2_packed_k = g_sk_s2_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_tmp[8 * threadIdx.x + i];
        g_sk_s2_packed_k[3 * threadIdx.x + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s2_packed_k[3 * threadIdx.x + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s2_packed_k[3 * threadIdx.x + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_tmp[2 * i + 0];
            regs[1] = ETA - s_tmp[2 * i + 1];
            g_sk_s2_packed_k[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();

        int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i) {
            regs[i] = g_t1_k[8 * threadIdx.x + i];
            regs[i] = reduce32(regs[i]);
        }
        invntt_inner(regs, s_tmp2);

        for (size_t i = 0; i < 8; ++i) {
            // add s2 to t1
            regs[i] += s_tmp[i * 32 + threadIdx.x];
            regs[i] = caddq(regs[i]);
            s_tmp[i * 32 + threadIdx.x] = power2round(&s_tmp2[i * 32 + threadIdx.x], regs[i]);
            // after that, s_tmp stores t1, s_tmp2 stores t0
        }
        __syncwarp();

        // pack t1 into g_pk_t1_packed
        uint8_t *g_sk_t1_packed_k = g_pk_t1_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYT1_PACKEDBYTES;
        for (size_t i = threadIdx.x; i < DILITHIUM_N / 4; i += 32) {
            g_sk_t1_packed_k[5 * i + 0] = (s_tmp[4 * i + 0] >> 0);
            g_sk_t1_packed_k[5 * i + 1] = (s_tmp[4 * i + 0] >> 8) | (s_tmp[4 * i + 1] << 2);
            g_sk_t1_packed_k[5 * i + 2] = (s_tmp[4 * i + 1] >> 6) | (s_tmp[4 * i + 2] << 4);
            g_sk_t1_packed_k[5 * i + 3] = (s_tmp[4 * i + 2] >> 4) | (s_tmp[4 * i + 3] << 6);
            g_sk_t1_packed_k[5 * i + 4] = (s_tmp[4 * i + 3] >> 2);
        }
        __syncwarp();

        // pack t0 into g_sk_t0_packed
        uint8_t *g_sk_t0_packed_k = g_sk_t0_packed + blockIdx.x * keypair_mem_pool_pitch + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = (1 << (DILITHIUM_D - 1)) - s_tmp2[8 * threadIdx.x + i];
        g_sk_t0_packed_k[13 * threadIdx.x + 0] = regs[0];
        g_sk_t0_packed_k[13 * threadIdx.x + 1] = regs[0] >> 8;
        g_sk_t0_packed_k[13 * threadIdx.x + 1] |= regs[1] << 5;
        g_sk_t0_packed_k[13 * threadIdx.x + 2] = regs[1] >> 3;
        g_sk_t0_packed_k[13 * threadIdx.x + 3] = regs[1] >> 11;
        g_sk_t0_packed_k[13 * threadIdx.x + 3] |= regs[2] << 2;
        g_sk_t0_packed_k[13 * threadIdx.x + 4] = regs[2] >> 6;
        g_sk_t0_packed_k[13 * threadIdx.x + 4] |= regs[3] << 7;
        g_sk_t0_packed_k[13 * threadIdx.x + 5] = regs[3] >> 1;
        g_sk_t0_packed_k[13 * threadIdx.x + 6] = regs[3] >> 9;
        g_sk_t0_packed_k[13 * threadIdx.x + 6] |= regs[4] << 4;
        g_sk_t0_packed_k[13 * threadIdx.x + 7] = regs[4] >> 4;
        g_sk_t0_packed_k[13 * threadIdx.x + 8] = regs[4] >> 12;
        g_sk_t0_packed_k[13 * threadIdx.x + 8] |= regs[5] << 1;
        g_sk_t0_packed_k[13 * threadIdx.x + 9] = regs[5] >> 7;
        g_sk_t0_packed_k[13 * threadIdx.x + 9] |= regs[6] << 6;
        g_sk_t0_packed_k[13 * threadIdx.x + 10] = regs[6] >> 2;
        g_sk_t0_packed_k[13 * threadIdx.x + 11] = regs[6] >> 10;
        g_sk_t0_packed_k[13 * threadIdx.x + 11] |= regs[7] << 3;
        g_sk_t0_packed_k[13 * threadIdx.x + 12] = regs[7] >> 5;
        __syncwarp();
    }

    // tr = H(rho, t1)
    // reuse s_buf_gens_out
    shake<SHAKE256_RATE, 0x1f>(s_buf_gens_out, 1, g_pk_rho + blockIdx.x * keypair_mem_pool_pitch, CRYPTO_PUBLICKEYBYTES);
    __syncwarp();
    // write tr to g_sk_tr
    for (size_t i = threadIdx.x; i < TRBYTES; i += 32)
        g_sk_tr[blockIdx.x * keypair_mem_pool_pitch + i] = s_buf_gens_out[i];
}

#ifdef USE_128T_KEYPAIR
// 128-thread (4-warp) keypair kernel using warp-shuffle NTT.
// Layout: regs[i] = coeff[i*32 + lane_id]  (cyclic-striped, no-transpose).
// NTT/INTT are register-only via ntt_inner_shuffle_notranspose.
//
// Structure:
//   Warp 0: seed expansion
//   Phase 1 (warp 0): sample s1[l], pack, NTT(s1) -> s_shared.s1_ntt[l]
//   Phase 2 (All warps): expand A[k,l] + accumulate g_t1[k] (striped over K)
//   Phase 3 (All warps): s2 sample, invNTT(t1), power2round, pack (striped over K)
//   Warp 0: tr = H(pk)
__global__ void gpu_keypair_128t(uint8_t *g_pk_rho, uint8_t *g_pk_t1_packed,
                                  uint8_t *g_sk_rho, uint8_t *g_sk_key, uint8_t *g_sk_tr,
                                  uint8_t *g_sk_s1_packed, uint8_t *g_sk_s2_packed, uint8_t *g_sk_t0_packed,
                                  int32_t *g_t1, size_t keypair_mem_pool_pitch, size_t rand_index) {
    // shared memory
    __shared__ uint8_t s_random_buf[8];
    __shared__ uint8_t s_seedbuf[SHAKE256_RATE];            // rho(32) || rhoprime(64) || key(32)
    __shared__ uint8_t s_rho[SEEDBYTES];
    __shared__ uint8_t s_rhoprime[CRHBYTES];
    // Per-warp SHAKE buffers (indexed by warp_id)
    __shared__ uint8_t s_gens_in[4][CRHBYTES + 8];         // rhoprime || 2-byte nonce, +6 pad
    __shared__ uint8_t s_gens_out[4][POLY_UNIFORM_ETA_BUFLEN];
    __shared__ uint8_t s_buf_gena[4][(POLY_UNIFORM_NBLOCKS + 1) * SHAKE128_RATE];
    
    // Union to save shared memory: s1_ntt is used in Phase 1 & 2; tmp is used in Phase 3.
    // manual aliasing via pointer
    __shared__ int32_t s_shared_mem[DILITHIUM_L * DILITHIUM_N]; // size max(L*N, 4*N)
    struct {
        int32_t (*s1_ntt)[DILITHIUM_N];
        int32_t (*tmp)[DILITHIUM_N];
    } s_shared = { 
        (int32_t(*)[DILITHIUM_N])s_shared_mem, 
        (int32_t(*)[DILITHIUM_N])s_shared_mem 
    };


    int32_t regs[8];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f;

    // Seed expansion (warp 0 only)
    if (warp_id == 0) {
        uint64_t random_ctr = rand_index + blockIdx.x * 2 + 1;
        if (lane_id < 8)
            s_random_buf[lane_id] = random_ctr >> (8 * lane_id);
        __syncwarp();
        // Reuse s_buf_gena[0] as a temp for SHAKE128 output (>=32 bytes needed)
        shake<SHAKE128_RATE, 0x1F>(s_buf_gena[0], 1, s_random_buf, 8);
        __syncwarp();
        s_seedbuf[lane_id] = s_buf_gena[0][lane_id];
        // FIPS 204 domain separation: H(xi || k || l)
        if (lane_id == 0) {
            s_seedbuf[SEEDBYTES + 0] = DILITHIUM_K;
            s_seedbuf[SEEDBYTES + 1] = DILITHIUM_L;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_seedbuf, 1, s_seedbuf, SEEDBYTES + 2);
        __syncwarp();

        // Extract rho (bytes 0..31), rhoprime (bytes 32..95), key (bytes 96..127)
        s_rho[lane_id]      = s_seedbuf[lane_id];
        s_rhoprime[lane_id] = s_seedbuf[SEEDBYTES + lane_id];
        if (lane_id < 32)
            s_rhoprime[32 + lane_id] = s_seedbuf[SEEDBYTES + 32 + lane_id];
        __syncwarp();

        // Copy rho and key to global output
        g_pk_rho[blockIdx.x * keypair_mem_pool_pitch + lane_id] = s_rho[lane_id];
        g_sk_rho[blockIdx.x * keypair_mem_pool_pitch + lane_id] = s_rho[lane_id];
        g_sk_key[blockIdx.x * keypair_mem_pool_pitch + lane_id] = s_seedbuf[SEEDBYTES + CRHBYTES + lane_id];
    }
    __syncthreads(); // Make rhoprime and rho available to all warps

    // Prepare rhoprime in per-warp s_gens_in
    s_gens_in[warp_id][lane_id]      = s_rhoprime[lane_id];
    s_gens_in[warp_id][32 + lane_id] = s_rhoprime[32 + lane_id];
    __syncwarp();

    // Phase 1: sample s1[l], pack, NTT(s1) (Warp 0 only)
    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        if (warp_id == 0) {
        if (lane_id == 0) {
            s_gens_in[warp_id][CRHBYTES + 0] = l & 0xFF;
            s_gens_in[warp_id][CRHBYTES + 1] = (l >> 8) & 0xFF;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_gens_out[warp_id], POLY_UNIFORM_ETA_NBLOCKS,
                                    s_gens_in[warp_id], CRHBYTES + 2);
        __syncwarp();

        // Sample directly into s_shared.s1_ntt[l] as natural order (temporarily)
        unsigned int ctr = 0, pos = 0;
        while (ctr < DILITHIUM_N && pos + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_gens_out[warp_id][pos + (lane_id >> 1)];
            uint32_t t  = (buf >> (4 * (lane_id & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - lane_id;
            unsigned int ctr_off = __popc(good_mask);
            if (ctr + ctr_off <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_shared.s1_ntt[l][ctr + ctr_off - 1] = 2 - (int32_t)t;
#elif ETA == 4
                s_shared.s1_ntt[l][ctr + ctr_off - 1] = 4 - (int32_t)t;
#endif
            }
            ctr += __shfl_sync(0xFFFFFFFF, ctr_off, 31);
            pos += 16;
        }
        __syncwarp();

        // Pack s1[l] from natural layout s_shared.s1_ntt[l]
        uint8_t *g_sk_s1_packed_l = g_sk_s1_packed + blockIdx.x * keypair_mem_pool_pitch
                                     + l * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_shared.s1_ntt[l][8 * lane_id + i];
        g_sk_s1_packed_l[3 * lane_id + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s1_packed_l[3 * lane_id + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s1_packed_l[3 * lane_id + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = lane_id; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_shared.s1_ntt[l][2 * i + 0];
            regs[1] = ETA - s_shared.s1_ntt[l][2 * i + 1];
            g_sk_s1_packed_l[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();

        // NTT(s1): read natural layout -> cyclic-striped regs -> shuffle NTT
        for (size_t i = 0; i < 8; i++)
            regs[i] = s_shared.s1_ntt[l][i * 32 + lane_id];
        ntt_inner_shuffle_notranspose(regs);
        // Write back in cyclic-striped layout
        for (size_t i = 0; i < 8; i++)
            s_shared.s1_ntt[l][i * 32 + lane_id] = regs[i];
        }
    }
    __syncthreads(); // Wait for warp 0 to finish s1 generation

    // Phase 2: expand A[k,l] + accumulate g_t1[k] (all 4 warps stripe over K)
    for (unsigned int l = 0; l < DILITHIUM_L; ++l) {
        for (unsigned int k = warp_id; k < DILITHIUM_K; k += 4) {
        int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t) + k * DILITHIUM_N;
        uint8_t *s_gena_w = s_buf_gena[warp_id];
            // Load rho and set nonce
            s_gena_w[lane_id] = s_rho[lane_id];
            if (lane_id == 0) {
                s_gena_w[SEEDBYTES]     = (k << 8) + l;
                s_gena_w[SEEDBYTES + 1] = k;
            }
            __syncwarp();
            shake<SHAKE128_RATE, 0x1f>(s_gena_w, POLY_UNIFORM_NBLOCKS, s_gena_w, SEEDBYTES + 2);
            __syncwarp();

            // Rejection sampling -> store A[k,l] coefficients in cyclic-striped g_t1_k
            size_t ctr = 0;
            for (size_t round = 0; round < 8; round++) {
                size_t off = round * 32 * 3 + lane_id * 3;
                uint8_t t0 = s_gena_w[off + 0];
                uint8_t t1 = s_gena_w[off + 1];
                uint8_t t2 = s_gena_w[off + 2] & 0x7F;
                uint32_t tv = t0 | (t1 << 8) | (t2 << 16);
                int good = (static_cast<int32_t>(tv - DILITHIUM_Q) >> 31) & 1;
                unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                if (good_mask == 0xFFFFFFFF) {
                    // Accumulate: g_t1_k[ctr+lane_id] += A * NTT(s1)
                    g_t1_k[ctr + lane_id] += gpu_montgomery_multiply(
                        (int32_t)tv, s_shared.s1_ntt[l][ctr + lane_id]);
                    ctr += 32;
                } else {
                    good_mask <<= 31 - lane_id;
                    unsigned int ctr_off = __popc(good_mask);
                    if (ctr + ctr_off <= DILITHIUM_N && good) {
                        g_t1_k[ctr + ctr_off - 1] += gpu_montgomery_multiply(
                            (int32_t)tv, s_shared.s1_ntt[l][ctr + ctr_off - 1]);
                    }
                    ctr += __shfl_sync(0xFFFFFFFF, ctr_off, 31);
                }
            }
            if (ctr < DILITHIUM_N) {
                size_t off = 8 * 32 * 3 + lane_id * 3;
                uint8_t t0 = s_gena_w[off + 0];
                uint8_t t1 = s_gena_w[off + 1];
                uint8_t t2 = s_gena_w[off + 2] & 0x7F;
                uint32_t tv = t0 | (t1 << 8) | (t2 << 16);
                int good = (static_cast<int32_t>(tv - DILITHIUM_Q) >> 31) & 1;
                unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
                good_mask <<= 31 - lane_id;
                unsigned int ctr_off = __popc(good_mask);
                if (ctr + ctr_off <= DILITHIUM_N && good) {
                    g_t1_k[ctr + ctr_off - 1] += gpu_montgomery_multiply(
                        (int32_t)tv, s_shared.s1_ntt[l][ctr + ctr_off - 1]);
                }
            }
            __syncwarp();
        }
    }
    __syncthreads(); // Phase 2 complete. Safe to overwrite s_shared.s1_ntt with tmp now.

    // Phase 3: s2 sample + invNTT(t1) + power2round + pack (all 4 warps stripe over K)
    for (unsigned int k = warp_id; k < DILITHIUM_K; k += 4) {
        // -- Sample s2[k] using rhoprime || (L + k) --
        if (lane_id == 0) {
            s_gens_in[warp_id][CRHBYTES + 0] = (DILITHIUM_L + k) & 0xFF;
            s_gens_in[warp_id][CRHBYTES + 1] = (DILITHIUM_L + k) >> 8;
        }
        __syncwarp();
        shake<SHAKE256_RATE, 0x1f>(s_gens_out[warp_id], POLY_UNIFORM_ETA_NBLOCKS,
                                    s_gens_in[warp_id], CRHBYTES + 2);
        __syncwarp();

        unsigned int ctr = 0, pos = 0;
        while (ctr < DILITHIUM_N && pos + 16 <= POLY_UNIFORM_ETA_BUFLEN) {
            uint8_t buf = s_gens_out[warp_id][pos + (lane_id >> 1)];
            uint32_t t  = (buf >> (4 * (lane_id & 1))) & 0x0F;
#if ETA == 2
            int good = t < 15;
#elif ETA == 4
            int good = t < 9;
#endif
            unsigned int good_mask = __ballot_sync(0xFFFFFFFF, good);
            good_mask <<= 31 - lane_id;
            unsigned int ctr_off = __popc(good_mask);
            if (ctr + ctr_off <= DILITHIUM_N && good) {
#if ETA == 2
                t = t - (205 * t >> 10) * 5;
                s_shared.tmp[warp_id][ctr + ctr_off - 1] = 2 - (int32_t)t;
#elif ETA == 4
                s_shared.tmp[warp_id][ctr + ctr_off - 1] = 4 - (int32_t)t;
#endif
            }
            ctr += __shfl_sync(0xFFFFFFFF, ctr_off, 31);
            pos += 16;
        }
        __syncwarp();

        // Pack s2[k] from natural-order s_shared.tmp[warp_id]
        uint8_t *g_sk_s2_packed_k = g_sk_s2_packed + blockIdx.x * keypair_mem_pool_pitch
                                     + k * POLYETA_PACKEDBYTES;
#if ETA == 2
        for (size_t i = 0; i < 8; i++)
            regs[i] = ETA - s_shared.tmp[warp_id][8 * lane_id + i];
        g_sk_s2_packed_k[3 * lane_id + 0] = (regs[0] >> 0) | (regs[1] << 3) | (regs[2] << 6);
        g_sk_s2_packed_k[3 * lane_id + 1] = (regs[2] >> 2) | (regs[3] << 1) | (regs[4] << 4) | (regs[5] << 7);
        g_sk_s2_packed_k[3 * lane_id + 2] = (regs[5] >> 1) | (regs[6] << 2) | (regs[7] << 5);
#elif ETA == 4
        for (size_t i = lane_id; i < DILITHIUM_N / 2; i += 32) {
            regs[0] = ETA - s_shared.tmp[warp_id][2 * i + 0];
            regs[1] = ETA - s_shared.tmp[warp_id][2 * i + 1];
            g_sk_s2_packed_k[i] = regs[0] | (regs[1] << 4);
        }
#endif
        __syncwarp();
        
        // Load s2 into registers (cyclic-striped) for addition later
        int32_t regs_s2[8];
        for (size_t i = 0; i < 8; ++i)
            regs_s2[i] = s_shared.tmp[warp_id][i * 32 + lane_id];
        // Now s_shared.tmp[warp_id] is FREE to be overwritten

        // invNTT(t1[k]): g_t1[k] in cyclic-striped layout -> cyclic-striped regs
        int32_t *g_t1_k = g_t1 + blockIdx.x * keypair_mem_pool_pitch / sizeof(int32_t)
                           + k * DILITHIUM_N;
        for (size_t i = 0; i < 8; ++i) {
            regs[i] = g_t1_k[i * 32 + lane_id];
            regs[i] = reduce32(regs[i]);
        }
        invntt_inner_shuffle_notranspose(regs);

        // Add s2, caddq, power2round -> keep t1 and t0 in registers
        int32_t regs_t0[8];
        for (size_t i = 0; i < 8; ++i) {
            regs[i] += regs_s2[i];
            regs[i] = caddq(regs[i]);
            regs[i] = power2round(&regs_t0[i], regs[i]); // regs[i] becomes t1
        }
        
        // Stage t1 into s_shared.tmp (natural layout)
        for (size_t i = 0; i < 8; ++i)
            s_shared.tmp[warp_id][i * 32 + lane_id] = regs[i];
        __syncwarp();

        // Pack t1 (10-bit, 5 bytes per 4 coefficients, natural layout)
        uint8_t *g_pk_t1_packed_k = g_pk_t1_packed + blockIdx.x * keypair_mem_pool_pitch
                                     + k * POLYT1_PACKEDBYTES;
        for (size_t i = lane_id; i < DILITHIUM_N / 4; i += 32) {
            g_pk_t1_packed_k[5 * i + 0] =  s_shared.tmp[warp_id][4 * i + 0];
            g_pk_t1_packed_k[5 * i + 1] = (s_shared.tmp[warp_id][4 * i + 0] >> 8) | (s_shared.tmp[warp_id][4 * i + 1] << 2);
            g_pk_t1_packed_k[5 * i + 2] = (s_shared.tmp[warp_id][4 * i + 1] >> 6) | (s_shared.tmp[warp_id][4 * i + 2] << 4);
            g_pk_t1_packed_k[5 * i + 3] = (s_shared.tmp[warp_id][4 * i + 2] >> 4) | (s_shared.tmp[warp_id][4 * i + 3] << 6);
            g_pk_t1_packed_k[5 * i + 4] =  s_shared.tmp[warp_id][4 * i + 3] >> 2;
        }
        __syncwarp(); // ensure t1 is packed before overwriting with t0

        // Stage t0 into s_shared.tmp (natural layout)
        for (size_t i = 0; i < 8; ++i)
            s_shared.tmp[warp_id][i * 32 + lane_id] = regs_t0[i];
        __syncwarp();

        // Pack t0 (13-bit, 13 bytes per 8 coefficients)
        uint8_t *g_sk_t0_packed_k = g_sk_t0_packed + blockIdx.x * keypair_mem_pool_pitch
                                     + k * POLYT0_PACKEDBYTES;
        for (size_t i = 0; i < 8; ++i)
            regs[i] = (1 << (DILITHIUM_D - 1)) - s_shared.tmp[warp_id][8 * lane_id + i];
        g_sk_t0_packed_k[13 * lane_id +  0] =  regs[0];
        g_sk_t0_packed_k[13 * lane_id +  1] = (regs[0] >> 8) | (regs[1] << 5);
        g_sk_t0_packed_k[13 * lane_id +  2] =  regs[1] >> 3;
        g_sk_t0_packed_k[13 * lane_id +  3] = (regs[1] >> 11) | (regs[2] << 2);
        g_sk_t0_packed_k[13 * lane_id +  4] = (regs[2] >>  6) | (regs[3] << 7);
        g_sk_t0_packed_k[13 * lane_id +  5] =  regs[3] >> 1;
        g_sk_t0_packed_k[13 * lane_id +  6] = (regs[3] >>  9) | (regs[4] << 4);
        g_sk_t0_packed_k[13 * lane_id +  7] =  regs[4] >> 4;
        g_sk_t0_packed_k[13 * lane_id +  8] = (regs[4] >> 12) | (regs[5] << 1);
        g_sk_t0_packed_k[13 * lane_id +  9] = (regs[5] >>  7) | (regs[6] << 6);
        g_sk_t0_packed_k[13 * lane_id + 10] =  regs[6] >> 2;
        g_sk_t0_packed_k[13 * lane_id + 11] = (regs[6] >> 10) | (regs[7] << 3);
        g_sk_t0_packed_k[13 * lane_id + 12] =  regs[7] >> 5;
        __syncwarp();
    } // end k loop

    // tr = H(rho || t1_packed) (warp 0 only)
    __syncthreads();
    if (warp_id == 0) {
        shake<SHAKE256_RATE, 0x1f>(s_gens_out[0], 1,
                                    g_pk_rho + blockIdx.x * keypair_mem_pool_pitch,
                                    CRYPTO_PUBLICKEYBYTES);
        __syncwarp();
        for (size_t i = lane_id; i < TRBYTES; i += 32)
            g_sk_tr[blockIdx.x * keypair_mem_pool_pitch + i] = s_gens_out[0][i];
    }
}
#endif // USE_128T_KEYPAIR
