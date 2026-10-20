/* Modified form https://github.com/pq-crystals/dilithium.
 * Under Apache 2.0 License.
 *
 * Modified in 2026: FIPS 204 (ML-DSA) parameters and the options of the fused
 * and staged signing kernels.
 */

#pragma once

#define SEEDBYTES 32
#define CRHBYTES 64
#define TRBYTES 64
#define RNDBYTES 32
#define DILITHIUM_N 256
#define DILITHIUM_Q 8380417
#define DILITHIUM_D 13
#define ROOT_OF_UNITY 1753

#if DILITHIUM_MODE == 2
#define DILITHIUM_K 4
#define DILITHIUM_L 4
#define ETA 2
#define TAU 39
#define BETA 78
#define GAMMA1 (1 << 17)
#define GAMMA2 ((DILITHIUM_Q - 1) / 88)
#define OMEGA 80
#define CTILDEBYTES 32
#define HINT_PAD_SIZE 4
#define DILITHIUM_WARPS 4
#define DILITHIUM_SIGN_WARPS 4

#elif DILITHIUM_MODE == 3
#define DILITHIUM_K 6
#define DILITHIUM_L 5
#define ETA 4
#define TAU 49
#define BETA 196
#define GAMMA1 (1 << 19)
#define GAMMA2 ((DILITHIUM_Q - 1) / 32)
#define OMEGA 55
#define CTILDEBYTES 48
#define HINT_PAD_SIZE 3
#define DILITHIUM_WARPS 4
#define DILITHIUM_SIGN_WARPS 3

#elif DILITHIUM_MODE == 5
#define DILITHIUM_K 8
#define DILITHIUM_L 7
#define ETA 2
#define TAU 60
#define BETA 120
#define GAMMA1 (1 << 19)
#define GAMMA2 ((DILITHIUM_Q - 1) / 32)
#define OMEGA 75
#define CTILDEBYTES 64
#define HINT_PAD_SIZE 5
#define DILITHIUM_WARPS 4
#define DILITHIUM_SIGN_WARPS 4

#endif

// fused signing kernel: one block per signature
// FUSED_TRIM_WARPS: launch only the warps that do work (only matters for
// ML-DSA-65, where the 4th warp is idle)
#ifdef FUSED_TRIM_WARPS
#define FUSED_WARPS DILITHIUM_SIGN_WARPS
#else
#define FUSED_WARPS 4
#endif
#define FUSED_BLOCK_SIZE (FUSED_WARPS * 32)
// min. blocks per SM for __launch_bounds__ (caps the registers per thread);
// 1024 threads per SM = 64 regs for 128-thread blocks, 3-11% faster on A100
#ifndef FUSED_MIN_BLOCKS
#define FUSED_MIN_BLOCKS (1024 / FUSED_BLOCK_SIZE)
#endif
#define LAUNCH_BOUNDS_SIGN __launch_bounds__(FUSED_BLOCK_SIZE, FUSED_MIN_BLOCKS)

// FUSED_PREGEN_Y: packed y of the first FUSED_PREGEN_ATTEMPTS attempts is
// computed beforehand (one Keccak state per thread) and stored at the end of
// each sign pool row. Later attempts compute y in the kernel as before.
#ifdef FUSED_PREGEN_Y
#ifndef FUSED_PREGEN_ATTEMPTS
#define FUSED_PREGEN_ATTEMPTS 8
#endif
#define SIGN_POOL_YPRE_DATA_BYTES                                              \
  (FUSED_PREGEN_ATTEMPTS * DILITHIUM_L * POLYZ_PACKEDBYTES)
#else
#define SIGN_POOL_YPRE_DATA_BYTES 0
#endif

// FUSED_STAGED: first FUSED_STAGED_ROUNDS attempts are done in rounds of
// commit / challenge (one thread per signature) / respond; the fused kernel
// finishes the rest. Per row: attempt, done flag, SHAKE256(c~) block.
#ifdef FUSED_STAGED
#ifndef FUSED_PREGEN_Y
#error "FUSED_STAGED requires FUSED_PREGEN_Y"
#endif
#ifdef FUSED_OVERLAP_Y
#error "FUSED_STAGED and FUSED_OVERLAP_Y cannot be combined"
#endif
#ifndef FUSED_STAGED_ROUNDS
#define FUSED_STAGED_ROUNDS FUSED_PREGEN_ATTEMPTS
#endif
#define SIGN_POOL_STAGE_BYTES 256
#else
#define SIGN_POOL_STAGE_BYTES 0
#endif
#define SIGN_POOL_YPRE_BYTES (SIGN_POOL_YPRE_DATA_BYTES + SIGN_POOL_STAGE_BYTES)

#define POLYT1_PACKEDBYTES 320
#define POLYT0_PACKEDBYTES 416
#define POLYVECH_PACKEDBYTES (OMEGA + DILITHIUM_K)

#if GAMMA1 == (1 << 17)
#define POLYZ_PACKEDBYTES 576
#elif GAMMA1 == (1 << 19)
#define POLYZ_PACKEDBYTES 640
#endif

#if GAMMA2 == (DILITHIUM_Q - 1) / 88
#define POLYW1_PACKEDBYTES 192
#elif GAMMA2 == (DILITHIUM_Q - 1) / 32
#define POLYW1_PACKEDBYTES 128
#endif

#if ETA == 2
#define POLYETA_PACKEDBYTES 96
#elif ETA == 4
#define POLYETA_PACKEDBYTES 128
#endif

#define CRYPTO_PUBLICKEYBYTES (SEEDBYTES + DILITHIUM_K * POLYT1_PACKEDBYTES)
// FIPS 204: sk = rho || K || tr || s1 || s2 || t0, sig = c_tilde || z || h
#define CRYPTO_SECRETKEYBYTES (2 * SEEDBYTES + TRBYTES + DILITHIUM_L * POLYETA_PACKEDBYTES + DILITHIUM_K * POLYETA_PACKEDBYTES + DILITHIUM_K * POLYT0_PACKEDBYTES)
#define CRYPTO_BYTES (CTILDEBYTES + DILITHIUM_L * POLYZ_PACKEDBYTES + POLYVECH_PACKEDBYTES)

// Pure ML-DSA with an empty context string: M' = 0 || 0 || M
#define MSG_PREFIX_BYTES 2
