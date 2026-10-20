/* Modified form https://github.com/pq-crystals/dilithium.
 * Under Apache 2.0 License.
 */

#pragma once

#include <cstdint>

#include "params.h"

typedef struct {
    int32_t coeffs[DILITHIUM_N];
} poly;

/* Vectors of polynomials of length L */
typedef struct {
    poly vec[DILITHIUM_L];
} polyvecl;

/* Vectors of polynomials of length K */
typedef struct {
    poly vec[DILITHIUM_K];
} polyveck;

/**
 * check is norm
 * @tparam B
 * @param a
 * @return True/False
 */
template<int32_t B>
__device__ int chknorm(const int32_t a) {
    unsigned int i;
    int32_t t;

    if (B > (DILITHIUM_Q - 1) / 8)
        return 1;

    /* It is ok to leak which coefficient violates the bound since
     * the probability for each coefficient is independent of secret
     * data, but we must not leak the sign of the centralized representative.
     */
    for (i = 0; i < DILITHIUM_N; ++i) {
        /* Absolute value */
        t = a >> 31;
        t = a - (t & 2 * a);

        if (t >= B) {
            return 1;
        }
    }

    return 0;
}
#ifdef USE_128T_SIGN
__device__ __forceinline__ void poly_challenge_shake256_opt(int32_t regs[8],
                                                            uint8_t *s_buf,
                                                            int lane,
                                                            int32_t *s_poly) {
  for (int i = 0; i < 8; i++)
    regs[i] = 0;
  for (int i = 0; i < 8; i++)
    s_poly[i * 32 + lane] = 0;
  __syncwarp();

  uint64_t signs = ((uint64_t *)s_buf)[0];
  int pos = 8;
  for (int i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
    int b = -1;
    while (true) {
      int candidate = (pos + lane < 136) ? s_buf[pos + lane] : 255;
      unsigned int mask = __ballot_sync(0xFFFFFFFF, candidate <= i);
      if (mask != 0) {
        int first = __ffs(mask) - 1;
        b = __shfl_sync(0xFFFFFFFF, candidate, first);
        pos += first + 1;
        break;
      } else {
        pos += 32;
      }
    }
    if (lane == 0) {
      s_poly[i] = s_poly[b];
      s_poly[b] = 1 - 2 * (signs & 1);
      signs >>= 1;
    }
  }
  __syncwarp();

  for (int i = 0; i < 8; i++) {
    regs[i] = s_poly[i * 32 + lane];
  }
}
#else
__device__ __forceinline__ void poly_challenge_shake256_opt(int32_t regs[8],
                                                            uint8_t *s_buf,
                                                            int lane,
                                                            int32_t *s_poly) {
  for (int i = 0; i < 8; i++)
    regs[i] = 0;
  for (int i = 0; i < 8; i++)
    s_poly[i * 32 + lane] = 0;
  __syncwarp();

  if (lane == 0) {
    uint64_t signs = ((uint64_t *)s_buf)[0];
    int pos = 8;
    for (int i = DILITHIUM_N - TAU; i < DILITHIUM_N; ++i) {
      int b;
      do {
        b = s_buf[pos++];
      } while (b > i);
      s_poly[i] = s_poly[b];
      s_poly[b] = 1 - 2 * (signs & 1);
      signs >>= 1;
    }
  }
  __syncwarp();

  for (int i = 0; i < 8; i++) {
    regs[i] = s_poly[i * 32 + lane];
  }
}
#endif
