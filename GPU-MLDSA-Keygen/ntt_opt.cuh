/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Radix-4 NTT (NTT_OPT=1) and radix-4 SDFS NTT (NTT_OPT=2) */
#ifndef NTT_OPT_CUH
#define NTT_OPT_CUH

#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"

#define CT(a,b,z) do {                                       \
    int32_t _t = montgomery_reduce((int64_t)(z)*(b));        \
    (b) = (a) - _t;                                          \
    (a) = (a) + _t;                                          \
} while(0)

#define GS(a,b,z) do {                                       \
    int32_t _t = (a);                                        \
    (a) = _t + (b);                                          \
    (b) = _t - (b);                                          \
    (b) = montgomery_reduce((int64_t)(z)*(b));               \
} while(0)

#define RADIX4_BF(x0, x1, x2, x3, z1, z2, z3) do { \
    int32_t _t0 = montgomery_reduce((int64_t)(z1) * (x2)); \
    int32_t _t1 = montgomery_reduce((int64_t)(z1) * (x3)); \
    int32_t _x0_p = (x0) + _t0; \
    int32_t _x2_p = (x0) - _t0; \
    int32_t _x1_p = (x1) + _t1; \
    int32_t _x3_p = (x1) - _t1; \
    int32_t _t2 = montgomery_reduce((int64_t)(z2) * _x1_p); \
    int32_t _t3 = montgomery_reduce((int64_t)(z3) * _x3_p); \
    (x0) = _x0_p + _t2; \
    (x1) = _x0_p - _t2; \
    (x2) = _x2_p + _t3; \
    (x3) = _x2_p - _t3; \
} while(0)

#define RADIX4_GS(x0, x1, x2, x3, za, zb, zc) do { \
    int32_t _x0_p = (x0) + (x1); \
    int32_t _x1_p = montgomery_reduce((int64_t)(za) * ((x0) - (x1))); \
    int32_t _x2_p = (x2) + (x3); \
    int32_t _x3_p = montgomery_reduce((int64_t)(zb) * ((x2) - (x3))); \
    (x0) = _x0_p + _x2_p; \
    (x2) = montgomery_reduce((int64_t)(zc) * (_x0_p - _x2_p)); \
    (x1) = _x1_p + _x3_p; \
    (x3) = montgomery_reduce((int64_t)(zc) * (_x1_p - _x3_p)); \
} while(0)

__device__ static void ntt_radix4(int32_t a[N]) {
    // Stage 1 (Layer 1 & 2): stride 64
    for (int j = 0; j < 64; ++j) {
        RADIX4_BF(a[j], a[j+64], a[j+128], a[j+192], dev_zetas[1], dev_zetas[2], dev_zetas[3]);
    }
    // Stage 2 (Layer 3 & 4): stride 16
    for (int g = 0; g < 4; ++g) {
        int start = 64 * g;
        int32_t z1 = dev_zetas[4 + g];
        int32_t z2 = dev_zetas[8 + 2*g];
        int32_t z3 = dev_zetas[8 + 2*g + 1];
        for (int j = 0; j < 16; ++j) {
            RADIX4_BF(a[start + j], a[start + j + 16], a[start + j + 32], a[start + j + 48], z1, z2, z3);
        }
    }
    // Stage 3 (Layer 5 & 6): stride 4
    for (int g = 0; g < 16; ++g) {
        int start = 16 * g;
        int32_t z1 = dev_zetas[16 + g];
        int32_t z2 = dev_zetas[32 + 2*g];
        int32_t z3 = dev_zetas[32 + 2*g + 1];
        for (int j = 0; j < 4; ++j) {
            RADIX4_BF(a[start + j], a[start + j + 4], a[start + j + 8], a[start + j + 12], z1, z2, z3);
        }
    }
    // Stage 4 (Layer 7 & 8): stride 1
    for (int g = 0; g < 64; ++g) {
        int start = 4 * g;
        RADIX4_BF(a[start], a[start + 1], a[start + 2], a[start + 3], dev_zetas[64 + g], dev_zetas[128 + 2*g], dev_zetas[128 + 2*g + 1]);
    }
}

__device__ static void invntt_radix4(int32_t a[N]) {
    // Stage 1 (Layer 1 & 2): stride 1
    for (int g = 0; g < 64; ++g) {
        int start = 4 * g;
        RADIX4_GS(a[start], a[start + 1], a[start + 2], a[start + 3],
                  -dev_zetas[255 - 2*g], -dev_zetas[254 - 2*g], -dev_zetas[127 - g]);
    }
    // Stage 2 (Layer 3 & 4): stride 4
    for (int g = 0; g < 16; ++g) {
        int start = 16 * g;
        int32_t za = -dev_zetas[63 - 2*g];
        int32_t zb = -dev_zetas[62 - 2*g];
        int32_t zc = -dev_zetas[31 - g];
        for (int j = 0; j < 4; ++j) {
            RADIX4_GS(a[start + j], a[start + j + 4], a[start + j + 8], a[start + j + 12], za, zb, zc);
        }
    }
    // Stage 3 (Layer 5 & 6): stride 16
    for (int g = 0; g < 4; ++g) {
        int start = 64 * g;
        int32_t za = -dev_zetas[15 - 2*g];
        int32_t zb = -dev_zetas[14 - 2*g];
        int32_t zc = -dev_zetas[7 - g];
        for (int j = 0; j < 16; ++j) {
            RADIX4_GS(a[start + j], a[start + j + 16], a[start + j + 32], a[start + j + 48], za, zb, zc);
        }
    }
    // Stage 4 (Layer 7 & 8): stride 64
    int32_t za = -dev_zetas[3];
    int32_t zb = -dev_zetas[2];
    int32_t zc = -dev_zetas[1];
    for (int j = 0; j < 64; ++j) {
        RADIX4_GS(a[j], a[j + 64], a[j + 128], a[j + 192], za, zb, zc);
    }

    // final scaling factor: f = mont^2/256 = 41978
    const int32_t f = 41978;
    for (int i = 0; i < N; ++i) {
        a[i] = montgomery_reduce((int64_t)f * a[i]);
    }
}

/* SDFS: 16 coefficients at a time in registers, two radix-4 stages each */
__device__ static void ntt_radix4_sdfs(int32_t a[N]) {
    // Phase A: Stage 1 & 2 (stride-16 groups)
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int32_t b[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) b[i] = a[k + i*16];

        for (int i = 0; i < 4; ++i) {
            RADIX4_BF(b[i], b[i+4], b[i+8], b[i+12], dev_zetas[1], dev_zetas[2], dev_zetas[3]);
        }

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            RADIX4_BF(b[4*g], b[4*g+1], b[4*g+2], b[4*g+3],
                      dev_zetas[4 + g], dev_zetas[8 + 2*g], dev_zetas[8 + 2*g + 1]);
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) a[k + i*16] = b[i];
    }

    // Phase B: Stage 3 & 4 (consecutive 16-element blocks)
    #pragma unroll
    for (int g = 0; g < 16; ++g) {
        int32_t b[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) b[i] = a[g*16 + i];

        int32_t z1 = dev_zetas[16 + g];
        int32_t z2 = dev_zetas[32 + 2*g];
        int32_t z3 = dev_zetas[32 + 2*g + 1];
        for (int i = 0; i < 4; ++i) {
            RADIX4_BF(b[i], b[i+4], b[i+8], b[i+12], z1, z2, z3);
        }

        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            int idx = 4*g + h;
            RADIX4_BF(b[4*h], b[4*h+1], b[4*h+2], b[4*h+3],
                      dev_zetas[64 + idx], dev_zetas[128 + 2*idx], dev_zetas[128 + 2*idx + 1]);
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) a[g*16 + i] = b[i];
    }
}

__device__ static void invntt_radix4_sdfs(int32_t a[N]) {
    // Phase A: Stage 1 & 2 (consecutive 16-element blocks)
    #pragma unroll
    for (int g = 0; g < 16; ++g) {
        int32_t b[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) b[i] = a[g*16 + i];

        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            int idx = 4*g + h;
            RADIX4_GS(b[4*h], b[4*h+1], b[4*h+2], b[4*h+3],
                      -dev_zetas[255 - 2*idx], -dev_zetas[254 - 2*idx], -dev_zetas[127 - idx]);
        }

        int32_t za = -dev_zetas[63 - 2*g];
        int32_t zb = -dev_zetas[62 - 2*g];
        int32_t zc = -dev_zetas[31 - g];
        for (int i = 0; i < 4; ++i) {
            RADIX4_GS(b[i], b[i+4], b[i+8], b[i+12], za, zb, zc);
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) a[g*16 + i] = b[i];
    }

    // Phase B: Stage 3 & 4 (stride-16 groups)
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int32_t b[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) b[i] = a[k + i*16];

        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            RADIX4_GS(b[4*h], b[4*h+1], b[4*h+2], b[4*h+3],
                      -dev_zetas[15 - 2*h], -dev_zetas[14 - 2*h], -dev_zetas[7 - h]);
        }

        int32_t za = -dev_zetas[3];
        int32_t zb = -dev_zetas[2];
        int32_t zc = -dev_zetas[1];
        for (int i = 0; i < 4; ++i) {
            RADIX4_GS(b[i], b[i+4], b[i+8], b[i+12], za, zb, zc);
        }

        const int32_t f = 41978;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            b[i] = montgomery_reduce((int64_t)f * b[i]);
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) a[k + i*16] = b[i];
    }
}

__device__ __forceinline__ void poly_ntt_radix4(poly *p) { ntt_radix4(p->coeffs); }
__device__ __forceinline__ void poly_invntt_radix4(poly *p) { invntt_radix4(p->coeffs); }

__device__ __forceinline__ void poly_ntt_radix4_sdfs(poly *p) { ntt_radix4_sdfs(p->coeffs); }
__device__ __forceinline__ void poly_invntt_radix4_sdfs(poly *p) { invntt_radix4_sdfs(p->coeffs); }

#endif /* NTT_OPT_CUH */