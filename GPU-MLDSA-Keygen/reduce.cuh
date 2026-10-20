/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <stdint.h>
#include "params.h"

#define MONT (-4186625)   // 2^32 % Q
#define QINV (58728449)   // q^(-1) mod 2^32

__device__ __forceinline__
int32_t montgomery_reduce(int64_t a) {
    int32_t t = (int64_t)(int32_t)a * QINV;
    t = (a - (int64_t)t * Q) >> 32;
    return t;
}

__device__ __forceinline__
int32_t reduce32(int32_t a) {
    int32_t t = (a + (1 << 22)) >> 23;
    t = a - t * Q;
    return t;
}

__device__ __forceinline__
int32_t caddq(int32_t a) {
    a += (a >> 31) & Q;
    return a;
}

__device__ __forceinline__
int32_t freeze(int32_t a) {
    a = reduce32(a);
    a = caddq(a);
    return a;
}

#endif // REDUCE_CUH
