/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef PACKING_CUH
#define PACKING_CUH

#include <stdint.h>
#include "params.h"
#include "poly.cuh"
#include "polyvec.cuh"

__device__ __forceinline__
void pack_pk(uint8_t pk[CRYPTO_PUBLICKEYBYTES],
             const uint8_t rho[SEEDBYTES],
             const polyveck *t1) {
    for (unsigned int i = 0; i < SEEDBYTES; ++i) pk[i] = rho[i];
    for (unsigned int i = 0; i < K; ++i)
        polyt1_pack(pk + SEEDBYTES + i*POLYT1_PACKEDBYTES, &t1->vec[i]);
}

__device__ __forceinline__
void unpack_pk(uint8_t rho[SEEDBYTES], polyveck *t1,
               const uint8_t pk[CRYPTO_PUBLICKEYBYTES]) {
    for (unsigned int i = 0; i < SEEDBYTES; ++i) rho[i] = pk[i];
    for (unsigned int i = 0; i < K; ++i)
        polyt1_unpack(&t1->vec[i], pk + SEEDBYTES + i*POLYT1_PACKEDBYTES);
}

__device__ __forceinline__
void pack_sk(uint8_t sk[CRYPTO_SECRETKEYBYTES],
             const uint8_t rho[SEEDBYTES],
             const uint8_t tr[TRBYTES],
             const uint8_t key[SEEDBYTES],
             const polyveck *t0,
             const polyvecl *s1,
             const polyveck *s2) {
    unsigned int i;
    uint8_t *ptr = sk;
    for (i = 0; i < SEEDBYTES; ++i) *ptr++ = rho[i];
    for (i = 0; i < SEEDBYTES; ++i) *ptr++ = key[i];
    for (i = 0; i < TRBYTES;   ++i) *ptr++ = tr[i];
    for (i = 0; i < L; ++i) { polyeta_pack(ptr, &s1->vec[i]); ptr += POLYETA_PACKEDBYTES; }
    for (i = 0; i < K; ++i) { polyeta_pack(ptr, &s2->vec[i]); ptr += POLYETA_PACKEDBYTES; }
    for (i = 0; i < K; ++i) { polyt0_pack(ptr,  &t0->vec[i]); ptr += POLYT0_PACKEDBYTES;  }
}

__device__ __forceinline__
void unpack_sk(uint8_t rho[SEEDBYTES],
               uint8_t tr[TRBYTES],
               uint8_t key[SEEDBYTES],
               polyveck *t0,
               polyvecl *s1,
               polyveck *s2,
               const uint8_t sk[CRYPTO_SECRETKEYBYTES]) {
    unsigned int i;
    const uint8_t *ptr = sk;
    for (i = 0; i < SEEDBYTES; ++i) rho[i] = *ptr++;
    for (i = 0; i < SEEDBYTES; ++i) key[i] = *ptr++;
    for (i = 0; i < TRBYTES;   ++i) tr[i]  = *ptr++;
    for (i = 0; i < L; ++i) { polyeta_unpack(&s1->vec[i], ptr); ptr += POLYETA_PACKEDBYTES; }
    for (i = 0; i < K; ++i) { polyeta_unpack(&s2->vec[i], ptr); ptr += POLYETA_PACKEDBYTES; }
    for (i = 0; i < K; ++i) { polyt0_unpack( &t0->vec[i], ptr); ptr += POLYT0_PACKEDBYTES;  }
}

__device__ __forceinline__
void pack_sig(uint8_t sig[CRYPTO_BYTES],
              const uint8_t c[CTILDEBYTES],
              const polyvecl *z,
              const polyveck *h) {
    unsigned int i, j, k;
    for (i = 0; i < CTILDEBYTES; ++i) sig[i] = c[i];
    for (i = 0; i < L; ++i)
        polyz_pack(sig + CTILDEBYTES + i*POLYZ_PACKEDBYTES, &z->vec[i]);
    uint8_t *hints = sig + CTILDEBYTES + L*POLYZ_PACKEDBYTES;
    for (i = 0; i < OMEGA + K; ++i) hints[i] = 0;
    k = 0;
    for (i = 0; i < K; ++i) {
        for (j = 0; j < N; ++j)
            if (h->vec[i].coeffs[j]) hints[k++] = j;
        hints[OMEGA + i] = k;
    }
}

__device__ __forceinline__
int unpack_sig(uint8_t c[CTILDEBYTES],
               polyvecl *z,
               polyveck *h,
               const uint8_t sig[CRYPTO_BYTES]) {
    unsigned int i, j, k;
    for (i = 0; i < CTILDEBYTES; ++i) c[i] = sig[i];
    for (i = 0; i < L; ++i)
        polyz_unpack(&z->vec[i], sig + CTILDEBYTES + i*POLYZ_PACKEDBYTES);
    const uint8_t *hints = sig + CTILDEBYTES + L*POLYZ_PACKEDBYTES;
    k = 0;
    for (i = 0; i < K; ++i) {
        for (j = 0; j < N; ++j) h->vec[i].coeffs[j] = 0;
        if (hints[OMEGA + i] < k || hints[OMEGA + i] > OMEGA) return 1;
        for (j = k; j < hints[OMEGA + i]; ++j) {
            if (j > k && hints[j] <= hints[j-1]) return 1;
            h->vec[i].coeffs[hints[j]] = 1;
        }
        k = hints[OMEGA + i];
    }
    for (j = k; j < OMEGA; ++j) if (hints[j]) return 1;
    return 0;
}

#endif // PACKING_CUH
