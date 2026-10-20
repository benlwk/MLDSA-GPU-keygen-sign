/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef POLYVEC_CUH
#define POLYVEC_CUH

#include <stdint.h>
#include "params.h"
#include "poly.cuh"

typedef struct { poly vec[L]; } polyvecl;
typedef struct { poly vec[K]; } polyveck;

/* ---- polyvecl ---- */

static __device__ __noinline__
void polyvecl_uniform_eta(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    for (unsigned int i = 0; i < L; ++i)
        poly_uniform_eta(&v->vec[i], seed, nonce++);
}

static __device__ __noinline__
void polyvecl_uniform_gamma1(polyvecl *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    for (unsigned int i = 0; i < L; ++i)
        poly_uniform_gamma1(&v->vec[i], seed, L*nonce + i);
}

__device__ __forceinline__
void polyvecl_reduce(polyvecl *v) {
    for (unsigned int i = 0; i < L; ++i) poly_reduce(&v->vec[i]);
}

__device__ __forceinline__
void polyvecl_add(polyvecl *w, const polyvecl *u, const polyvecl *v) {
    for (unsigned int i = 0; i < L; ++i) poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
}

__device__ __forceinline__
void polyvecl_ntt(polyvecl *v) {
    for (unsigned int i = 0; i < L; ++i) poly_ntt(&v->vec[i]);
}

__device__ __forceinline__
void polyvecl_invntt_tomont(polyvecl *v) {
    for (unsigned int i = 0; i < L; ++i) poly_invntt_tomont(&v->vec[i]);
}

__device__ __forceinline__
void polyvecl_pointwise_poly_montgomery(polyvecl *r, const poly *a, const polyvecl *v) {
    for (unsigned int i = 0; i < L; ++i)
        poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
}

static __device__ __noinline__
void polyvecl_pointwise_acc_montgomery(poly *w, const polyvecl *u, const polyvecl *v) {
    poly t;
    poly_pointwise_montgomery(w, &u->vec[0], &v->vec[0]);
    for (unsigned int i = 1; i < L; ++i) {
        poly_pointwise_montgomery(&t, &u->vec[i], &v->vec[i]);
        poly_add(w, w, &t);
    }
}

static __device__ __noinline__
int polyvecl_chknorm(const polyvecl *v, int32_t bound) {
    for (unsigned int i = 0; i < L; ++i)
        if (poly_chknorm(&v->vec[i], bound)) return 1;
    return 0;
}

/* ---- polyveck ---- */

static __device__ __noinline__
void polyveck_uniform_eta(polyveck *v, const uint8_t seed[CRHBYTES], uint16_t nonce) {
    for (unsigned int i = 0; i < K; ++i)
        poly_uniform_eta(&v->vec[i], seed, nonce++);
}

__device__ __forceinline__
void polyveck_reduce(polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_reduce(&v->vec[i]);
}

__device__ __forceinline__
void polyveck_caddq(polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_caddq(&v->vec[i]);
}

__device__ __forceinline__
void polyveck_add(polyveck *w, const polyveck *u, const polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_add(&w->vec[i], &u->vec[i], &v->vec[i]);
}

__device__ __forceinline__
void polyveck_sub(polyveck *w, const polyveck *u, const polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_sub(&w->vec[i], &u->vec[i], &v->vec[i]);
}

__device__ __forceinline__
void polyveck_shiftl(polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_shiftl(&v->vec[i]);
}

__device__ __forceinline__
void polyveck_ntt(polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_ntt(&v->vec[i]);
}

__device__ __forceinline__
void polyveck_invntt_tomont(polyveck *v) {
    for (unsigned int i = 0; i < K; ++i) poly_invntt_tomont(&v->vec[i]);
}

__device__ __forceinline__
void polyveck_pointwise_poly_montgomery(polyveck *r, const poly *a, const polyveck *v) {
    for (unsigned int i = 0; i < K; ++i)
        poly_pointwise_montgomery(&r->vec[i], a, &v->vec[i]);
}

static __device__ __noinline__
int polyveck_chknorm(const polyveck *v, int32_t bound) {
    for (unsigned int i = 0; i < K; ++i)
        if (poly_chknorm(&v->vec[i], bound)) return 1;
    return 0;
}

__device__ __forceinline__
void polyveck_power2round(polyveck *v1, polyveck *v0, const polyveck *v) {
    for (unsigned int i = 0; i < K; ++i)
        poly_power2round(&v1->vec[i], &v0->vec[i], &v->vec[i]);
}

__device__ __forceinline__
void polyveck_decompose(polyveck *v1, polyveck *v0, const polyveck *v) {
    for (unsigned int i = 0; i < K; ++i)
        poly_decompose(&v1->vec[i], &v0->vec[i], &v->vec[i]);
}

static __device__ __noinline__
unsigned int polyveck_make_hint(polyveck *h, const polyveck *v0, const polyveck *v1) {
    unsigned int s = 0;
    for (unsigned int i = 0; i < K; ++i)
        s += poly_make_hint(&h->vec[i], &v0->vec[i], &v1->vec[i]);
    return s;
}

__device__ __forceinline__
void polyveck_use_hint(polyveck *w, const polyveck *u, const polyveck *h) {
    for (unsigned int i = 0; i < K; ++i)
        poly_use_hint(&w->vec[i], &u->vec[i], &h->vec[i]);
}

static __device__ __noinline__
void polyveck_pack_w1(uint8_t r[K*POLYW1_PACKEDBYTES], const polyveck *w1) {
    for (unsigned int i = 0; i < K; ++i)
        polyw1_pack(&r[i*POLYW1_PACKEDBYTES], &w1->vec[i]);
}

/* ---- Matrix operations ---- */

static __device__ __noinline__
void polyvec_matrix_expand(polyvecl mat[K], const uint8_t rho[SEEDBYTES]) {
    for (unsigned int i = 0; i < K; ++i)
        for (unsigned int j = 0; j < L; ++j)
            poly_uniform(&mat[i].vec[j], rho, (i << 8) + j);
}

static __device__ __noinline__
void polyvec_matrix_pointwise_montgomery(polyveck *t,
                                          const polyvecl mat[K],
                                          const polyvecl *v) {
    for (unsigned int i = 0; i < K; ++i)
        polyvecl_pointwise_acc_montgomery(&t->vec[i], &mat[i], v);
}

#endif // POLYVEC_CUH
