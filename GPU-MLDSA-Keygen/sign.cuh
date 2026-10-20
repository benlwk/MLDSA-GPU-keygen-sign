/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SIGN_CUH
#define SIGN_CUH

#include <stdint.h>
#include "params.h"
#include "packing.cuh"
#include "polyvec.cuh"
#include "poly.cuh"
#include "fips202.cuh"


/* ML-DSA.KeyGen of one key pair in one thread, from the 32-byte seed xi */
static __device__
int dilithium_keygen_device(const uint8_t seed[SEEDBYTES],
                             uint8_t pk[CRYPTO_PUBLICKEYBYTES],
                             uint8_t sk[CRYPTO_SECRETKEYBYTES]) {
    uint8_t seedbuf[2*SEEDBYTES + CRHBYTES];
    uint8_t tr[TRBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl mat[K];
    polyvecl s1, s1hat;
    polyveck s2, t1, t0;

    for (unsigned int i = 0; i < SEEDBYTES; ++i) seedbuf[i] = seed[i];
    seedbuf[SEEDBYTES+0] = K;
    seedbuf[SEEDBYTES+1] = L;
    shake256(seedbuf, 2*SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES + 2);
    rho      = seedbuf;
    rhoprime = rho + SEEDBYTES;
    key      = rhoprime + CRHBYTES;

    polyvec_matrix_expand(mat, rho);
    polyvecl_uniform_eta(&s1, rhoprime, 0);
    polyveck_uniform_eta(&s2, rhoprime, L);

    s1hat = s1;
    polyvecl_ntt(&s1hat);
    polyvec_matrix_pointwise_montgomery(&t1, mat, &s1hat);
    polyveck_reduce(&t1);
    polyveck_invntt_tomont(&t1);
    polyveck_add(&t1, &t1, &s2);
    polyveck_caddq(&t1);
    polyveck_power2round(&t1, &t0, &t1);
    pack_pk(pk, rho, &t1);

    shake256(tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
    pack_sk(sk, rho, tr, key, &t0, &s1, &s2);
    return 0;
}



#endif // SIGN_CUH
