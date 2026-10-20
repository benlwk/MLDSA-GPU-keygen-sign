/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Thin wrapper around the pq-crystals FIPS 204 reference (ref/), compiled
 * once per parameter set. Randomness is supplied by the caller so the GPU
 * output can be compared byte for byte.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t ref_publickeybytes(void);
size_t ref_secretkeybytes(void);
size_t ref_signaturebytes(void);

/* ML-DSA.KeyGen with seed xi (32 bytes). */
int ref_keypair_from_seed(uint8_t *pk, uint8_t *sk, const uint8_t xi[32]);

/* ML-DSA.Sign, pure, empty context. rnd == NULL selects the deterministic
 * variant (rnd = 0^32). */
int ref_sign(uint8_t *sig, const uint8_t *m, size_t mlen, const uint8_t *sk,
             const uint8_t *rnd);

/* ML-DSA.Verify, pure, empty context. Returns 0 if valid. */
int ref_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
               const uint8_t *pk);

#ifdef __cplusplus
}
#endif
