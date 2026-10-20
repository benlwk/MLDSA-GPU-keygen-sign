/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <string.h>

#include "../ref/sign.h"
#include "ref_wrap.h"

/* The reference draws xi (keygen) and rnd (sign) through randombytes();
 * serve the caller-provided bytes, or zeros when none are set. */
static const uint8_t *next_random = NULL;

void randombytes(uint8_t *out, size_t outlen) {
  if (next_random)
    memcpy(out, next_random, outlen);
  else
    memset(out, 0, outlen);
}

size_t ref_publickeybytes(void) { return CRYPTO_PUBLICKEYBYTES; }
size_t ref_secretkeybytes(void) { return CRYPTO_SECRETKEYBYTES; }
size_t ref_signaturebytes(void) { return CRYPTO_BYTES; }

int ref_keypair_from_seed(uint8_t *pk, uint8_t *sk, const uint8_t xi[32]) {
  next_random = xi;
  int ret = crypto_sign_keypair(pk, sk);
  next_random = NULL;
  return ret;
}

int ref_sign(uint8_t *sig, const uint8_t *m, size_t mlen, const uint8_t *sk,
             const uint8_t *rnd) {
  size_t siglen;
  next_random = rnd;
  int ret = crypto_sign_signature(sig, &siglen, m, mlen, NULL, 0, sk);
  next_random = NULL;
  return ret;
}

int ref_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen,
               const uint8_t *pk) {
  return crypto_sign_verify(sig, siglen, m, mlen, NULL, 0, pk);
}
