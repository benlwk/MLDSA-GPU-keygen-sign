/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "params.h"
#include "dilithium_gpu.h"
#include "../ref/config.h"
#include "../ref/api.h"
#include "../ref/sign.h"

void randombytes(uint8_t *out, size_t outlen);

// the reference draws xi from randombytes()
static const uint8_t *global_seed = NULL;
static const uint8_t *global_rnd = NULL;

void randombytes(uint8_t *out, size_t outlen) {
    if (outlen == SEEDBYTES && global_seed != NULL) {
        memcpy(out, global_seed, SEEDBYTES);
    } else if (outlen == RNDBYTES && global_rnd != NULL) {
        memcpy(out, global_rnd, RNDBYTES);
    } else {
        memset(out, 0, outlen);
    }
}

#define MSG_LEN 59

int main(int argc, char **argv) {
    int batch = 100;
    int tpb = 128;
    unsigned int seed = 42;

    if (argc > 1) {
        batch = atoi(argv[1]);
    }
    if (argc > 2) {
        tpb = atoi(argv[2]);
    }
    if (argc > 3) {
        seed = (unsigned int)strtoul(argv[3], NULL, 10);
    }

#ifndef NTT_OPT
#define NTT_OPT 0
#endif
    const char *opt_names[] = {
        "Baseline",
        "Radix-4",
        "Radix-4 SDFS"
    };
    printf("Starting verification for DILITHIUM_MODE=%d (%s variant) (batch=%d, tpb=%d, seed=%u)\n",
           DILITHIUM_MODE, opt_names[NTT_OPT], batch, tpb, seed);

    srand(seed);

    uint8_t *h_seeds    = (uint8_t*)malloc((size_t)batch * SEEDBYTES);
    uint8_t *h_pk_gpu   = (uint8_t*)malloc((size_t)batch * CRYPTO_PUBLICKEYBYTES);
    uint8_t *h_sk_gpu   = (uint8_t*)malloc((size_t)batch * CRYPTO_SECRETKEYBYTES);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < SEEDBYTES; j++) h_seeds[i * SEEDBYTES + j] = rand() & 0xFF;
    }

    printf("Running GPU batch keygen...\n");
    gpu_batch_keygen(h_seeds, h_pk_gpu, h_sk_gpu, batch, tpb);

    printf("   Verifying GPU keys against CPU keygen...\n");
    for (int i = 0; i < batch; i++) {
        global_seed = h_seeds + (size_t)i * SEEDBYTES;
        uint8_t pk_cpu[CRYPTO_PUBLICKEYBYTES];
        uint8_t sk_cpu[CRYPTO_SECRETKEYBYTES];
        
        crypto_sign_keypair(pk_cpu, sk_cpu);

        if (memcmp(pk_cpu, h_pk_gpu + (size_t)i * CRYPTO_PUBLICKEYBYTES, CRYPTO_PUBLICKEYBYTES) != 0) {
            fprintf(stderr, "   [ERROR] Public key mismatch at batch index %d\n", i);
            fprintf(stderr, "   CPU pk: ");
            for (int k = 0; k < 16; k++) fprintf(stderr, "%02x", pk_cpu[k]);
            fprintf(stderr, "\n   GPU pk: ");
            for (int k = 0; k < 16; k++) fprintf(stderr, "%02x", h_pk_gpu[(size_t)i * CRYPTO_PUBLICKEYBYTES + k]);
            fprintf(stderr, "\n");
            return 1;
        }

        if (memcmp(sk_cpu, h_sk_gpu + (size_t)i * CRYPTO_SECRETKEYBYTES, CRYPTO_SECRETKEYBYTES) != 0) {
            fprintf(stderr, "   [ERROR] Secret key mismatch at batch index %d\n", i);
            return 1;
        }
    }
    printf("   [PASS] Key generation matches CPU ref.\n");

    printf("ALL TESTS PASSED\n");

    free(h_seeds); free(h_pk_gpu); free(h_sk_gpu);

    return 0;
}
