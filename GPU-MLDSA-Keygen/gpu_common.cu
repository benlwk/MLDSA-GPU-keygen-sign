/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdlib.h>
#include <string.h>
#include "params.h"
#include "gpu_utils.h"
#include "gpu_private.h"
#include "dilithium_gpu.h"

extern "C" void bench_ctx_alloc(BenchCtx **ctx_out,
                                 int batch, int tpb,
                                 const uint8_t *h_seeds,
                                 const uint8_t *h_pk,
                                 const uint8_t *h_sk)
{
    BenchCtx *c = (BenchCtx*)malloc(sizeof(BenchCtx));
    c->batch   = batch;
    c->tpb     = tpb;
    c->blocks  = (batch + tpb - 1) / tpb;

    CUDA_CHECK(cudaMalloc(&c->d_seeds,   (size_t)batch * SEEDBYTES));
    CUDA_CHECK(cudaMalloc(&c->d_pk,      (size_t)batch * CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&c->d_sk,      (size_t)batch * CRYPTO_SECRETKEYBYTES));

    CUDA_CHECK(cudaMemcpy(c->d_seeds, h_seeds, (size_t)batch * SEEDBYTES,             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_pk,    h_pk,    (size_t)batch * CRYPTO_PUBLICKEYBYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->d_sk,    h_sk,    (size_t)batch * CRYPTO_SECRETKEYBYTES, cudaMemcpyHostToDevice));

    *ctx_out = c;
}

extern "C" void bench_ctx_free(BenchCtx *c) {
    cudaFree(c->d_seeds); cudaFree(c->d_pk);  cudaFree(c->d_sk);
    free(c);
}
