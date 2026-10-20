/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdint.h>
#include <stddef.h>
#include "params.h"
#include "reduce.cuh"
#include "rounding.cuh"
#include "fips202.cuh"
#include "symmetric.cuh"
#include "ntt.cuh"
#include "poly.cuh"
#include "polyvec.cuh"
#include "packing.cuh"
#include "sign.cuh"

#include "gpu_utils.h"
#include "gpu_private.h"
#include "dilithium_gpu.h"

__global__
void keygen_batch_kernel(const uint8_t * __restrict__ seeds,
                          uint8_t       * __restrict__ pk_out,
                          uint8_t       * __restrict__ sk_out,
                          int            batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch) return;

    dilithium_keygen_device(seeds   + (size_t)tid * SEEDBYTES,
                             pk_out  + (size_t)tid * CRYPTO_PUBLICKEYBYTES,
                             sk_out  + (size_t)tid * CRYPTO_SECRETKEYBYTES);
}

extern "C" void gpu_batch_keygen(const uint8_t *seeds,
                                  uint8_t       *pk_out,
                                  uint8_t       *sk_out,
                                  int            batch,
                                  int            threads_per_block) {
    uint8_t *d_seeds, *d_pk, *d_sk;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)batch * SEEDBYTES));
    CUDA_CHECK(cudaMalloc(&d_pk,    (size_t)batch * CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&d_sk,    (size_t)batch * CRYPTO_SECRETKEYBYTES));

    CUDA_CHECK(cudaMemcpy(d_seeds, seeds, (size_t)batch * SEEDBYTES, cudaMemcpyHostToDevice));

    int blocks = (batch + threads_per_block - 1) / threads_per_block;
    keygen_batch_kernel<<<blocks, threads_per_block>>>(d_seeds, d_pk, d_sk, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pk_out, d_pk, (size_t)batch * CRYPTO_PUBLICKEYBYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sk_out, d_sk, (size_t)batch * CRYPTO_SECRETKEYBYTES, cudaMemcpyDeviceToHost));

    cudaFree(d_seeds); cudaFree(d_pk); cudaFree(d_sk);
}

extern "C" float bench_keygen(BenchCtx *c, int iters) {
    cudaEvent_t s, e; float ms;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++)
        keygen_batch_kernel<<<c->blocks, c->tpb>>>(c->d_seeds, c->d_pk, c->d_sk, c->batch);
    cudaEventRecord(e);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

// seeds in, keygen, keys out, split over n_streams streams with pinned host
// buffers (the setup of Shen et al.)
extern "C" void bench_keygen_e2e_n(BenchCtx *c, int n_streams, int n,
                                   float *out_ms) {
    uint8_t *h_seeds, *h_pk, *h_sk;
    CUDA_CHECK(cudaMallocHost(&h_seeds, (size_t)c->batch * SEEDBYTES));
    CUDA_CHECK(cudaMallocHost(&h_pk, (size_t)c->batch * CRYPTO_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMallocHost(&h_sk, (size_t)c->batch * CRYPTO_SECRETKEYBYTES));
    CUDA_CHECK(cudaMemcpy(h_seeds, c->d_seeds, (size_t)c->batch * SEEDBYTES,
                          cudaMemcpyDeviceToHost));
    cudaStream_t *st = new cudaStream_t[n_streams];
    for (int s = 0; s < n_streams; s++)
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    const int chunk = (c->batch + n_streams - 1) / n_streams;
    for (int it = 0; it < n; it++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int s = 0; s < n_streams; s++) {
            int off = s * chunk;
            int cnt = c->batch - off < chunk ? c->batch - off : chunk;
            if (cnt <= 0) break;
            cudaMemcpyAsync(c->d_seeds + (size_t)off * SEEDBYTES,
                            h_seeds + (size_t)off * SEEDBYTES,
                            (size_t)cnt * SEEDBYTES, cudaMemcpyHostToDevice, st[s]);
            keygen_batch_kernel<<<(cnt + c->tpb - 1) / c->tpb, c->tpb, 0, st[s]>>>(
                c->d_seeds + (size_t)off * SEEDBYTES,
                c->d_pk + (size_t)off * CRYPTO_PUBLICKEYBYTES,
                c->d_sk + (size_t)off * CRYPTO_SECRETKEYBYTES, cnt);
            cudaMemcpyAsync(h_pk + (size_t)off * CRYPTO_PUBLICKEYBYTES,
                            c->d_pk + (size_t)off * CRYPTO_PUBLICKEYBYTES,
                            (size_t)cnt * CRYPTO_PUBLICKEYBYTES, cudaMemcpyDeviceToHost, st[s]);
            cudaMemcpyAsync(h_sk + (size_t)off * CRYPTO_SECRETKEYBYTES,
                            c->d_sk + (size_t)off * CRYPTO_SECRETKEYBYTES,
                            (size_t)cnt * CRYPTO_SECRETKEYBYTES, cudaMemcpyDeviceToHost, st[s]);
        }
        // e0/e1 are on the legacy stream, which waits for the non-blocking
        // streams only through this device-wide sync.
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&out_ms[it], e0, e1);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(st[s]);
    delete[] st;
    cudaFreeHost(h_seeds); cudaFreeHost(h_pk); cudaFreeHost(h_sk);
}

extern "C" void bench_keygen_n(BenchCtx *c, int n, float *out_ms) {
    cudaEvent_t *starts = new cudaEvent_t[n];
    cudaEvent_t *ends   = new cudaEvent_t[n];
    for (int i = 0; i < n; i++) {
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&ends[i]);
    }
    for (int i = 0; i < n; i++) {
        cudaEventRecord(starts[i]);
        keygen_batch_kernel<<<c->blocks, c->tpb>>>(c->d_seeds, c->d_pk, c->d_sk, c->batch);
        cudaEventRecord(ends[i]);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
        cudaEventElapsedTime(&out_ms[i], starts[i], ends[i]);
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(ends[i]);
    }
    delete[] starts;
    delete[] ends;
}
