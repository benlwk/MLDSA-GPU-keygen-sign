/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "dilithium_gpu.h"
#include "params.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_random(uint8_t *buf, size_t len) {
  for (size_t i = 0; i < len; i++)
    buf[i] = (uint8_t)(rand() >> 7);
}

int main(void) {
  srand(11967);
  {
    int dev;
    struct cudaDeviceProp p;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
#ifndef NTT_OPT
#define NTT_OPT 0
#endif
    const char *opt_names[] = {"Baseline",
                               "Radix-4",
                               "Radix-4 SDFS"};
    printf("=== Dilithium%d GPU Benchmark (%s variant) ===\nDevice: %s "
           "(sm_%d%d)\n\n",
           DILITHIUM_MODE, opt_names[NTT_OPT], p.name, p.major, p.minor);
  }

  const int batch_sizes[] = {4096, 8192, 16384, 32768, 65536, 131072};
  const int n_batches = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));
  const int TPB = 128, WARMUP = 2, ITERS = 20, N_STREAMS = 10;
  float *kg_ms = (float *)malloc(ITERS * sizeof(float));
  float *e2e_ms = (float *)malloc(ITERS * sizeof(float));

  /* keygen: kernel only; e2e: including transfers, 10 streams */
  printf("%-8s | %-12s | %-12s\n", "batch", "keygen", "keygen_e2e");
  printf("n_batches: %u\n", n_batches);
  printf("---------|--------------|--------------\n");

  for (int bi = 0; bi < n_batches; bi++) {
    int batch = batch_sizes[bi];

    uint8_t *h_seeds = (uint8_t *)malloc((size_t)batch * SEEDBYTES);
    uint8_t *h_pk = (uint8_t *)malloc((size_t)batch * CRYPTO_PUBLICKEYBYTES);
    uint8_t *h_sk = (uint8_t *)malloc((size_t)batch * CRYPTO_SECRETKEYBYTES);

    fill_random(h_seeds, (size_t)batch * SEEDBYTES);
    gpu_batch_keygen(h_seeds, h_pk, h_sk, batch, TPB);

    BenchCtx *ctx;
    bench_ctx_alloc(&ctx, batch, TPB, h_seeds, h_pk, h_sk);

    /* warm-up */
    bench_keygen(ctx, WARMUP);

    bench_keygen_n(ctx, ITERS, kg_ms);
    bench_keygen_e2e_n(ctx, N_STREAMS, ITERS, e2e_ms);

    for (int i = 0; i < ITERS; i++) {
      double kg = (double)batch / (kg_ms[i] / 1000.0);
      double e2e = (double)batch / (e2e_ms[i] / 1000.0);
      printf("%-8d | %-12.0f | %-12.0f\n", batch, kg, e2e);
    }

    bench_ctx_free(ctx);
    free(h_seeds);
    free(h_pk);
    free(h_sk);
  }
  printf("\nDone.\n");
  free(kg_ms);
  free(e2e_ms);
  return 0;
}
