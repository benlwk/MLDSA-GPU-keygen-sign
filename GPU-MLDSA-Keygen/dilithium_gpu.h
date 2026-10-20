/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef DILITHIUM_GPU_H
#define DILITHIUM_GPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_batch_keygen(const uint8_t *seeds,
                       uint8_t       *pk_out,
                       uint8_t       *sk_out,
                       int            batch,
                       int            threads_per_block);

/* opaque for C code; the definition is in gpu_private.h */
#ifndef GPU_PRIVATE_H
struct BenchCtx;
typedef struct BenchCtx BenchCtx;
#endif

void  bench_ctx_alloc(BenchCtx **ctx_out,
                       int batch, int tpb,
                       const uint8_t *h_seeds,
                       const uint8_t *h_pk,
                       const uint8_t *h_sk);

void  bench_ctx_free(BenchCtx *c);

float bench_keygen (BenchCtx *c, int iters);

/* kernel time of each of n launches in out_ms[0..n-1] (ms) */
void bench_keygen_n  (BenchCtx *c, int n, float *out_ms);

/* the same, including seed upload and key download, over n_streams streams */
void bench_keygen_e2e_n(BenchCtx *c, int n_streams, int n, float *out_ms);

#ifdef __cplusplus
}
#endif

#endif // DILITHIUM_GPU_H
