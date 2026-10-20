/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Isolated comparison of the NTTs of the signing kernels:
 *   smem      the shared-memory radix-2 NTT of the multi-kernel code (128
 *             threads per polynomial, ntt_radix2_inner / intt_radix2_inner)
 *   warp      the warp-shuffle NTT of the fused kernel (32 threads per
 *             polynomial, twiddles from __constant__ memory)
 *   warp_s    the warp-shuffle NTT with twiddles in shared memory
 *   warp_g    the warp-shuffle NTT with twiddles through the read-only cache
 *   warp_x    the exchange-then-butterfly warp NTT (ntt_warpx), shared twiddles
 * Each kernel loads a polynomial from global memory, applies REPS transforms
 * in registers, and stores it. REPS = 1 includes the global memory traffic;
 * larger REPS isolates the transform. All variants are checked against each
 * other (mod q) before timing.
 * Usage: bench_ntt_compare [n_polys] (default 262144)
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ntt.cuh"
#include "params.h"
#include "util.cuh"

enum { Z_CONST = 0, Z_SHARED = 1, Z_GLOBAL = 2 };

template <int ZS>
__device__ __forceinline__ int32_t zeta_at(int idx, const int32_t *s_z,
                                           const int32_t *g_z) {
  if (ZS == Z_SHARED)
    return s_z[idx];
  if (ZS == Z_GLOBAL)
    return __ldg(&g_z[idx]);
  return c_zetas[idx];
}

// Same algorithm as ntt_inner_shuffle_notranspose, twiddle source as template.
template <int ZS>
__device__ __forceinline__ void ntt_warp(int32_t r[8], const int32_t *s_z,
                                         const int32_t *g_z) {
  const int lane = threadIdx.x & 0x1f;
  auto butt = [](int32_t &a, int32_t &b, int32_t z) {
    int32_t t = gpu_montgomery_multiply(z, b);
    b = a - t;
    a = a + t;
  };
  const int32_t z1 = zeta_at<ZS>(1, s_z, g_z);
  butt(r[0], r[4], z1); butt(r[1], r[5], z1); butt(r[2], r[6], z1); butt(r[3], r[7], z1);
  const int32_t z2 = zeta_at<ZS>(2, s_z, g_z), z3 = zeta_at<ZS>(3, s_z, g_z);
  butt(r[0], r[2], z2); butt(r[1], r[3], z2); butt(r[4], r[6], z3); butt(r[5], r[7], z3);
  butt(r[0], r[1], zeta_at<ZS>(4, s_z, g_z)); butt(r[2], r[3], zeta_at<ZS>(5, s_z, g_z));
  butt(r[4], r[5], zeta_at<ZS>(6, s_z, g_z)); butt(r[6], r[7], zeta_at<ZS>(7, s_z, g_z));
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int distance = 16 >> i;
    const int base = (8 << i) + (lane >> (5 - i));
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int32_t zeta = zeta_at<ZS>(base + (j << i), s_z, g_z);
      int32_t other = __shfl_xor_sync(0xFFFFFFFF, r[j], distance);
      if ((lane & distance) == 0)
        r[j] = r[j] + gpu_montgomery_multiply(zeta, other);
      else
        r[j] = other - gpu_montgomery_multiply(zeta, r[j]);
    }
  }
}

// Same algorithm as invntt_inner_shuffle_notranspose, twiddle source as template.
template <int ZS>
__device__ __forceinline__ void intt_warp(int32_t r[8], const int32_t *s_z,
                                          const int32_t *g_z) {
  const int lane = threadIdx.x & 0x1f;
  auto butt = [](int32_t &a, int32_t &b, int32_t z) {
    int32_t t = a;
    a = t + b;
    b = gpu_montgomery_multiply(z, t - b);
  };
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int distance = 1 << i;
    const int base = ((256 >> i) - 1) - (lane >> (i + 1));
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int32_t other = __shfl_xor_sync(0xFFFFFFFF, r[j], distance);
      const int32_t zeta = -zeta_at<ZS>(base - (j << (4 - i)), s_z, g_z);
      if ((lane & distance) == 0)
        r[j] = r[j] + other;
      else
        r[j] = gpu_montgomery_multiply(zeta, other - r[j]);
    }
  }
  butt(r[0], r[1], -zeta_at<ZS>(7, s_z, g_z)); butt(r[2], r[3], -zeta_at<ZS>(6, s_z, g_z));
  butt(r[4], r[5], -zeta_at<ZS>(5, s_z, g_z)); butt(r[6], r[7], -zeta_at<ZS>(4, s_z, g_z));
  const int32_t z3 = -zeta_at<ZS>(3, s_z, g_z), z2 = -zeta_at<ZS>(2, s_z, g_z);
  butt(r[0], r[2], z3); butt(r[1], r[3], z3); butt(r[4], r[6], z2); butt(r[5], r[7], z2);
  const int32_t z1 = -zeta_at<ZS>(1, s_z, g_z);
  butt(r[0], r[4], z1); butt(r[1], r[5], z1); butt(r[2], r[6], z1); butt(r[3], r[7], z1);
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = gpu_montgomery_multiply(41978, r[i]);
}

// 128 threads, one polynomial per block, shared-memory radix-2 NTT.
template <bool INV>
__global__ void k_smem(int32_t *g, int reps) {
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  __shared__ int32_t s_zetas[DILITHIUM_N];
  s_zetas[threadIdx.x] = c_zetas[threadIdx.x];
  s_zetas[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();
  int32_t *p = g + (size_t)blockIdx.x * DILITHIUM_N;
  int32_t r0 = p[threadIdx.x], r1 = p[128 + threadIdx.x];
  for (int k = 0; k < reps; k++) {
    if (INV)
      intt_radix2_inner(r0, r1, s_ntt, s_zetas);
    else
      ntt_radix2_inner(r0, r1, s_ntt, s_zetas);
  }
  p[threadIdx.x] = r0;
  p[128 + threadIdx.x] = r1;
}

// 128 threads, four polynomials per block (one per warp), warp-shuffle NTT.
// ZS < 0: the library function (constant-memory twiddles), as in signing.
template <int ZS, bool INV>
__global__ void k_warp(int32_t *g, int reps, const int32_t *g_z) {
  __shared__ int32_t s_z[DILITHIUM_N];
  if (ZS == Z_SHARED) {
    s_z[threadIdx.x] = c_zetas[threadIdx.x];
    s_z[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
    __syncthreads();
  }
  const int lane = threadIdx.x & 31;
  int32_t *p = g + ((size_t)blockIdx.x * 4 + threadIdx.x / 32) * DILITHIUM_N;
  int32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    r[i] = p[i * 32 + lane];
  for (int k = 0; k < reps; k++) {
    if (ZS < 0) {
      if (INV)
        invntt_inner_shuffle_notranspose(r);
      else
        ntt_inner_shuffle_notranspose(r);
    } else {
      if (INV)
        intt_warp<ZS < 0 ? 0 : ZS>(r, s_z, g_z);
      else
        ntt_warp<ZS < 0 ? 0 : ZS>(r, s_z, g_z);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++)
    p[i * 32 + lane] = r[i];
}

// 128 threads, four polynomials per block, exchange-then-butterfly warp NTT
// (ntt_warpx / invntt_warpx, twiddles in shared memory). NTT-domain data is
// read and written in natural order through register layout B.
template <bool INV>
__global__ void k_warpx(int32_t *g, int reps, const int32_t *) {
  __shared__ int32_t s_z[DILITHIUM_N];
  s_z[threadIdx.x] = c_zetas[threadIdx.x];
  s_z[128 + threadIdx.x] = c_zetas[128 + threadIdx.x];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  int32_t *p = g + ((size_t)blockIdx.x * 4 + threadIdx.x / 32) * DILITHIUM_N;
  int32_t r[8];
  if (INV) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int2 v = reinterpret_cast<const int2 *>(p)[32 * q + lane];
      r[2 * q] = v.x;
      r[2 * q + 1] = v.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++)
      r[i] = p[i * 32 + lane];
  }
  // reps > 1 only times the transform (the output is then not meaningful,
  // as for the other variants); reps = 1 is checked
  for (int k = 0; k < reps; k++) {
    if (INV)
      invntt_warpx(r, s_z);
    else
      ntt_warpx(r, s_z);
  }
  if (INV) { // INTT output: layout A
#pragma unroll
    for (int i = 0; i < 8; i++)
      p[i * 32 + lane] = r[i];
  } else {
#pragma unroll
    for (int q = 0; q < 4; q++)
      reinterpret_cast<int2 *>(p)[32 * q + lane] = make_int2(r[2 * q], r[2 * q + 1]);
  }
}

struct Variant {
  const char *name;
  void (*launch)(int32_t *, size_t, int, bool, const int32_t *);
};

template <bool INV> void l_smem(int32_t *g, size_t n, int reps, const int32_t *) {
  k_smem<INV><<<n, 128>>>(g, reps);
}
template <int ZS, bool INV> void l_warp(int32_t *g, size_t n, int reps, const int32_t *gz) {
  k_warp<ZS, INV><<<n / 4, 128>>>(g, reps, gz);
}
#define DISPATCH(F) [](int32_t *g, size_t n, int r, bool inv, const int32_t *z) { \
    if (inv) F<true>(g, n, r, z); else F<false>(g, n, r, z); }
template <bool INV> void l_warpx(int32_t *g, size_t n, int reps, const int32_t *gz) {
  k_warpx<INV><<<n / 4, 128>>>(g, reps, gz);
}
#define DISPATCH_W(ZS)[](int32_t *g, size_t n, int r, bool inv, const int32_t *z) { \
    if (inv) l_warp<ZS, true>(g, n, r, z); else l_warp<ZS, false>(g, n, r, z); }

int main(int argc, char **argv) {
  const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 262144;
  const Variant vars[] = {{"smem", DISPATCH(l_smem)},
                          {"warp", DISPATCH_W(-1)},
                          {"warp_s", DISPATCH_W(Z_SHARED)},
                          {"warp_g", DISPATCH_W(Z_GLOBAL)},
                          {"warp_x", DISPATCH(l_warpx)}};
  const int nv = sizeof(vars) / sizeof(vars[0]);

  std::vector<int32_t> h_in(n * DILITHIUM_N), h_ref, h_out(n * DILITHIUM_N);
  srand(7);
  for (auto &v : h_in)
    v = (int32_t)(rand() % DILITHIUM_Q) - DILITHIUM_Q / 2;
  int32_t *d, *d_z;
  CHECK_CUDA_ERROR(cudaMalloc(&d, n * DILITHIUM_N * sizeof(int32_t)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_z, DILITHIUM_N * sizeof(int32_t)));
  CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(d_z, c_zetas, DILITHIUM_N * sizeof(int32_t)));

  // correctness: every variant equals the first one (mod q), forward and inverse
  for (int inv = 0; inv < 2; inv++) {
    for (int v = 0; v < nv; v++) {
      CHECK_CUDA_ERROR(cudaMemcpy(d, h_in.data(), n * DILITHIUM_N * 4, cudaMemcpyHostToDevice));
      vars[v].launch(d, n, 1, inv, d_z);
      CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d, n * DILITHIUM_N * 4, cudaMemcpyDeviceToHost));
      if (v == 0) {
        h_ref = h_out;
        continue;
      }
      size_t bad = 0;
      for (size_t i = 0; i < h_out.size(); i++)
        if (((int64_t)h_out[i] - h_ref[i]) % DILITHIUM_Q != 0)
          bad++;
      printf("check %s %s vs smem: %s\n", inv ? "INTT" : "NTT", vars[v].name,
             bad ? "MISMATCH" : "ok");
      if (bad)
        return 1;
    }
  }

  printf("transform,variant,reps,n_polys,median_ms,ns_per_transform,Mtransforms_per_s\n");
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int inv = 0; inv < 2; inv++)
    for (int reps : {1, 8}) {
      for (int v = 0; v < nv; v++) {
        std::vector<float> t;
        for (int it = 0; it < 22; it++) {
          cudaEventRecord(e0);
          vars[v].launch(d, n, reps, inv, d_z);
          cudaEventRecord(e1);
          cudaEventSynchronize(e1);
          float ms;
          cudaEventElapsedTime(&ms, e0, e1);
          if (it >= 2)
            t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        float med = t[t.size() / 2];
        double per = med * 1e6 / ((double)n * reps);
        printf("%s,%s,%d,%zu,%.4f,%.3f,%.1f\n", inv ? "INTT" : "NTT", vars[v].name, reps, n,
               med, per, 1e3 / per);
      }
    }
  CHECK_CUDA_ERROR(cudaGetLastError());
  cudaFree(d);
  cudaFree(d_z);
  return 0;
}
