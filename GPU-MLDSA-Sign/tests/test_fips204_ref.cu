/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Cross-check the GPU implementation against the pq-crystals FIPS 204
 * reference (ref/): keygen, deterministic signing (byte-exact), and verify
 * in both directions, including rejection of tampered signatures.
 */

#include <cstdio>
#include <cstring>
#include <vector>

#include "api.cuh"
#include "ref_wrap.h"

extern "C" {
#include "fips202/fips202.h"
}

#ifndef NREF
#define NREF 4096
#endif

#ifdef USE_CACHED_SK
void crypto_sign_invalidate_cache(void *d_sign_mem_pool);
#endif

static int g_failures = 0;

static void report(const char *what, size_t ok, size_t total) {
  bool pass = ok == total;
  if (!pass)
    g_failures++;
  printf("  %-48s %6zu / %-6zu %s\n", what, ok, total, pass ? "PASS" : "FAIL");
}

static bool cuda_ok(const char *where) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess)
    err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("  CUDA error after %s: %s\n", where, cudaGetErrorString(err));
    g_failures++;
    return false;
  }
  return true;
}

// The keypair kernel derives xi = SHAKE128(le64(rand_index + 2 * i + 1))[0:32].
static void gpu_keygen_seed(uint8_t xi[32], size_t i, size_t rand_index) {
  uint8_t ctr[8];
  uint64_t v = rand_index + 2 * i + 1;
  for (int b = 0; b < 8; ++b)
    ctr[b] = (uint8_t)(v >> (8 * b));
  shake128(xi, 32, ctr, 8);
}

static void gpu_verify_batch(int *h_ret, const uint8_t *h_sig,
                             const uint8_t *h_m, size_t mlen,
                             const uint8_t *h_pk, size_t n) {
  uint8_t *d_pool;
  size_t pitch;
  size_t mem_size = VERIFY_POOL_BYTE_SIZE(mlen) +
                    DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // mat
                    DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // z
                    DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // t1
                    DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // w1prime
                    DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // h
                    DILITHIUM_N * sizeof(int32_t) +                             // cp
                    sizeof(int);                                                // ret
  cudaMallocPitch(&d_pool, &pitch, mem_size, n);
  crypto_sign_verify(h_ret, h_sig, CRYPTO_BYTES, CRYPTO_BYTES, h_m, mlen, mlen,
                     h_pk, d_pool, pitch, n);
  cuda_ok("crypto_sign_verify");
  cudaFree(d_pool);
}

int main() {
  const size_t n = NREF;
  const size_t rand_index = 0;

  printf("ML-DSA-%s GPU vs FIPS 204 reference, %zu vectors\n",
         DILITHIUM_MODE == 2 ? "44" : DILITHIUM_MODE == 3 ? "65" : "87", n);

  bool sizes_ok = ref_publickeybytes() == CRYPTO_PUBLICKEYBYTES &&
                  ref_secretkeybytes() == CRYPTO_SECRETKEYBYTES &&
                  ref_signaturebytes() == CRYPTO_BYTES;
  printf("  sizes pk/sk/sig = %d/%d/%d bytes (reference %zu/%zu/%zu)\n",
         CRYPTO_PUBLICKEYBYTES, CRYPTO_SECRETKEYBYTES, CRYPTO_BYTES,
         ref_publickeybytes(), ref_secretkeybytes(), ref_signaturebytes());
  report("sizes match reference", sizes_ok, 1);
  if (!sizes_ok)
    return 1;

  // KeyGen
  uint8_t *h_pk, *h_sk;
  cudaMallocHost(&h_pk, n * CRYPTO_PUBLICKEYBYTES);
  cudaMallocHost(&h_sk, n * CRYPTO_SECRETKEYBYTES);

  uint8_t *d_keypair_pool;
  size_t keypair_pitch;
  size_t keypair_mem_size = ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) +
                            ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES) +
                            DILITHIUM_K * DILITHIUM_N * sizeof(int32_t);
  cudaMallocPitch(&d_keypair_pool, &keypair_pitch, keypair_mem_size, n);
  crypto_sign_keypair(h_pk, h_sk, d_keypair_pool, keypair_pitch, n, nullptr,
                      rand_index);
  cuda_ok("crypto_sign_keypair");
  cudaFree(d_keypair_pool);

  {
    std::vector<uint8_t> pk(CRYPTO_PUBLICKEYBYTES), sk(CRYPTO_SECRETKEYBYTES);
    size_t pk_ok = 0, sk_ok = 0;
    for (size_t i = 0; i < n; ++i) {
      uint8_t xi[32];
      gpu_keygen_seed(xi, i, rand_index);
      ref_keypair_from_seed(pk.data(), sk.data(), xi);
      pk_ok += memcmp(pk.data(), h_pk + i * CRYPTO_PUBLICKEYBYTES,
                      CRYPTO_PUBLICKEYBYTES) == 0;
      sk_ok += memcmp(sk.data(), h_sk + i * CRYPTO_SECRETKEYBYTES,
                      CRYPTO_SECRETKEYBYTES) == 0;
    }
    report("keygen: pk identical to reference", pk_ok, n);
    report("keygen: sk identical to reference", sk_ok, n);
  }

#ifdef USE_CACHED_SK_SINGLE
  // The single-key cache signs every message with one key.
  for (size_t i = 1; i < n; ++i) {
    memcpy(h_pk + i * CRYPTO_PUBLICKEYBYTES, h_pk, CRYPTO_PUBLICKEYBYTES);
    memcpy(h_sk + i * CRYPTO_SECRETKEYBYTES, h_sk, CRYPTO_SECRETKEYBYTES);
  }
  printf("  (USE_CACHED_SK_SINGLE: signing with key 0 for all messages)\n");
#endif

  // Sign / Verify for several message lengths
  const size_t mlens[] = {1, 33, 300};
  for (size_t mlen : mlens) {
    printf(" message length %zu bytes\n", mlen);

    uint8_t *h_m, *h_sig;
    cudaMallocHost(&h_m, n * mlen);
    cudaMallocHost(&h_sig, n * CRYPTO_BYTES);
    for (size_t i = 0; i < n; ++i) {
      uint8_t in[16];
      for (int b = 0; b < 8; ++b) {
        in[b] = (uint8_t)(i >> (8 * b));
        in[8 + b] = (uint8_t)(mlen >> (8 * b));
      }
      shake128(h_m + i * mlen, mlen, in, sizeof(in));
    }

    uint8_t *d_sign_pool;
    size_t sign_pitch;
    size_t sign_mem_size = SIGN_POOL_BYTE_SIZE(mlen) +
                           DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // mat
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // s1
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // y
                           DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // z
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // t0
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // s2
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // w1
                           DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // w0
                           DILITHIUM_N * sizeof(int32_t) +                             // cp
                           SIGN_POOL_YPRE_BYTES;                                       // pre-generated y
    cudaMallocPitch(&d_sign_pool, &sign_pitch, sign_mem_size, n);

    size_t exec_threshold = 2048;
    task_lut lut{};
    lut.sign_lut.resize(n);
    cudaMallocHost(&lut.h_exec_lut, sizeof(uint32_t) * exec_threshold);
    cudaMallocHost(&lut.h_done_lut, sizeof(uint8_t) * exec_threshold);
    cudaMallocHost(&lut.h_copy_lut, sizeof(copy_lut_element) * exec_threshold);
    cudaMalloc(&lut.d_exec_lut, sizeof(uint32_t) * exec_threshold);
    cudaMalloc(&lut.d_done_lut, sizeof(uint8_t) * exec_threshold);
    cudaMalloc(&lut.d_copy_lut, sizeof(copy_lut_element) * exec_threshold);

    uint8_t *d_temp_pool;
    size_t temp_pitch;
    size_t temp_mem_size =
        ALIGN_TO_256_BYTES(CRYPTO_BYTES) +
        ALIGN_TO_256_BYTES(CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES) +
        DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) * 2 + // y, z
        DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) * 3;  // w0, w1, cp
    cudaMallocPitch(&d_temp_pool, &temp_pitch, temp_mem_size, exec_threshold);

    // The pool layout depends on mlen, so never reuse a cached setup here.
#ifdef USE_CACHED_SK
    crypto_sign_invalidate_cache(d_sign_pool);
#endif
#ifdef USE_CACHED_SK_SINGLE
    crypto_sign_invalidate_single_cache();
#endif

    size_t siglen = 0;
    crypto_sign_signature(h_sig, CRYPTO_BYTES, &siglen, h_m, mlen, mlen, h_sk,
                          d_sign_pool, sign_pitch, d_temp_pool, temp_pitch, lut,
                          exec_threshold, n);
    cuda_ok("crypto_sign_signature");
    report("sign: siglen == CRYPTO_BYTES", siglen == CRYPTO_BYTES, 1);

    cudaFree(d_sign_pool);
    cudaFree(d_temp_pool);
    cudaFreeHost(lut.h_exec_lut);
    cudaFreeHost(lut.h_done_lut);
    cudaFreeHost(lut.h_copy_lut);
    cudaFree(lut.d_exec_lut);
    cudaFree(lut.d_done_lut);
    cudaFree(lut.d_copy_lut);

    std::vector<uint8_t> sig_ref(CRYPTO_BYTES);
    size_t sig_ok = 0, refver_ok = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t *sig = h_sig + i * CRYPTO_BYTES;
      const uint8_t *m = h_m + i * mlen;
      const uint8_t *sk = h_sk + i * CRYPTO_SECRETKEYBYTES;
      const uint8_t *pk = h_pk + i * CRYPTO_PUBLICKEYBYTES;
#ifndef DILITHIUM_RANDOMIZED_SIGNING
      ref_sign(sig_ref.data(), m, mlen, sk, nullptr);
      sig_ok += memcmp(sig_ref.data(), sig, CRYPTO_BYTES) == 0;
#endif
      refver_ok += ref_verify(sig, CRYPTO_BYTES, m, mlen, pk) == 0;
    }
#ifndef DILITHIUM_RANDOMIZED_SIGNING
    report("sign: signature identical to reference", sig_ok, n);
#endif
    report("verify(ref) accepts GPU signature", refver_ok, n);

    std::vector<int> ret(n);
    gpu_verify_batch(ret.data(), h_sig, h_m, mlen, h_pk, n);
    size_t gpuver_ok = 0;
    for (size_t i = 0; i < n; ++i)
      gpuver_ok += ret[i] == 0;
    report("verify(GPU) accepts GPU signature", gpuver_ok, n);

    // Tamper with one byte: in c_tilde, in z, or in the hint encoding.
    std::vector<uint8_t> bad(n * CRYPTO_BYTES);
    memcpy(bad.data(), h_sig, n * CRYPTO_BYTES);
    for (size_t i = 0; i < n; ++i) {
      size_t pos;
      switch (i % 3) {
      case 0: pos = i % CTILDEBYTES; break;
      case 1: pos = CTILDEBYTES + (i * 7) % (DILITHIUM_L * POLYZ_PACKEDBYTES); break;
      default: pos = CRYPTO_BYTES - 1 - (i % DILITHIUM_K); break;
      }
      bad[i * CRYPTO_BYTES + pos] ^= (uint8_t)(1u << (i % 8));
    }
    gpu_verify_batch(ret.data(), bad.data(), h_m, mlen, h_pk, n);
    size_t gpu_rej = 0, ref_rej = 0;
    for (size_t i = 0; i < n; ++i) {
      gpu_rej += ret[i] != 0;
      ref_rej += ref_verify(bad.data() + i * CRYPTO_BYTES, CRYPTO_BYTES,
                            h_m + i * mlen, mlen,
                            h_pk + i * CRYPTO_PUBLICKEYBYTES) != 0;
    }
    report("verify(GPU) rejects tampered signature", gpu_rej, n);
    report("verify(ref) rejects tampered signature", ref_rej, n);

    // Wrong message: flip one bit of m.
    std::vector<uint8_t> m_bad(h_m, h_m + n * mlen);
    for (size_t i = 0; i < n; ++i)
      m_bad[i * mlen + (i % mlen)] ^= 0x01;
    gpu_verify_batch(ret.data(), h_sig, m_bad.data(), mlen, h_pk, n);
    size_t gpu_rej_m = 0;
    for (size_t i = 0; i < n; ++i)
      gpu_rej_m += ret[i] != 0;
    report("verify(GPU) rejects signature on wrong message", gpu_rej_m, n);

    cudaFreeHost(h_m);
    cudaFreeHost(h_sig);
  }

#ifdef USE_CACHED_SK_SINGLE
  crypto_sign_destroy_single_cache();
#endif
  cudaFreeHost(h_pk);
  cudaFreeHost(h_sk);

  printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "ALL PASSED",
         g_failures);
  return g_failures ? 1 : 0;
}
