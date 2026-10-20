/*
 * Copyright 2022-2023 [Anonymous].
 *
 * This file is part of cuDilithium.
 *
 * cuDilithium is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * cuDilithium is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * cuDilithium. If not, see <https://www.gnu.org/licenses/>.
 *
 * Modified in 2026: ported to FIPS 204 (ML-DSA); fused and staged signing,
 * warp NTT and key caching.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

extern "C" {
#include "randombytes.h"
}

#include "api.cuh"
#include "util.cuh"

#define MLEN 32

class DilithiumThroughputTimer {
public:
  explicit DilithiumThroughputTimer(std::string func_name, size_t batch_size,
                                    size_t n_streams) {
    func_name_ = std::move(func_name);
    batch_size_ = batch_size;
    n_streams_ = n_streams;
  }

  ~DilithiumThroughputTimer() {
    if (time_.empty())
      return;

    std::vector<double> throughput;
    for (float t : time_) {
      if (t > 0)
        throughput.push_back((double)batch_size_ / (t / 1e6));
      else
        throughput.push_back(0);
    }

    auto n_trials = throughput.size();
    auto median_tp = median(throughput);
    auto min_tp = min(throughput);
    auto max_tp = max(throughput);
    auto mean_tp = mean(throughput);
    auto stddev = std_dev(throughput);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << batch_size_ << ",\t" << n_streams_ << ",\t" << func_name_
              << ",\t" << n_trials << ",\t"
              << min_tp << ",\t" << mean_tp << ",\t" << median_tp << ",\t"
              << stddev << std::endl;
  }

  inline void start() { start_point_ = std::chrono::steady_clock::now(); }

  inline void stop() {
    stop_point_ = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::micro> elapsed_time =
        stop_point_ - start_point_;
    time_.emplace_back(elapsed_time.count());
  }

private:
  std::string func_name_;
  size_t batch_size_;
  size_t n_streams_;
  std::chrono::time_point<std::chrono::steady_clock> start_point_, stop_point_;
  std::vector<float> time_;

  static double mean(std::vector<double> const &v) {
    if (v.empty())
      return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  }

  static double median(std::vector<double> v) {
    size_t size = v.size();
    if (size == 0)
      return 0;
    sort(v.begin(), v.end());
    if (size % 2 == 0)
      return (v[size / 2 - 1] + v[size / 2]) / 2.0;
    return v[size / 2];
  }

  static double min(std::vector<double> v) {
    if (v.empty())
      return 0;
    sort(v.begin(), v.end());
    return v.front();
  }

  static double max(std::vector<double> v) {
    if (v.empty())
      return 0;
    sort(v.begin(), v.end());
    return v.back();
  }

  static double std_dev(std::vector<double> const &v) {
    if (v.empty())
      return 0;
    double m = mean(v);
    double sq_sum = 0;
    for (double x : v)
      sq_sum += (x - m) * (x - m);
    return std::sqrt(sq_sum / v.size());
  }
};

// BENCH_TRIALS / BENCH_STREAMS override the defaults (50 trials, 10 streams),
// e.g. to capture a single signing call under a profiler.
static size_t env_or(const char *name, size_t def) {
  const char *v = std::getenv(name);
  return v ? std::strtoull(v, nullptr, 10) : def;
}

int bench_cudilithium(size_t batch_size, size_t exec_threshold,
                      size_t n_streams) {
  size_t smlen;

  uint8_t *h_pk, *h_sk, *h_m, *h_sm;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_pk, batch_size * CRYPTO_PUBLICKEYBYTES));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_sk, batch_size * CRYPTO_SECRETKEYBYTES));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_m, batch_size * MLEN));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_sm, batch_size * CRYPTO_BYTES));

  int *h_ret;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_ret, batch_size * sizeof(int)));

  // Keypair

  uint8_t *d_keypair_mem_pool;
  size_t d_keypair_mem_pool_pitch;
  size_t byte_size_per_keypair =
      ALIGN_TO_256_BYTES(CRYPTO_PUBLICKEYBYTES) + // pk
      ALIGN_TO_256_BYTES(CRYPTO_SECRETKEYBYTES);  // sk

  size_t mem_size_per_keypair =
      byte_size_per_keypair +                      // align to 256 bytes
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t); // d_as

  CHECK_CUDA_ERROR(cudaMallocPitch(&d_keypair_mem_pool,
                                   &d_keypair_mem_pool_pitch,
                                   mem_size_per_keypair, batch_size));

  // Sign

  uint8_t *d_sign_mem_pool;
  size_t d_sign_mem_pool_pitch;
  size_t byte_size_per_sign = SIGN_POOL_BYTE_SIZE(MLEN);

  size_t mem_size_per_sign =
      byte_size_per_sign + // align to 8 bytes
      DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // mat
      DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // d_s1
      DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // d_y
      DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // d_z
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // d_t0
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // d_s2
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // d_w1
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // d_w0
      DILITHIUM_N * sizeof(int32_t) +                             // d_cp
      SIGN_POOL_YPRE_BYTES;                                       // pre-generated y

  CHECK_CUDA_ERROR(cudaMallocPitch(&d_sign_mem_pool, &d_sign_mem_pool_pitch,
                                   mem_size_per_sign, batch_size));

  // sign temp pool

  task_lut lut{};
  lut.sign_lut.resize(batch_size);
  CHECK_CUDA_ERROR(
      cudaMallocHost(&lut.h_exec_lut, sizeof(uint32_t) * exec_threshold));
  CHECK_CUDA_ERROR(
      cudaMallocHost(&lut.h_done_lut, sizeof(uint8_t) * exec_threshold));
  CHECK_CUDA_ERROR(cudaMallocHost(&lut.h_copy_lut,
                                  sizeof(copy_lut_element) * exec_threshold));
  CHECK_CUDA_ERROR(
      cudaMalloc(&lut.d_exec_lut, sizeof(uint32_t) * exec_threshold));
  CHECK_CUDA_ERROR(
      cudaMalloc(&lut.d_done_lut, sizeof(uint8_t) * exec_threshold));
  CHECK_CUDA_ERROR(
      cudaMalloc(&lut.d_copy_lut, sizeof(copy_lut_element) * exec_threshold));

  uint8_t *d_temp_mem_pool;
  size_t d_temp_mem_pool_pitch;
  size_t byte_size_per_sign_temp =
      ALIGN_TO_256_BYTES(CRYPTO_BYTES) + // sig
      ALIGN_TO_256_BYTES(
          CRHBYTES + DILITHIUM_K * POLYW1_PACKEDBYTES); // d_mu || d_w1_packed

  size_t temp_mem_size = byte_size_per_sign_temp + // align to 256 bytes
                         DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // d_y
                         DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // d_z
                         DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) + // d_w0
                         DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) + // d_w1
                         DILITHIUM_K * DILITHIUM_N * sizeof(int32_t);  // d_cp

  CHECK_CUDA_ERROR(cudaMallocPitch(&d_temp_mem_pool, &d_temp_mem_pool_pitch,
                                   temp_mem_size, exec_threshold));

  // Verify

  uint8_t *d_verify_mem_pool;
  size_t d_verify_mem_pool_pitch;
  size_t byte_size_per_verify = VERIFY_POOL_BYTE_SIZE(MLEN);

  size_t mem_size_per_verify =
      byte_size_per_verify + // align to 256 bytes
      DILITHIUM_K * DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) + // mat
      DILITHIUM_L * DILITHIUM_N * sizeof(int32_t) +               // z
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // t1
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // w1prime
      DILITHIUM_K * DILITHIUM_N * sizeof(int32_t) +               // h
      DILITHIUM_N * sizeof(int32_t) +                             // cp
      sizeof(int);                                                // ret

  CHECK_CUDA_ERROR(cudaMallocPitch(&d_verify_mem_pool, &d_verify_mem_pool_pitch,
                                   mem_size_per_verify, batch_size));

  // DilithiumThroughputTimer timer_verify_batch("verify batch", batch_size,
  //                                             n_streams);
  // DilithiumThroughputTimer timer_sign_batch("sign_batch", batch_size,
  //                                           n_streams);
  // DilithiumThroughputTimer timer_keypair_batch("keypair batch", batch_size,
  //                                              n_streams);

  // Distinct random messages: with identical messages (and, under
  // USE_CACHED_SK_SINGLE, one key) every signature would take the same number
  // of rejection iterations, making throughput depend on one lucky/unlucky input.
  randombytes(h_m, batch_size * MLEN);

  // for (size_t test_idx = 0; test_idx < 100; test_idx++) {
  //   timer_keypair_batch.start();
  //   crypto_sign_keypair(h_pk, h_sk, d_keypair_mem_pool,
  //                       d_keypair_mem_pool_pitch, batch_size);
  //   cudaDeviceSynchronize();
  //   timer_keypair_batch.stop();

  //   timer_sign_batch.start();
  //   crypto_sign_signature(h_sm, CRYPTO_BYTES, &smlen, h_m, MLEN, MLEN, h_sk,
  //                         d_sign_mem_pool, d_sign_mem_pool_pitch,
  //                         d_temp_mem_pool, d_temp_mem_pool_pitch, lut,
  //                         exec_threshold, batch_size);
  //   cudaDeviceSynchronize();
  //   timer_sign_batch.stop();

  //   timer_verify_batch.start();
  //   crypto_sign_verify(h_ret, h_sm, CRYPTO_BYTES, CRYPTO_BYTES, h_m, MLEN,
  //   MLEN,
  //                      h_pk, d_verify_mem_pool, d_verify_mem_pool_pitch,
  //                      batch_size);
  //   cudaDeviceSynchronize();
  //   timer_verify_batch.stop();

  //   for (size_t i = 0; i < batch_size; ++i) {
  //     if (h_ret[i]) {
  //       fprintf(stderr, "Verification failed: %zu\n", i);
  //       return -1;
  //     }
  //   }
  // }

  DilithiumThroughputTimer timer_verify_stream("verify stream", batch_size,
                                               n_streams);
  DilithiumThroughputTimer timer_sign_stream("sign stream", batch_size,
                                             n_streams);
  DilithiumThroughputTimer timer_keypair_stream("keypair stream", batch_size,
                                                n_streams);

  // cuda stream init
  std::vector<cudaStream_t> streams(n_streams);
  for (auto &stream : streams) {
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  const size_t n_trials = env_or("BENCH_TRIALS", 50);
  for (size_t test_idx = 0; test_idx < n_trials; test_idx++) {
    timer_keypair_stream.start();
#ifdef USE_CACHED_SK_SINGLE
    crypto_sign_keypair(h_pk, h_sk, d_keypair_mem_pool,
                        d_keypair_mem_pool_pitch, 1, streams[0]);
    cudaStreamSynchronize(streams[0]);
    for (size_t i = 1; i < batch_size; ++i) {
      memcpy(h_sk + i * CRYPTO_SECRETKEYBYTES, h_sk, CRYPTO_SECRETKEYBYTES);
      memcpy(h_pk + i * CRYPTO_PUBLICKEYBYTES, h_pk, CRYPTO_PUBLICKEYBYTES);
    }
#else
    for (size_t i = 0; i < n_streams; ++i) {
      size_t keypair_stream_batch_size = batch_size / n_streams;
      size_t last_keypair_stream_batch_size = keypair_stream_batch_size;
      last_keypair_stream_batch_size +=
          (i == n_streams - 1) ? (batch_size % n_streams) : 0;
      crypto_sign_keypair(
          h_pk + i * keypair_stream_batch_size * CRYPTO_PUBLICKEYBYTES,
          h_sk + i * keypair_stream_batch_size * CRYPTO_SECRETKEYBYTES,
          d_keypair_mem_pool +
              i * keypair_stream_batch_size * d_keypair_mem_pool_pitch,
          d_keypair_mem_pool_pitch, last_keypair_stream_batch_size, streams[i]);
    }
#endif
    cudaDeviceSynchronize();
    timer_keypair_stream.stop();

    timer_sign_stream.start();
#pragma omp parallel for default(none)                                         \
    shared(h_sm, smlen, h_m, h_sk, d_sign_mem_pool, d_sign_mem_pool_pitch,     \
               d_temp_mem_pool, d_temp_mem_pool_pitch, lut, streams,           \
               batch_size, exec_threshold, n_streams)
    for (size_t i = 0; i < n_streams; ++i) {
      size_t sign_stream_batch_size = batch_size / n_streams;
      size_t last_sign_stream_batch_size = sign_stream_batch_size;
      last_sign_stream_batch_size +=
          (i == n_streams - 1) ? (batch_size % n_streams) : 0;
      size_t stream_exec_threshold = exec_threshold / n_streams;
      size_t last_stream_exec_threshold = stream_exec_threshold;
      last_stream_exec_threshold +=
          (i == n_streams - 1) ? (exec_threshold % n_streams) : 0;

      task_lut lut_i = {};
      lut_i.sign_lut.resize(last_sign_stream_batch_size);
      lut_i.h_exec_lut = lut.h_exec_lut + i * stream_exec_threshold;
      lut_i.h_done_lut = lut.h_done_lut + i * stream_exec_threshold;
      lut_i.h_copy_lut = lut.h_copy_lut + i * stream_exec_threshold;
      lut_i.d_exec_lut = lut.d_exec_lut + i * stream_exec_threshold;
      lut_i.d_done_lut = lut.d_done_lut + i * stream_exec_threshold;
      lut_i.d_copy_lut = lut.d_copy_lut + i * stream_exec_threshold;
      crypto_sign_signature(
          h_sm + i * sign_stream_batch_size * CRYPTO_BYTES, CRYPTO_BYTES,
          &smlen, h_m + i * sign_stream_batch_size * MLEN, MLEN, MLEN,
          h_sk + i * sign_stream_batch_size * CRYPTO_SECRETKEYBYTES,
          d_sign_mem_pool + i * sign_stream_batch_size * d_sign_mem_pool_pitch,
          d_sign_mem_pool_pitch,
          d_temp_mem_pool + i * stream_exec_threshold * d_temp_mem_pool_pitch,
          d_temp_mem_pool_pitch, lut_i, last_stream_exec_threshold,
          last_sign_stream_batch_size, streams[i]);
    }
    cudaDeviceSynchronize();
    timer_sign_stream.stop();

    timer_verify_stream.start();
    for (size_t i = 0; i < n_streams; ++i) {
      size_t verify_stream_batch_size = batch_size / n_streams;
      size_t last_verify_stream_batch_size = verify_stream_batch_size;
      last_verify_stream_batch_size +=
          (i == n_streams - 1) ? (batch_size % n_streams) : 0;
      crypto_sign_verify(
          h_ret + i * verify_stream_batch_size,
          h_sm + i * verify_stream_batch_size * CRYPTO_BYTES, CRYPTO_BYTES,
          CRYPTO_BYTES, h_m + i * verify_stream_batch_size * MLEN, MLEN, MLEN,
          h_pk + i * verify_stream_batch_size * CRYPTO_PUBLICKEYBYTES,
          d_verify_mem_pool +
              i * verify_stream_batch_size * d_verify_mem_pool_pitch,
          d_verify_mem_pool_pitch, last_verify_stream_batch_size, streams[i]);
    }
    cudaDeviceSynchronize();
    timer_verify_stream.stop();

    for (size_t i = 0; i < batch_size; ++i) {
      if (h_ret[i]) {
        fprintf(stderr, "Verification failed: %zu\n", i);
        return -1;
      }
    }
  }
#if defined(SIGN_PHASE_PROFILE) &&                                             \
    (defined(USE_PERSISTENT_WARPNTT) || defined(USE_PERSISTENT))
  sign_phase_profile_dump();
  sign_phase_profile_reset();
#endif
  // cuda stream destroy
  for (auto &stream : streams) {
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  cudaFree(d_keypair_mem_pool);

  cudaFree(d_sign_mem_pool);
  cudaFree(d_temp_mem_pool);
  cudaFreeHost(lut.h_exec_lut);
  cudaFreeHost(lut.h_done_lut);
  cudaFreeHost(lut.h_copy_lut);
  cudaFree(lut.d_exec_lut);
  cudaFree(lut.d_done_lut);
  cudaFree(lut.d_copy_lut);

  cudaFree(d_verify_mem_pool);

  cudaFreeHost(h_pk);
  cudaFreeHost(h_sk);
  cudaFreeHost(h_m);
  cudaFreeHost(h_sm);
  cudaFreeHost(h_ret);

#ifdef USE_CACHED_SK_SINGLE
  crypto_sign_destroy_single_cache();
#endif

  return 0;
}

int main(int argc, char **argv) {
  // Optional arguments: batch sizes to run, e.g. bench 1024 4096 16384
  std::vector<size_t> batch_sizes = {5000};
  if (argc > 1) {
    batch_sizes.clear();
    for (int i = 1; i < argc; ++i)
      batch_sizes.push_back(std::strtoull(argv[i], nullptr, 10));
  }
  // Concurrent signing attempts of the multi-kernel (non-fused) path, split
  // over the streams; BENCH_EXEC_THRESHOLD overrides it.
  std::vector<size_t> exec_thresholds = {env_or("BENCH_EXEC_THRESHOLD", 2048)};
  std::vector<size_t> v_n_streams = {env_or("BENCH_STREAMS", 10)};
  //    for (size_t i = 1; i <= 16; i += 1)
  //        v_n_streams.push_back(i);
  //    std::cout << "threshold,batch,function,trials,min,mean,median,stddev."
  //    << std::endl;
  std::cout << "batch,n_streams,function,trials,min (op/s),mean (op/s),median "
               "(op/s),stddev (op/s)"
            << std::endl;
  for (auto &batch_size : batch_sizes) {
    for (auto &exec_threshold : exec_thresholds) {
      for (auto &n_streams : v_n_streams) {
        bench_cudilithium(batch_size, exec_threshold, n_streams);
      }
    }
  }
  return 0;
}
