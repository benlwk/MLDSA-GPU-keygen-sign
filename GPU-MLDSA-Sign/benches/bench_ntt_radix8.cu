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
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "params.h"
#include "randombytes.h"
#include "util.cuh"

#define NTESTS 16384

#define QINV 58728449 // q^(-1) mod 2^32

__device__ static const int32_t c_zetas[DILITHIUM_N] = {
    0,        25847,    -2608894, -518909,  237124,   -777960,  -876248,
    466468,   1826347,  2353451,  -359251,  -2091905, 3119733,  -2884855,
    3111497,  2680103,  2725464,  1024112,  -1079900, 3585928,  -549488,
    -1119584, 2619752,  -2108549, -2118186, -3859737, -1399561, -3277672,
    1757237,  -19422,   4010497,  280005,   2706023,  95776,    3077325,
    3530437,  -1661693, -3592148, -2537516, 3915439,  -3861115, -3043716,
    3574422,  -2867647, 3539968,  -300467,  2348700,  -539299,  -1699267,
    -1643818, 3505694,  -3821735, 3507263,  -2140649, -1600420, 3699596,
    811944,   531354,   954230,   3881043,  3900724,  -2556880, 2071892,
    -2797779, -3930395, -1528703, -3677745, -3041255, -1452451, 3475950,
    2176455,  -1585221, -1257611, 1939314,  -4083598, -1000202, -3190144,
    -3157330, -3632928, 126922,   3412210,  -983419,  2147896,  2715295,
    -2967645, -3693493, -411027,  -2477047, -671102,  -1228525, -22981,
    -1308169, -381987,  1349076,  1852771,  -1430430, -3343383, 264944,
    508951,   3097992,  44288,    -1100098, 904516,   3958618,  -3724342,
    -8578,    1653064,  -3249728, 2389356,  -210977,  759969,   -1316856,
    189548,   -3553272, 3159746,  -1851402, -2409325, -177440,  1315589,
    1341330,  1285669,  -1584928, -812732,  -1439742, -3019102, -3881060,
    -3628969, 3839961,  2091667,  3407706,  2316500,  3817976,  -3342478,
    2244091,  -2446433, -3562462, 266997,   2434439,  -1235728, 3513181,
    -3520352, -3759364, -1197226, -3193378, 900702,   1859098,  909542,
    819034,   495491,   -1613174, -43260,   -522500,  -655327,  -3122442,
    2031748,  3207046,  -3556995, -525098,  -768622,  -3595838, 342297,
    286988,   -2437823, 4108315,  3437287,  -3342277, 1735879,  203044,
    2842341,  2691481,  -2590150, 1265009,  4055324,  1247620,  2486353,
    1595974,  -3767016, 1250494,  2635921,  -3548272, -2994039, 1869119,
    1903435,  -1050970, -1333058, 1237275,  -3318210, -1430225, -451100,
    1312455,  3306115,  -1962642, -1279661, 1917081,  -2546312, -1374803,
    1500165,  777191,   2235880,  3406031,  -542412,  -2831860, -1671176,
    -1846953, -2584293, -3724270, 594136,   -3776993, -2013608, 2432395,
    2454455,  -164721,  1957272,  3369112,  185531,   -1207385, -3183426,
    162844,   1616392,  3014001,  810149,   1652634,  -3694233, -1799107,
    -3038916, 3523897,  3866901,  269760,   2213111,  -975884,  1717735,
    472078,   -426683,  1723600,  -1803090, 1910376,  -1667432, -1104333,
    -260646,  -3833893, -2939036, -2235985, -420899,  -2286327, 183443,
    -976891,  1612842,  -3545687, -554416,  3919660,  -48306,   -1362209,
    3937738,  1400424,  -846154,  1976782};

__device__ __forceinline__ int32_t gpu_montgomery_multiply(int32_t x,
                                                           int32_t y) {
  int32_t a_hi = __mulhi(x, y);
  int32_t a_lo = x * y;
  int32_t t = a_lo * QINV;
  t = __mulhi(t, DILITHIUM_Q);
  t = a_hi - t;
  return t;
}

__device__ __inline__ static void ntt_butt(int32_t &a, int32_t &b,
                                           const int32_t zeta) {
  int32_t t = gpu_montgomery_multiply(zeta, b);
  b = a - t;
  a = a + t;
}

__device__ __inline__ static void invntt_butt(int32_t &a, int32_t &b,
                                              const int32_t zeta) {
  int32_t t = a;
  a = t + b;
  b = t - b;
  b = gpu_montgomery_multiply(zeta, b);
}

__device__ __inline__ static void ntt_butt_shfl(int32_t &reg,
                                                const int distance,
                                                const int32_t zeta,
                                                const bool is_even) {
  int32_t other = __shfl_xor_sync(0xFFFFFFFF, reg, distance);
  int32_t t = gpu_montgomery_multiply(zeta, is_even ? other : reg);
  reg = is_even ? (reg + t) : (other - t);
}

__device__ __inline__ static void invntt_butt_shfl(int32_t &reg,
                                                   const int distance,
                                                   const int32_t zeta,
                                                   const bool is_even) {
  int32_t other = __shfl_xor_sync(0xFFFFFFFF, reg, distance);
  if (is_even) {
    reg = reg + other;
  } else {
    int32_t diff = other - reg;
    reg = gpu_montgomery_multiply(zeta, diff);
  }
}

__device__ void ntt_inner(int32_t regs[8], int32_t *s_ntt) {
  const int laneId = threadIdx.x & 0x1f;
  // level 1
  ntt_butt(regs[0], regs[4], c_zetas[1]);
  ntt_butt(regs[1], regs[5], c_zetas[1]);
  ntt_butt(regs[2], regs[6], c_zetas[1]);
  ntt_butt(regs[3], regs[7], c_zetas[1]);
  // level 2
  ntt_butt(regs[0], regs[2], c_zetas[2]);
  ntt_butt(regs[1], regs[3], c_zetas[2]);
  ntt_butt(regs[4], regs[6], c_zetas[3]);
  ntt_butt(regs[5], regs[7], c_zetas[3]);
  // level 3
  ntt_butt(regs[0], regs[1], c_zetas[4]);
  ntt_butt(regs[2], regs[3], c_zetas[5]);
  ntt_butt(regs[4], regs[5], c_zetas[6]);
  ntt_butt(regs[6], regs[7], c_zetas[7]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_ntt[i * 32 + laneId] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[(laneId / 4) * 32 + (laneId & 3) + i * 4];
  // level 4
  ntt_butt(regs[0], regs[4], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[1], regs[5], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[2], regs[6], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[3], regs[7], c_zetas[8 + laneId / 4]);
  // level 5
  ntt_butt(regs[0], regs[2], c_zetas[16 + (laneId / 4) * 2]);
  ntt_butt(regs[1], regs[3], c_zetas[16 + (laneId / 4) * 2]);
  ntt_butt(regs[4], regs[6], c_zetas[17 + (laneId / 4) * 2]);
  ntt_butt(regs[5], regs[7], c_zetas[17 + (laneId / 4) * 2]);
  // level 6
  ntt_butt(regs[0], regs[1], c_zetas[32 + (laneId / 4) * 4]);
  ntt_butt(regs[2], regs[3], c_zetas[33 + (laneId / 4) * 4]);
  ntt_butt(regs[4], regs[5], c_zetas[34 + (laneId / 4) * 4]);
  ntt_butt(regs[6], regs[7], c_zetas[35 + (laneId / 4) * 4]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_ntt[(laneId / 4) * 32 + (laneId & 3) + i * 4] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[laneId * 8 + i];
  // level 7
  ntt_butt(regs[0], regs[2], c_zetas[64 + laneId * 2]);
  ntt_butt(regs[1], regs[3], c_zetas[64 + laneId * 2]);
  ntt_butt(regs[4], regs[6], c_zetas[65 + laneId * 2]);
  ntt_butt(regs[5], regs[7], c_zetas[65 + laneId * 2]);
  // level 8
  ntt_butt(regs[0], regs[1], c_zetas[128 + laneId * 4]);
  ntt_butt(regs[2], regs[3], c_zetas[129 + laneId * 4]);
  ntt_butt(regs[4], regs[5], c_zetas[130 + laneId * 4]);
  ntt_butt(regs[6], regs[7], c_zetas[131 + laneId * 4]);
}

__device__ void ntt_inner_shuffle(int32_t regs[8], int32_t *s_poly) {
  const int laneId = threadIdx.x & 0x1f;

  // Levels 1-3 (Internal to thread: Stride 128, 64, 32)
  ntt_butt(regs[0], regs[4], c_zetas[1]);
  ntt_butt(regs[1], regs[5], c_zetas[1]);
  ntt_butt(regs[2], regs[6], c_zetas[1]);
  ntt_butt(regs[3], regs[7], c_zetas[1]);

  ntt_butt(regs[0], regs[2], c_zetas[2]);
  ntt_butt(regs[1], regs[3], c_zetas[2]);
  ntt_butt(regs[4], regs[6], c_zetas[3]);
  ntt_butt(regs[5], regs[7], c_zetas[3]);

  ntt_butt(regs[0], regs[1], c_zetas[4]);
  ntt_butt(regs[2], regs[3], c_zetas[5]);
  ntt_butt(regs[4], regs[5], c_zetas[6]);
  ntt_butt(regs[6], regs[7], c_zetas[7]);

  // Levels 4-8 (Warp shuffles: Stride 16, 8, 4, 2, 1)
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int distance = 16 >> i;
    const int zeta_offset = 8 << i;
    const int log_2s = 5 - i;
    const int base_zeta = zeta_offset + (laneId >> log_2s);
    const bool is_even = (laneId & distance) == 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int32_t zeta = c_zetas[base_zeta + (j << i)];
      ntt_butt_shfl(regs[j], distance, zeta, is_even);
    }
  }

  // Final transpose to natural order (8t+i) using bank-conflict-free stride
#pragma unroll
  for (int i = 0; i < 8; i++)
    s_poly[i * 33 + laneId] = regs[i];
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 8; i++)
    regs[i] = s_poly[laneId * 8 + (laneId >> 2) + i];
}

__device__ __inline__ void ntt_inner_shuffle_notranspose(int32_t regs[8]) {
  const int laneId = threadIdx.x & 0x1f;

  // Levels 1-3 (Internal to thread: Stride 128, 64, 32)
  ntt_butt(regs[0], regs[4], c_zetas[1]);
  ntt_butt(regs[1], regs[5], c_zetas[1]);
  ntt_butt(regs[2], regs[6], c_zetas[1]);
  ntt_butt(regs[3], regs[7], c_zetas[1]);

  ntt_butt(regs[0], regs[2], c_zetas[2]);
  ntt_butt(regs[1], regs[3], c_zetas[2]);
  ntt_butt(regs[4], regs[6], c_zetas[3]);
  ntt_butt(regs[5], regs[7], c_zetas[3]);

  ntt_butt(regs[0], regs[1], c_zetas[4]);
  ntt_butt(regs[2], regs[3], c_zetas[5]);
  ntt_butt(regs[4], regs[5], c_zetas[6]);
  ntt_butt(regs[6], regs[7], c_zetas[7]);

  // Levels 4-8 (Warp shuffles: Stride 16, 8, 4, 2, 1)
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int distance = 16 >> i;
    const int zeta_offset = 8 << i;
    const int log_2s = 5 - i;
    const int base_zeta = zeta_offset + (laneId >> log_2s);
    const bool is_even = (laneId & distance) == 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int32_t zeta = c_zetas[base_zeta + (j << i)];
      ntt_butt_shfl(regs[j], distance, zeta, is_even);
    }
  }
}

class NTTThroughputTimer {
public:
  explicit NTTThroughputTimer(std::string func_name, size_t n_tests) {
    func_name_ = std::move(func_name);
    n_tests_ = n_tests;
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
  }

  ~NTTThroughputTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    if (time_.empty())
      return;

    std::vector<double> throughput;
    for (float t : time_) {
      if (t > 0)
        throughput.push_back((double)n_tests_ / (t * 1000.0));
      else
        throughput.push_back(0);
    }

    auto n_trials = throughput.size();
    auto median_tp = median(throughput);
    auto min_tp = min(throughput);
    auto mean_tp = mean(throughput);
    auto stddev = std_dev(throughput);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << func_name_ << ",\t" << n_trials << ",\t" << min_tp << ",\t"
              << mean_tp << ",\t" << median_tp << ",\t" << stddev << std::endl;
  }

  inline void start() const { cudaEventRecord(start_event_); }

  inline void stop() {
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
    time_.push_back(milliseconds);
  }

private:
  std::string func_name_;
  size_t n_tests_;
  cudaEvent_t start_event_{}, stop_event_{};
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

__device__ void ntt_inner_1(int32_t regs[8], int32_t *s_ntt) {
  const int laneId = threadIdx.x & 0x1f;
  // level 1
  ntt_butt(regs[0], regs[4], c_zetas[1]);
  ntt_butt(regs[1], regs[5], c_zetas[1]);
  ntt_butt(regs[2], regs[6], c_zetas[1]);
  ntt_butt(regs[3], regs[7], c_zetas[1]);
  // level 2
  ntt_butt(regs[0], regs[2], c_zetas[2]);
  ntt_butt(regs[1], regs[3], c_zetas[2]);
  ntt_butt(regs[4], regs[6], c_zetas[3]);
  ntt_butt(regs[5], regs[7], c_zetas[3]);
  // level 3
  ntt_butt(regs[0], regs[1], c_zetas[4]);
  ntt_butt(regs[2], regs[3], c_zetas[5]);
  ntt_butt(regs[4], regs[5], c_zetas[6]);
  ntt_butt(regs[6], regs[7], c_zetas[7]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_ntt[i * 36 + laneId] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[(laneId / 4) * 36 + (laneId & 3) + i * 4];
  // level 4
  ntt_butt(regs[0], regs[4], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[1], regs[5], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[2], regs[6], c_zetas[8 + laneId / 4]);
  ntt_butt(regs[3], regs[7], c_zetas[8 + laneId / 4]);
  // level 5
  ntt_butt(regs[0], regs[2], c_zetas[16 + (laneId / 4) * 2]);
  ntt_butt(regs[1], regs[3], c_zetas[16 + (laneId / 4) * 2]);
  ntt_butt(regs[4], regs[6], c_zetas[17 + (laneId / 4) * 2]);
  ntt_butt(regs[5], regs[7], c_zetas[17 + (laneId / 4) * 2]);
  // level 6
  ntt_butt(regs[0], regs[1], c_zetas[32 + (laneId / 4) * 4]);
  ntt_butt(regs[2], regs[3], c_zetas[33 + (laneId / 4) * 4]);
  ntt_butt(regs[4], regs[5], c_zetas[34 + (laneId / 4) * 4]);
  ntt_butt(regs[6], regs[7], c_zetas[35 + (laneId / 4) * 4]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_ntt[(laneId / 4) * 36 + ((laneId & 3) + i * 4) / 8 + (laneId & 3) +
          i * 4] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[laneId * 9 + i];
  // level 7
  ntt_butt(regs[0], regs[2], c_zetas[64 + laneId * 2]);
  ntt_butt(regs[1], regs[3], c_zetas[64 + laneId * 2]);
  ntt_butt(regs[4], regs[6], c_zetas[65 + laneId * 2]);
  ntt_butt(regs[5], regs[7], c_zetas[65 + laneId * 2]);
  // level 8
  ntt_butt(regs[0], regs[1], c_zetas[128 + laneId * 4]);
  ntt_butt(regs[2], regs[3], c_zetas[129 + laneId * 4]);
  ntt_butt(regs[4], regs[5], c_zetas[130 + laneId * 4]);
  ntt_butt(regs[6], regs[7], c_zetas[131 + laneId * 4]);
}

__device__ void invntt_inner(int32_t regs[8], int32_t *s_poly) {
  const int laneId = threadIdx.x & 0x1f;
  // level 1
  invntt_butt(regs[0], regs[1], -c_zetas[256 - laneId * 4 - 1]);
  invntt_butt(regs[2], regs[3], -c_zetas[256 - laneId * 4 - 2]);
  invntt_butt(regs[4], regs[5], -c_zetas[256 - laneId * 4 - 3]);
  invntt_butt(regs[6], regs[7], -c_zetas[256 - laneId * 4 - 4]);
  // level 2
  invntt_butt(regs[0], regs[2], -c_zetas[128 - laneId * 2 - 1]);
  invntt_butt(regs[1], regs[3], -c_zetas[128 - laneId * 2 - 1]);
  invntt_butt(regs[4], regs[6], -c_zetas[128 - laneId * 2 - 2]);
  invntt_butt(regs[5], regs[7], -c_zetas[128 - laneId * 2 - 2]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_poly[laneId * 9 + i] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++) {
    size_t offset_per_row = (laneId & 3) + i * 4;
    regs[i] = s_poly[(laneId / 4) * 36 + offset_per_row / 8 + offset_per_row];
  }
  // level 3
  invntt_butt(regs[0], regs[1], -c_zetas[64 - (laneId / 4) * 4 - 1]);
  invntt_butt(regs[2], regs[3], -c_zetas[64 - (laneId / 4) * 4 - 2]);
  invntt_butt(regs[4], regs[5], -c_zetas[64 - (laneId / 4) * 4 - 3]);
  invntt_butt(regs[6], regs[7], -c_zetas[64 - (laneId / 4) * 4 - 4]);
  // level 4
  invntt_butt(regs[0], regs[2], -c_zetas[32 - (laneId / 4) * 2 - 1]);
  invntt_butt(regs[1], regs[3], -c_zetas[32 - (laneId / 4) * 2 - 1]);
  invntt_butt(regs[4], regs[6], -c_zetas[32 - (laneId / 4) * 2 - 2]);
  invntt_butt(regs[5], regs[7], -c_zetas[32 - (laneId / 4) * 2 - 2]);
  // level 5
  invntt_butt(regs[0], regs[4], -c_zetas[16 - laneId / 4 - 1]);
  invntt_butt(regs[1], regs[5], -c_zetas[16 - laneId / 4 - 1]);
  invntt_butt(regs[2], regs[6], -c_zetas[16 - laneId / 4 - 1]);
  invntt_butt(regs[3], regs[7], -c_zetas[16 - laneId / 4 - 1]);
  // SMEM exchange
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    s_poly[(laneId / 4) * 36 + (laneId & 3) + i * 4] = regs[i];
  __syncwarp();
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_poly[i * 36 + laneId];
  // level 6
  invntt_butt(regs[0], regs[1], -c_zetas[7]);
  invntt_butt(regs[2], regs[3], -c_zetas[6]);
  invntt_butt(regs[4], regs[5], -c_zetas[5]);
  invntt_butt(regs[6], regs[7], -c_zetas[4]);
  // level 7
  invntt_butt(regs[0], regs[2], -c_zetas[3]);
  invntt_butt(regs[1], regs[3], -c_zetas[3]);
  invntt_butt(regs[4], regs[6], -c_zetas[2]);
  invntt_butt(regs[5], regs[7], -c_zetas[2]);
  // level 8
  invntt_butt(regs[0], regs[4], -c_zetas[1]);
  invntt_butt(regs[1], regs[5], -c_zetas[1]);
  invntt_butt(regs[2], regs[6], -c_zetas[1]);
  invntt_butt(regs[3], regs[7], -c_zetas[1]);
  // reduce
  const int32_t f = 41978; // mont^2/256
#pragma unroll
  for (size_t i = 0; i < 8; i++)
    regs[i] = gpu_montgomery_multiply(f, regs[i]);
}

__device__ void invntt_inner_shuffle(int32_t regs[8], int32_t *s_poly) {
  const int laneId = threadIdx.x & 0x1f;

  // Initial transpose from natural order (8t+i) using bank-conflict-free stride
#pragma unroll
  for (int i = 0; i < 8; i++)
    s_poly[laneId * 8 + (laneId >> 2) + i] = regs[i];
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 8; i++)
    regs[i] = s_poly[i * 33 + laneId];

  // Levels 1-5 (Warp shuffles: Stride 1, 2, 4, 8, 16)
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int distance = 1 << i;
    const int zeta_offset = (256 >> i) - 1;
    const int log_2s = i + 1;
    const int base_zeta = zeta_offset - (laneId >> log_2s);
    const bool is_even = (laneId & distance) == 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int32_t zeta = -c_zetas[base_zeta - (j << (4 - i))];
      invntt_butt_shfl(regs[j], distance, zeta, is_even);
    }
  }

  // Levels 6-8 (Internal to thread: Stride 32, 64, 128)
  invntt_butt(regs[0], regs[1], -c_zetas[7]);
  invntt_butt(regs[2], regs[3], -c_zetas[6]);
  invntt_butt(regs[4], regs[5], -c_zetas[5]);
  invntt_butt(regs[6], regs[7], -c_zetas[4]);

  invntt_butt(regs[0], regs[2], -c_zetas[3]);
  invntt_butt(regs[1], regs[3], -c_zetas[3]);
  invntt_butt(regs[4], regs[6], -c_zetas[2]);
  invntt_butt(regs[5], regs[7], -c_zetas[2]);

  invntt_butt(regs[0], regs[4], -c_zetas[1]);
  invntt_butt(regs[1], regs[5], -c_zetas[1]);
  invntt_butt(regs[2], regs[6], -c_zetas[1]);
  invntt_butt(regs[3], regs[7], -c_zetas[1]);

  // Final reduction
  const int32_t f = 41978; // mont^2/256
#pragma unroll
  for (int i = 0; i < 8; i++)
    regs[i] = gpu_montgomery_multiply(f, regs[i]);
}

__global__ void k0_ntt(int32_t *g_polyvec, size_t g_polyvec_pitch) {
  __shared__ int32_t s_poly[DILITHIUM_N];
  int32_t regs[8];
  for (int k = 0; k < DILITHIUM_K; ++k) {
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; ++i)
      regs[i] = g_poly[i * 32 + threadIdx.x];
    ntt_inner(regs, s_poly);
    for (size_t i = 0; i < 8; ++i)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

__global__ void k0_unpack(int32_t *g_polyvec, size_t g_polyvec_pitch,
                          const uint8_t *g_polyvec_packed,
                          size_t g_polyvec_packed_pitch) {
  for (int k = 0; k < DILITHIUM_K; k++) {
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;

    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      g_poly[i * 32 + threadIdx.x] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
  }
}

// kernel fusing
__global__ void k1_unpack_fuse_ntt(int32_t *g_polyvec, size_t g_polyvec_pitch,
                                   const uint8_t *g_polyvec_packed,
                                   size_t g_polyvec_packed_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N];
  int32_t regs[8];

  // unpack
  for (int k = 0; k < DILITHIUM_K; ++k) {
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      g_poly[i * 32 + threadIdx.x] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
  }

  // ntt
  for (int k = 0; k < DILITHIUM_K; ++k) {
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; ++i)
      regs[i] = g_poly[i * 32 + threadIdx.x];
    ntt_inner(regs, s_ntt);
    for (size_t i = 0; i < 8; i++)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

// merge two loops into one and use registers to store intermediate poly
__global__ void k4_unpack_fuse_ntt_shuffle(int32_t *g_polyvec,
                                           size_t g_polyvec_pitch,
                                           const uint8_t *g_polyvec_packed,
                                           size_t g_polyvec_packed_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N + 32];
  int32_t regs[8];

  // unpack
  for (int k = 0; k < DILITHIUM_K; ++k) {
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      g_poly[i * 32 + threadIdx.x] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
  }

  // ntt
  for (int k = 0; k < DILITHIUM_K; ++k) {
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; ++i)
      regs[i] = g_poly[i * 32 + threadIdx.x];
    ntt_inner_shuffle(regs, s_ntt);
    for (size_t i = 0; i < 8; i++)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

// scale to 128 threads to solve low-occupancy issue, keeping poly in registers
__global__ void k5_unpack_fuse_ntt_shuffle_128t(int32_t *g_polyvec,
                                                size_t g_polyvec_pitch,
                                                const uint8_t *g_polyvec_packed,
                                                size_t g_polyvec_packed_pitch) {
  __shared__ int32_t s_ntt_total[4 * (DILITHIUM_N + 32)];
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x & 0x1f;
  int32_t *s_ntt = &s_ntt_total[warpId * (DILITHIUM_N + 32)];
  int32_t regs[8];

  for (int k = warpId; k < DILITHIUM_K; k += 4) {
    // unpack
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (laneId & 7) * 13 - ((laneId & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    // ntt
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    ntt_inner_shuffle(regs, s_ntt);
    for (size_t i = 0; i < 8; i++)
      g_poly[laneId * 8 + i] = regs[i];
  }
}

// zero shared memory, fully coalesced global memory access
__global__ void k6_unpack_fuse_ntt_shuffle_128t_coalesced(
    int32_t *g_polyvec, size_t g_polyvec_pitch, const uint8_t *g_polyvec_packed,
    size_t g_polyvec_packed_pitch) {
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x & 0x1f;
  int32_t regs[8];

  for (int k = warpId; k < DILITHIUM_K; k += 4) {
    // unpack
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (laneId & 7) * 13 - ((laneId & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    // ntt
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    ntt_inner_shuffle_notranspose(regs);
    // write coalesced!
    for (size_t i = 0; i < 8; i++)
      g_poly[i * 32 + laneId] = regs[i];
  }
}

// 4 warps per block, each warp processes all DILITHIUM_K polys for one vector.
// No warp idling.
__global__ void k7_unpack_fuse_ntt_shuffle_128t_k_per_warp(
    int32_t *g_polyvec, size_t g_polyvec_pitch, const uint8_t *g_polyvec_packed,
    size_t g_polyvec_packed_pitch) {
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x & 0x1f;
  int32_t regs[8];

  const int vectorIdx = blockIdx.x * 4 + warpId;

  for (int k = 0; k < DILITHIUM_K; ++k) {
    // unpack
    auto *g_poly_packed = g_polyvec_packed +
                          vectorIdx * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (laneId / 8) * 13 +
                                  ((laneId & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (laneId & 7) * 13 - ((laneId & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    // ntt
    int32_t *g_poly = g_polyvec +
                      vectorIdx * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    ntt_inner_shuffle_notranspose(regs);
    // write coalesced!
    for (size_t i = 0; i < 8; i++)
      g_poly[i * 32 + laneId] = regs[i];
  }
}

// Shared key: all 4 warps load the SAME packed polynomial collaboratively (128t
// = 4x bandwidth), then each warp independently NTTs and writes to its own
// output slot. Assumes same key across all 4 vectors in the block (realistic
// for batch signing).
__global__ void k8_shared_key_load(int32_t *g_polyvec, size_t g_polyvec_pitch,
                                   const uint8_t *g_polyvec_packed,
                                   size_t g_polyvec_packed_pitch) {
  constexpr int TOTAL_PACKED = DILITHIUM_K * POLYT0_PACKEDBYTES;
  __shared__ uint8_t s_packed[TOTAL_PACKED];

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x & 0x1f;
  int32_t regs[8];

  const int baseVectorIdx = blockIdx.x * 4;
  const int myVectorIdx = baseVectorIdx + warpId;

  // All 128 threads collaboratively load ALL K polynomials from baseVectorIdx.
  // 4x faster than a single warp, and only 1 global load for 4 signatures.
  const uint8_t *src =
      g_polyvec_packed + baseVectorIdx * g_polyvec_packed_pitch;
  for (int byte = threadIdx.x; byte < TOTAL_PACKED; byte += 128)
    s_packed[byte] = src[byte];
  __syncthreads();

  // Each warp independently unpacks from SMEM (L1 broadcast) and runs NTT.
  for (int k = 0; k < DILITHIUM_K; ++k) {
    const uint8_t *kpacked = &s_packed[k * POLYT0_PACKEDBYTES];
    for (size_t i = 0; i < 8; i++) {
      uint32_t t =
          (kpacked[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 0]) |
          (kpacked[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 1]
           << 8) |
          (kpacked[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 2]
           << 16) |
          (kpacked[i * 52 + (laneId / 8) * 13 + ((laneId & 7) / 2) * 3 + 3]
           << 24);
      t >>= (laneId & 7) * 13 - ((laneId & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    ntt_inner_shuffle_notranspose(regs);
    int32_t *g_poly = g_polyvec +
                      myVectorIdx * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    for (size_t i = 0; i < 8; i++)
      g_poly[i * 32 + laneId] = regs[i];
  }
}

// merge two loops into one and use registers to store intermediate poly
__global__ void k2(int32_t *g_polyvec, size_t g_polyvec_pitch,
                   const uint8_t *g_polyvec_packed,
                   size_t g_polyvec_packed_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N];
  int32_t regs[8];

  for (int k = 0; k < DILITHIUM_K; ++k) {
    // unpack
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    // ntt
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    ntt_inner(regs, s_ntt);
    for (size_t i = 0; i < 8; i++)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

// avoid smem bank conflict in ntt
__global__ void k3(int32_t *g_polyvec, size_t g_polyvec_pitch,
                   const uint8_t *g_polyvec_packed,
                   size_t g_polyvec_packed_pitch) {
  __shared__ int32_t s_ntt[DILITHIUM_N + 32];
  int32_t regs[8];

  for (int k = 0; k < DILITHIUM_K; ++k) {
    // unpack
    auto *g_poly_packed = g_polyvec_packed +
                          blockIdx.x * g_polyvec_packed_pitch +
                          k * POLYT0_PACKEDBYTES;
    for (size_t i = 0; i < 8; i++) {
      uint32_t t = (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 0]) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 1]
                    << 8) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 2]
                    << 16) |
                   (g_poly_packed[i * 52 + (threadIdx.x / 8) * 13 +
                                  ((threadIdx.x & 7) / 2) * 3 + 3]
                    << 24);
      t >>= (threadIdx.x & 7) * 13 - ((threadIdx.x & 7) / 2) * 3 * 8;
      t &= 0x1FFF;
      regs[i] = (1 << (DILITHIUM_D - 1)) - (int32_t)t;
    }
    // ntt
    int32_t *g_poly = g_polyvec +
                      blockIdx.x * g_polyvec_pitch / sizeof(int32_t) +
                      k * DILITHIUM_N;
    ntt_inner_1(regs, s_ntt);
    for (size_t i = 0; i < 8; i++)
      g_poly[threadIdx.x * 8 + i] = regs[i];
  }
}

__global__ void test_ntt_correctness() {
  __shared__ int32_t s_ntt[DILITHIUM_N + 32];
  int32_t regs[8];

  for (size_t i = 0; i < 8; ++i)
    regs[i] = 32 * i + threadIdx.x;

  ntt_inner(regs, s_ntt);

  printf("%d ", regs[0]);
  if (threadIdx.x == 0)
    printf("\n");

  for (size_t i = 0; i < 8; ++i)
    regs[i] = 32 * i + threadIdx.x;

  ntt_inner_1(regs, s_ntt);

  printf("%d ", regs[0]);
  if (threadIdx.x == 0)
    printf("\n");
}

__global__ void bench_ntt_radix8() {
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  int32_t regs[8];
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[32 * i + threadIdx.x];
  ntt_inner(regs, s_ntt);
  for (size_t i = 0; i < 8; i++)
    s_ntt[8 * threadIdx.x + i] = regs[i];
}

__global__ void bench_intt_radix8() {
  __shared__ int32_t s_ntt[DILITHIUM_N + 128];
  int32_t regs[8];
  for (size_t i = 0; i < 8; i++)
    regs[i] = s_ntt[8 * threadIdx.x + i];
  invntt_inner(regs, s_ntt);
  for (size_t i = 0; i < 8; i++)
    s_ntt[32 * i + threadIdx.x] = regs[i];
}

int main() {
  uint8_t *d_polyveck_packed;
  int32_t *d_polyveck;
  size_t d_polyveck_packed_pitch;
  size_t d_polyveck_pitch;

  cudaMallocPitch(&d_polyveck_packed, &d_polyveck_packed_pitch,
                  DILITHIUM_K * POLYT0_PACKEDBYTES, NTESTS);
  cudaMallocPitch(&d_polyveck, &d_polyveck_pitch,
                  DILITHIUM_K * DILITHIUM_N * sizeof(int32_t), NTESTS);

  // print header
  std::cout << "function,\t trials,min (M op/s),mean (M op/s),median (M "
               "op/s),stddev (M op/s)"
            << std::endl;

  NTTThroughputTimer timer_ntt("ntt radix8", NTESTS);
  NTTThroughputTimer timer_k6("k6_shfl_coal", NTESTS);
  NTTThroughputTimer timer_k3("k3_bank", NTESTS);
  NTTThroughputTimer timer_k2("k2_xxxx", NTESTS);

  for (size_t i = 0; i < 10000; ++i) {
    // timer_k0.start();
    // k0_unpack<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch,
    // d_polyveck_packed,
    //                           d_polyveck_packed_pitch);
    // k0_ntt<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch);
    // cudaDeviceSynchronize();
    // timer_k0.stop();

    timer_k2.start();
    k2<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch, d_polyveck_packed,
                       d_polyveck_packed_pitch);
    cudaDeviceSynchronize();
    timer_k2.stop();

    timer_k3.start();
    k3<<<NTESTS, 32>>>(d_polyveck, d_polyveck_pitch, d_polyveck_packed,
                       d_polyveck_packed_pitch);
    cudaDeviceSynchronize();
    timer_k3.stop();

    timer_k6.start();
    k6_unpack_fuse_ntt_shuffle_128t_coalesced<<<NTESTS, 128>>>(
        d_polyveck, d_polyveck_pitch, d_polyveck_packed,
        d_polyveck_packed_pitch);
    cudaDeviceSynchronize();
    timer_k6.stop();

    timer_ntt.start();
    bench_ntt_radix8<<<NTESTS, 32>>>();
    cudaDeviceSynchronize();
    timer_ntt.stop();
  }

  cudaFree(d_polyveck_packed);
  cudaFree(d_polyveck);

  CHECK_LAST_CUDA_ERROR();

  return 0;
}
