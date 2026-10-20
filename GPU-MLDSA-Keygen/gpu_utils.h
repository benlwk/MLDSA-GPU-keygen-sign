/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
      do {                                                                  \
          cudaError_t _e = (call);                                          \
          if (_e != cudaSuccess) {                                          \
              fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                      __FILE__, __LINE__, cudaGetErrorString(_e));          \
              exit(1);                                                      \
          }                                                                 \
      } while (0)

#endif // GPU_UTILS_H
