/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_PRIVATE_H
#define GPU_PRIVATE_H

#include <stdint.h>
#include <stddef.h>

struct BenchCtx {
    int       batch;
    int       tpb;           /* threads per block */
    int       blocks;
    size_t    msg_len;

    /* device buffers */
    uint8_t  *d_seeds;
    uint8_t  *d_pk, *d_sk;
};

#endif // GPU_PRIVATE_H
