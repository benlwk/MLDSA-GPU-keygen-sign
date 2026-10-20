# SPDX-License-Identifier: GPL-3.0-or-later
NVCC     := nvcc
CC       ?= gcc
LD       := g++
MODE     ?= 2
OPT_FLAG ?= -O3
NTT_OPT  ?= 0

ARCH     := -arch=sm_89 -gencode arch=compute_89,code=sm_89
CFLAGS   := $(OPT_FLAG) -std=c++17 $(ARCH) -DDILITHIUM_MODE=$(MODE) -DNTT_OPT=$(NTT_OPT) -Xcompiler -fopenmp -Xcompiler -U_GNU_SOURCE
CFLAGS_C := -O3 -Wall -Wextra -fPIC -DDILITHIUM_MODE=$(MODE) -DNTT_OPT=$(NTT_OPT) -I/usr/local/cuda/include

CPU_REF_DIR  ?= ../ref
CPU_REF_SRCS := $(CPU_REF_DIR)/sign.c \
                $(CPU_REF_DIR)/packing.c \
                $(CPU_REF_DIR)/polyvec.c \
                $(CPU_REF_DIR)/poly.c \
                $(CPU_REF_DIR)/ntt.c \
                $(CPU_REF_DIR)/reduce.c \
                $(CPU_REF_DIR)/rounding.c \
                $(CPU_REF_DIR)/fips202.c \
                $(CPU_REF_DIR)/symmetric-shake.c

CPU_REF_OBJS := $(notdir $(CPU_REF_SRCS:.c=.o))

GPU_OBJS := gpu_common.o gpu_keygen.o
VERIFY_TARGET := verify

TARGET   := benchmark

.SECONDARY: $(GPU_OBJS) $(CPU_REF_OBJS) benchmark.o verify.o

all: $(TARGET) $(VERIFY_TARGET)

%.o: $(CPU_REF_DIR)/%.c
	$(CC) $(CFLAGS_C) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

benchmark.o: benchmark.c
	$(CC) $(CFLAGS_C) -c $< -o $@

verify.o: verify.c
	$(CC) $(CFLAGS_C) -c $< -o $@

$(TARGET): benchmark.o $(GPU_OBJS)
	$(NVCC) -o $@ $^ -L/usr/local/cuda/lib64 -lcudart -Xcompiler -fopenmp

$(VERIFY_TARGET): verify.o $(GPU_OBJS) $(CPU_REF_OBJS)
	$(NVCC) -o $@ $^ -L/usr/local/cuda/lib64 -lcudart -Xcompiler -fopenmp

clean:
	rm -f $(TARGET) $(VERIFY_TARGET) *.o

.PHONY: all clean
