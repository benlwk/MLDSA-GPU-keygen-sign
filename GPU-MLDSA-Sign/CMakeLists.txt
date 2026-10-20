# Modified in 2026: options for the fused and staged signing kernels, the warp
# NTT and key caching; the final design of the paper is the default.

# deprecate old CMake versions
cmake_minimum_required(VERSION 3.20)

# set the project name, languages, version, and description
project(cuDilithium LANGUAGES C CXX CUDA VERSION 1.0.0 DESCRIPTION "Dilithium in CUDA")

# CUDA only: Enables separate compilation of device code
# enable separable compilation for all CUDA files in this project
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# deprecate old CUDA versions
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA Toolkit 11.0 or higher is required")
endif()

# if CUDA architecture is not specified, use the following
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# The paper's final design (warp NTT with whole butterflies, mask
# pre-generation, staged challenge hashing, single-key cache) is the default.
# Configure a fresh build directory with -DFINAL_DESIGN=OFF to select another
# configuration from the individual options below.
option(FINAL_DESIGN "Default the options below to the paper's final design" ON)
if(FINAL_DESIGN)
    set(FINAL_DEFAULT ON)
else()
    set(FINAL_DEFAULT OFF)
endif()

# Options for different optimizations
option(USE_128T_SIGN "Use 128T signature optimizations" OFF)
option(USE_CP_NTT "Use CP NTT kernel" OFF)
option(USE_128T_KEYPAIR "Use 128 threads with warp-shuffle NTT for gpu_keypair" OFF)
option(USE_PERSISTENT "Use persistent kernel for signing" OFF)
option(USE_PERSISTENT_WARPNTT "Use persistent kernel for signing with warp NTT" ${FINAL_DEFAULT})
option(USE_CACHED_SK_SINGLE "Use single shared SK cached on GPU" ${FINAL_DEFAULT})
option(USE_CACHED_SK "Cache an expanded SK in every signature slot" OFF)
if(USE_PERSISTENT AND USE_PERSISTENT_WARPNTT)
    message(FATAL_ERROR "USE_PERSISTENT and USE_PERSISTENT_WARPNTT cannot be combined (use -DFINAL_DESIGN=OFF)")
endif()
if(USE_CACHED_SK AND USE_CACHED_SK_SINGLE)
    message(FATAL_ERROR "USE_CACHED_SK and USE_CACHED_SK_SINGLE cannot be combined (use -DUSE_CACHED_SK_SINGLE=OFF)")
endif()


if(USE_128T_SIGN)
    add_compile_definitions(USE_128T_SIGN)
endif()
if(USE_CACHED_SK)
    add_compile_definitions(USE_CACHED_SK)
endif()
if(USE_CACHED_SK_SINGLE)
    add_compile_definitions(USE_CACHED_SK_SINGLE)
endif()
# Test-only: sign every test vector with one key (no caching), giving a
# reference output to check USE_CACHED_SK_SINGLE against.
option(TEST_SINGLE_KEY "Test vectors use a single key for all signatures" OFF)
if(TEST_SINGLE_KEY)
    add_compile_definitions(TEST_SINGLE_KEY)
endif()
# Fused signing kernel tuning (see params.h): launch only the working warps
# (ML-DSA-65), and/or request more resident blocks per SM (fewer registers).
option(FUSED_TRIM_WARPS "Fused sign kernel: launch only working warps" OFF)
if(FUSED_TRIM_WARPS)
    add_compile_definitions(FUSED_TRIM_WARPS)
endif()
set(FUSED_MIN_BLOCKS "" CACHE STRING "Fused sign kernel: min resident blocks per SM (empty = default)")
if(FUSED_MIN_BLOCKS)
    add_compile_definitions(FUSED_MIN_BLOCKS=${FUSED_MIN_BLOCKS})
endif()
# Fused sign kernel (warp NTT): exchange-then-butterfly warp NTT.
option(FUSED_WARPX "Fused sign kernel: exchange-then-butterfly warp NTT" ${FINAL_DEFAULT})
if(FUSED_WARPX)
    add_compile_definitions(FUSED_WARPX)
endif()
# Fused sign kernel: generate the next attempt's y during the challenge phase.
option(FUSED_OVERLAP_Y "Fused sign kernel: overlap next ExpandMask with challenge" OFF)
if(FUSED_OVERLAP_Y)
    add_compile_definitions(FUSED_OVERLAP_Y)
endif()
# Fused sign kernel: pre-generate y of the first attempts in a separate kernel
# (one Keccak state per thread). Pool rows grow by SIGN_POOL_YPRE_BYTES.
option(FUSED_PREGEN_Y "Fused sign kernel: pre-generate y with thread-per-state Keccak" ${FINAL_DEFAULT})
if(FUSED_PREGEN_Y)
    add_compile_definitions(FUSED_PREGEN_Y)
endif()
set(FUSED_PREGEN_ATTEMPTS "" CACHE STRING "Attempts with pre-generated y (empty = default 8)")
if(FUSED_PREGEN_ATTEMPTS)
    add_compile_definitions(FUSED_PREGEN_ATTEMPTS=${FUSED_PREGEN_ATTEMPTS})
endif()
# Fused sign kernel: run the first attempts as commit / batched challenge hash /
# respond passes over the batch (requires FUSED_PREGEN_Y).
option(FUSED_STAGED "Fused sign kernel: staged rounds with batched challenge hashing" ${FINAL_DEFAULT})
if(FUSED_STAGED)
    add_compile_definitions(FUSED_STAGED)
endif()
set(FUSED_STAGED_ROUNDS "" CACHE STRING "Staged rounds (empty = FUSED_PREGEN_ATTEMPTS)")
if(FUSED_STAGED_ROUNDS)
    add_compile_definitions(FUSED_STAGED_ROUNDS=${FUSED_STAGED_ROUNDS})
endif()
# Profiling only: per-phase cycle counters in the fused sign kernel, printed
# by the benchmark (adds clock reads and atomics; do not use for throughput).
option(SIGN_PHASE_PROFILE "Fused sign kernel: per-phase cycle counters" OFF)
if(SIGN_PHASE_PROFILE)
    add_compile_definitions(SIGN_PHASE_PROFILE)
endif()

# Test-only: override the number of test vectors (default 10000), e.g. to
# keep compute-sanitizer racecheck within host memory.
set(TEST_NVECTORS "" CACHE STRING "Number of test vectors (empty = default)")
if(TEST_NVECTORS)
    add_compile_definitions(NVECTORS=${TEST_NVECTORS})
endif()
if(USE_CP_NTT)
    # The multi-kernel signing path hangs with the CP NTT kernel but without
    # the 128-thread kernels; the persistent kernels set both.
    if(NOT USE_128T_SIGN AND NOT USE_PERSISTENT AND NOT USE_PERSISTENT_WARPNTT)
        message(FATAL_ERROR "USE_CP_NTT requires USE_128T_SIGN")
    endif()
    add_compile_definitions(USE_CP_NTT)
endif()
if(USE_128T_KEYPAIR)
    add_compile_definitions(USE_128T_KEYPAIR)
endif()

if(USE_PERSISTENT)
    add_compile_definitions(USE_PERSISTENT)
    # USE_PERSISTENT requires these options
    add_compile_definitions(USE_128T_SIGN)
    add_compile_definitions(USE_CP_NTT)
endif()
if(USE_PERSISTENT_WARPNTT)
    add_compile_definitions(USE_PERSISTENT_WARPNTT)
    # USE_PERSISTENT_WARPNTT requires these options
    add_compile_definitions(USE_128T_SIGN)
    add_compile_definitions(USE_CP_NTT)
endif()
# enable CTest testing
enable_testing()

# set the output directories, so that the executables are in bin/ and the libraries are in lib/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# include headers in this project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# use OpenMP for CPU multi-threading
find_package(OpenMP REQUIRED)
if(OPENMP_C_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    link_libraries(${OpenMP_C_LIBRARIES})
endif()

# add subdirectory of library source files
add_subdirectory(src)

# add subdirectory of test source files
add_subdirectory(tests)

# add subdirectory of benchmark source files
add_subdirectory(benches)
