# Modified in 2026: tests against the FIPS 204 reference implementation.

# test source files
set(TEST_SRCS test_vectors.cu prng.c)

# Dilithium 2 test vectors
add_executable(test_cuDilithium2 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium2 PRIVATE DILITHIUM_MODE=2)
target_link_libraries(test_cuDilithium2 cuDilithium2)

# Dilithium 3 test vectors
add_executable(test_cuDilithium3 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium3 PRIVATE DILITHIUM_MODE=3)
target_link_libraries(test_cuDilithium3 cuDilithium3)

# Dilithium 5 test vectors
add_executable(test_cuDilithium5 ${TEST_SRCS})
target_compile_definitions(test_cuDilithium5 PRIVATE DILITHIUM_MODE=5)
target_link_libraries(test_cuDilithium5 cuDilithium5)

# FIPS 204 reference (pq-crystals, ref/) and the GPU-vs-reference cross-check
set(REF_DIR ${CMAKE_SOURCE_DIR}/ref)
set(REF_SRCS ${REF_DIR}/sign.c ${REF_DIR}/packing.c ${REF_DIR}/polyvec.c
             ${REF_DIR}/poly.c ${REF_DIR}/ntt.c ${REF_DIR}/reduce.c
             ${REF_DIR}/rounding.c ${REF_DIR}/fips202.c
             ${REF_DIR}/symmetric-shake.c ref_wrap.c)
foreach(mode 2 3 5)
    add_library(mldsa_ref${mode} STATIC ${REF_SRCS})
    target_compile_definitions(mldsa_ref${mode} PRIVATE DILITHIUM_MODE=${mode})

    add_executable(test_fips204_ref${mode} test_fips204_ref.cu)
    target_compile_definitions(test_fips204_ref${mode} PRIVATE DILITHIUM_MODE=${mode})
    target_link_libraries(test_fips204_ref${mode} cuDilithium${mode} mldsa_ref${mode} fips202)
    add_test(NAME test_fips204_ref${mode} COMMAND test_fips204_ref${mode})
endforeach()

# add CTests for test vectors
# we cut off the official test vectors to only check m,pk,sk,sig
# FIPS 204 (ML-DSA) hashes, generated after test_fips204_ref passed for all
# configurations; they do not apply to TEST_SINGLE_KEY/USE_CACHED_SK_SINGLE.
add_test(NAME test_cuDilithium2 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium2>\" && echo \"844db26d80ea7b213087b4cc290cac9e58761ee717e798c59b60522f0b61394f tvecs2\" | sha256sum --check")
add_test(NAME test_cuDilithium3 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium3>\" && echo \"7cd026f4877f0bd870452fd300b53b5eb1e00fd3ce48f043da75f9d310b422c1 tvecs3\" | sha256sum --check")
add_test(NAME test_cuDilithium5 COMMAND sh -c "\"$<TARGET_FILE:test_cuDilithium5>\" && echo \"3953cd89d88a576567ebc06d9ecda634593e28cd2d5fc08a2de4cead7454da6c tvecs5\" | sha256sum --check")
