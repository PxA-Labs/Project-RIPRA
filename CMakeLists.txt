cmake_minimum_required(VERSION 3.18)
project(RIPRA LANGUAGES C)

# Linear algebra: la.c is a vendored, dependency-free implementation providing
# LU decomposition (Doolittle with partial pivoting) and pseudo-inverse via
# one-sided Jacobi SVD. No BLAS/LAPACK is required or used. See rippra/src/la.c
# for design notes.

# ---- Options ---------------------------------------------------------------
option(RIPRA_BUILD_SHARED      "Build shared library (DLL/SO)" OFF)
option(RIPRA_BUILD_TESTS       "Build test executables"       ON)
option(RIPRA_BUILD_BENCHMARKS  "Build benchmark executables"  OFF)

# ---- Dependencies ----------------------------------------------------------
find_package(OpenMP REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_compile_definitions(HAVE_CUDA)
endif()

# ---- Library --------------------------------------------------------------
set(RIPRA_SOURCES
  rippra/src/centroid.c rippra/src/recon.c rippra/src/io.c
  rippra/src/la.c rippra/src/stream.c rippra/src/rippra_api.c
  rippra/src/simd.c
)

# simd_avx2.c contains AVX2 intrinsics. It is intentionally limited to
# non-Windows x86 GCC/Clang targets; ARM and Windows use the scalar path.
if(NOT WIN32 AND
   CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x86|i[3-6]86)$")
  list(APPEND RIPRA_SOURCES rippra/src/simd_avx2.c)
  set_property(SOURCE rippra/src/simd_avx2.c APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma)
endif()

# Add CUDA kernel sources if CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
  set(RIPPRA_CUDA_SOURCES
    rippra/cuda/centroid_kernels.cu
    rippra/cuda/dm_kernels.cu
    rippra/cuda/matrix_kernels.cu
  )
  list(APPEND RIPRA_SOURCES ${RIPPRA_CUDA_SOURCES})
endif()

if(RIPRA_BUILD_SHARED)
  add_library(ripra SHARED ${RIPRA_SOURCES})
  target_compile_definitions(ripra PRIVATE BUILD_RIPRA_DLL)
else()
  add_library(ripra STATIC ${RIPRA_SOURCES})
endif()

target_compile_definitions(ripra PRIVATE _POSIX_SOURCE)
target_link_libraries(ripra PUBLIC OpenMP::OpenMP_C)
if(UNIX)
  target_link_libraries(ripra PUBLIC m)
endif()
target_include_directories(ripra PUBLIC rippra/include/)

# ---- Tests -----------------------------------------------------------------
if(RIPRA_BUILD_TESTS)
  enable_testing()

  macro(ripra_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ripra)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra")
  endmacro()

  ripra_add_test(test_recon          rippra/tests/test_recon.c)
  ripra_add_test(test_full_pipeline  rippra/tests/test_full_pipeline.c)
  ripra_add_test(test_golden_values  rippra/tests/test_golden_values.c)
  ripra_add_test(test_closed_loop    rippra/tests/test_closed_loop.c)
  ripra_add_test(test_malformed_input rippra/tests/test_malformed_input.c)
  ripra_add_test(test_centroid       rippra/tests/test_centroid.c)
  ripra_add_test(test_la             rippra/tests/test_la.c)
  ripra_add_test(test_stream         rippra/tests/test_stream.c)
  ripra_add_test(test_io             rippra/tests/test_io.c)
  ripra_add_test(test_minimal        rippra/tests/test_minimal.c)
  ripra_add_test(doc_example         rippra/tests/doc_example.c)
  ripra_add_test(test_simd           rippra/tests/test_simd.c)
  ripra_add_test(test_dm_safety      rippra/tests/test_dm_safety.c)

  # CUDA test: compile check (requires GPU to actually run; CI runs
  # continue-on-error).  Mark .c as CUDA so nvcc compiles it.
  if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rippra/cuda/rippra_cuda.h")
    set_source_files_properties(rippra/tests/test_cuda.c PROPERTIES LANGUAGE CUDA)
    ripra_add_test(test_cuda rippra/tests/test_cuda.c)

    add_executable(test_cuda_dm_safety rippra/tests/test_cuda_dm_safety.cu)
    target_link_libraries(test_cuda_dm_safety PRIVATE ripra)
    add_test(NAME test_cuda_dm_safety COMMAND test_cuda_dm_safety)
    set_tests_properties(test_cuda_dm_safety PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra"
      SKIP_RETURN_CODE 77)
  endif()
endif()

# ---- Benchmarks ------------------------------------------------------------
if(RIPRA_BUILD_BENCHMARKS)
  add_executable(benchmark_centroid rippra/tests/benchmark_centroid.c)
  target_link_libraries(benchmark_centroid PRIVATE ripra)

  add_executable(benchmark_openmp rippra/tests/benchmark_openmp.c)
  target_link_libraries(benchmark_openmp PRIVATE ripra)

  add_executable(benchmark_e2e rippra/tests/benchmark_end_to_end.c)
  target_link_libraries(benchmark_e2e PRIVATE ripra)
  set_target_properties(benchmark_e2e PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra/bin")

  add_executable(benchmark_simd rippra/tests/benchmark_simd.c)
  target_link_libraries(benchmark_simd PRIVATE ripra)

  if(CMAKE_CUDA_COMPILER)
    add_executable(benchmark_cuda_vs_cpu rippra/tests/benchmark_cuda_vs_cpu.cu)
    target_link_libraries(benchmark_cuda_vs_cpu PRIVATE ripra)
    set_target_properties(benchmark_cuda_vs_cpu PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra/bin")
  endif()
endif()
