cmake_minimum_required(VERSION 3.18)
project(RIPRA LANGUAGES C)

# Find necessary packages
find_package(OpenMP REQUIRED)

# Try to find CUDA if available, but make it optional if it is just a skeleton for now
# The audit mentioned it was a skeleton. We'll enable CUDA if possible.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_compile_definitions(HAVE_CUDA)
endif()

add_library(ripra SHARED
  rippra/src/centroid.c rippra/src/recon.c rippra/src/io.c
  rippra/src/la.c rippra/src/stream.c rippra/src/rippra_api.c
)

target_link_libraries(ripra PUBLIC OpenMP::OpenMP_C)
target_include_directories(ripra PUBLIC rippra/include/)

# Tests
enable_testing()

add_executable(test_recon rippra/tests/test_recon.c)
target_link_libraries(test_recon PRIVATE ripra)
add_test(NAME recon_test COMMAND test_recon)

add_executable(test_full_pipeline rippra/tests/test_full_pipeline.c)
target_link_libraries(test_full_pipeline PRIVATE ripra)
add_test(NAME full_pipeline_test COMMAND test_full_pipeline)

# Set working directory for tests so they can load system.conf and data files
set_tests_properties(recon_test PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra")
set_tests_properties(full_pipeline_test PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rippra")
