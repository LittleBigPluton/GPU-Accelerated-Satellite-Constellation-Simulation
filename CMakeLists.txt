cmake_minimum_required(VERSION 3.20)

project(
    GPUAcceleratedSatelliteConstellationSimulation
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)

# ---------------------------------------------------------------------------
# Shared simulation code
# ---------------------------------------------------------------------------

add_library(
    simulation_core
    STATIC
    src/satellite.cu
    src/ground_point.cu
    src/check_covered.cu
)

target_include_directories(
    simulation_core
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(
    simulation_core
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# CPU satellite-position simulation
# ---------------------------------------------------------------------------

add_executable(
    cpu_simulation
    cpu_simulation.cpp
)

target_link_libraries(
    cpu_simulation
    PRIVATE
    simulation_core
)

# ---------------------------------------------------------------------------
# CPU coverage simulation
# ---------------------------------------------------------------------------

add_executable(
    cpu_coverage_simulation
    cpu_coverage_simulation.cpp
)

target_link_libraries(
    cpu_coverage_simulation
    PRIVATE
    simulation_core
)

# ---------------------------------------------------------------------------
# GPU simulation
# ---------------------------------------------------------------------------

add_executable(
    gpu_simulation
    gpu_simulation.cu
    src/cuda_info.cu
)

target_include_directories(
    gpu_simulation
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    gpu_simulation
    PRIVATE
    simulation_core
)

set_target_properties(
    gpu_simulation
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

enable_testing()


# Satellite unit tests
add_executable(
    test_satellite
    tests/test_satellite.cpp
)

target_link_libraries(
    test_satellite
    PRIVATE
    simulation_core
)

add_test(
    NAME satellite
    COMMAND test_satellite
)


# Ground-point unit tests
add_executable(
    test_ground_point
    tests/test_ground_point.cpp
)

target_link_libraries(
    test_ground_point
    PRIVATE
    simulation_core
)

add_test(
    NAME ground_point
    COMMAND test_ground_point
)


# Coverage unit tests
add_executable(
    test_coverage
    tests/test_coverage.cpp
)

target_link_libraries(
    test_coverage
    PRIVATE
    simulation_core
)

add_test(
    NAME coverage
    COMMAND test_coverage
)


# CPU/GPU consistency integration test
add_executable(
    test_cpu_gpu_consistency
    tests/test_cpu_gpu_consistency.cu
)

target_link_libraries(
    test_cpu_gpu_consistency
    PRIVATE
    simulation_core
)

target_include_directories(
    test_cpu_gpu_consistency
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(
    test_cpu_gpu_consistency
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(
    NAME cpu_gpu_consistency
    COMMAND test_cpu_gpu_consistency
)

# Return code 77 means the test was skipped because
# no CUDA-capable device was available.
set_tests_properties(
    cpu_gpu_consistency
    PROPERTIES
    SKIP_RETURN_CODE 77
)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

add_executable(
    satellite_benchmark
    benchmarks/benchmark.cu
)

target_include_directories(
    satellite_benchmark
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    satellite_benchmark
    PRIVATE
    simulation_core
)

set_target_properties(
    satellite_benchmark
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
