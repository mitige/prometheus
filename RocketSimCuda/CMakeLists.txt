cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# RSCUDA_BUILD_CPU_PARITY: build ONLY the GPU-less CPU<->CUDA parity harness
# (OPT-105). It host-compiles the real .cu/.cuh kernels with a plain C++17
# compiler, so it must configure on machines with NO CUDA toolkit / GPU.
# Declared before project() so we can skip enabling the CUDA language. Default
# OFF => the existing CUDA-required targets below are completely unaffected.
# ---------------------------------------------------------------------------
option(RSCUDA_BUILD_CPU_PARITY
    "Build the GPU-less CPU parity harness only (no CUDA toolkit required)" OFF)

if(RSCUDA_BUILD_CPU_PARITY)
    project(RocketSimCuda LANGUAGES CXX)

    # Single translation unit by design: cuda_host_shim.cuh turns __device__/etc.
    # into no-ops (so the real kernels host-compile and run bit-for-bit), then the
    # harness diffs them against an independent float64 reference of the same
    # algorithm. Exit code gates CI.
    add_executable(RocketSimCudaCpuParity tests/cpu_parity_harness.cpp)
    target_include_directories(RocketSimCudaCpuParity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(RocketSimCudaCpuParity PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)
    # Strict parity: disable FP contraction (FMA) so host rounding matches the
    # f64 reference. Verified identical PASS at -O0/-O2 with and without this.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(RocketSimCudaCpuParity PRIVATE -ffp-contract=off)
    endif()

    enable_testing()
    add_test(NAME RocketSimCudaCpuParity COMMAND RocketSimCudaCpuParity)

    return()  # skip all CUDA-requiring targets below
endif()

project(RocketSimCuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE CUDA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
                               "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

add_library(RocketSimCuda STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})

target_include_directories(RocketSimCuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(RocketSimCuda PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "75;86;89"
)

# Physics parity: MSVC (/fp:precise) does not contract mul+add into FMA, but
# nvcc does by default. Disable contraction so GPU arithmetic matches the
# reference RocketSim build operation-for-operation.
target_compile_options(RocketSimCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

target_link_libraries(RocketSimCuda PUBLIC CUDA::cudart)

# ---- Test executable ----
add_executable(RocketSimCudaTest tests/test_physics.cpp)
target_link_libraries(RocketSimCudaTest PRIVATE RocketSimCuda)
set_target_properties(RocketSimCudaTest PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

add_executable(RocketSimCudaGaeParity tests/test_gae_parity.cpp)
target_link_libraries(RocketSimCudaGaeParity PRIVATE RocketSimCuda)
set_target_properties(RocketSimCudaGaeParity PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

add_executable(RocketSimCudaBenchmark1024 tests/benchmark_1024.cpp)
target_link_libraries(RocketSimCudaBenchmark1024 PRIVATE RocketSimCuda)
target_compile_definitions(RocketSimCudaBenchmark1024 PRIVATE
    RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
set_target_properties(RocketSimCudaBenchmark1024 PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "75;86;89"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory("../GigaLearnCPP/RLGymCPP" RLGymCPPBuild)

    add_executable(RocketSimCpuCudaCompare tests/compare_cpu_cuda.cpp)
    target_include_directories(RocketSimCpuCudaCompare PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCpuCudaCompare PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCpuCudaCompare PRIVATE RocketSimCuda RocketSim)
    set_target_properties(RocketSimCpuCudaCompare PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimEnvPipelineBenchmark tests/benchmark_env_pipeline.cpp)
    target_include_directories(RocketSimEnvPipelineBenchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimEnvPipelineBenchmark PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimEnvPipelineBenchmark PRIVATE RocketSimCuda RLGymCPP RocketSim)
    set_target_properties(RocketSimEnvPipelineBenchmark PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimCudaTraceDivergence tests/trace_divergence.cpp)
    target_include_directories(RocketSimCudaTraceDivergence PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCudaTraceDivergence PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCudaTraceDivergence PRIVATE RocketSimCuda RocketSim)
    set_target_properties(RocketSimCudaTraceDivergence PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )

    add_executable(RocketSimCudaGameStateBridgeTest tests/test_gamestate_cuda.cpp)
    target_include_directories(RocketSimCudaGameStateBridgeTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../GigaLearnCPP/RLGymCPP/RocketSim/src
    )
    target_compile_definitions(RocketSimCudaGameStateBridgeTest PRIVATE
        RSCUDA_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(RocketSimCudaGameStateBridgeTest PRIVATE RocketSimCuda RLGymCPP RocketSim)
    set_target_properties(RocketSimCudaGameStateBridgeTest PROPERTIES
        CXX_STANDARD 20
        CUDA_ARCHITECTURES "75;86;89"
    )
endif()
