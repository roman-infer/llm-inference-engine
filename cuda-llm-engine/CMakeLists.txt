cmake_minimum_required(VERSION 3.18)

project(cuda_llm_engine LANGUAGES CUDA CXX)

# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 目标 GPU 架构。RTX 40 系笔记本为 sm_89（Ada）。
# 75=Turing, 80=A100, 86=RTX30, 89=RTX40, 90=H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Release 模式下保留行号信息，供 Nsight Compute 性能分析使用。
# 注意：使用 -lineinfo（单横杠）—— --lineinfo 不是有效的 nvcc 选项，
# nvcc 会将其转发给主机编译器（gcc），导致报错。
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# 引擎静态库：核心运行时 + kernel + 模型
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cu")
file(GLOB_RECURSE KERNEL_SOURCES  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")
file(GLOB_RECURSE MODEL_SOURCES   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cu")

add_library(cudallm STATIC
  ${CORE_SOURCES}
  ${KERNEL_SOURCES}
  ${MODEL_SOURCES}
)
target_include_directories(cudallm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cudallm PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cudallm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ---------------------------------------------------------------------------
# 应用程序
# ---------------------------------------------------------------------------
add_executable(generate     src/apps/generate.cu)
target_link_libraries(generate PRIVATE cudallm)

add_executable(bench_gemm   src/apps/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE cudallm)

add_executable(bench_attention src/apps/bench_attention.cu)
target_link_libraries(bench_attention PRIVATE cudallm)

# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
add_executable(test_gemm    tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE cudallm)

add_executable(test_ops     tests/test_ops.cu)
target_link_libraries(test_ops PRIVATE cudallm)

# 便捷目标：运行所有测试
enable_testing()
add_test(NAME gemm COMMAND test_gemm)
add_test(NAME ops  COMMAND test_ops)
