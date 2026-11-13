add_library(ocu_bench STATIC ocu_bench.c)
target_include_directories(ocu_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OCU_BACKEND_CPU)
  add_executable(bench_blas bench_blas.c)
  target_link_libraries(bench_blas PRIVATE ocu_bench ocu_blas)
endif()

# GEMM throughput of the BLAS library API on an NVIDIA GPU.
if(TARGET ocu_cublas_nvidia)
  ocu_add_nvidia_cu_executable(bench_cublas bench_cublas.cu LIBRARIES ocu_cublas_nvidia ocu_bench)
endif()
# The same benchmark on the CPU path (pass small sizes, e.g. 256 512 1024).
if(TARGET ocu_cublas_cpu)
  ocu_add_cpu_cu_executable(bench_cublas_cpu bench_cublas.cu LIBRARIES ocu_cublas_cpu ocu_bench)
endif()

# Stage 0 of docs/cpu-region-loops.md: llama.cpp's mul_mat_vec_q as fibers
# and as region loops (written by hand), same inputs, same bits. Not part
# of the runtime. The loop form is compiled three times: vectoriser off,
# default, x86-64-v3 (contraction off so that the bits stay the same).
# -DOCU_RL_VEC_REPORT=ON prints the vectoriser's remarks while building.
if(OCU_BACKEND_CPU AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT WIN32)
  option(OCU_RL_VEC_REPORT "print vectoriser remarks for the region-loop benchmark" OFF)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_rl_novec -fno-vectorize -fno-slp-vectorize)
    set(_rl_report -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
  else()
    set(_rl_novec -fno-tree-vectorize -fno-tree-slp-vectorize)
    set(_rl_report -fopt-info-vec-optimized -fopt-info-vec-missed)
  endif()
  if(NOT OCU_RL_VEC_REPORT)
    set(_rl_report)
  endif()
  foreach(v scalar vec avx2)
    add_library(rl_mmvq_${v} STATIC region_loops/mmvq_loop.cpp)
    target_compile_definitions(rl_mmvq_${v} PRIVATE MMVQ_LOOP_NAME=mmvq_loop_${v})
    target_compile_options(rl_mmvq_${v} PRIVATE -O3)
  endforeach()
  target_compile_options(rl_mmvq_scalar PRIVATE ${_rl_novec})
  target_compile_options(rl_mmvq_vec PRIVATE ${_rl_report})
  target_compile_options(rl_mmvq_avx2 PRIVATE -march=x86-64-v3 -ffp-contract=off ${_rl_report})
  find_package(Threads REQUIRED)
  ocu_add_cpu_cu_executable(bench_region_loops region_loops/bench_region_loops.cu region_loops/mmvq_pool.cpp
                            LIBRARIES rl_mmvq_scalar rl_mmvq_vec rl_mmvq_avx2 Threads::Threads)
endif()
