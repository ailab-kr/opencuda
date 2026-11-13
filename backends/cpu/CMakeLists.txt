find_package(Threads REQUIRED)

# Fiber switch: hand-written on x86-64 and AArch64, ucontext elsewhere or
# when OCU_CPU_UCONTEXT is ON (useful to compare the two).
option(OCU_CPU_UCONTEXT "Use ucontext for CPU backend fibers" OFF)
set(_ocu_fast_switch OFF)
if(NOT WIN32 AND NOT OCU_CPU_UCONTEXT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64|ARM64)$")
  set(_ocu_fast_switch ON)
endif()

set(_ocu_cpu_sources
  cpu_backend.c
  cpu_memory.c
  cpu_queue.c
  cpu_exec.c
  cpu_fiber.c
  cpu_blas.c
  cpu_ops.c
  cpu_gemm.c
  cpu_gemm_generic.c
  cpu_gemm_avx2.c)
# The op table's GEMM has a variant for x86-64 with AVX2, FMA and F16C,
# chosen at run time when the processor has them (cpu_gemm.c).
set(_ocu_gemm_avx2 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set(_ocu_gemm_avx2 ON)
  set_source_files_properties(cpu_gemm_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
if(_ocu_fast_switch)
  enable_language(ASM)
  list(APPEND _ocu_cpu_sources cpu_fiber_switch.S)
endif()

add_library(ocu_backend_cpu STATIC ${_ocu_cpu_sources})
target_include_directories(ocu_backend_cpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_backend_cpu PROPERTIES EXPORT_NAME backend_cpu)
install(TARGETS ocu_backend_cpu EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
# Public headers only: the device-side API and the backend entry point.
install(FILES ocu_backend_cpu.h ocu_cpu_kernel.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
target_link_libraries(ocu_backend_cpu PUBLIC ocu_backend_interface Threads::Threads)
target_link_libraries(ocu_backend_cpu PRIVATE ocu_ops_ref)
if(NOT APPLE)
  target_link_libraries(ocu_backend_cpu PUBLIC m)
endif()
if(_ocu_gemm_avx2)
  target_compile_definitions(ocu_backend_cpu PRIVATE OCU_CPU_GEMM_AVX2)
endif()
if(_ocu_fast_switch)
  target_compile_definitions(ocu_backend_cpu PRIVATE OCU_CPU_FAST_SWITCH)
  message(STATUS "ocu: CPU backend fibers use the fast register switch")
else()
  if(WIN32)
    message(STATUS "ocu: CPU backend fibers use Windows fibers")
  else()
    message(STATUS "ocu: CPU backend fibers use ucontext")
  endif()
endif()
