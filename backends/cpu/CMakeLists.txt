find_package(Threads REQUIRED)

# Fiber switch: hand-written on x86-64 and AArch64, ucontext elsewhere or
# when OCU_CPU_UCONTEXT is ON (useful to compare the two).
option(OCU_CPU_UCONTEXT "Use ucontext for CPU backend fibers" OFF)
set(_ocu_fast_switch OFF)
if(NOT OCU_CPU_UCONTEXT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64|ARM64)$")
  set(_ocu_fast_switch ON)
endif()

set(_ocu_cpu_sources
  cpu_backend.c
  cpu_memory.c
  cpu_queue.c
  cpu_exec.c
  cpu_fiber.c
  cpu_blas.c)
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
if(NOT APPLE)
  target_link_libraries(ocu_backend_cpu PUBLIC m)
endif()
if(_ocu_fast_switch)
  target_compile_definitions(ocu_backend_cpu PRIVATE OCU_CPU_FAST_SWITCH)
  message(STATUS "ocu: CPU backend fibers use the fast register switch")
else()
  message(STATUS "ocu: CPU backend fibers use ucontext")
endif()
