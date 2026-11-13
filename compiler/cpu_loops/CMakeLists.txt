# The CPU path's whole-block pass (ocu_cpu_loops.cpp, docs/cpu-region-loops.md
# stage 1): an LLVM pass plugin that Clang loads with -fpass-plugin while it
# compiles CPU-path .cu files (compiler/probe/ocu_cc.py).
#
# Built when LLVM 18 or later is found with its CMake package and the clang++
# of the same LLVM, on a host that is not cross-compiling and not Windows.
# Elsewhere (MinGW, the ARM64 cross build, no LLVM development files) the
# CPU path builds as before and every kernel runs on fibers.
#
# Sets, for cmake/OcuCpuPath.cmake and the toolkit folder:
#   OCU_CPU_LOOPS_PLUGIN  the plugin file (empty when not built)
#   OCU_CPU_LOOPS_ENV     environment for ocu_cc: OCU_CPU_LOOPS=<plugin|off>
set(OCU_CPU_LOOPS_PLUGIN "" CACHE INTERNAL "")
set(OCU_CPU_LOOPS_ENV "OCU_CPU_LOOPS=off" CACHE INTERNAL "")
if(NOT OCU_CPU_LOOPS)
  message(STATUS "ocu: CPU-path whole-block pass off (OCU_CPU_LOOPS=OFF): kernels run on fibers")
  return()
endif()
if(WIN32 OR CMAKE_CROSSCOMPILING)
  message(STATUS "ocu: CPU-path whole-block pass not built (Windows or cross build): kernels run on fibers")
  return()
endif()

find_package(LLVM CONFIG QUIET HINTS ${OCU_LLVM_DIR} /usr/lib/llvm-18/lib/cmake/llvm)
if(NOT LLVM_FOUND OR LLVM_VERSION_MAJOR LESS 18)
  message(STATUS "ocu: CPU-path whole-block pass not built (no LLVM 18+ CMake package; "
                 "set OCU_LLVM_DIR): kernels run on fibers")
  return()
endif()
set(_ocu_loops_clang ${LLVM_TOOLS_BINARY_DIR}/clang++)
if(NOT EXISTS ${_ocu_loops_clang})
  message(STATUS "ocu: CPU-path whole-block pass not built (no ${_ocu_loops_clang}): kernels run on fibers")
  return()
endif()

add_library(ocu_cpu_loops MODULE ocu_cpu_loops.cpp)
target_include_directories(ocu_cpu_loops SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(_ocu_llvm_defs NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(ocu_cpu_loops PRIVATE ${_ocu_llvm_defs})
# Loaded into Clang: no sanitizer runtime there, and LLVM's own settings.
target_compile_options(ocu_cpu_loops PRIVATE -fno-sanitize=all)
target_link_options(ocu_cpu_loops PRIVATE -fno-sanitize=all)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ocu_cpu_loops PRIVATE -fno-rtti)
endif()
if(APPLE)
  target_link_options(ocu_cpu_loops PRIVATE -undefined dynamic_lookup)
endif()
# The same file name on every host (ocu_cc.py looks for it).
set_target_properties(ocu_cpu_loops PROPERTIES
  PREFIX "lib" SUFFIX ".so" OUTPUT_NAME ocu_cpu_loops
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  CXX_VISIBILITY_PRESET hidden)
# The Clang the plugin loads into, next to it.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/libocu_cpu_loops.clang "${_ocu_loops_clang}\n")

set(OCU_CPU_LOOPS_PLUGIN ${CMAKE_CURRENT_BINARY_DIR}/libocu_cpu_loops.so CACHE INTERNAL "")
set(OCU_CPU_LOOPS_ENV "OCU_CPU_LOOPS=${OCU_CPU_LOOPS_PLUGIN}" CACHE INTERNAL "")
message(STATUS "ocu: CPU-path whole-block pass: LLVM ${LLVM_PACKAGE_VERSION}, ${_ocu_loops_clang}")

install(TARGETS ocu_cpu_loops LIBRARY DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ocu)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libocu_cpu_loops.clang
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ocu)
