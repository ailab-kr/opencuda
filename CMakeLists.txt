cmake_minimum_required(VERSION 3.20)
project(ocu VERSION 0.0.1 LANGUAGES C CXX)

option(OCU_BACKEND_CPU "Build the CPU reference backend" ON)
option(OCU_BACKEND_NVIDIA "Build the NVIDIA GPU backend (uses the installed driver at run time)" ${UNIX})
option(OCU_BACKEND_METAL "Build the Apple GPU (Metal) backend" ${APPLE})
option(OCU_BACKEND_AMD "Build the AMD GPU backend (not available yet)" OFF)
option(OCU_BACKEND_INTEL "Build the Intel GPU backend (not available yet)" OFF)
option(OCU_BUILD_TESTS "Build smoke tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Python3 COMPONENTS Interpreter)

include(GNUInstallDirs)
# All public headers go to one folder, so `#include <cuda_runtime.h>` finds
# ours and never a vendor's.
set(OCU_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/ocu)
set(OCU_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Ocu)
# ocu_add_cpu_cu_executable() in this tree: the driver from the source tree.
set(OCU_CC_SCRIPT ${PROJECT_SOURCE_DIR}/compiler/probe/ocu_cc.py)
set(OCU_CUDART_TARGET ocu_cudart)
file(GLOB OCU_CC_DEPENDS ${PROJECT_SOURCE_DIR}/runtime/include/*.h)
list(APPEND OCU_CC_DEPENDS ${PROJECT_SOURCE_DIR}/backends/cpu/ocu_cpu_kernel.h)

add_subdirectory(backends)
add_subdirectory(runtime)
add_subdirectory(libs)
add_subdirectory(tuning)
# The toolkit folder for other build systems: ocu-nvcc, headers, libraries
# and the CMake files that make them a project's CUDA toolkit.
include(OcuToolkit)

# Install: libraries, headers, the CPU-path driver, and a CMake package so
# other projects can `find_package(Ocu)` and use ocu::cudart and
# ocu_add_cpu_cu_executable().
install(PROGRAMS compiler/probe/ocu_cc.py DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ocu)
install(FILES compiler/probe/cpu_rewrite.py DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ocu)
install(EXPORT OcuTargets NAMESPACE ocu:: DESTINATION ${OCU_INSTALL_CMAKEDIR})
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/OcuConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OcuConfig.cmake
  INSTALL_DESTINATION ${OCU_INSTALL_CMAKEDIR}
  PATH_VARS CMAKE_INSTALL_LIBEXECDIR OCU_INSTALL_INCLUDEDIR)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OcuConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OcuConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/OcuConfigVersion.cmake
              cmake/OcuCpuPath.cmake
        DESTINATION ${OCU_INSTALL_CMAKEDIR})

if(OCU_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
