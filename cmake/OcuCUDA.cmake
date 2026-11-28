# OcuCUDA.cmake: build a CMake project's CUDA code with the ocu toolkit.
#
# This file sits in a toolkit folder (<build>/toolkit/cmake in this build
# tree, <prefix>/lib/ocu/toolkit/cmake when installed). Use it either as
# the toolchain file
#
#   cmake -S <project> -B <build> \
#         -DCMAKE_TOOLCHAIN_FILE=<toolkit>/cmake/OcuCUDA.cmake \
#         [-DOCU_TARGET=cpu|nvidia] [-DCMAKE_CUDA_ARCHITECTURES=80]
#
# or include() it before enable_language(CUDA) / project(... CUDA), or name
# it in CMAKE_PROJECT_TOP_LEVEL_INCLUDES (CMake 3.24 and later).
#
# What it sets, and why:
#   CMAKE_CUDA_COMPILER   <toolkit>/bin/ocu-nvcc-<OCU_TARGET> (a link to
#                         ocu-nvcc; the name selects the backend, so the
#                         choice holds whatever CMAKE_CUDA_FLAGS the project
#                         or the user sets). CMake identifies it from
#                         its --version output as a Clang-based CUDA
#                         compiler, which it is (LLVM Clang compiles host
#                         and device code), and then uses Clang's CUDA
#                         rules. The driver accepts those options as well
#                         as nvcc's, which projects add themselves.
#   CUDAToolkit_ROOT      <toolkit>: CMake looks for the toolkit there
#                         first. The toolkit has the files CMake checks
#                         for (bin/nvcc, a link to ocu-nvcc under the name
#                         CMake searches for; include/cuda_runtime.h, our
#                         header; lib/; an empty nvvm/libdevice folder).
#                         Nothing outside the toolkit is searched or run.
#   CMAKE_MODULE_PATH     <toolkit>/cmake/modules first: our
#                         FindCUDAToolkit, whose CUDA::cudart, CUDA::cublas
#                         ... targets name our libraries.
#   CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA
#                         OcuCUDARules.cmake: the CUDA runtime CMake links
#                         by itself (CMAKE_CUDA_RUNTIME_LIBRARY) is ours.
#   OCU_TARGET            cpu or nvidia (default: the environment
#                         variable OCU_TARGET, else cpu).
get_filename_component(OCU_TOOLKIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(NOT DEFINED OCU_TARGET)
  if(DEFINED ENV{OCU_TARGET} AND NOT "$ENV{OCU_TARGET}" STREQUAL "")
    set(OCU_TARGET "$ENV{OCU_TARGET}")
  else()
    set(OCU_TARGET cpu)
  endif()
endif()
if(NOT OCU_TARGET MATCHES "^(cpu|nvidia|opencl|intel|spirv)$")
  message(FATAL_ERROR "OcuCUDA: OCU_TARGET must be cpu, nvidia, opencl or intel, not '${OCU_TARGET}'")
endif()
set(OCU_TARGET "${OCU_TARGET}" CACHE STRING "Backend for CUDA code built by ocu-nvcc (cpu or nvidia)")

set(CMAKE_CUDA_COMPILER "${OCU_TOOLKIT_ROOT}/bin/ocu-nvcc-${OCU_TARGET}")
set(CUDAToolkit_ROOT "${OCU_TOOLKIT_ROOT}")
list(PREPEND CMAKE_MODULE_PATH "${OCU_TOOLKIT_ROOT}/cmake/modules")
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA "${OCU_TOOLKIT_ROOT}/cmake/OcuCUDARules.cmake")
# Targets without a CUDA_STANDARD: C++17, the language level nvcc compiles
# by default. Left alone, some CMake versions (seen on macOS) pass
# -std=gnu++03, which the CUDA headers cannot be compiled with. A project or
# target that sets its own standard still wins.
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()
# try_compile projects see the same choice.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES OCU_TARGET)
