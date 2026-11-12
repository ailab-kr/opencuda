# An ordinary CMake CUDA project, built with the ocu toolkit:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<toolkit>/cmake/OcuCUDA.cmake
# Nothing in it names ocu.
cmake_minimum_required(VERSION 3.20)
project(ocu_driver_consumer LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
enable_language(CUDA)
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}, "
               "toolkit ${CUDAToolkit_VERSION}")

# A shared library with kernels, with options as projects pass them to nvcc.
add_library(kernels SHARED kernels.cu)
target_compile_options(kernels PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math;-extended-lambda;--expt-relaxed-constexpr;-Xcompiler;-Wno-pedantic>")
target_compile_definitions(kernels PRIVATE SCALE=2)
target_link_libraries(kernels PRIVATE CUDA::cudart)

# A host program in C++ that calls the runtime API and the library.
add_executable(host_main main.cpp)
target_link_libraries(host_main PRIVATE kernels CUDA::cudart)

# A program that is a single .cu file, linked by the CUDA compiler.
add_executable(cu_main cu_main.cu)
set_target_properties(cu_main PROPERTIES CUDA_STANDARD 17)
