# FindCUDAToolkit for the ocu toolkit (on CMAKE_MODULE_PATH through
# OcuCUDA.cmake). Same result variables and CUDA::<name> targets that
# CMake's FindCUDAToolkit documents, pointing at our libraries:
#
#   CUDA::cudart, CUDA::cudart_static, CUDA::cuda_driver, CUDA::cudadevrt
#                       our runtime (ocu_cudart and the backends)
#   CUDA::cublas, CUDA::cublas_static
#                       ocu_cublas_<OCU_TARGET> (cublas_v2.h, built for the
#                       same target as the CUDA code: cpu or nvidia), else
#                       ocu_blas (functions it does not have stay
#                       unresolved at link time)
#   CUDA::toolkit       our include folder
#
# Libraries without an implementation get no target, so a project that
# needs one fails at configure time with CMake's own message, unless the
# name is in OCU_CUDA_PLACEHOLDER_LIBS (default: cublasLt, which projects
# usually link next to cuBLAS): those get an empty target, a status
# message, and their functions stay unresolved at link time.
#
# Nothing outside the toolkit folder is searched or run.
get_filename_component(_ocu_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(CUDAToolkit_ROOT_DIR "${_ocu_root}")
set(CUDAToolkit_BIN_DIR "${_ocu_root}/bin")
set(CUDAToolkit_NVCC_EXECUTABLE "${_ocu_root}/bin/ocu-nvcc")
set(CUDAToolkit_INCLUDE_DIRS "${_ocu_root}/include")
set(CUDAToolkit_LIBRARY_DIR "${_ocu_root}/lib")
set(CUDAToolkit_LIBRARY_ROOT "${_ocu_root}")
set(CUDAToolkit_TARGET_DIR "${_ocu_root}")

# Version: the CUDA API level our headers declare.
set(CUDAToolkit_VERSION "")
if(EXISTS "${_ocu_root}/include/cuda_runtime_api.h")
  file(STRINGS "${_ocu_root}/include/cuda_runtime_api.h" _ocu_v
       REGEX "^#define[ \t]+CUDART_VERSION[ \t]+[0-9]+")
  if(_ocu_v MATCHES "([0-9]+)$")
    math(EXPR CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1} / 1000")
    math(EXPR CUDAToolkit_VERSION_MINOR "(${CMAKE_MATCH_1} % 1000) / 10")
    set(CUDAToolkit_VERSION_PATCH 0)
    set(CUDAToolkit_VERSION
        "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}")
  endif()
endif()

set(_ocu_runtime)
foreach(_ocu_name ocu_cudart ocu_core ocu_backend_cpu ocu_backend_nvidia ocu_backend_opencl ocu_backend_metal ocu_backend_npusim ocu_ops_ref)
  if(EXISTS "${_ocu_root}/lib/lib${_ocu_name}.a")
    list(APPEND _ocu_runtime "${_ocu_root}/lib/lib${_ocu_name}.a")
  endif()
endforeach()
if(_ocu_runtime)
  list(GET _ocu_runtime 0 CUDA_cudart_LIBRARY)
  set(CUDA_cudart_static_LIBRARY "${CUDA_cudart_LIBRARY}")
endif()
if(APPLE)
  if(EXISTS "${_ocu_root}/lib/libocu_backend_metal.a")
    list(APPEND _ocu_runtime "-framework Metal" "-framework Foundation")
  endif()
  list(APPEND _ocu_runtime pthread m)
else()
  list(APPEND _ocu_runtime pthread dl m)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDAToolkit
  REQUIRED_VARS CUDAToolkit_INCLUDE_DIRS CUDA_cudart_LIBRARY
  VERSION_VAR CUDAToolkit_VERSION)

if(CUDAToolkit_FOUND)
  if(NOT TARGET CUDA::toolkit)
    add_library(CUDA::toolkit INTERFACE IMPORTED)
    set_target_properties(CUDA::toolkit PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
  endif()
  foreach(_ocu_t cudart cudart_static cuda_driver cudadevrt)
    if(NOT TARGET CUDA::${_ocu_t})
      add_library(CUDA::${_ocu_t} INTERFACE IMPORTED)
      set_target_properties(CUDA::${_ocu_t} PROPERTIES
        INTERFACE_LINK_LIBRARIES "CUDA::toolkit;${_ocu_runtime}")
    endif()
    set(CUDAToolkit_${_ocu_t}_FOUND TRUE)
  endforeach()
  set(_ocu_target "${OCU_TARGET}")
  # intel and spirv programs run on the OpenCL backend: its cuBLAS library
  if(_ocu_target STREQUAL "intel" OR _ocu_target STREQUAL "spirv")
    set(_ocu_target opencl)
  endif()
  if(NOT _ocu_target)
    set(_ocu_target cpu)
  endif()
  if(EXISTS "${_ocu_root}/lib/libocu_cublas_${_ocu_target}.a")
    set(CUDA_cublas_LIBRARY "${_ocu_root}/lib/libocu_cublas_${_ocu_target}.a")
  elseif(EXISTS "${_ocu_root}/lib/libocu_blas.a")
    set(CUDA_cublas_LIBRARY "${_ocu_root}/lib/libocu_blas.a")
  endif()
  if(CUDA_cublas_LIBRARY)
    foreach(_ocu_t cublas cublas_static)
      if(NOT TARGET CUDA::${_ocu_t})
        add_library(CUDA::${_ocu_t} INTERFACE IMPORTED)
        set_target_properties(CUDA::${_ocu_t} PROPERTIES
          INTERFACE_LINK_LIBRARIES "CUDA::toolkit;${CUDA_cublas_LIBRARY};${_ocu_runtime}")
      endif()
      set(CUDAToolkit_${_ocu_t}_FOUND TRUE)
    endforeach()
  endif()
  if(NOT DEFINED OCU_CUDA_PLACEHOLDER_LIBS)
    set(OCU_CUDA_PLACEHOLDER_LIBS cublasLt cublasLt_static)
  endif()
  foreach(_ocu_t IN LISTS OCU_CUDA_PLACEHOLDER_LIBS)
    if(NOT TARGET CUDA::${_ocu_t})
      add_library(CUDA::${_ocu_t} INTERFACE IMPORTED)
      set_target_properties(CUDA::${_ocu_t} PROPERTIES INTERFACE_LINK_LIBRARIES "CUDA::toolkit")
      message(STATUS "ocu: CUDA::${_ocu_t} has no implementation yet (empty target)")
    endif()
    set(CUDAToolkit_${_ocu_t}_FOUND FALSE)
  endforeach()
endif()

unset(_ocu_t)
unset(_ocu_v)
unset(_ocu_name)
unset(_ocu_runtime)
unset(_ocu_root)
unset(_ocu_target)
