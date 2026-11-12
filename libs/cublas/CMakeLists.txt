# The BLAS library API (cublas_v2.h), one library per target from the same
# CUDA source (cublas.cu): ocu_cublas_cpu for CPU-path programs,
# ocu_cublas_nvidia for programs built for NVIDIA GPUs. Both carry their
# own kernels, compiled by ocu_cc for that target; neither uses a vendor
# library. A program links the variant matching how it was built (see
# ocu_add_*_cu_executable(... LIBRARIES ...)).
if(NOT Python3_FOUND)
  return()
endif()
include(OcuCpuPath)

set(OCU_CUBLAS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OCU_CUBLAS_VARIANTS)
if(OCU_BACKEND_CPU)
  list(APPEND OCU_CUBLAS_VARIANTS cpu)
endif()
if(OCU_BACKEND_NVIDIA)
  # Device code needs an LLVM Clang with the NVPTX target.
  find_program(OCU_CUBLAS_CLANGXX NAMES clang++-20 clang++-19 clang++-18 clang++
               PATHS /usr/lib/llvm-18/bin)
  if(OCU_CUBLAS_CLANGXX)
    execute_process(COMMAND ${OCU_CUBLAS_CLANGXX} --print-targets
                    OUTPUT_VARIABLE _targets ERROR_QUIET)
    if(_targets MATCHES "nvptx64")
      list(APPEND OCU_CUBLAS_VARIANTS nvidia)
    endif()
  endif()
endif()

foreach(kind IN LISTS OCU_CUBLAS_VARIANTS)
  ocu_add_cu_library(ocu_cublas_${kind} ${kind} SOURCES cublas.cu INCLUDES ${OCU_CUBLAS_INCLUDE})
  target_include_directories(ocu_cublas_${kind} PUBLIC
    $<BUILD_INTERFACE:${OCU_CUBLAS_INCLUDE}>
    $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
  set_target_properties(ocu_cublas_${kind} PROPERTIES EXPORT_NAME cublas_${kind})
  install(TARGETS ocu_cublas_${kind} EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()
if(OCU_CUBLAS_VARIANTS)
  install(FILES include/cublas_v2.h include/cublas_api.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
endif()
