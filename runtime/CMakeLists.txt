# ocu_core: backend registry and other chip-neutral plumbing.
find_package(Threads REQUIRED)
add_library(ocu_core STATIC core/registry.c)
target_include_directories(ocu_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_core PROPERTIES EXPORT_NAME core)
target_link_libraries(ocu_core PUBLIC ocu_backend_interface Threads::Threads)
if(OCU_BACKEND_CPU)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_CPU)
  target_link_libraries(ocu_core PRIVATE ocu_backend_cpu)
endif()
if(OCU_BACKEND_METAL)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_METAL)
  target_link_libraries(ocu_core PRIVATE ocu_backend_metal)
endif()
if(OCU_BACKEND_NVIDIA)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_NVIDIA)
  target_link_libraries(ocu_core PRIVATE ocu_backend_nvidia)
endif()

# ocu_cudart: the runtime API (cuda_runtime_api.h) on the backend interface.
add_library(ocu_cudart STATIC cudart/cudart.c cudart/fatbin.c)
target_include_directories(ocu_cudart PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_cudart PROPERTIES EXPORT_NAME cudart)
target_link_libraries(ocu_cudart PUBLIC ocu_core ocu_backend_interface)
if(OCU_BACKEND_CPU)
  # Programs built for the CPU path also need the CPU device-side API.
  target_link_libraries(ocu_cudart PUBLIC ocu_backend_cpu)
endif()
if(OCU_BACKEND_NVIDIA)
  target_link_libraries(ocu_cudart PUBLIC ocu_backend_nvidia)
endif()

# CUDA's fma/fmaf round once (IEEE 754 fusedMultiplyAdd). Some C libraries'
# do not (MinGW-w64). A configure check compares the C library with the
# exact software fma (include/ocu_fma_exact.h); when they differ, ocu_cudart
# carries replacements that programs link ahead of the C library. When the
# check cannot run (cross build without an emulator), the replacements are
# used: they are exact everywhere. -DOCU_LIBM_FMA_EXACT=ON/OFF overrides.
if(NOT DEFINED OCU_LIBM_FMA_EXACT)
  set(_ocu_fma_exact OFF)
  if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    set(_ocu_fma_why "cross build without an emulator")
  else()
    set(_ocu_libm "")
    if(UNIX)
      set(_ocu_libm m)
    endif()
    try_run(_ocu_fma_run _ocu_fma_compile ${CMAKE_CURRENT_BINARY_DIR}/check_libm_fma
      ${CMAKE_CURRENT_SOURCE_DIR}/cudart/check_libm_fma.c
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include"
      LINK_LIBRARIES ${_ocu_libm}
      RUN_OUTPUT_VARIABLE _ocu_fma_out)
    string(REGEX MATCH "fma/fmaf: [a-z ]+" _ocu_fma_why "${_ocu_fma_out}")
    if(_ocu_fma_compile AND "${_ocu_fma_run}" STREQUAL "0")
      set(_ocu_fma_exact ON)
    endif()
  endif()
  message(STATUS "C library fma/fmaf correctly rounded: ${_ocu_fma_exact} (${_ocu_fma_why})")
  set(OCU_LIBM_FMA_EXACT ${_ocu_fma_exact} CACHE BOOL
    "The C library's fma and fmaf are correctly rounded (else ocu_cudart replaces them)")
endif()
# The headers' own fma (ocu_fma_exact.h) follows the same result in every
# program that links ocu_cudart.
if(OCU_LIBM_FMA_EXACT)
  target_compile_definitions(ocu_cudart PUBLIC OCU_LIBM_FMA_EXACT=1)
else()
  target_compile_definitions(ocu_cudart PUBLIC OCU_LIBM_FMA_EXACT=0)
  target_sources(ocu_cudart PRIVATE cudart/libm_fma.c)
endif()

install(TARGETS ocu_core ocu_cudart EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES core/ocu_core.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
file(GLOB _ocu_rt_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
install(FILES ${_ocu_rt_headers} DESTINATION ${OCU_INSTALL_INCLUDEDIR})
