# OcuCUDARules.cmake: loaded by CMake after its CUDA compiler rules
# (CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA, set by OcuCUDA.cmake).
#
# CMake links a CUDA runtime into every target with CUDA sources
# (CMAKE_CUDA_RUNTIME_LIBRARY, STATIC by default). With this toolkit that
# runtime is ours: the static libraries in <toolkit>/lib, in link order,
# and the system libraries they use.
get_filename_component(_ocu_lib "${CMAKE_CURRENT_LIST_DIR}/../lib" ABSOLUTE)
set(_ocu_runtime)
foreach(_ocu_name ocu_cudart ocu_core ocu_backend_cpu ocu_backend_nvidia ocu_backend_metal)
  if(EXISTS "${_ocu_lib}/lib${_ocu_name}.a")
    list(APPEND _ocu_runtime "${_ocu_lib}/lib${_ocu_name}.a")
  endif()
endforeach()
if(APPLE)
  if(EXISTS "${_ocu_lib}/libocu_backend_metal.a")
    list(APPEND _ocu_runtime "-framework Metal" "-framework Foundation")
  endif()
  list(APPEND _ocu_runtime pthread m)
else()
  list(APPEND _ocu_runtime pthread dl m)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC "${_ocu_runtime}")
set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED "${_ocu_runtime}")
unset(_ocu_runtime)
unset(_ocu_name)
unset(_ocu_lib)
