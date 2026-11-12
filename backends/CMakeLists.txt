# The interface is a single header; every other target sees backends only
# through it.
add_library(ocu_backend_interface INTERFACE)
target_include_directories(ocu_backend_interface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/interface)

if(OCU_BACKEND_CPU)
  add_subdirectory(cpu)
endif()

if(OCU_BACKEND_AMD)
  message(FATAL_ERROR "The AMD backend is not implemented yet (see backends/amd/README.md).")
endif()

if(OCU_BACKEND_INTEL)
  message(FATAL_ERROR "The Intel backend is not implemented yet (see backends/intel/README.md).")
endif()
