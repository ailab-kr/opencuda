# The interface is a single header; every other target sees backends only
# through it.
add_library(ocu_backend_interface INTERFACE)
target_include_directories(ocu_backend_interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_backend_interface PROPERTIES EXPORT_NAME backend_interface)
install(TARGETS ocu_backend_interface EXPORT OcuTargets)
install(FILES interface/ocu_backend.h interface/ocu_blas_provider.h
        DESTINATION ${OCU_INSTALL_INCLUDEDIR})

if(OCU_BACKEND_CPU)
  add_subdirectory(cpu)
endif()

if(OCU_BACKEND_NVIDIA)
  find_package(Threads REQUIRED)
  add_subdirectory(nvidia)
endif()

if(OCU_BACKEND_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "The Metal backend builds on Apple platforms only.")
  endif()
  find_package(Threads REQUIRED)
  add_subdirectory(metal)
endif()

if(OCU_BACKEND_AMD)
  message(FATAL_ERROR "The AMD backend is not implemented yet (see backends/amd/README.md).")
endif()

if(OCU_BACKEND_INTEL)
  message(FATAL_ERROR "The Intel backend is not implemented yet (see backends/intel/README.md).")
endif()
