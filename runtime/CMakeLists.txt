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
if(OCU_BACKEND_NVIDIA)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_NVIDIA)
  target_link_libraries(ocu_core PRIVATE ocu_backend_nvidia)
endif()

# ocu_cudart: the runtime API (cuda_runtime_api.h) on the backend interface.
add_library(ocu_cudart STATIC cudart/cudart.c)
target_include_directories(ocu_cudart PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_cudart PROPERTIES EXPORT_NAME cudart)
target_link_libraries(ocu_cudart PUBLIC ocu_core ocu_backend_interface)
if(OCU_BACKEND_CPU)
  # Programs built for the CPU path also need the CPU device-side API.
  target_link_libraries(ocu_cudart PUBLIC ocu_backend_cpu)
endif()

install(TARGETS ocu_core ocu_cudart EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES core/ocu_core.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
file(GLOB _ocu_rt_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
install(FILES ${_ocu_rt_headers} DESTINATION ${OCU_INSTALL_INCLUDEDIR})
