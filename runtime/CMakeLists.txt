# ocu_core: backend registry and other chip-neutral plumbing.
find_package(Threads REQUIRED)
add_library(ocu_core STATIC core/registry.c)
target_include_directories(ocu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(ocu_core PUBLIC ocu_backend_interface Threads::Threads)
if(OCU_BACKEND_CPU)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_CPU)
  target_link_libraries(ocu_core PRIVATE ocu_backend_cpu)
endif()

# ocu_cudart: the runtime API (cuda_runtime_api.h) on the backend interface.
add_library(ocu_cudart STATIC cudart/cudart.c)
target_include_directories(ocu_cudart PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ocu_cudart PUBLIC ocu_core ocu_backend_interface)
if(OCU_BACKEND_CPU)
  # Programs built for the CPU path also need the CPU device-side API.
  target_link_libraries(ocu_cudart PUBLIC ocu_backend_cpu)
endif()
