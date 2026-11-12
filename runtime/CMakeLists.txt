# ocu_core: backend registry and other chip-neutral plumbing.
# The user-facing runtime API library is added here later.
add_library(ocu_core STATIC core/registry.c)
target_include_directories(ocu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(ocu_core PUBLIC ocu_backend_interface)

if(OCU_BACKEND_CPU)
  target_compile_definitions(ocu_core PRIVATE OCU_HAVE_BACKEND_CPU)
  target_link_libraries(ocu_core PRIVATE ocu_backend_cpu)
endif()
