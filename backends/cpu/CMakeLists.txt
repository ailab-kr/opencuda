find_package(Threads REQUIRED)

add_library(ocu_backend_cpu STATIC
  cpu_backend.c
  cpu_memory.c
  cpu_queue.c
  cpu_exec.c)
target_include_directories(ocu_backend_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ocu_backend_cpu PUBLIC ocu_backend_interface Threads::Threads)
