add_library(ocu_bench STATIC ocu_bench.c)
target_include_directories(ocu_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OCU_BACKEND_CPU)
  add_executable(bench_blas bench_blas.c)
  target_link_libraries(bench_blas PRIVATE ocu_bench ocu_blas)
endif()
