add_library(ocu_blas STATIC blas.c)
target_include_directories(ocu_blas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ocu_blas PUBLIC ocu_core ocu_backend_interface)
