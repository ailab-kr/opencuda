add_library(ocu_blas STATIC blas.c)
target_include_directories(ocu_blas PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
set_target_properties(ocu_blas PROPERTIES EXPORT_NAME blas)
install(TARGETS ocu_blas EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_blas.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
target_link_libraries(ocu_blas PUBLIC ocu_core ocu_backend_interface)
