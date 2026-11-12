# npusim: a host backend that enforces NPU-like limits (no kernels, operator
# table only, separate device memory, scratchpad-bounded plans).
add_library(ocu_backend_npusim STATIC npusim.c)
target_include_directories(ocu_backend_npusim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
target_link_libraries(ocu_backend_npusim PUBLIC ocu_backend_interface Threads::Threads)
target_link_libraries(ocu_backend_npusim PRIVATE ocu_ops_ref)
set_target_properties(ocu_backend_npusim PROPERTIES EXPORT_NAME backend_npusim)
install(TARGETS ocu_backend_npusim EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_backend_npusim.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
