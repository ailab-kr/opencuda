add_library(ocu_ops_ref STATIC ocu_ops_ref.c)
target_include_directories(ocu_ops_ref PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(ocu_ops_ref PUBLIC ocu_backend_interface)
if(NOT APPLE)
  target_link_libraries(ocu_ops_ref PUBLIC m)
endif()
set_target_properties(ocu_ops_ref PROPERTIES EXPORT_NAME ops_ref)
install(TARGETS ocu_ops_ref EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
