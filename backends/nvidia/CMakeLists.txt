# NVIDIA GPU backend: the installed driver's public API, found at run time
# (dlopen), so building needs nothing from NVIDIA. Without a driver the
# backend reports no devices.
add_library(ocu_backend_nvidia STATIC nv_backend.c nv_driver.c)
target_include_directories(ocu_backend_nvidia PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
target_link_libraries(ocu_backend_nvidia PUBLIC ocu_backend_interface Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(ocu_backend_nvidia PROPERTIES EXPORT_NAME backend_nvidia)
install(TARGETS ocu_backend_nvidia EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_backend_nvidia.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
