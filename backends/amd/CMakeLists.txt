# AMD GPU backend: the installed HIP runtime's public API, found at run time
# (amdhip64 on Windows, shipped with the GPU driver; libamdhip64 on Linux),
# so building needs nothing from AMD. Without a runtime the backend reports
# no devices.
add_library(ocu_backend_amd STATIC amd_backend.c amd_hip.c amd_meta.c)
target_include_directories(ocu_backend_amd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
target_link_libraries(ocu_backend_amd PUBLIC ocu_backend_interface Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(ocu_backend_amd PROPERTIES EXPORT_NAME backend_amd)
install(TARGETS ocu_backend_amd EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_backend_amd.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
