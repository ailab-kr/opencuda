# Apple GPU backend through the public Metal API (macOS 13 or later at run
# time; older systems report no devices). Built only on Apple platforms.
enable_language(OBJC)
add_library(ocu_backend_metal STATIC mt_backend.m)
# Manual reference counting (the code retains and releases explicitly).
target_compile_options(ocu_backend_metal PRIVATE -fno-objc-arc)
target_include_directories(ocu_backend_metal PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
target_link_libraries(ocu_backend_metal PUBLIC ocu_backend_interface Threads::Threads
  "-framework Metal" "-framework Foundation")
set_target_properties(ocu_backend_metal PROPERTIES EXPORT_NAME backend_metal)
install(TARGETS ocu_backend_metal EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_backend_metal.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
