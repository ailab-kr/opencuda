# OpenCL backend: the OpenCL library (ICD loader or a driver) is found at run
# time (dlopen), so building needs no OpenCL SDK. Without a usable device the
# backend reports none.
add_library(ocu_backend_opencl STATIC ocl_backend.c ocl_load.c)
target_include_directories(ocu_backend_opencl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${OCU_INSTALL_INCLUDEDIR}>)
target_link_libraries(ocu_backend_opencl PUBLIC ocu_backend_interface Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(ocu_backend_opencl PROPERTIES EXPORT_NAME backend_opencl)
install(TARGETS ocu_backend_opencl EXPORT OcuTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ocu_backend_opencl.h DESTINATION ${OCU_INSTALL_INCLUDEDIR})
