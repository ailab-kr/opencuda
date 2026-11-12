# The toolkit folder: what a CMake project (or a plain Makefile) points at
# to use ocu-nvcc as its CUDA compiler.
#
#   bin/ocu-nvcc            the driver (compiler/driver/ocu-nvcc)
#   bin/nvcc                link to ocu-nvcc under the name CMake's toolkit
#                           search looks for (it prints nothing of NVIDIA)
#   bin/ocu-nvcc-cpu, bin/ocu-nvcc-nvidia
#                           links to ocu-nvcc; the name picks the target
#   include/                our headers (cuda_runtime.h ...)
#   lib/                    libocu_cudart.a, libocu_core.a, backends, ocu_blas,
#                           libocu_cublas_<target>.a (cublas_v2.h, per target)
#   nvvm/libdevice/         empty: CMake checks that it exists; never read
#   cmake/OcuCUDA.cmake     toolchain file / module (see there)
#   cmake/OcuCUDARules.cmake, cmake/modules/FindCUDAToolkit.cmake
#
# In this build tree: ${PROJECT_BINARY_DIR}/toolkit (target ocu_toolkit,
# built by default). Installed: <prefix>/<libdir>/ocu/toolkit, made of
# links to the installed driver, headers and libraries.
# Unix only (links); Clang's CUDA mode in CMake is not available on Windows.
if(WIN32)
  return()
endif()

set(OCU_TOOLKIT_DIR ${PROJECT_BINARY_DIR}/toolkit)
set(_tk ${OCU_TOOLKIT_DIR})
file(MAKE_DIRECTORY ${_tk}/bin ${_tk}/include ${_tk}/lib ${_tk}/nvvm/libdevice ${_tk}/cmake/modules)
file(CREATE_LINK ${PROJECT_SOURCE_DIR}/compiler/driver/ocu-nvcc ${_tk}/bin/ocu-nvcc SYMBOLIC)
foreach(n nvcc ocu-nvcc-cpu ocu-nvcc-nvidia)
  file(CREATE_LINK ocu-nvcc ${_tk}/bin/${n} SYMBOLIC)
endforeach()

# The headers an installed package has, one folder.
file(GLOB _ocu_tk_headers
  ${PROJECT_SOURCE_DIR}/runtime/include/*.h
  ${PROJECT_SOURCE_DIR}/runtime/core/ocu_core.h
  ${PROJECT_SOURCE_DIR}/backends/interface/*.h
  ${PROJECT_SOURCE_DIR}/backends/cpu/ocu_backend_cpu.h
  ${PROJECT_SOURCE_DIR}/backends/cpu/ocu_cpu_kernel.h
  ${PROJECT_SOURCE_DIR}/backends/nvidia/ocu_backend_nvidia.h
  ${PROJECT_SOURCE_DIR}/backends/opencl/ocu_backend_opencl.h
  ${PROJECT_SOURCE_DIR}/libs/blas/ocu_blas.h
  ${PROJECT_SOURCE_DIR}/libs/cublas/include/*.h)
foreach(h IN LISTS _ocu_tk_headers)
  get_filename_component(n ${h} NAME)
  file(CREATE_LINK ${h} ${_tk}/include/${n} SYMBOLIC)
endforeach()
# Header folders included as <name/file.h> (cooperative_groups/reduce.h).
file(GLOB _ocu_tk_header_dirs LIST_DIRECTORIES true ${PROJECT_SOURCE_DIR}/runtime/include/*)
foreach(d IN LISTS _ocu_tk_header_dirs)
  if(IS_DIRECTORY ${d})
    get_filename_component(n ${d} NAME)
    file(CREATE_LINK ${d} ${_tk}/include/${n} SYMBOLIC)
  endif()
endforeach()
foreach(f OcuCUDA.cmake OcuCUDARules.cmake)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/${f} ${_tk}/cmake/${f} COPYONLY)
endforeach()
configure_file(${PROJECT_SOURCE_DIR}/cmake/cuda-modules/FindCUDAToolkit.cmake
               ${_tk}/cmake/modules/FindCUDAToolkit.cmake COPYONLY)

# Libraries: links to the built archives, refreshed after every build.
set(_ocu_tk_libs ocu_cudart ocu_core ocu_blas)
foreach(t ocu_backend_cpu ocu_backend_nvidia ocu_backend_opencl ocu_backend_metal ocu_backend_npusim ocu_ops_ref ocu_cublas_cpu ocu_cublas_nvidia)
  if(TARGET ${t})
    list(APPEND _ocu_tk_libs ${t})
  endif()
endforeach()
set(_ocu_tk_cmds)
foreach(t IN LISTS _ocu_tk_libs)
  list(APPEND _ocu_tk_cmds COMMAND ${CMAKE_COMMAND} -E create_symlink
       $<TARGET_FILE:${t}> ${_tk}/lib/$<TARGET_FILE_NAME:${t}>)
endforeach()
add_custom_target(ocu_toolkit ALL ${_ocu_tk_cmds}
  DEPENDS ${_ocu_tk_libs}
  COMMENT "ocu toolkit folder: ${_tk}")

# ---- install ---------------------------------------------------------------
install(PROGRAMS compiler/driver/ocu-nvcc DESTINATION ${CMAKE_INSTALL_BINDIR})
set(_ocu_tk_rel ${CMAKE_INSTALL_LIBDIR}/ocu/toolkit)
install(FILES cmake/OcuCUDA.cmake cmake/OcuCUDARules.cmake DESTINATION ${_ocu_tk_rel}/cmake)
install(FILES cmake/cuda-modules/FindCUDAToolkit.cmake DESTINATION ${_ocu_tk_rel}/cmake/modules)
# Relative links, so the installed tree can be moved.
file(RELATIVE_PATH _ocu_to_bin /r/${_ocu_tk_rel}/bin /r/${CMAKE_INSTALL_BINDIR}/ocu-nvcc)
file(RELATIVE_PATH _ocu_to_inc /r/${_ocu_tk_rel} /r/${OCU_INSTALL_INCLUDEDIR})
file(RELATIVE_PATH _ocu_to_lib /r/${_ocu_tk_rel} /r/${CMAKE_INSTALL_LIBDIR})
install(CODE "
  set(_tk \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_ocu_tk_rel}\")
  file(MAKE_DIRECTORY \"\${_tk}/bin\" \"\${_tk}/nvvm/libdevice\")
  file(REMOVE \"\${_tk}/bin/ocu-nvcc\" \"\${_tk}/include\" \"\${_tk}/lib\")
  file(CREATE_LINK ${_ocu_to_bin} \"\${_tk}/bin/ocu-nvcc\" SYMBOLIC)
  foreach(n nvcc ocu-nvcc-cpu ocu-nvcc-nvidia)
    file(REMOVE \"\${_tk}/bin/\${n}\")
    file(CREATE_LINK ocu-nvcc \"\${_tk}/bin/\${n}\" SYMBOLIC)
  endforeach()
  file(CREATE_LINK ${_ocu_to_inc} \"\${_tk}/include\" SYMBOLIC)
  file(CREATE_LINK ${_ocu_to_lib} \"\${_tk}/lib\" SYMBOLIC)
  message(STATUS \"Toolkit folder: \${_tk}\")
")
