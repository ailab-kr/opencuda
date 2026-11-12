# Configure, build and run tests/driver/cmake_project, an ordinary CMake
# CUDA project, with the ocu toolkit as its CUDA toolkit.
#   cmake -DSRC=<cmake_project> -DWORK=<dir> -DTARGET=cpu|nvidia
#         (-DTOOLKIT=<toolkit dir> | -DINSTALL_FROM=<build dir> -DLIBDIR=lib)
#         [-DRUN_WRAPPER=<prog>] -DCC=.. -DCXX=.. [-DCFLAGS=..] [-DCXXFLAGS=..]
#         [-DLDFLAGS=..] -P cmake_project_test.cmake
# With INSTALL_FROM the build is installed into <WORK>/prefix first and the
# installed toolkit is used.
file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
macro(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed (${rc}): ${ARGN}\n${out}\n${err}")
  endif()
endmacro()

if(INSTALL_FROM)
  run(${CMAKE_COMMAND} --install ${INSTALL_FROM} --prefix ${WORK}/prefix)
  set(TOOLKIT ${WORK}/prefix/${LIBDIR}/ocu/toolkit)
endif()
foreach(f bin/ocu-nvcc bin/nvcc include/cuda_runtime.h lib/libocu_cudart.a nvvm/libdevice
          cmake/OcuCUDA.cmake cmake/modules/FindCUDAToolkit.cmake)
  if(NOT EXISTS ${TOOLKIT}/${f})
    message(FATAL_ERROR "toolkit ${TOOLKIT} has no ${f}")
  endif()
endforeach()

# The driver never says it is NVIDIA's.
run(${TOOLKIT}/bin/nvcc --version)
if(out MATCHES "NVIDIA")
  message(FATAL_ERROR "--version names NVIDIA:\n${out}")
endif()

# CUDAFLAGS from the caller's shell would leak into the project.
unset(ENV{CUDAFLAGS})
unset(ENV{CUDACXX})
run(${CMAKE_COMMAND} -S ${SRC} -B ${WORK}/build
    -DCMAKE_TOOLCHAIN_FILE=${TOOLKIT}/cmake/OcuCUDA.cmake -DOCU_TARGET=${TARGET}
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX}
    "-DCMAKE_C_FLAGS=${CFLAGS}" "-DCMAKE_CXX_FLAGS=${CXXFLAGS}" "-DCMAKE_CUDA_FLAGS=${CXXFLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${LDFLAGS}" "-DCMAKE_SHARED_LINKER_FLAGS=${LDFLAGS}")
if(NOT out MATCHES "CUDA compiler: Clang [0-9.]+, toolkit [0-9]+\\.[0-9]+")
  message(FATAL_ERROR "unexpected CUDA compiler identification:\n${out}")
endif()
file(READ ${WORK}/build/CMakeCache.txt cache)
if(cache MATCHES "/usr/local/cuda")
  message(FATAL_ERROR "the configured project refers to a vendor SDK")
endif()
run(${CMAKE_COMMAND} --build ${WORK}/build)

foreach(exe host_main cu_main)
  if(RUN_WRAPPER)
    execute_process(COMMAND ${RUN_WRAPPER} ${WORK}/build/${exe} RESULT_VARIABLE rc
                    OUTPUT_VARIABLE out ERROR_VARIABLE err)
  else()
    execute_process(COMMAND ${WORK}/build/${exe} RESULT_VARIABLE rc
                    OUTPUT_VARIABLE out ERROR_VARIABLE err)
  endif()
  message(STATUS "${out}${err}")
  if(rc EQUAL 77)
    message(FATAL_ERROR "OCU_SKIP")
  endif()
  if(NOT rc EQUAL 0 OR NOT out MATCHES ": ok")
    message(FATAL_ERROR "${exe} failed (${rc})")
  endif()
endforeach()
