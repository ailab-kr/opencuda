# Install the build into a fresh prefix, then configure, build and run a
# separate project against it with find_package(Ocu).
#   cmake -DBUILD=<build dir> -DSRC=<this folder> -DWORK=<scratch> -P install_test.cmake
file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
macro(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed: ${ARGN}\n${out}\n${err}")
  endif()
endmacro()
run(${CMAKE_COMMAND} --install ${BUILD} --prefix ${WORK}/prefix)
run(${CMAKE_COMMAND} -S ${SRC}/consumer -B ${WORK}/build -DCMAKE_PREFIX_PATH=${WORK}/prefix)
run(${CMAKE_COMMAND} --build ${WORK}/build)
foreach(exe consumer_saxpy consumer_host)
  run(${WORK}/build/${exe})
  message(STATUS "${out}")
endforeach()
