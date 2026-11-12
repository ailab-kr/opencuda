# Build a CUDA program with ocu-nvcc the way a Makefile calls nvcc
# (compile each source with -c, then link), then run it.
#   cmake -DDRIVER=<toolkit>/bin/ocu-nvcc -DTARGET=cpu|nvidia -DWORK=<dir>
#         -DSOURCES=a.cu,b.cu -DINCLUDES=<dir> [-DRUN_WRAPPER=<prog>]
#         [-DFLAGS=<host flags>] -P run_driver.cmake
# Exit code 77 from the wrapper (no device) prints OCU_SKIP, which the test
# reports as skipped (SKIP_REGULAR_EXPRESSION).
file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
string(REPLACE "," ";" SOURCES "${SOURCES}")

macro(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err
                  WORKING_DIRECTORY ${WORK})
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed (${rc}): ${ARGN}\n${out}\n${err}")
  endif()
endmacro()

set(objs)
set(incs)
foreach(i IN LISTS INCLUDES)
  list(APPEND incs -I ${i})
endforeach()
foreach(src IN LISTS SOURCES)
  get_filename_component(stem ${src} NAME_WE)
  set(obj ${WORK}/${stem}.o)
  # Options as projects write them for nvcc; several are only accepted.
  run(${DRIVER} --ocu-target=${TARGET} -c -x cu ${src} -o ${obj}
      -O2 -std=c++17 -arch=sm_80 -gencode arch=compute_80,code=sm_80
      --use_fast_math --expt-relaxed-constexpr --extended-lambda -lineinfo
      -Xcompiler -Wall,-Wno-unused-function -Xptxas -v -DOCU_DRIVER_TEST=1 ${incs}
      -MD -MF ${WORK}/${stem}.d -MT ${obj} ${FLAGS})
  if(NOT EXISTS ${obj})
    message(FATAL_ERROR "no object ${obj}")
  endif()
  file(READ ${WORK}/${stem}.d deps)
  get_filename_component(name ${src} NAME)
  if(NOT deps MATCHES "${stem}\\.o:" OR NOT deps MATCHES "${name}" OR NOT deps MATCHES "cuda_runtime\\.h")
    message(FATAL_ERROR "dependency file ${WORK}/${stem}.d lacks the target, source or our header:\n${deps}")
  endif()
  list(APPEND objs ${obj})
endforeach()

run(${DRIVER} --ocu-target=${TARGET} -o ${WORK}/prog ${objs} -lcudart ${FLAGS})
if(RUN_WRAPPER)
  execute_process(COMMAND ${RUN_WRAPPER} ${WORK}/prog RESULT_VARIABLE rc
                  OUTPUT_VARIABLE out ERROR_VARIABLE err)
else()
  execute_process(COMMAND ${WORK}/prog RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
endif()
message(STATUS "${out}${err}")
if(rc EQUAL 77)
  message(STATUS "SKIP: ${out}")
  # cmake -P cannot choose its exit code; the caller matches this line.
  message(FATAL_ERROR "OCU_SKIP")
endif()
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "program failed (${rc})")
endif()
