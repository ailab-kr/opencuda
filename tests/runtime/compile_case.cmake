# Compiles cg_compile_cases.cu with -DCASE=<n> through ocu_cc.py and checks
# the outcome.
#   -DPY=<python> -DCC=<ocu_cc.py> -DSRC=<cases.cu> -DOUT=<object> -DTARGET=cpu|nvidia
#   -DCASE=<n> -DEXPECT=ok|fail
# A "fail" case must fail because of the compiler (an error message about
# the source), not because the driver is broken: the control case (0) has
# to compile in the same configuration, which is another test.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env CXX=${CXX} CC=${CC_COMPILER}
          ${PY} ${CC} --target ${TARGET} -O0 -D CASE=${CASE} -c ${SRC} -o ${OUT}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("exit: ${rc}\n${out}\n${err}")
if(EXPECT STREQUAL "ok")
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} (${TARGET}) must compile, the compiler refused it")
  endif()
else()
  if(rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} (${TARGET}) must NOT compile, but it did")
  endif()
  if(NOT "${out}${err}" MATCHES "error")
    message(FATAL_ERROR "case ${CASE} (${TARGET}) failed without a compiler error message")
  endif()
endif()
