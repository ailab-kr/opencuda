# Translate SRC to an MSL image with cu2msl.py, then run the GPU test on it.
execute_process(COMMAND ${CMAKE_COMMAND} -E env OCU_CLANG=${CLANG} ${PY} ${TOOL} ${SRC} -o ${IMG}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "cu2msl failed:\n${out}\n${err}")
endif()
execute_process(COMMAND ${RUN} ${IMG} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}${err}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "test_metal_cu2msl failed")
endif()
