# Translate each of SRCS (a ;-list, passed as a |-list) with cu2msl.py and
# check the generated MSL with msl_check.py (no Metal compiler needed).
# EXPECT is the number of kernels the images must hold in all.
string(REPLACE "|" ";" srcs "${SRCS}")
set(imgs)
foreach(src ${srcs})
  get_filename_component(name ${src} NAME_WE)
  set(img ${WORK}/${name}.metal.img)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env OCU_CLANG=${CLANG} ${PY} ${TOOL} ${src} -o ${img}
                  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "cu2msl failed on ${src}:\n${out}\n${err}")
  endif()
  list(APPEND imgs ${img})
endforeach()
execute_process(COMMAND ${PY} ${CHECK} --clang ${CLANG} --expect-kernels ${EXPECT} ${imgs}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}${err}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "msl_check failed")
endif()
