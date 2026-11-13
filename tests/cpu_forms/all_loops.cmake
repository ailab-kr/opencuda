# all_loops.cmake: a program whose kernels have no synchronisation points must
# run every launch as a whole-block loop under OCU_CPU_FORM=loop (so the
# differential tests compare two different forms) and none under =fiber.
#   cmake -DRUN=<program> -P all_loops.cmake
cmake_minimum_required(VERSION 3.20)
foreach(form loop fiber)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_FORM=${form} OCU_CPU_THREADS=1 OCU_CPU_STATS=1 ${RUN}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(SEND_ERROR "${form}: exit ${rc}\n${out}")
  endif()
  if(err MATCHES "ocu cpu stats: ([0-9]+) launches \\(([0-9]+) as loops\\)")
    set(launches ${CMAKE_MATCH_1})
    set(loops ${CMAKE_MATCH_2})
  else()
    message(SEND_ERROR "${form}: no stats summary:\n${err}")
    continue()
  endif()
  if(form STREQUAL "loop" AND NOT loops EQUAL launches)
    message(SEND_ERROR "loop form: ${loops} of ${launches} launches ran as loops, expected all")
  elseif(form STREQUAL "fiber" AND NOT loops EQUAL 0)
    message(SEND_ERROR "fiber form: ${loops} launches ran as loops, expected none")
  else()
    message(STATUS "${form} form: ${loops} of ${launches} launches as loops")
  endif()
endforeach()
