# seed_check.cmake: a last-writer-wins kernel (o[0] = threadIdx.x, 64 threads)
# under both forms. Fixed order runs threads 0, 1, 2, ... so the last
# writer is thread 63; with random order the winner depends on the seed,
# and for the same seed both forms must pick the same winner.
#   cmake -DRUN=<cpu_form_cases> -P seed_check.cmake
cmake_minimum_required(VERSION 3.20)
function(winner form order seed out_var)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_FORM=${form} OCU_CPU_ORDER=${order} OCU_CPU_SEED=${seed}
            OCU_CPU_THREADS=1 ${RUN} order
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
  if(NOT rc EQUAL 0)
    message(SEND_ERROR "${form} ${order} ${seed}: exit ${rc}\n${out}\n${err}")
  endif()
  if(out MATCHES "lastwriter ([0-9]+)")
    set(${out_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
  else()
    message(SEND_ERROR "${form} ${order} ${seed}: no lastwriter line in\n${out}")
    set(${out_var} -1 PARENT_SCOPE)
  endif()
endfunction()

foreach(form fiber loop)
  winner(${form} fixed 1 w_fixed_${form})
  if(NOT w_fixed_${form} EQUAL 63)
    message(SEND_ERROR "fixed order, ${form} form: last writer ${w_fixed_${form}}, expected 63")
  endif()
endforeach()
set(distinct "")
foreach(seed 1 2 3 4 5 6 7 8 9 10 11 12)
  winner(fiber random ${seed} wf)
  winner(loop random ${seed} wl)
  if(NOT wf EQUAL wl)
    message(SEND_ERROR "random order seed ${seed}: last writer ${wf} (fiber) against ${wl} (loop)")
  endif()
  list(APPEND distinct ${wf})
endforeach()
list(REMOVE_DUPLICATES distinct)
list(LENGTH distinct nd)
if(nd LESS 4)
  message(SEND_ERROR "random order: only ${nd} different winners over 12 seeds (${distinct})")
endif()
message(STATUS "last writer: fixed order 63 in both forms; random order, 12 seeds, ${nd} different winners, identical in both forms (${distinct})")
