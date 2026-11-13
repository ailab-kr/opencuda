# form_diff.cmake: differential run of a CPU-path program in its two kernel
# forms (docs/cpu-region-loops.md): OCU_CPU_FORM=fiber (every thread a
# fiber) against OCU_CPU_FORM=loop (whole-block loop entries where the
# compiler made them). For each thread order (fixed, and random with two
# seeds) both runs must give
#   - the same exit code,
#   - the same standard output and standard error,
#   - the same device-memory trace: after every launch, a hash of all
#     device and pinned memory (OCU_CPU_TRACE), so every launch's results
#     are compared bit for bit, not only what the program checks.
# One OS thread (block order fixed). Address-space randomisation is turned
# off where `setarch -R` exists, so pointers stored in device memory hash
# the same in both runs.
#
#   cmake -DRUN=<program> -DNAME=<name> -DWORK=<dir> [-DARGS=<a;b>]
#         [-DSETARCH=<setarch program> -DARCH=<machine>] -P form_diff.cmake
#
# Prints how many launches ran as loops (OCU_CPU_STATS) for the record.
cmake_minimum_required(VERSION 3.20)
foreach(v RUN NAME WORK)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "form_diff: ${v} not set")
  endif()
endforeach()
set(dir ${WORK}/${NAME})
file(REMOVE_RECURSE ${dir})
file(MAKE_DIRECTORY ${dir})
set(prefix)
if(SETARCH AND ARCH)
  set(prefix ${SETARCH} ${ARCH} -R)
endif()

set(failed 0)
foreach(order "fixed|1" "random|7" "random|1234")
  string(REPLACE "|" ";" order "${order}")
  list(GET order 0 ord)
  list(GET order 1 seed)
  foreach(form fiber loop)
    set(tag ${ord}-${seed}-${form})
    file(REMOVE ${dir}/${tag}.trace)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_FORM=${form} OCU_CPU_ORDER=${ord}
              OCU_CPU_SEED=${seed} OCU_CPU_THREADS=1 OCU_CPU_TRACE=${dir}/${tag}.trace
              ${prefix} ${RUN} ${ARGS}
      RESULT_VARIABLE rc_${form}
      OUTPUT_VARIABLE out_${form}
      ERROR_VARIABLE err_${form}
      TIMEOUT 600)
    set(trace_${form} "")
    if(EXISTS ${dir}/${tag}.trace)
      file(READ ${dir}/${tag}.trace trace_${form})
      # Which form ran each launch is recorded, not compared.
      string(REGEX REPLACE " form [a-z]+" "" trace_${form} "${trace_${form}}")
    endif()
    file(WRITE ${dir}/${tag}.out "${out_${form}}")
    file(WRITE ${dir}/${tag}.err "${err_${form}}")
  endforeach()
  string(REGEX MATCHALL "\n" lines "${trace_fiber}")
  list(LENGTH lines launches)
  set(what "")
  if(NOT rc_fiber STREQUAL rc_loop)
    string(APPEND what " exit code ${rc_fiber} vs ${rc_loop};")
  endif()
  if(NOT out_fiber STREQUAL out_loop)
    string(APPEND what " standard output differs;")
  endif()
  if(NOT err_fiber STREQUAL err_loop)
    string(APPEND what " standard error differs;")
  endif()
  if(NOT trace_fiber STREQUAL trace_loop)
    string(APPEND what " device memory after a launch differs;")
  endif()
  if(what STREQUAL "")
    message(STATUS "${NAME} ${ord} order seed ${seed}: same in both forms "
                   "(exit ${rc_fiber}, ${launches} launches)")
  else()
    message(SEND_ERROR "${NAME} ${ord} order seed ${seed}:${what} files in ${dir}")
    set(failed 1)
  endif()
endforeach()

# How much of the program the loop form covered.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_FORM=loop OCU_CPU_THREADS=1 OCU_CPU_STATS=1
          ${prefix} ${RUN} ${ARGS}
  OUTPUT_QUIET ERROR_VARIABLE stats TIMEOUT 600)
if(stats MATCHES "ocu cpu stats: ([0-9]+) launches \\(([0-9]+) as loops\\)")
  message(STATUS "${NAME}: ${CMAKE_MATCH_2} of ${CMAKE_MATCH_1} launches ran as loops")
endif()
if(failed)
  message(FATAL_ERROR "${NAME}: the two kernel forms differ")
endif()
