# variants.cmake: how the whole-block pass behaves under other ways of
# building (docs/cpu-region-loops.md "Tests for an independent session"
# item 10): -O0 (no conversion), -g (still converts, builds), OCU_CPU_LOOPS=off,
# a wrong OCU_CPU_LOOPS path (a warning, fibers), -ccbin g++ (converted
# modules still use the pass's Clang), the toolkit's driver finding the
# pass, an installed package finding it, and objects with and without the
# pass linked together (the same inline kernel launched from both).
#
# variant.cu launches 4 kernels: one without a barrier, one with, and the
# inline kernel from two translation units. With the pass 3 launches run as
# loops (the barrier one never does).
#
#   cmake -DNVCC=<ocu-nvcc> -DSRC=<tests/cpu_forms> -DWORK=<dir>
#         -DBUILD=<build dir> -DINSTALL_PREFIX=<dir> -P variants.cmake
cmake_minimum_required(VERSION 3.20)
foreach(v NVCC SRC WORK BUILD)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "variants: ${v} not set")
  endif()
endforeach()
file(REMOVE_RECURSE ${WORK})
file(MAKE_DIRECTORY ${WORK})
set(failed 0)

function(check name want_min want_max)
  # runs ${WORK}/${name}; the program's own checks must pass; loops counted
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_FORM=loop OCU_CPU_STATS=1 ${WORK}/${name}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
  if(NOT rc EQUAL 0 OR NOT out MATCHES "variant ok")
    message(SEND_ERROR "${name}: wrong results (exit ${rc}): ${out}")
    set(failed 1 PARENT_SCOPE)
    return()
  endif()
  if(err MATCHES "ocu cpu stats: ([0-9]+) launches \\(([0-9]+) as loops\\)")
    set(n ${CMAKE_MATCH_2})
    if(NOT CMAKE_MATCH_1 EQUAL 4)
      message(SEND_ERROR "${name}: ${CMAKE_MATCH_1} launches, expected 4")
      set(failed 1 PARENT_SCOPE)
    endif()
    if(n LESS ${want_min} OR n GREATER ${want_max})
      message(SEND_ERROR "${name}: ${n} launches ran as loops, expected ${want_min} to ${want_max}")
      set(failed 1 PARENT_SCOPE)
    else()
      message(STATUS "${name}: results right, ${n} of 4 launches as loops (expected ${want_min}..${want_max})")
    endif()
  else()
    message(SEND_ERROR "${name}: no stats summary:\n${err}")
    set(failed 1 PARENT_SCOPE)
  endif()
endfunction()

function(build name)  # build(name [ENV X=Y]... [FLAGS ...]) : variant.cu + cases_ext.cu
  cmake_parse_arguments(A "" "" "ENV;FLAGS;WARN" ${ARGN})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env OCU_TARGET=cpu ${A_ENV} ${NVCC} ${A_FLAGS} -I${SRC}
            ${SRC}/variant.cu ${SRC}/cases_ext.cu -o ${WORK}/${name}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 600)
  if(NOT rc EQUAL 0)
    message(SEND_ERROR "${name}: build failed (${rc}):\n${err}")
    set(failed 1 PARENT_SCOPE)
    set(${name}_built 0 PARENT_SCOPE)
    return()
  endif()
  set(${name}_built 1 PARENT_SCOPE)
  set(${name}_err "${err}" PARENT_SCOPE)
endfunction()

build(v_default FLAGS -O2)
if(v_default_built)
  check(v_default 3 3)
endif()
build(v_g FLAGS -g -O2)
if(v_g_built)
  check(v_g 3 3)
endif()
build(v_o0 FLAGS -O0)
if(v_o0_built)
  check(v_o0 0 0)
endif()
build(v_off ENV OCU_CPU_LOOPS=off FLAGS -O2)
if(v_off_built)
  check(v_off 0 0)
endif()
build(v_wrong ENV OCU_CPU_LOOPS=${WORK}/no/such/plugin.so FLAGS -O2)
if(v_wrong_built)
  if(NOT v_wrong_err MATCHES "OCU_CPU_LOOPS")
    message(SEND_ERROR "wrong OCU_CPU_LOOPS path: no warning in the compiler output:\n${v_wrong_err}")
    set(failed 1)
  endif()
  check(v_wrong 0 0)
endif()
build(v_ccbin FLAGS -ccbin g++ -O2)
if(v_ccbin_built)
  check(v_ccbin 3 3)
endif()

# objects with and without the pass, linked together: the inline kernel
# exists in both objects; the linker keeps one definition. Results must be
# right whichever it keeps.
foreach(order "pass_first|1" "pass_second|0")
  string(REPLACE "|" ";" order "${order}")
  list(GET order 0 name)
  list(GET order 1 first_with_pass)
  if(first_with_pass)
    set(env_a "OCU_CPU_LOOPS_UNUSED=1")
    set(env_b "OCU_CPU_LOOPS=off")
  else()
    set(env_a "OCU_CPU_LOOPS=off")
    set(env_b "OCU_CPU_LOOPS_UNUSED=1")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} -E env OCU_TARGET=cpu ${env_a} ${NVCC} -O2 -I${SRC} -c ${SRC}/variant.cu -o ${WORK}/${name}_a.o
    RESULT_VARIABLE r1 ERROR_VARIABLE e1)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env OCU_TARGET=cpu ${env_b} ${NVCC} -O2 -I${SRC} -c ${SRC}/cases_ext.cu -o ${WORK}/${name}_b.o
    RESULT_VARIABLE r2 ERROR_VARIABLE e2)
  execute_process(COMMAND ${CMAKE_COMMAND} -E env OCU_TARGET=cpu ${NVCC} ${WORK}/${name}_a.o ${WORK}/${name}_b.o -o ${WORK}/${name}
    RESULT_VARIABLE r3 ERROR_VARIABLE e3)
  if(NOT r1 EQUAL 0 OR NOT r2 EQUAL 0 OR NOT r3 EQUAL 0)
    message(SEND_ERROR "${name}: build failed:\n${e1}\n${e2}\n${e3}")
    set(failed 1)
  else()
    # with the pass in one object only: the barrier kernel never loops, the
    # other kernel does when its object has the pass; the inline kernel is
    # one definition (either object's), so 0 to 3
    if(first_with_pass)
      check(${name} 1 3)
    else()
      check(${name} 0 2)
    endif()
  endif()
endforeach()

# The installed package finds the pass without OCU_CPU_LOOPS (the report
# must show k_lp_v1 as loop and k_fb_v2 as fiber).
if(DEFINED INSTALL_PREFIX)
  file(REMOVE_RECURSE ${INSTALL_PREFIX})
  execute_process(COMMAND ${CMAKE_COMMAND} --install ${BUILD} --prefix ${INSTALL_PREFIX}
    RESULT_VARIABLE ri OUTPUT_QUIET ERROR_VARIABLE ei)
  find_program(PY python3)
  if(NOT ri EQUAL 0 OR NOT EXISTS ${INSTALL_PREFIX}/libexec/ocu/ocu_cc.py)
    message(SEND_ERROR "install failed (${ri}): ${ei}")
    set(failed 1)
  else()
    set(rep ${WORK}/installed_report.txt)
    file(REMOVE ${rep})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env OCU_CPU_LOOPS_REPORT=${rep}
              ${PY} ${INSTALL_PREFIX}/libexec/ocu/ocu_cc.py --target cpu -O2 -I${SRC}
              -c ${SRC}/variant.cu -o ${WORK}/installed.o
      RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0 OR NOT EXISTS ${rep})
      message(SEND_ERROR "installed ocu_cc failed (${rc}):\n${err}")
      set(failed 1)
    else()
      file(READ ${rep} content)
      if(content MATCHES "loop\t[^\n]*k_lp_v1" AND content MATCHES "fiber\t[^\n]*k_fb_v2")
        message(STATUS "installed package: the pass is found and used (k_lp_v1 loop, k_fb_v2 fiber)")
      else()
        message(SEND_ERROR "installed package did not use the pass:\n${content}")
        set(failed 1)
      endif()
    endif()
  endif()
endif()
if(failed)
  message(FATAL_ERROR "variants: failed")
endif()
