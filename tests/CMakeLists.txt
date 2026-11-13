add_executable(smoke_registry smoke/test_registry.c)
target_link_libraries(smoke_registry PRIVATE ocu_core)
add_test(NAME smoke_registry COMMAND smoke_registry)

add_executable(test_bench backend/test_bench.c)
target_link_libraries(test_bench PRIVATE ocu_bench m)
add_test(NAME bench_statistics COMMAND test_bench)

add_executable(smoke_registry_threads smoke/test_registry_threads.c)
target_link_libraries(smoke_registry_threads PRIVATE ocu_core)
add_test(NAME smoke_registry_threads COMMAND smoke_registry_threads)

if(OCU_BACKEND_CPU)
  add_executable(test_cpu_backend backend/test_cpu_backend.c)
  target_link_libraries(test_cpu_backend PRIVATE ocu_backend_cpu)

  # The same tests under each CPU backend mode.
  add_test(NAME cpu_backend COMMAND test_cpu_backend)
  add_test(NAME cpu_backend_random COMMAND test_cpu_backend)
  set_tests_properties(cpu_backend_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_SEED=7")
  add_test(NAME cpu_backend_strict_warp64_1thread COMMAND test_cpu_backend)
  set_tests_properties(cpu_backend_strict_warp64_1thread PROPERTIES
    ENVIRONMENT "OCU_CPU_STRICT=1;OCU_CPU_WARP=64;OCU_CPU_THREADS=1")
  # Many workers: memory regions per launch must stay small (review finding 4).
  add_test(NAME cpu_backend_64workers COMMAND test_cpu_backend)
  set_tests_properties(cpu_backend_64workers PROPERTIES
    ENVIRONMENT "OCU_CPU_THREADS=64")

  add_executable(test_cpu_warp backend/test_cpu_warp.c)
  target_link_libraries(test_cpu_warp PRIVATE ocu_backend_cpu)
  foreach(mode "warp32;OCU_CPU_WARP=32" "warp64;OCU_CPU_WARP=64"
               "warp32_random;OCU_CPU_WARP=32\;OCU_CPU_ORDER=random"
               "warp64_random;OCU_CPU_WARP=64\;OCU_CPU_ORDER=random\;OCU_CPU_SEED=5"
               "warp32_strict;OCU_CPU_WARP=32\;OCU_CPU_STRICT=1")
    list(GET mode 0 tag)
    list(GET mode 1 envs)
    add_test(NAME cpu_${tag} COMMAND test_cpu_warp)
    set_tests_properties(cpu_${tag} PROPERTIES ENVIRONMENT "${envs}")
  endforeach()

  add_executable(test_cpu_pool backend/test_cpu_pool.c)
  target_link_libraries(test_cpu_pool PRIVATE ocu_backend_cpu)
  add_test(NAME cpu_pool COMMAND test_cpu_pool)
  add_test(NAME cpu_pool_random_4workers COMMAND test_cpu_pool)
  set_tests_properties(cpu_pool_random_4workers PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=4")

  if(NOT WIN32) # no fork() on Windows
    add_executable(test_cpu_fork backend/test_cpu_fork.c)
    target_link_libraries(test_cpu_fork PRIVATE ocu_backend_cpu)
    add_test(NAME cpu_fork COMMAND test_cpu_fork)
  endif()

  # Fast paths (card backend-cpu, tester list 1-9): independent tests.
  if(NOT WIN32)
    add_executable(test_cpu_fastpath backend/test_cpu_fastpath.c)
    target_link_libraries(test_cpu_fastpath PRIVATE ocu_backend_cpu)
    foreach(mode
        "fixed_w32|OCU_CPU_WARP=32"
        "fixed_w64|OCU_CPU_WARP=64"
        "random_w32|OCU_CPU_ORDER=random;OCU_CPU_SEED=3"
        "random_w64|OCU_CPU_ORDER=random;OCU_CPU_SEED=4;OCU_CPU_WARP=64"
        "fixed_w32_1thread|OCU_CPU_THREADS=1"
        "fixed_w64_1thread|OCU_CPU_THREADS=1;OCU_CPU_WARP=64"
        "random_w32_1thread|OCU_CPU_THREADS=1;OCU_CPU_ORDER=random;OCU_CPU_SEED=9"
        "random_w64_1thread|OCU_CPU_THREADS=1;OCU_CPU_ORDER=random;OCU_CPU_SEED=2;OCU_CPU_WARP=64"
        "fixed_w32_4threads|OCU_CPU_THREADS=4"
        "random_w64_4threads|OCU_CPU_THREADS=4;OCU_CPU_ORDER=random;OCU_CPU_WARP=64"
        "strict_w32|OCU_CPU_STRICT=1"
        "strict_w64_random|OCU_CPU_STRICT=1;OCU_CPU_WARP=64;OCU_CPU_ORDER=random;OCU_CPU_SEED=6"
        "spin0_w32|OCU_CPU_SPIN_US=0"
        "spin0_w64_random_4threads|OCU_CPU_SPIN_US=0;OCU_CPU_WARP=64;OCU_CPU_ORDER=random;OCU_CPU_THREADS=4"
        "small_stack_1thread|OCU_CPU_STACK_KB=16;OCU_CPU_THREADS=1")
      string(REPLACE "|" ";" mode "${mode}")
      list(GET mode 0 tag)
      list(GET mode 1 envs)
      add_test(NAME cpu_fastpath_${tag} COMMAND test_cpu_fastpath)
      set_tests_properties(cpu_fastpath_${tag} PROPERTIES ENVIRONMENT "${envs}" TIMEOUT 600)
    endforeach()
    # Card item 6: the lane id is defined outside a kernel (0 from host code).
    add_test(NAME cpu_fastpath_lane_id_outside_kernel COMMAND test_cpu_fastpath lane_outside)
    # Card item 2 / header: mask 0 names the whole warp, and with
    # OCU_CPU_STRICT=1 a named lane that already returned is an error, as with
    # mask 0xffffffff.
    add_test(NAME cpu_fastpath_strict_mask0_lane_gone COMMAND test_cpu_fastpath strict_mask0)
    set_tests_properties(cpu_fastpath_strict_mask0_lane_gone PROPERTIES
      ENVIRONMENT "OCU_CPU_STRICT=1")
    # Card item 9: control registers changed by one thread must not reach the
    # next one: a thread that ends with a changed rounding mode / FTZ+DAZ
    # must not pass it to threads started after it on the same stack.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      add_test(NAME cpu_fastpath_fpu_state_reaches_next_thread COMMAND test_cpu_fastpath fpu_dirty)
      add_test(NAME cpu_fastpath_fpu_state_reaches_next_thread_random COMMAND test_cpu_fastpath fpu_dirty)
      set_tests_properties(cpu_fastpath_fpu_state_reaches_next_thread_random PROPERTIES
        ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_SEED=5")
      # Random order starts every thread through the scheduler: no leak there.
      # Sanitizer builds always start threads through the scheduler: no leak there.
    endif()
  endif()

  add_executable(test_cpu_regressions backend/test_cpu_regressions.c)
  target_link_libraries(test_cpu_regressions PRIVATE ocu_backend_cpu)
  add_executable(test_cpu_tsan_fibers backend/test_cpu_tsan_fibers.c)
  target_link_libraries(test_cpu_tsan_fibers PRIVATE ocu_backend_cpu)
  add_test(NAME cpu_tsan_fibers COMMAND test_cpu_tsan_fibers)
  add_test(NAME cpu_tsan_fibers_1worker COMMAND test_cpu_tsan_fibers)
  # One worker thread runs every block: the defect showed there first. A
  # sanitizer slowed by stale shadow frames fails by timeout.
  set_tests_properties(cpu_tsan_fibers PROPERTIES TIMEOUT 120)
  set_tests_properties(cpu_tsan_fibers_1worker PROPERTIES TIMEOUT 120 ENVIRONMENT "OCU_CPU_THREADS=1")
  add_test(NAME cpu_regressions COMMAND test_cpu_regressions)
  add_test(NAME cpu_regressions_small_stack COMMAND test_cpu_regressions)
  set_tests_properties(cpu_regressions_small_stack PROPERTIES
    ENVIRONMENT "OCU_CPU_STACK_KB=16;OCU_CPU_THREADS=1")
endif()

# ---- compiler probe, CPU path (PLAN P0-W1-11) ----------------------------
# Unmodified .cu files are rewritten and compiled as plain C++.
find_package(Python3 COMPONENTS Interpreter)
if(OCU_BACKEND_CPU AND Python3_FOUND)
  set(OCU_PROBE_DIR ${PROJECT_SOURCE_DIR}/compiler/probe)
  set(OCU_PROBE_GEN ${CMAKE_CURRENT_BINARY_DIR}/probe_gen)
  set(OCU_PROBE_SOURCES)
  foreach(name probe_kernel probe_features probe_review probe_warp)
    set(out ${OCU_PROBE_GEN}/${name}.cpp)
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OCU_PROBE_GEN}
      COMMAND ${Python3_EXECUTABLE} ${OCU_PROBE_DIR}/cpu_rewrite.py
              ${OCU_PROBE_DIR}/${name}.cu ${out}
      DEPENDS ${OCU_PROBE_DIR}/${name}.cu ${OCU_PROBE_DIR}/cpu_rewrite.py
      COMMENT "Rewriting ${name}.cu for the CPU path")
    list(APPEND OCU_PROBE_SOURCES ${out})
  endforeach()

  add_executable(test_probe_cpu probe/test_probe_cpu.cpp ${OCU_PROBE_SOURCES})
  target_include_directories(test_probe_cpu PRIVATE ${OCU_PROBE_DIR})
  target_compile_definitions(test_probe_cpu PRIVATE OCU_CPU_PATH)
  target_link_libraries(test_probe_cpu PRIVATE ocu_backend_cpu)
  add_test(NAME probe_cpu COMMAND test_probe_cpu)
  add_test(NAME probe_cpu_random COMMAND test_probe_cpu)
  set_tests_properties(probe_cpu_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_SEED=11")
  add_test(NAME probe_cpu_warp64 COMMAND test_probe_cpu)
  set_tests_properties(probe_cpu_warp64 PROPERTIES
    ENVIRONMENT "OCU_CPU_WARP=64;OCU_CPU_ORDER=random;OCU_CPU_SEED=3")

  add_test(NAME probe_cpu_rewrite_unit
    COMMAND ${Python3_EXECUTABLE} -m unittest test_cpu_rewrite
    WORKING_DIRECTORY ${OCU_PROBE_DIR})
endif()

# ---- conformance: BLAS reference cases on the CPU backend ---------------
if(OCU_BACKEND_CPU AND Python3_FOUND)
  add_executable(test_blas_args backend/test_blas_args.c)
  target_link_libraries(test_blas_args PRIVATE ocu_blas)
  add_test(NAME blas_args COMMAND test_blas_args)
  set_tests_properties(blas_args PROPERTIES ENVIRONMENT "OCU_BACKEND=cpu")

  add_executable(test_blas_large backend/test_blas_large.c)
  target_link_libraries(test_blas_large PRIVATE ocu_blas)
  add_test(NAME blas_large COMMAND test_blas_large)
  set_tests_properties(blas_large PROPERTIES ENVIRONMENT "OCU_BACKEND=cpu")

  add_executable(blas_driver conformance/blas_driver.c)
  target_link_libraries(blas_driver PRIVATE ocu_blas)
  add_test(NAME conformance_blas_cpu
    COMMAND ${Python3_EXECUTABLE} -m ocu_conf.blas_run $<TARGET_FILE:blas_driver>
            --out ${CMAKE_CURRENT_BINARY_DIR}/blas_cpu_result.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/conformance/harness)
  # A cross-compiled driver runs through the emulator CMake knows (Wine).
  set(_ocu_launch "")
  if(CMAKE_CROSSCOMPILING_EMULATOR)
    string(JOIN " " _ocu_emu ${CMAKE_CROSSCOMPILING_EMULATOR})
    set(_ocu_launch ";OCU_DRIVER_LAUNCHER=${_ocu_emu}")
  endif()
  set_tests_properties(conformance_blas_cpu PROPERTIES ENVIRONMENT "OCU_BACKEND=cpu${_ocu_launch}")
  add_test(NAME conformance_blas_cpu_queue
    COMMAND ${Python3_EXECUTABLE} -m ocu_conf.blas_run $<TARGET_FILE:blas_driver>
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/conformance/harness)
  set_tests_properties(conformance_blas_cpu_queue PROPERTIES
    ENVIRONMENT "OCU_BACKEND=cpu;OCU_DRIVER_QUEUE=1;OCU_CPU_ORDER=random${_ocu_launch}")
endif()

# ---- ocu_cc.py (probe driver): every target, and a CPU program end to end
if(OCU_BACKEND_CPU AND Python3_FOUND)
  set(OCU_CC ${PROJECT_SOURCE_DIR}/compiler/probe/ocu_cc.py)
  set(OCU_CC_OUT ${CMAKE_CURRENT_BINARY_DIR}/ocu_cc)

  # Ordinary CUDA programs on the CPU path: runtime API + launches.
  include(OcuCpuPath)
  ocu_add_cpu_cu_executable(rt_api runtime/rt_api.cu)
  add_test(NAME runtime_api COMMAND rt_api)
  add_test(NAME runtime_api_random COMMAND rt_api)
  set_tests_properties(runtime_api_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3")
  ocu_add_cpu_cu_executable(rt_lang runtime/rt_lang.cu)
  add_test(NAME runtime_lang COMMAND rt_lang)
  add_test(NAME runtime_lang_random COMMAND rt_lang)
  set_tests_properties(runtime_lang_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3;OCU_CPU_WARP=64")
  # card backend-cpu tester list item 5: shuffles and votes as CUDA code.
  ocu_add_cpu_cu_executable(rt_warp_lang runtime/rt_warp_lang.cu)
  foreach(mode "w32|OCU_CPU_WARP=32" "w64|OCU_CPU_WARP=64"
               "random_w32|OCU_CPU_ORDER=random;OCU_CPU_SEED=2"
               "random_w64|OCU_CPU_ORDER=random;OCU_CPU_SEED=8;OCU_CPU_WARP=64"
               "w32_1thread|OCU_CPU_THREADS=1" "w64_random_1thread|OCU_CPU_THREADS=1;OCU_CPU_WARP=64;OCU_CPU_ORDER=random"
               "w32_4threads|OCU_CPU_THREADS=4" "w32_strict|OCU_CPU_STRICT=1")
    string(REPLACE "|" ";" mode "${mode}")
    list(GET mode 0 tag)
    list(GET mode 1 envs)
    add_test(NAME runtime_warp_lang_${tag} COMMAND rt_warp_lang)
    set_tests_properties(runtime_warp_lang_${tag} PROPERTIES ENVIRONMENT "${envs}" TIMEOUT 600)
  endforeach()
  ocu_add_cpu_cu_executable(rt_mem runtime/rt_mem.cu)
  add_test(NAME runtime_memory COMMAND rt_mem)
  ocu_add_cpu_cu_executable(rt_misc runtime/rt_misc.cu)
  add_test(NAME runtime_misc COMMAND rt_misc)
  ocu_add_cpu_cu_executable(rt_fp16 runtime/rt_fp16.cu)
  add_test(NAME runtime_fp16 COMMAND rt_fp16)
  ocu_add_cpu_cu_executable(rt_bf16 runtime/rt_bf16.cu)
  add_test(NAME runtime_bf16 COMMAND rt_bf16)
  # card compiler-ptx-shift: --ptx-shift semantics (tests/compiler/ptx_shift/).
  ocu_add_cpu_cu_executable(ptx_shift_run compiler/ptx_shift/ptx_shift_run.cu PTX_SHIFT)
  add_test(NAME ptx_shift_cpu COMMAND ptx_shift_run)
  # Control: the same program without the option must fail (C++ leaves the
  # out-of-range shifts undefined; hardware takes the amount modulo the
  # width), so the matrix would notice a rewrite that does nothing.
  ocu_add_cpu_cu_executable(ptx_shift_run_plain compiler/ptx_shift/ptx_shift_run.cu)
  add_test(NAME ptx_shift_cpu_control_without_option COMMAND ptx_shift_run_plain)
  set_tests_properties(ptx_shift_cpu_control_without_option PROPERTIES WILL_FAIL TRUE)
  # Evaluation order: ISO C++17 [expr.shift] sequences the left operand of << and >>
  # before the right one, [expr.ass] the right of <<= and >>= before the left.
  add_test(NAME ptx_shift_cpu_eval_order COMMAND ptx_shift_run order)
  set(_psdir ${CMAKE_CURRENT_SOURCE_DIR}/compiler/ptx_shift)
  function(ocu_ptx_shift_cmd name target)
    cmake_parse_arguments(A "" "OUTEXT" "DEFS" ${ARGN})
    if(NOT A_OUTEXT)
      set(A_OUTEXT o)
    endif()
    add_test(NAME ${name}
      COMMAND ${CMAKE_COMMAND} -DPY=${Python3_EXECUTABLE} -DCC=${OCU_CC} -DCXX=${CMAKE_CXX_COMPILER}
              -DCC_COMPILER=${CMAKE_C_COMPILER} -DDIR=${_psdir} -DTGT=${target}
              -DOUT=${CMAKE_CURRENT_BINARY_DIR}/${name}.${A_OUTEXT} ${A_DEFS}
              -P ${_psdir}/ptx_shift_cmd.cmake)
  endfunction()
  foreach(c "2|warn|ptx_shift_cases.cu" "3|warn|outside_shift.h"
            "4|warn|ptx_shift_cases.cu" "9|warn|ptx_shift_cases.cu" "1|quiet|-" "8|quiet|-" "5|quiet|-" "0|quiet|-" "6|error|-")
    string(REPLACE "|" ";" c "${c}")
    list(GET c 0 cn)
    list(GET c 1 cx)
    list(GET c 2 cf)
    ocu_ptx_shift_cmd(ptx_shift_cmd_case${cn}_${cx}_cpu cpu DEFS -DMODE=warn -DCASE=${cn} -DEXPECT=${cx} -DMARK_FILE=${cf})
  endforeach()
  # Without the option the same sources compile and say nothing.
  ocu_ptx_shift_cmd(ptx_shift_cmd_case10_warn_cpu cpu DEFS -DMODE=warn -DCASE=10 -DEXPECT=warn -DMARK_FILE=ptx_shift_cases_if.cu -DSRC_NAME=ptx_shift_cases_if.cu)
  ocu_ptx_shift_cmd(ptx_shift_cmd_unsupported_opencl cpu DEFS -DMODE=unsupported -DARGS=--target|opencl)
  ocu_ptx_shift_cmd(ptx_shift_cmd_case1_no_option_cpu cpu DEFS -DMODE=warn -DCASE=1 -DEXPECT=quiet -DNO_OPTION=1)
  ocu_ptx_shift_cmd(ptx_shift_cmd_print_cpu cpu DEFS -DMODE=print)
  ocu_ptx_shift_cmd(ptx_shift_cmd_unsupported_amd cpu DEFS -DMODE=unsupported -DARGS=--target|amd)
  ocu_ptx_shift_cmd(ptx_shift_cmd_unsupported_nvidia_ptx cpu DEFS -DMODE=unsupported "-DARGS=--target|nvidia|--arch|sm_80" OUTEXT ptx)

  # Exact software fma/fmaf (runtime/include/ocu_fma_exact.h): kernel functions,
  # host std::fma with ocu_cudart linked, the header's software functions, and
  # powf within its documented bound.  Tables: runtime/gen_fma_tables.py.
  ocu_add_cpu_cu_executable(fma_exact_test runtime/fma_exact_test.cu)
  add_test(NAME fma_exact COMMAND fma_exact_test)
  add_test(NAME fma_exact_powf COMMAND fma_exact_test powf)
  # What the configure check decided against what the C library does, and
  # against the platform: glibc exact, MinGW-w64 not.
  add_executable(fma_libm_probe runtime/fma_libm_probe.c)
  if(UNIX)
    target_link_libraries(fma_libm_probe PRIVATE m)
  endif()
  string(REGEX REPLACE "^(ON|TRUE|1|Y|YES)$" "1" _fma_got "${OCU_LIBM_FMA_EXACT}")
  string(REGEX REPLACE "^(OFF|FALSE|0|N|NO)$" "0" _fma_got "${_fma_got}")
  add_test(NAME fma_configure_matches_libm COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:fma_libm_probe> ${_fma_got})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_fma_want ON)
  elseif(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(_fma_want OFF)
  endif()
  if(DEFINED _fma_want)
    add_test(NAME fma_configure_platform
      COMMAND ${CMAKE_COMMAND} -DGOT=${OCU_LIBM_FMA_EXACT} -DWANT=${_fma_want}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime/fma_configure_check.cmake)
  endif()
  # Runtime API gaps (card runtime-llama-gaps): data types, attribute
  # policy, PCI bus id, peer access, cooperative launch. The compute
  # capability variants pass the expected value as arguments.
  # Kernels that stop the process: one process per test, the outcome is
  # checked by trap_test.cmake. rt_trap_arch is built with the macros the
  # CPU path needs for a NO_DEVICE_CODE pattern (card: boundary conditions).
  ocu_add_cpu_cu_executable(rt_trap runtime/rt_trap.cu)
  set(_trap_arch_obj ${CMAKE_CURRENT_BINARY_DIR}/rt_trap_arch.dir/rt_trap.cu.o)
  add_custom_command(
    OUTPUT ${_trap_arch_obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rt_trap_arch.dir
    COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
            ${Python3_EXECUTABLE} ${OCU_CC} --target cpu -O2 -D __CUDA_ARCH__=500 -D __CUDA_ARCH_LIST__=500
            -c ${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_trap.cu -o ${_trap_arch_obj}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_trap.cu ${OCU_CC} ${PROJECT_SOURCE_DIR}/compiler/probe/cpu_rewrite.py
    IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_trap.cu
    COMMENT "ocu_cc --target cpu -D__CUDA_ARCH__=500 rt_trap.cu")
  add_executable(rt_trap_arch ${_trap_arch_obj})
  set_target_properties(rt_trap_arch PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(rt_trap_arch PRIVATE ${OCU_CUDART_TARGET})
  foreach(t "trap|rt_trap|trap|__trap\\(\\).*block \\(1,0,0\\) thread \\(2,0,0\\)|before trap block 1 thread 2"
            "nodevcode|rt_trap_arch|nodevcode|__trap\\(\\).*block \\(1,0,0\\) thread \\(2,0,0\\)|compiled without support"
            "gridsync|rt_trap|gridsync|__trap\\(\\)|")
    string(REPLACE "|" ";" t "${t}")
    list(GET t 0 tname)
    list(GET t 1 exe)
    list(GET t 2 mode)
    list(GET t 3 errre)
    list(GET t 4 outre)
    add_test(NAME runtime_trap_${tname}
      COMMAND ${CMAKE_COMMAND} -DPROG=$<TARGET_FILE:${exe}> -DMODE=${mode} -DKIND=cpu
              "-DEMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}"
              "-DERR_REGEX=${errre}" "-DOUT_REGEX=${outre}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime/trap_test.cmake)
  endforeach()
  # Compile-time cases: what must and must not compile (runtime/cg_compile_cases.cu).
  foreach(c "0|ok" "1|fail" "2|fail" "3|fail" "4|fail" "5|fail" "6|fail" "7|fail" "20|ok" "21|ok" "22|ok")
    string(REPLACE "|" ";" c "${c}")
    list(GET c 0 cn)
    list(GET c 1 cx)
    add_test(NAME runtime_compile_case${cn}_cpu
      COMMAND ${CMAKE_COMMAND} -DPY=${Python3_EXECUTABLE} -DCC=${OCU_CC} -DCXX=${CMAKE_CXX_COMPILER}
              -DCC_COMPILER=${CMAKE_C_COMPILER}
              -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/runtime/cg_compile_cases.cu
              -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cg_case${cn}_cpu.o -DTARGET=cpu -DCASE=${cn} -DEXPECT=${cx}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime/compile_case.cmake)
  endforeach()
  ocu_add_cpu_cu_executable(rt_gaps_intrin runtime/rt_gaps_intrin.cu)
  add_test(NAME runtime_gaps_intrin COMMAND rt_gaps_intrin)
  add_test(NAME runtime_gaps_intrin_random COMMAND rt_gaps_intrin)
  set_tests_properties(runtime_gaps_intrin_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3;OCU_CPU_SEED=9")
  ocu_add_cpu_cu_executable(rt_gaps_api runtime/rt_gaps_api.cu)
  add_test(NAME runtime_gaps_api COMMAND rt_gaps_api)
  add_test(NAME runtime_gaps_api_cc80 COMMAND rt_gaps_api 8 0)
  set_tests_properties(runtime_gaps_api_cc80 PROPERTIES ENVIRONMENT "OCU_COMPUTE_CAPABILITY=8.0")
  add_test(NAME runtime_gaps_api_cc_bad_x COMMAND rt_gaps_api 5 0)
  set_tests_properties(runtime_gaps_api_cc_bad_x PROPERTIES ENVIRONMENT "OCU_COMPUTE_CAPABILITY=x")
  add_test(NAME runtime_gaps_api_cc_bad_50 COMMAND rt_gaps_api 5 0)
  set_tests_properties(runtime_gaps_api_cc_bad_50 PROPERTIES ENVIRONMENT "OCU_COMPUTE_CAPABILITY=50")
  ocu_add_cpu_cu_executable(rt_review2 runtime/rt_review2.cu)
  add_test(NAME runtime_review2 COMMAND rt_review2)
  ocu_add_cpu_cu_executable(rt_launch_api runtime/rt_launch_api.cu)
  add_test(NAME runtime_launch_api COMMAND rt_launch_api)
  # card libs-cub-subset: programmatic dependent launch, cudaLaunchKernelEx.
  ocu_add_cpu_cu_executable(rt_pdl runtime/rt_pdl.cu)
  add_test(NAME runtime_pdl COMMAND rt_pdl)
  add_test(NAME runtime_pdl_random COMMAND rt_pdl)
  set_tests_properties(runtime_pdl_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3")
  add_executable(rt_pdl_c runtime/rt_pdl_c.c)
  target_link_libraries(rt_pdl_c PRIVATE ${OCU_CUDART_TARGET})
  add_test(NAME runtime_pdl_c COMMAND rt_pdl_c)
  # card libs-cub-subset: CUB subset (cub/cub.cuh), device-wide and block-wide.
  ocu_add_cpu_cu_executable(cub_device libs/cub_device.cu)
  add_test(NAME cub_device COMMAND cub_device)
  set_tests_properties(cub_device PROPERTIES TIMEOUT 1200)
  add_test(NAME cub_device_random COMMAND cub_device)
  set_tests_properties(cub_device_random PROPERTIES TIMEOUT 1200
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3")
  ocu_add_cpu_cu_executable(cub_block libs/cub_block.cu)
  add_test(NAME cub_block COMMAND cub_block)
  add_test(NAME cub_block_random COMMAND cub_block)
  set_tests_properties(cub_block_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=3")
  add_test(NAME cub_block_warp64 COMMAND cub_block)
  set_tests_properties(cub_block_warp64 PROPERTIES
    ENVIRONMENT "OCU_CPU_WARP=64;OCU_CPU_ORDER=random;OCU_CPU_SEED=3")
  ocu_add_cpu_cu_executable(rt_review runtime/rt_review.cu)
  add_test(NAME runtime_review COMMAND rt_review)
  add_test(NAME runtime_review_random COMMAND rt_review)
  set_tests_properties(runtime_review_random PROPERTIES
    ENVIRONMENT "OCU_CPU_ORDER=random;OCU_CPU_THREADS=2")
  ocu_add_cpu_cu_executable(ocu_cc_saxpy probe/main_saxpy.cu
    ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu)
  add_test(NAME ocu_cc_cpu_saxpy COMMAND ocu_cc_saxpy)

  # The Metal backend on a real Apple GPU (SKIP without one).
  if(OCU_BACKEND_METAL)
    add_executable(test_metal_backend backend/test_metal_backend.c)
    target_link_libraries(test_metal_backend PRIVATE ocu_backend_metal)
    add_test(NAME metal_backend COMMAND test_metal_backend)
    # CUDA kernels translated to MSL by cu2msl.py (with the C++ compiler's
    # CUDA front end), then run on the GPU. Translation happens in the test,
    # so a compiler without CUDA support fails this test, not the build.
    add_executable(test_metal_cu2msl backend/test_metal_cu2msl.c)
    target_link_libraries(test_metal_cu2msl PRIVATE ocu_backend_metal)
    add_test(NAME metal_cu2msl COMMAND ${CMAKE_COMMAND}
      -DPY=${Python3_EXECUTABLE} -DCLANG=${CMAKE_CXX_COMPILER}
      -DTOOL=${PROJECT_SOURCE_DIR}/compiler/metal/cu2msl.py
      -DSRC=${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu
      -DIMG=${CMAKE_CURRENT_BINARY_DIR}/probe_kernel.metal.img
      -DRUN=$<TARGET_FILE:test_metal_cu2msl>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_cu2msl_test.cmake)
    # Translator features (address-space overloads, templates, module
    # variables, struct arguments, dynamic shared memory) on the GPU.
    add_executable(test_metal_features metal/test_metal_features.c)
    target_link_libraries(test_metal_features PRIVATE ocu_backend_metal)
    add_test(NAME metal_cu2msl_features COMMAND ${CMAKE_COMMAND}
      -DPY=${Python3_EXECUTABLE} -DCLANG=${CMAKE_CXX_COMPILER}
      -DTOOL=${PROJECT_SOURCE_DIR}/compiler/metal/cu2msl.py
      -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/metal/cu2msl_features.cu
      -DIMG=${CMAKE_CURRENT_BINARY_DIR}/cu2msl_features.metal.img
      -DRUN=$<TARGET_FILE:test_metal_features>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal_cu2msl_test.cmake)
  endif()

  # Device math library, compiled for the host, against long double.
  add_executable(test_device_math runtime/test_device_math.cpp)
  target_include_directories(test_device_math PRIVATE ${PROJECT_SOURCE_DIR}/runtime/include)
  add_test(NAME device_math COMMAND test_device_math)
  # The float functions against the host's double ones, on all cores; the
  # one-argument ones on every 257th float (OCU_MATHF_EXHAUSTIVE=1: all).
  find_package(Threads REQUIRED)
  add_executable(test_device_mathf runtime/test_device_mathf.cpp)
  target_include_directories(test_device_mathf PRIVATE ${PROJECT_SOURCE_DIR}/runtime/include)
  target_compile_options(test_device_mathf PRIVATE -O2 -ffp-contract=off)
  target_link_libraries(test_device_mathf PRIVATE Threads::Threads m)
  add_test(NAME device_mathf COMMAND test_device_mathf)

  # The installed package works for a project outside this tree, built
  # with the same compilers and flags (sanitizers included). Not when
  # cross-compiling: the consumer would need the whole toolchain setup.
  if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME install_package
      COMMAND ${CMAKE_COMMAND} -DBUILD=${PROJECT_BINARY_DIR}
              -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/install
              -DWORK=${CMAKE_CURRENT_BINARY_DIR}/install_test
              -DCC=${CMAKE_C_COMPILER} -DCXX=${CMAKE_CXX_COMPILER}
              "-DCFLAGS=${CMAKE_C_FLAGS}" "-DCXXFLAGS=${CMAKE_CXX_FLAGS}"
              "-DLDFLAGS=${CMAKE_EXE_LINKER_FLAGS}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/install/install_test.cmake)
  endif()

  # GPU-target compiles need an LLVM clang with the NVPTX and AMDGPU
  # back ends (Apple's clang has neither).
  find_program(OCU_CLANGXX NAMES clang++-18 clang++ PATHS /usr/lib/llvm-18/bin)
  set(OCU_CLANG_GPU_TARGETS OFF)
  if(OCU_CLANGXX)
    execute_process(COMMAND ${OCU_CLANGXX} --print-targets
                    OUTPUT_VARIABLE _ocu_targets ERROR_QUIET)
    if(_ocu_targets MATCHES "nvptx64" AND _ocu_targets MATCHES "amdgcn")
      set(OCU_CLANG_GPU_TARGETS ON)
    endif()
  endif()

  # cu2msl.py on any host (no Metal device or compiler needed): what it
  # refuses, the shape of what it emits, and the generated MSL compiled
  # as C++ for OpenCL against tests/metal/msl_stub.h (address spaces are
  # checked as MSL checks them). Needs a Clang with the CUDA and C++ for
  # OpenCL front ends.
  set(_ocu_msl_clang "")
  foreach(_c ${OCU_CLANGXX} ${CMAKE_CXX_COMPILER})
    if(NOT _ocu_msl_clang)
      execute_process(COMMAND ${_c} -x clcpp -cl-std=clc++2021 -fsyntax-only -
                      INPUT_FILE /dev/null RESULT_VARIABLE _ocu_rc OUTPUT_QUIET ERROR_QUIET)
      if(_ocu_rc EQUAL 0)
        set(_ocu_msl_clang ${_c})
      endif()
    endif()
  endforeach()
  if(_ocu_msl_clang)
    add_test(NAME metal_cu2msl_host
      COMMAND ${CMAKE_COMMAND} -E env OCU_CLANG=${_ocu_msl_clang}
              ${Python3_EXECUTABLE} -m unittest test_cu2msl
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/metal)
    set(_ocu_msl_srcs
      ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu
      ${PROJECT_SOURCE_DIR}/compiler/probe/probe_warp.cu
      ${PROJECT_SOURCE_DIR}/compiler/probe/probe_features.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/metal/cu2msl_features.cu)
    string(REPLACE ";" "|" _ocu_msl_srcs "${_ocu_msl_srcs}")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msl_check)
    add_test(NAME metal_msl_check COMMAND ${CMAKE_COMMAND}
      -DPY=${Python3_EXECUTABLE} -DCLANG=${_ocu_msl_clang}
      -DTOOL=${PROJECT_SOURCE_DIR}/compiler/metal/cu2msl.py
      -DCHECK=${CMAKE_CURRENT_SOURCE_DIR}/metal/msl_check.py
      -DSRCS=${_ocu_msl_srcs} -DEXPECT=19
      -DWORK=${CMAKE_CURRENT_BINARY_DIR}/msl_check
      -P ${CMAKE_CURRENT_SOURCE_DIR}/metal/msl_check_test.cmake)
  endif()

  # The OpenCL backend. Hand-written OpenCL C kernels need only a device;
  # CUDA translated by cu2ocl.py needs the Clang found above, and SPIR-V
  # (for devices that take only SPIR-V, such as PoCL) needs llvm-spirv.
  # Device tests exit 77 (skipped) without a usable device.
  # OCU_OPENCL_TEST_LIBRARY names the OpenCL library the tests load.
  if(OCU_BACKEND_OPENCL)
    set(OCU_OPENCL_TEST_LIBRARY "" CACHE FILEPATH
        "OpenCL library (ICD loader or driver) the tests load; empty: libOpenCL.so.1")
    find_program(OCU_LLVM_SPIRV NAMES llvm-spirv llvm-spirv-20 llvm-spirv-19 llvm-spirv-18
                 DOC "SPIR-V translator used by cu2ocl.py")
    set(_ocu_ocl_env "")
    if(OCU_OPENCL_TEST_LIBRARY)
      set(_ocu_ocl_env "OCU_OPENCL_LIBRARY=${OCU_OPENCL_TEST_LIBRARY}")
    endif()
    set(_ocu_spirv none)
    if(OCU_LLVM_SPIRV)
      set(_ocu_spirv ${OCU_LLVM_SPIRV})
    endif()
    add_executable(test_opencl_backend backend/test_opencl_backend.c)
    target_link_libraries(test_opencl_backend PRIVATE ocu_backend_opencl)
    add_test(NAME opencl_backend COMMAND test_opencl_backend)
    set_tests_properties(opencl_backend PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "${_ocu_ocl_env}")
    if(_ocu_msl_clang)
      add_test(NAME opencl_cu2ocl_host
        COMMAND ${CMAKE_COMMAND} -E env OCU_CLANG=${_ocu_msl_clang} OCU_LLVM_SPIRV=${_ocu_spirv}
                ${Python3_EXECUTABLE} -m unittest test_cu2ocl
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/opencl)
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ocl_img)
      foreach(_t "probe_kernel|${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu|test_opencl_cu2ocl"
                 "cu2msl_features|${CMAKE_CURRENT_SOURCE_DIR}/metal/cu2msl_features.cu|test_opencl_features")
        string(REPLACE "|" ";" _t "${_t}")
        list(GET _t 0 _name)
        list(GET _t 1 _src)
        list(GET _t 2 _exe)
        set(_img ${CMAKE_CURRENT_BINARY_DIR}/ocl_img/${_name}.ocl.img)
        add_test(NAME opencl_translate_${_name}
          COMMAND ${CMAKE_COMMAND} -E env OCU_LLVM_SPIRV=${_ocu_spirv}
                  ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/compiler/opencl/cu2ocl.py
                  --clang ${_ocu_msl_clang} ${_src} -o ${_img})
        set_tests_properties(opencl_translate_${_name} PROPERTIES FIXTURES_SETUP ocl_${_name})
        add_executable(${_exe} opencl/${_exe}.c)
        target_link_libraries(${_exe} PRIVATE ocu_backend_opencl)
        add_test(NAME opencl_run_${_name} COMMAND ${_exe} ${_img})
        set_tests_properties(opencl_run_${_name} PROPERTIES FIXTURES_REQUIRED ocl_${_name}
          SKIP_RETURN_CODE 77 ENVIRONMENT "${_ocu_ocl_env}")
      endforeach()
    endif()
  endif()
  if(OCU_CLANG_GPU_TARGETS)
    # The same compile-time cases for the NVIDIA path (case 8 is only
    # enforceable there: device-only functions called from host code).
    foreach(c "0|ok" "1|fail" "2|fail" "3|fail" "4|fail" "5|fail" "6|fail" "7|fail" "8|fail" "20|ok" "21|ok" "22|ok")
      string(REPLACE "|" ";" c "${c}")
      list(GET c 0 cn)
      list(GET c 1 cx)
      add_test(NAME runtime_compile_case${cn}_nvidia
        COMMAND ${CMAKE_COMMAND} -DPY=${Python3_EXECUTABLE} -DCC=${OCU_CC} -DCXX=${CMAKE_CXX_COMPILER}
                -DCC_COMPILER=${CMAKE_C_COMPILER}
                -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/runtime/cg_compile_cases.cu
                -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cg_case${cn}_nvidia.o -DTARGET=nvidia -DCASE=${cn} -DEXPECT=${cx}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime/compile_case.cmake)
    endforeach()
    foreach(c "2|warn|ptx_shift_cases.cu" "3|warn|outside_shift.h"
              "4|warn|ptx_shift_cases.cu" "9|warn|ptx_shift_cases.cu" "1|quiet|-" "8|quiet|-" "5|quiet|-" "0|quiet|-" "6|error|-")
      string(REPLACE "|" ";" c "${c}")
      list(GET c 0 cn)
      list(GET c 1 cx)
      list(GET c 2 cf)
      ocu_ptx_shift_cmd(ptx_shift_cmd_case${cn}_${cx}_nvidia nvidia DEFS -DMODE=warn -DCASE=${cn} -DEXPECT=${cx} -DMARK_FILE=${cf})
    endforeach()
    ocu_ptx_shift_cmd(ptx_shift_cmd_case10_warn_nvidia nvidia DEFS -DMODE=warn -DCASE=10 -DEXPECT=warn -DMARK_FILE=ptx_shift_cases_if.cu -DSRC_NAME=ptx_shift_cases_if.cu)
    ocu_ptx_shift_cmd(ptx_shift_cmd_print_nvidia nvidia DEFS -DMODE=print)
    foreach(t "nvidia;sm_80;ptx" "amd;gfx1100;co" "amd;gfx1201;co")
      list(GET t 0 target)
      list(GET t 1 arch)
      list(GET t 2 ext)
      foreach(src probe_kernel probe_features probe_review probe_warp)
        add_test(NAME ocu_cc_${target}_${arch}_${src}
          COMMAND ${Python3_EXECUTABLE} ${OCU_CC} --target ${target} --arch ${arch}
                  --cache ${OCU_CC_OUT}
                  -c ${PROJECT_SOURCE_DIR}/compiler/probe/${src}.cu
                  -o ${OCU_CC_OUT}/${src}.${arch}.${ext})
      endforeach()
    endforeach()
  endif()
  # The NVIDIA backend on a real GPU, with PTX Clang makes from the probe
  # kernel. Without an NVIDIA driver and device it prints SKIP and passes.
  if(OCU_CLANG_GPU_TARGETS AND OCU_BACKEND_NVIDIA)
    add_custom_command(
      OUTPUT ${OCU_CC_OUT}/probe_kernel.nvtest.ptx
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OCU_CC_OUT}
      COMMAND ${Python3_EXECUTABLE} ${OCU_CC} --target nvidia --arch sm_80 --cache ${OCU_CC_OUT}
              -c ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu
              -o ${OCU_CC_OUT}/probe_kernel.nvtest.ptx
      DEPENDS ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu ${OCU_CC})
    add_custom_target(nvtest_ptx ALL DEPENDS ${OCU_CC_OUT}/probe_kernel.nvtest.ptx)
    add_executable(test_nvidia_backend backend/test_nvidia_backend.c)
    target_link_libraries(test_nvidia_backend PRIVATE ocu_backend_nvidia m)
    add_dependencies(test_nvidia_backend nvtest_ptx)
    add_test(NAME nvidia_backend COMMAND test_nvidia_backend ${OCU_CC_OUT}/probe_kernel.nvtest.ptx)
    # The runtime test programs again, built for the GPU: the same CUDA
    # sources, on the installed NVIDIA GPU (skipped without one).
    add_executable(run_if_nvidia runtime/run_if_nvidia.c)
    target_link_libraries(run_if_nvidia PRIVATE ocu_backend_nvidia)
    foreach(t rt_api rt_lang rt_mem rt_misc rt_launch_api rt_review rt_review2 rt_fp16 rt_bf16 rt_pdl)
      ocu_add_nvidia_cu_executable(${t}_nv runtime/${t}.cu)
      add_test(NAME gpu_${t} COMMAND run_if_nvidia $<TARGET_FILE:${t}_nv>)
      set_tests_properties(gpu_${t} PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endforeach()
    foreach(t cub_device cub_block)
      ocu_add_nvidia_cu_executable(${t}_nv libs/${t}.cu)
      add_test(NAME gpu_${t} COMMAND run_if_nvidia $<TARGET_FILE:${t}_nv>)
      set_tests_properties(gpu_${t} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endforeach()
    ocu_add_nvidia_cu_executable(ptx_shift_run_nv compiler/ptx_shift/ptx_shift_run.cu PTX_SHIFT)
    add_test(NAME gpu_ptx_shift COMMAND run_if_nvidia $<TARGET_FILE:ptx_shift_run_nv>)
    set_tests_properties(gpu_ptx_shift PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    ocu_add_nvidia_cu_executable(fma_exact_test_nv runtime/fma_exact_test.cu)
    add_test(NAME gpu_fma_exact COMMAND run_if_nvidia $<TARGET_FILE:fma_exact_test_nv>)
    add_test(NAME gpu_fma_exact_powf COMMAND run_if_nvidia $<TARGET_FILE:fma_exact_test_nv> powf)
    set_tests_properties(gpu_fma_exact gpu_fma_exact_powf PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    add_test(NAME gpu_ptx_shift_eval_order COMMAND run_if_nvidia $<TARGET_FILE:ptx_shift_run_nv> order)
    set_tests_properties(gpu_ptx_shift_eval_order PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    ocu_add_nvidia_cu_executable(rt_trap_nv runtime/rt_trap.cu)
    foreach(mode trap nodevcode gridsync)
      add_test(NAME gpu_runtime_trap_${mode}
        COMMAND ${CMAKE_COMMAND} -DPROG=$<TARGET_FILE:rt_trap_nv> -DMODE=${mode} -DKIND=nvidia
                -DWRAPPER=$<TARGET_FILE:run_if_nvidia>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime/trap_test.cmake)
      set_tests_properties(gpu_runtime_trap_${mode} PROPERTIES SKIP_REGULAR_EXPRESSION "OCU_SKIP"
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endforeach()
    ocu_add_nvidia_cu_executable(rt_gaps_intrin_nv runtime/rt_gaps_intrin.cu)
    add_test(NAME gpu_rt_gaps_intrin COMMAND run_if_nvidia $<TARGET_FILE:rt_gaps_intrin_nv>)
    set_tests_properties(gpu_rt_gaps_intrin PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    ocu_add_nvidia_cu_executable(rt_gaps_api_nv runtime/rt_gaps_api.cu)
    foreach(v "default||5 0" "cc80|OCU_COMPUTE_CAPABILITY=8.0|8 0"
              "cc_bad_x|OCU_COMPUTE_CAPABILITY=x|5 0" "cc_bad_50|OCU_COMPUTE_CAPABILITY=50|5 0")
      string(REPLACE "|" ";" v "${v}")
      list(GET v 0 tag)
      list(GET v 1 venv)
      list(GET v 2 vargs)
      separate_arguments(vargs)
      add_test(NAME gpu_rt_gaps_api_${tag} COMMAND run_if_nvidia $<TARGET_FILE:rt_gaps_api_nv> ${vargs})
      set_tests_properties(gpu_rt_gaps_api_${tag} PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;${venv}")
    endforeach()
    # The float math functions on the GPU: same bits as on the host, and
    # kernel times against the double-based versions.
    ocu_add_nvidia_cu_executable(rt_mathf_gpu_nv runtime/rt_mathf_gpu.cu runtime/rt_mathf_ref.cpp)
    add_test(NAME gpu_rt_mathf COMMAND run_if_nvidia $<TARGET_FILE:rt_mathf_gpu_nv>)
    set_tests_properties(gpu_rt_mathf PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    ocu_add_nvidia_cu_executable(ocu_cc_saxpy_nv probe/main_saxpy.cu
      ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu)
    add_test(NAME gpu_saxpy COMMAND run_if_nvidia $<TARGET_FILE:ocu_cc_saxpy_nv>)
    set_tests_properties(gpu_saxpy PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    # Leak checking is off for this one test: the driver and the code
    # generator it loads at run time (no symbol names) keep allocations for
    # the life of the process. Memory errors are still checked.
    set_tests_properties(nvidia_backend PROPERTIES ENVIRONMENT
      "ASAN_OPTIONS=detect_leaks=0;LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_nvidia_driver.supp")
  endif()
  add_test(NAME ocu_cc_refuses_vendor_path
    COMMAND ${Python3_EXECUTABLE} ${OCU_CC} --target cpu -I /usr/local/cuda/include
            -c ${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu -o ${OCU_CC_OUT}/x.o)
  set_tests_properties(ocu_cc_refuses_vendor_path PROPERTIES WILL_FAIL TRUE)
endif()

# ---- the BLAS library API (cublas_v2.h): the same test program on the CPU
# path and on an NVIDIA GPU (skipped without one). OCU_CUBLAS_TILE forces
# each tile configuration, so both are checked whatever the device size.
if(TARGET ocu_cublas_cpu)
  ocu_add_cpu_cu_executable(cublas_accuracy libs/cublas_accuracy.cu LIBRARIES ocu_cublas_cpu)
  add_test(NAME cublas_cpu COMMAND cublas_accuracy)
  add_test(NAME cublas_cpu_tile64_random COMMAND cublas_accuracy)
  set_tests_properties(cublas_cpu_tile64_random PROPERTIES
    ENVIRONMENT "OCU_CUBLAS_TILE=64;OCU_CPU_ORDER=random;OCU_CPU_SEED=9")
  add_test(NAME cublas_cpu_tile128 COMMAND cublas_accuracy)
  set_tests_properties(cublas_cpu_tile128 PROPERTIES ENVIRONMENT "OCU_CUBLAS_TILE=128")
endif()
# The reference BLAS cases (conformance/oracle) through the library, for
# the functions it offers.
set(OCU_CUBLAS_CASES "gemm,gemv:f32,axpy:f32,dot:f32,scal:f32,nrm2:f32,trsm:f32")
if(TARGET ocu_cublas_cpu)
  ocu_add_cpu_cu_executable(cublas_driver conformance/cublas_driver.cu LIBRARIES ocu_cublas_cpu)
  add_test(NAME conformance_cublas_cpu
    COMMAND ${Python3_EXECUTABLE} -m ocu_conf.blas_run $<TARGET_FILE:cublas_driver>
            --only ${OCU_CUBLAS_CASES} --out ${CMAKE_CURRENT_BINARY_DIR}/cublas_cpu_result.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/conformance/harness)
  add_test(NAME conformance_cublas_cpu_stream
    COMMAND ${Python3_EXECUTABLE} -m ocu_conf.blas_run $<TARGET_FILE:cublas_driver>
            --only ${OCU_CUBLAS_CASES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/conformance/harness)
  set_tests_properties(conformance_cublas_cpu_stream PROPERTIES
    ENVIRONMENT "OCU_DRIVER_STREAM=1;OCU_CPU_ORDER=random${_ocu_launch}")
  if(_ocu_launch) # cross builds: the driver runs through the emulator
    string(REGEX REPLACE "^;" "" _ocu_launch_only "${_ocu_launch}")
    set_tests_properties(conformance_cublas_cpu PROPERTIES ENVIRONMENT "${_ocu_launch_only}")
  endif()
endif()
if(TARGET ocu_cublas_nvidia AND TARGET run_if_nvidia)
  ocu_add_nvidia_cu_executable(cublas_accuracy_nv libs/cublas_accuracy.cu LIBRARIES ocu_cublas_nvidia)
  foreach(tile default 64 128)
    add_test(NAME gpu_cublas_${tile} COMMAND run_if_nvidia $<TARGET_FILE:cublas_accuracy_nv>)
    set(_env "ASAN_OPTIONS=detect_leaks=0")
    if(NOT tile STREQUAL "default")
      list(APPEND _env "OCU_CUBLAS_TILE=${tile}")
    endif()
    set_tests_properties(gpu_cublas_${tile} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "${_env}")
  endforeach()
  ocu_add_nvidia_cu_executable(cublas_driver_nv conformance/cublas_driver.cu LIBRARIES ocu_cublas_nvidia)
  add_test(NAME gpu_conformance_cublas
    COMMAND run_if_nvidia ${Python3_EXECUTABLE} -m ocu_conf.blas_run
            $<TARGET_FILE:cublas_driver_nv> --backend nvidia --only ${OCU_CUBLAS_CASES}
            --out ${CMAKE_CURRENT_BINARY_DIR}/cublas_nvidia_result.json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/conformance/harness)
  set_tests_properties(gpu_conformance_cublas PROPERTIES SKIP_RETURN_CODE 77
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;OCU_DRIVER_STREAM=1")
endif()

# ---- ocu-nvcc: the nvcc-compatible driver and the toolkit folder ----------
if(OCU_BACKEND_CPU AND Python3_FOUND AND NOT WIN32 AND NOT CMAKE_CROSSCOMPILING)
  set(OCU_NVCC ${OCU_TOOLKIT_DIR}/bin/ocu-nvcc)
  set(_drv ${CMAKE_CURRENT_SOURCE_DIR}/driver)
  # The command line: spellings, phases, dependencies, libraries, refusals.
  add_test(NAME ocu_nvcc_options
    COMMAND ${Python3_EXECUTABLE} -m unittest -v test_ocu_nvcc
    WORKING_DIRECTORY ${_drv})
  # Sanitizer builds: the toolkit's libraries are instrumented, so programs
  # the driver builds must be too (Clang reads CCC_OVERRIDE_OPTIONS; "#"
  # keeps it quiet). The test programs' volatile host-flag waits (the CUDA
  # idiom for host-to-kernel signals) are races in C++ terms: suppressed.
  set(_ocu_san_env)
  string(REGEX MATCH "-fsanitize=[a-z,]+" _ocu_san "${CMAKE_CXX_FLAGS}")
  if(_ocu_san)
    list(APPEND _ocu_san_env "CCC_OVERRIDE_OPTIONS=#+${_ocu_san}")
    if(_ocu_san MATCHES "thread")
      list(APPEND _ocu_san_env
        "TSAN_OPTIONS=halt_on_error=1 suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan_cuda_idioms.supp")
    endif()
  endif()
  set_tests_properties(ocu_nvcc_options PROPERTIES ENVIRONMENT
    "OCU_NVCC=${OCU_NVCC};OCU_TARGET=cpu;${_ocu_san_env}")

  # The runtime test programs, compiled one file at a time and linked, as a
  # Makefile drives nvcc.
  set(_ocu_nvcc_programs
    "rt_api|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_api.cu"
    "rt_lang|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_lang.cu"
    "rt_mem|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_mem.cu"
    "rt_misc|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_misc.cu"
    "rt_fp16|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_fp16.cu"
    "rt_launch_api|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_launch_api.cu"
    "rt_review|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_review.cu"
    "rt_review2|${CMAKE_CURRENT_SOURCE_DIR}/runtime/rt_review2.cu"
    "saxpy|${CMAKE_CURRENT_SOURCE_DIR}/probe/main_saxpy.cu,${PROJECT_SOURCE_DIR}/compiler/probe/probe_kernel.cu")
  set(_ocu_nvcc_targets cpu)
  if(OCU_CLANG_GPU_TARGETS AND OCU_BACKEND_NVIDIA)
    list(APPEND _ocu_nvcc_targets nvidia)
  endif()
  foreach(target IN LISTS _ocu_nvcc_targets)
    set(_wrap)
    set(_env ${_ocu_san_env})
    if(target STREQUAL "nvidia")
      # Skipped (exit 77) without a usable NVIDIA device.
      set(_wrap -DRUN_WRAPPER=$<TARGET_FILE:run_if_nvidia>)
      list(APPEND _env "ASAN_OPTIONS=detect_leaks=0")
    endif()
    foreach(p IN LISTS _ocu_nvcc_programs)
      string(REPLACE "|" ";" p "${p}")
      list(GET p 0 name)
      list(GET p 1 srcs)
      add_test(NAME ocu_nvcc_${target}_${name}
        COMMAND ${CMAKE_COMMAND} -DDRIVER=${OCU_NVCC} -DTARGET=${target}
                -DWORK=${CMAKE_CURRENT_BINARY_DIR}/ocu_nvcc/${target}_${name}
                "-DSOURCES=${srcs}" -DINCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/runtime
                "-DFLAGS=${CMAKE_CXX_FLAGS}" ${_wrap}
                -P ${_drv}/run_driver.cmake)
      set_tests_properties(ocu_nvcc_${target}_${name} PROPERTIES
        SKIP_REGULAR_EXPRESSION "OCU_SKIP" ENVIRONMENT "${_env}")
    endforeach()

    # An ordinary CMake project with enable_language(CUDA) and
    # find_package(CUDAToolkit), using the toolkit through OcuCUDA.cmake.
    add_test(NAME ocu_nvcc_cmake_project_${target}
      COMMAND ${CMAKE_COMMAND} -DSRC=${_drv}/cmake_project
              -DWORK=${CMAKE_CURRENT_BINARY_DIR}/ocu_nvcc/cmake_${target}
              -DTARGET=${target} -DTOOLKIT=${OCU_TOOLKIT_DIR}
              -DCC=${CMAKE_C_COMPILER} -DCXX=${CMAKE_CXX_COMPILER}
              "-DCFLAGS=${CMAKE_C_FLAGS}" "-DCXXFLAGS=${CMAKE_CXX_FLAGS}"
              "-DLDFLAGS=${CMAKE_EXE_LINKER_FLAGS}" ${_wrap}
              -P ${_drv}/cmake_project_test.cmake)
    set_tests_properties(ocu_nvcc_cmake_project_${target} PROPERTIES
      SKIP_REGULAR_EXPRESSION "OCU_SKIP" ENVIRONMENT "${_env}")
  endforeach()

  # The same project with the installed toolkit (<prefix>/<libdir>/ocu/toolkit).
  add_test(NAME ocu_nvcc_cmake_project_installed
    COMMAND ${CMAKE_COMMAND} -DSRC=${_drv}/cmake_project
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/ocu_nvcc/cmake_installed
            -DTARGET=cpu -DINSTALL_FROM=${PROJECT_BINARY_DIR} -DLIBDIR=${CMAKE_INSTALL_LIBDIR}
            -DCC=${CMAKE_C_COMPILER} -DCXX=${CMAKE_CXX_COMPILER}
            "-DCFLAGS=${CMAKE_C_FLAGS}" "-DCXXFLAGS=${CMAKE_CXX_FLAGS}"
            "-DLDFLAGS=${CMAKE_EXE_LINKER_FLAGS}"
            -P ${_drv}/cmake_project_test.cmake)
endif()

# ---- operator table (docs/spec/op-table.md)------------------------------
# test_ops <backend>: conformance of one backend's operator table.
# test_npusim <mode>: the npusim profile (spec section 7), one entry per
# environment. op_support_table: writes the support table as markdown.
if(OCU_BACKEND_CPU OR OCU_BACKEND_NPUSIM)
  add_library(ocu_ops_test_common STATIC ops/ops_common.c)
  target_link_libraries(ocu_ops_test_common PUBLIC ocu_core m)

  add_executable(test_ops ops/test_ops.c)
  target_link_libraries(test_ops PRIVATE ocu_ops_test_common)

  add_executable(op_support_table ops/op_support_table.c)
  target_link_libraries(op_support_table PRIVATE ocu_core)
  add_test(NAME op_support_table
    COMMAND op_support_table ${CMAKE_CURRENT_BINARY_DIR}/op_support_table.md)
endif()
if(OCU_BACKEND_CPU)
  add_test(NAME ops_cpu COMMAND test_ops cpu)
  set_tests_properties(ops_cpu PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(OCU_BACKEND_NPUSIM)
  add_test(NAME ops_npusim COMMAND test_ops npusim)
  set_tests_properties(ops_npusim PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(test_npusim ops/test_npusim.c)
  target_link_libraries(test_npusim PRIVATE ocu_ops_test_common)
  add_test(NAME npusim_profile COMMAND test_npusim basic)
  set_tests_properties(npusim_profile PROPERTIES SKIP_RETURN_CODE 77)
  foreach(mode "scratch4;OCU_NPUSIM_SCRATCH_KB=4"
               "scratch1;OCU_NPUSIM_SCRATCH_KB=1"
               "tiling0;OCU_NPUSIM_TILING=0"
               "dtypes_f16;OCU_NPUSIM_DTYPES=f16"
               "env;OCU_NPUSIM_MEM_MB=8\;OCU_NPUSIM_PLAN_MS=40")
    list(GET mode 0 tag)
    list(GET mode 1 envs)
    add_test(NAME npusim_profile_${tag} COMMAND test_npusim ${tag})
    set_tests_properties(npusim_profile_${tag} PROPERTIES
      SKIP_RETURN_CODE 77 ENVIRONMENT "${envs}")
  endforeach()
endif()

# ---- the CPU path's two kernel forms (docs/cpu-region-loops.md, stage 1):
# every CPU-path test program run with OCU_CPU_FORM=fiber and =loop, fixed
# and random order, must exit the same, print the same and leave the same
# device memory after every launch (cpu_forms/form_diff.cmake). Only where
# the whole-block pass is built; elsewhere there is one form.
if(OCU_BACKEND_CPU AND OCU_CPU_LOOPS_PLUGIN)
  find_program(OCU_SETARCH setarch)
  # Not ptx_shift_run_plain: it shows what C++ makes of shifts by the width
  # or more (undefined behaviour), which may differ between the forms as
  # between optimisation levels.
  foreach(t rt_api rt_lang rt_warp_lang rt_mem rt_misc rt_fp16 rt_bf16 ptx_shift_run
            fma_exact_test rt_gaps_intrin rt_gaps_api rt_review rt_review2
            rt_launch_api rt_pdl cub_device cub_block ocu_cc_saxpy cublas_accuracy)
    if(TARGET ${t})
      add_test(NAME cpu_forms_${t}
        COMMAND ${CMAKE_COMMAND} -DRUN=$<TARGET_FILE:${t}> -DNAME=${t}
                -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cpu_forms
                -DSETARCH=${OCU_SETARCH} -DARCH=${CMAKE_HOST_SYSTEM_PROCESSOR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cpu_forms/form_diff.cmake)
    endif()
  endforeach()
endif()
