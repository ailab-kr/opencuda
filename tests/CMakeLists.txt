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

  add_executable(test_cpu_regressions backend/test_cpu_regressions.c)
  target_link_libraries(test_cpu_regressions PRIVATE ocu_backend_cpu)
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
  ocu_add_cpu_cu_executable(rt_mem runtime/rt_mem.cu)
  add_test(NAME runtime_memory COMMAND rt_mem)
  ocu_add_cpu_cu_executable(rt_misc runtime/rt_misc.cu)
  add_test(NAME runtime_misc COMMAND rt_misc)
  ocu_add_cpu_cu_executable(rt_fp16 runtime/rt_fp16.cu)
  add_test(NAME runtime_fp16 COMMAND rt_fp16)
  ocu_add_cpu_cu_executable(rt_review2 runtime/rt_review2.cu)
  add_test(NAME runtime_review2 COMMAND rt_review2)
  ocu_add_cpu_cu_executable(rt_launch_api runtime/rt_launch_api.cu)
  add_test(NAME runtime_launch_api COMMAND rt_launch_api)
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
  endif()

  # Device math library, compiled for the host, against long double.
  add_executable(test_device_math runtime/test_device_math.cpp)
  target_include_directories(test_device_math PRIVATE ${PROJECT_SOURCE_DIR}/runtime/include)
  add_test(NAME device_math COMMAND test_device_math)

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
  if(OCU_CLANG_GPU_TARGETS)
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
    foreach(t rt_api rt_lang rt_mem rt_misc rt_launch_api rt_review rt_review2 rt_fp16)
      ocu_add_nvidia_cu_executable(${t}_nv runtime/${t}.cu)
      add_test(NAME gpu_${t} COMMAND run_if_nvidia $<TARGET_FILE:${t}_nv>)
      set_tests_properties(gpu_${t} PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endforeach()
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
