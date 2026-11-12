add_executable(smoke_registry smoke/test_registry.c)
target_link_libraries(smoke_registry PRIVATE ocu_core)
add_test(NAME smoke_registry COMMAND smoke_registry)

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
endif()

# ---- compiler probe, CPU path (PLAN P0-W1-11) ----------------------------
# Unmodified .cu files are rewritten and compiled as plain C++.
find_package(Python3 COMPONENTS Interpreter)
if(OCU_BACKEND_CPU AND Python3_FOUND)
  set(OCU_PROBE_DIR ${PROJECT_SOURCE_DIR}/compiler/probe)
  set(OCU_PROBE_GEN ${CMAKE_CURRENT_BINARY_DIR}/probe_gen)
  set(OCU_PROBE_SOURCES)
  foreach(name probe_kernel probe_features)
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

  add_test(NAME probe_cpu_rewrite_unit
    COMMAND ${Python3_EXECUTABLE} -m unittest test_cpu_rewrite
    WORKING_DIRECTORY ${OCU_PROBE_DIR})
endif()
