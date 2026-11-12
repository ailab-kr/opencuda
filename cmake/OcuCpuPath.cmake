# ocu_add_cpu_cu_executable(<target> <source.cu|.c|.cpp>...)
#
# Builds a CUDA program for the CPU path: every .cu source goes through
# ocu_cc.py --target cpu (preprocess with cuda_runtime.h force-included as
# nvcc does, rewrite, compile as C++17), other sources are compiled
# normally; the result links with the runtime and the CPU backend.
#
# Works in this source tree and from an installed package (OcuConfig.cmake):
# both set OCU_CC_SCRIPT (the driver), OCU_CUDART_TARGET (the runtime
# target) and, optionally, OCU_CC_DEPENDS (files whose change rebuilds .cu
# objects).
if(NOT OCU_CC_SCRIPT OR NOT OCU_CUDART_TARGET)
  message(FATAL_ERROR "OcuCpuPath: OCU_CC_SCRIPT and OCU_CUDART_TARGET must be set")
endif()
if(NOT Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

function(ocu_add_cpu_cu_executable target)
  get_filename_component(_cc_dir ${OCU_CC_SCRIPT} DIRECTORY)
  set(objs)
  set(others)
  foreach(src IN LISTS ARGN)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(ext ${src} EXT)
    if(ext STREQUAL ".cu")
      get_filename_component(stem ${src} NAME_WE)
      set(obj ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir/${stem}.cu.o)
      add_custom_command(
        OUTPUT ${obj}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir
        COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
                ${Python3_EXECUTABLE} ${OCU_CC_SCRIPT}
                --target cpu -O2 -c ${abs} -o ${obj}
        DEPENDS ${abs} ${OCU_CC_SCRIPT} ${_cc_dir}/cpu_rewrite.py ${OCU_CC_DEPENDS}
        IMPLICIT_DEPENDS CXX ${abs}
        COMMENT "ocu_cc --target cpu ${src}")
      list(APPEND objs ${obj})
    else()
      list(APPEND others ${abs})
    endif()
  endforeach()
  add_executable(${target} ${others} ${objs})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PRIVATE ${OCU_CUDART_TARGET})
endfunction()
