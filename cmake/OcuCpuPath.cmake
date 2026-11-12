# ocu_add_cpu_cu_executable(<target> <source.cu|.cpp>...)
#
# Builds a CUDA program for the CPU path: every .cu source goes through
# compiler/probe/ocu_cc.py --target cpu (rewrite + C++17 compile with
# cuda_runtime.h force-included, as nvcc does), other sources are compiled
# normally; the result links with the runtime and the CPU backend.
function(ocu_add_cpu_cu_executable target)
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
                ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/compiler/probe/ocu_cc.py
                --target cpu -O2 -c ${abs} -o ${obj}
        DEPENDS ${abs} ${PROJECT_SOURCE_DIR}/compiler/probe/ocu_cc.py
                ${PROJECT_SOURCE_DIR}/compiler/probe/cpu_rewrite.py
                ${PROJECT_SOURCE_DIR}/runtime/include/cuda_runtime_api.h
                ${PROJECT_SOURCE_DIR}/runtime/include/ocu_cuda_lang_cpu.h
        IMPLICIT_DEPENDS CXX ${abs}
        COMMENT "ocu_cc --target cpu ${src}")
      list(APPEND objs ${obj})
    else()
      list(APPEND others ${abs})
    endif()
  endforeach()
  add_executable(${target} ${others} ${objs})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PRIVATE ocu_cudart)
endfunction()
