# ocu_add_cpu_cu_executable(<target> [PTX_SHIFT] <source.cu|.c|.cpp>...)
#
# Builds a CUDA program for the CPU path: every .cu source goes through
# ocu_cc.py --target cpu (preprocess with cuda_runtime.h force-included as
# nvcc does, rewrite, compile as C++17), other sources are compiled
# normally; the result links with the runtime and the CPU backend.
#
# PTX_SHIFT (or the variable OCU_PTX_SHIFT set true before the call) passes
# --ptx-shift: shifts in device code get the PTX shl/shr semantics (amounts
# clamped to the width) instead of standard C++ (compiler/probe/ptx_shift.py).
# Off by default. The same keyword works for ocu_add_nvidia_cu_executable.
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

# _ocu_shift_option(<out var> <PTX_SHIFT keyword given>)
macro(_ocu_shift_option out given)
  if(${given} OR OCU_PTX_SHIFT)
    set(${out} --ptx-shift)
  else()
    set(${out})
  endif()
endmacro()

function(ocu_add_cpu_cu_executable target)
  cmake_parse_arguments(PARSE_ARGV 1 _ocu "PTX_SHIFT" "" "")
  _ocu_shift_option(_shift _ocu_PTX_SHIFT)
  get_filename_component(_cc_dir ${OCU_CC_SCRIPT} DIRECTORY)
  set(objs)
  set(others)
  foreach(src IN LISTS _ocu_UNPARSED_ARGUMENTS)
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
                --target cpu -O2 ${_shift} -c ${abs} -o ${obj}
        DEPENDS ${abs} ${OCU_CC_SCRIPT} ${_cc_dir}/cpu_rewrite.py ${_cc_dir}/ptx_shift.py
                ${OCU_CC_DEPENDS}
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

# ocu_add_nvidia_cu_executable(<target> [PTX_SHIFT] <source.cu|.c|.cpp>...)
#
# The same program for NVIDIA GPUs: every .cu source goes through
# ocu_cc.py --target nvidia (device code as PTX in a container embedded in
# the host object, host code by Clang's CUDA mode); links the runtime with
# the NVIDIA backend. Needs an LLVM Clang with the NVPTX target.
function(ocu_add_nvidia_cu_executable target)
  cmake_parse_arguments(PARSE_ARGV 1 _ocu "PTX_SHIFT" "" "")
  _ocu_shift_option(_shift _ocu_PTX_SHIFT)
  get_filename_component(_cc_dir ${OCU_CC_SCRIPT} DIRECTORY)
  set(objs)
  foreach(src IN LISTS _ocu_UNPARSED_ARGUMENTS)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(name ${src} NAME)
    set(obj ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir/${name}.o)
    add_custom_command(
      OUTPUT ${obj}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir
      COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER}
              ${Python3_EXECUTABLE} ${OCU_CC_SCRIPT}
              --target nvidia -O2 ${_shift} -c ${abs} -o ${obj}
      DEPENDS ${abs} ${OCU_CC_SCRIPT} ${_cc_dir}/cpu_rewrite.py ${_cc_dir}/ptx_shift.py
              ${OCU_CC_DEPENDS}
      IMPLICIT_DEPENDS CXX ${abs}
      COMMENT "ocu_cc --target nvidia ${src}")
    list(APPEND objs ${obj})
  endforeach()
  add_executable(${target} ${objs})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PRIVATE ${OCU_CUDART_TARGET})
endfunction()
