# CMake functions that build CUDA sources with the project's toolchain
# (compiler/probe/ocu_cc.py), one function per kind of output and target:
#
#   ocu_add_cpu_cu_executable(<target> <sources>... [LIBRARIES <libs>...] [PTX_SHIFT])
#   ocu_add_nvidia_cu_executable(<target> <sources>... [LIBRARIES <libs>...] [PTX_SHIFT])
#   ocu_add_cu_library(<target> <cpu|nvidia> SOURCES <sources>...
#                      [INCLUDES <dirs>...] [DEFINES <NAME[=V]>...] [PTX_SHIFT])
#
# PTX_SHIFT (any of the three; or the variable OCU_PTX_SHIFT set true before
# the call) passes --ptx-shift: shifts in device code get the PTX shl/shr
# semantics (amounts clamped to the width) instead of standard C++
# (compiler/probe/ptx_shift.py). Off by default.
#
# Works in this source tree and from an installed package (OcuConfig.cmake):
# both set OCU_CC_SCRIPT (the driver), OCU_CUDART_TARGET (the runtime
# target) and, optionally, OCU_CC_DEPENDS (files whose change rebuilds .cu
# objects).
#
# LIBRARIES are linked, and their INTERFACE_INCLUDE_DIRECTORIES are passed
# to ocu_cc for the .cu sources (custom commands do not see target include
# directories otherwise). Use a library built for the same target: a
# CPU-path program links a cpu library, a GPU program an nvidia one.
if(NOT OCU_CC_SCRIPT OR NOT OCU_CUDART_TARGET)
  message(FATAL_ERROR "OcuCpuPath: OCU_CC_SCRIPT and OCU_CUDART_TARGET must be set")
endif()
if(NOT Python3_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

# _ocu_cu_objects(<objs-var> <others-var> <target> <cpu|nvidia>
#                 <include-flags> <define-flags> <sources>...)
# Custom commands that compile the .cu sources (on the nvidia path: every
# source) with ocu_cc; returns the object files and, on the cpu path, the
# sources left for the normal compiler.
function(_ocu_cu_objects objs_var others_var target kind incflags defflags)
  get_filename_component(_cc_dir ${OCU_CC_SCRIPT} DIRECTORY)
  set(objs)
  set(others)
  foreach(src IN LISTS ARGN)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(ext ${src} EXT)
    if(kind STREQUAL "cpu")
      if(NOT ext STREQUAL ".cu")
        list(APPEND others ${abs})
        continue()
      endif()
      get_filename_component(stem ${src} NAME_WE)
      set(obj ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir/${stem}.cu.o)
    else()
      get_filename_component(name ${src} NAME)
      set(obj ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir/${name}.o)
    endif()
    add_custom_command(
      OUTPUT ${obj}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${target}.dir
      COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER}
              ${Python3_EXECUTABLE} ${OCU_CC_SCRIPT}
              --target ${kind} -O2 ${incflags} ${defflags} -c ${abs} -o ${obj}
      DEPENDS ${abs} ${OCU_CC_SCRIPT} ${_cc_dir}/cpu_rewrite.py ${_cc_dir}/ptx_shift.py
              ${OCU_CC_DEPENDS}
      IMPLICIT_DEPENDS CXX ${abs}
      COMMAND_EXPAND_LISTS
      COMMENT "ocu_cc --target ${kind} ${src}")
    list(APPEND objs ${obj})
  endforeach()
  set(${objs_var} ${objs} PARENT_SCOPE)
  set(${others_var} ${others} PARENT_SCOPE)
endfunction()

# -I flags for the include directories the given libraries publish.
function(_ocu_lib_incflags out_var)
  set(flags)
  foreach(lib IN LISTS ARGN)
    set(dirs "$<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>")
    list(APPEND flags "$<$<BOOL:${dirs}>:-I$<JOIN:${dirs},$<SEMICOLON>-I>>")
  endforeach()
  set(${out_var} ${flags} PARENT_SCOPE)
endfunction()

# --ptx-shift when PTX_SHIFT was given or OCU_PTX_SHIFT is true.
macro(_ocu_shift_flags out given)
  if(${given} OR OCU_PTX_SHIFT)
    set(${out} --ptx-shift)
  else()
    set(${out})
  endif()
endmacro()

function(_ocu_add_cu_executable target kind)
  cmake_parse_arguments(PARSE_ARGV 2 ARG "PTX_SHIFT" "" "LIBRARIES")
  _ocu_lib_incflags(incflags ${ARG_LIBRARIES})
  _ocu_shift_flags(shiftflags ARG_PTX_SHIFT)
  _ocu_cu_objects(objs others ${target} ${kind} "${incflags}" "${shiftflags}"
                  ${ARG_UNPARSED_ARGUMENTS})
  add_executable(${target} ${others} ${objs})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES} ${OCU_CUDART_TARGET})
endfunction()

# The CPU path: every .cu source goes through ocu_cc.py --target cpu
# (preprocess with cuda_runtime.h force-included as nvcc does, rewrite,
# compile as C++17), other sources are compiled normally; the result links
# with the runtime and the CPU backend.
function(ocu_add_cpu_cu_executable target)
  _ocu_add_cu_executable(${target} cpu ${ARGN})
endfunction()

# The same program for NVIDIA GPUs: every source goes through ocu_cc.py
# --target nvidia (device code as PTX in a container embedded in the host
# object, host code by Clang's CUDA mode); links the runtime with the
# NVIDIA backend. Needs an LLVM Clang with the NVPTX target.
function(ocu_add_nvidia_cu_executable target)
  _ocu_add_cu_executable(${target} nvidia ${ARGN})
endfunction()

# A static library of CUDA sources for one target: the library a cpu or
# nvidia program links (see libs/cublas). INCLUDES are passed to ocu_cc for
# its own sources and published as the library's include directories.
function(ocu_add_cu_library target kind)
  cmake_parse_arguments(PARSE_ARGV 2 ARG "PTX_SHIFT" "" "SOURCES;INCLUDES;DEFINES")
  if(NOT kind MATCHES "^(cpu|nvidia)$")
    message(FATAL_ERROR "ocu_add_cu_library: target kind must be cpu or nvidia, not ${kind}")
  endif()
  set(incflags)
  foreach(d IN LISTS ARG_INCLUDES)
    get_filename_component(d ${d} ABSOLUTE)
    list(APPEND incflags -I ${d})
  endforeach()
  _ocu_shift_flags(defflags ARG_PTX_SHIFT)
  foreach(d IN LISTS ARG_DEFINES)
    list(APPEND defflags -D ${d})
  endforeach()
  _ocu_cu_objects(objs others ${target} ${kind} "${incflags}" "${defflags}" ${ARG_SOURCES})
  set_source_files_properties(${objs} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  add_library(${target} STATIC ${others} ${objs})
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PUBLIC ${OCU_CUDART_TARGET})
endfunction()
