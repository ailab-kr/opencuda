# Runs a program that is expected to trap, and checks how it ended.
#   -DPROG=<exe> -DMODE=<arg> -DKIND=cpu|nvidia [-DWRAPPER=<exe>]
#   [-DEMULATOR=<runner>] -DERR_REGEX=<regex on stderr, cpu only> -DOUT_REGEX=<regex on stdout, cpu only>
# CPU: the process must end abnormally, with the position message on
# stderr and the output printed before the trap on stdout.
# NVIDIA: the process runs on, the next synchronization reports 719
# (cudaErrorLaunchFailure) and nothing was written after the trap.
# EMULATOR: cross builds run the program through it (Wine).
execute_process(COMMAND ${EMULATOR} ${WRAPPER} ${PROG} ${MODE}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("exit: ${rc}\nstdout:\n${out}\nstderr:\n${err}")
if(WRAPPER AND rc EQUAL 77)
  message("OCU_SKIP: no usable NVIDIA device")
  return()
endif()
if(KIND STREQUAL "cpu")
  if(rc EQUAL 0 OR rc STREQUAL "")
    message(FATAL_ERROR "the process must end abnormally, it exited with ${rc}")
  endif()
  if(out MATCHES "after_trap_written")
    message(FATAL_ERROR "the kernel went on after the trap")
  endif()
  if(NOT err MATCHES "${ERR_REGEX}")
    message(FATAL_ERROR "stderr does not match '${ERR_REGEX}'")
  endif()
  if(OUT_REGEX AND NOT out MATCHES "${OUT_REGEX}")
    message(FATAL_ERROR "stdout lost the output printed before the trap (no match for '${OUT_REGEX}')")
  endif()
else()
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "the process itself must survive a failed launch (exit ${rc})")
  endif()
  if(NOT out MATCHES "sync=719 after_trap_written=(0|-1)")
    message(FATAL_ERROR "expected sync=719 and a flag that was not written (0, or -1 when the failed context cannot be read back)")
  endif()
endif()
