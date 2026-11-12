# -DGOT=<OCU_LIBM_FMA_EXACT as configured> -DWANT=<ON|OFF>: the configure check
# of runtime/CMakeLists.txt must find glibc's fma/fmaf correctly rounded (ON)
# and MinGW-w64's not (OFF) (header runtime/include/ocu_fma_exact.h).
if(GOT AND NOT WANT)
  message(FATAL_ERROR "configure check says the C library is correctly rounded (OCU_LIBM_FMA_EXACT=${GOT}); expected OFF on this platform")
endif()
if(NOT GOT AND WANT)
  message(FATAL_ERROR "configure check says the C library is NOT correctly rounded; expected ON on this platform")
endif()
message("OCU_LIBM_FMA_EXACT=${GOT} as expected")
