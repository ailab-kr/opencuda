# Cross-compile for 64-bit Windows with MinGW-w64 (POSIX thread model, for
# winpthreads), from Linux:
#   cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-w64-mingw32.cmake
# Programs are linked statically, so they run on a Windows machine with no
# compiler or runtime installed. When Wine is available, ctest runs the
# tests through it.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
find_program(OCU_WINE NAMES wine64 wine PATHS /usr/lib/wine)
if(OCU_WINE)
  set(CMAKE_CROSSCOMPILING_EMULATOR ${OCU_WINE})
endif()
