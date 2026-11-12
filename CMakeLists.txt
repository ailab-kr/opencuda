cmake_minimum_required(VERSION 3.20)
project(ocu VERSION 0.0.1 LANGUAGES C CXX)

option(OCU_BACKEND_CPU "Build the CPU reference backend" ON)
option(OCU_BACKEND_AMD "Build the AMD GPU backend (not available yet)" OFF)
option(OCU_BACKEND_INTEL "Build the Intel GPU backend (not available yet)" OFF)
option(OCU_BUILD_TESTS "Build smoke tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Python3 COMPONENTS Interpreter)

add_subdirectory(backends)
add_subdirectory(runtime)
add_subdirectory(libs)
add_subdirectory(tuning)

if(OCU_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
