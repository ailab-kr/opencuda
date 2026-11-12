# A project outside this tree that uses the installed package.
cmake_minimum_required(VERSION 3.20)
project(ocu_consumer LANGUAGES C CXX)
find_package(Ocu 0.0 REQUIRED)
ocu_add_cpu_cu_executable(consumer_saxpy saxpy.cu)
add_executable(consumer_host host.c)
target_link_libraries(consumer_host PRIVATE ocu::cudart)
