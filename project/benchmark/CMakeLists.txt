cmake_minimum_required(VERSION 3.10)

project(benchmark LANGUAGES CXX CUDA)

add_executable(benchmark)
add_custom_target(run-benchmark COMMAND benchmark)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cuh *.cpp *.cu)
target_sources(benchmark PRIVATE ${SOURCES})
target_link_libraries(benchmark PRIVATE aes cuaes utils)
