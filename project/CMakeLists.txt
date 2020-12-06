cmake_minimum_required(VERSION 3.0)

project(cuaes-top)
enable_testing()

add_subdirectory(utils)
add_subdirectory(pyaes)
add_subdirectory(libaes)
add_subdirectory(cuaes)
add_subdirectory(benchmark)
