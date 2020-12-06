cmake_minimum_required(VERSION 3.10)

project(cuaes LANGUAGES CXX CUDA)
enable_testing()

add_library(cuaes)
target_link_libraries(cuaes PUBLIC utils)

add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
