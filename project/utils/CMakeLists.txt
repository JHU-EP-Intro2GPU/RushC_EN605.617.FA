cmake_minimum_required(VERSION 3.0)

project(utils LANGUAGES CXX CUDA)

add_library(utils)

add_subdirectory(include)
add_subdirectory(src)
