cmake_minimum_required(VERSION 3.0)

project(libaes LANGUAGES CXX)
enable_testing()

add_library(aes)
target_link_libraries(aes PRIVATE utils)

add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)