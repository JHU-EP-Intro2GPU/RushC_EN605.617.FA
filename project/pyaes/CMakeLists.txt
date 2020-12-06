cmake_minimum_required(VERSION 3.0)

project(pyaes)
enable_testing()

find_package(Python 3.6 REQUIRED COMPONENTS Interpreter)
add_test(NAME test-pyaes 
    COMMAND ${Python_EXECUTABLE} -m unittest discover ${PROJECT_SOURCE_DIR}/test)
