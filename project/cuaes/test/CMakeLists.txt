add_executable(test-cuaes)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cuh *.cpp *.cu)
target_sources(test-cuaes PRIVATE ${SOURCES})
target_link_libraries(test-cuaes PRIVATE cuaes utils)

add_test(NAME test-cuaes COMMAND test-cuaes WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
