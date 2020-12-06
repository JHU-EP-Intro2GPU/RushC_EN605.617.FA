add_executable(test-aes)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cpp)
target_sources(test-aes PRIVATE ${SOURCES})
target_link_libraries(test-aes PRIVATE aes utils)

add_test(NAME test-aes COMMAND test-aes WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
