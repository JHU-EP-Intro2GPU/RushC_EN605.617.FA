add_executable(demo-aes)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cpp)

target_sources(demo-aes PRIVATE ${SOURCES})
target_link_libraries(demo-aes PRIVATE aes)