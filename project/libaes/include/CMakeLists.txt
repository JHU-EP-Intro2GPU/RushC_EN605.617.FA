file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp)

target_include_directories(aes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(aes PRIVATE ${HEADERS})