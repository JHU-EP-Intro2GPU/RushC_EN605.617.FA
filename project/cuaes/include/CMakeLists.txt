file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp *.cuh)

target_include_directories(cuaes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(cuaes PRIVATE ${HEADERS})