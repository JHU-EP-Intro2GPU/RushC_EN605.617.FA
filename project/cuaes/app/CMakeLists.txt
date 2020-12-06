add_executable(demo-cuaes)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cuh *.cu)

target_sources(demo-cuaes PRIVATE ${SOURCES})
target_link_libraries(demo-cuaes PRIVATE cuaes utils)