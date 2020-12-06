file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cuh *.cpp *.cu)

target_sources(cuaes PRIVATE ${SOURCES})