file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.hpp *.cpp)

target_sources(utils PRIVATE ${SOURCES})

# Filesystem support required to read the NIST KAT files. nvcc only supports up
# to C++14, which only has experimental filesystem support.
target_compile_features(utils PUBLIC cxx_std_14)

# Some compilers like to complain about using the experimental filesystem in 
# C++14. A special symbol can be defined to silence them.
target_compile_definitions(utils PRIVATE _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
