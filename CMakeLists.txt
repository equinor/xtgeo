# main CMakeLists.txt file

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(clib LANGUAGES C)

if(SKBUILD)
  message(STATUS "<0> The project is built using scikit-build")
endif()

find_package(PythonExtensions REQUIRED)

add_subdirectory(src/xtgeo/clib)
python_extension_module(cxtgeo)
message(STATUS "<0> Installing...")

install(TARGETS cxtgeo LIBRARY DESTINATION src/xtgeo/cxtgeo)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/src/xtgeo/clib/cxtgeo.py"
  DESTINATION src/xtgeo/cxtgeo)
