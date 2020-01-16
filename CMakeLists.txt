# main CMakeLists.txt file

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(clib LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(SKBUILD)
  message(STATUS "<0> The project is built using scikit-build")
endif()

find_package(PythonExtensions REQUIRED)

add_subdirectory(src/xtgeo/clib)
