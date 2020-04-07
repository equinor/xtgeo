# Main CMakeLists.txt file. Most of the work is inside
# the CMakeLists file under add_subdirectory()

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
if (CMAKE_VERSION VERSION_LESS 3)
  project(clib C)
else()
  project(clib LANGUAGES C)
endif()


set(CMAKE_VERBOSE_MAKEFILE ON)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

find_package(PythonExtensions REQUIRED)

add_subdirectory(src/clib)
