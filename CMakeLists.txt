cmake_minimum_required(VERSION 3.15.3)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

if(NOT SKBUILD)
  message(
    WARNING
    "This CMakeLists.txt file should not be used directly.\n"
    "Use 'pip install ${CMAKE_SOURCE_DIR}' to install this as a Python package.")
endif()

add_subdirectory(src/lib)
