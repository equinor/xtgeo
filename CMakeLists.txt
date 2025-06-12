cmake_minimum_required(VERSION 3.15.3)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(EIGEN_TEST_CXX11
    ON
    CACHE BOOL "" FORCE)
set(EIGEN_CXX_STANDARD
    17
    CACHE STRING "C++ standard for Eigen" FORCE)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT SKBUILD)
  message(
    WARNING "This CMakeLists.txt file should not be used directly.\n"
            "Use 'pip install ${CMAKE_SOURCE_DIR}' to install this as a Python "
            "package.")
endif()

add_subdirectory(src/lib)
