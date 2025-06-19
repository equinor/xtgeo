cmake_minimum_required(VERSION 3.15.3)

# Look for pre-compiled (vcpkg) stuff on Windows first. This _must_ stay before
# project()
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VCPKG_INSTALLATION_ROOT})
    set(VCPKG_ROOT "$ENV{VCPKG_INSTALLATION_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE
        "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
    message(STATUS "Using vcpkg from VCPKG_INSTALLATION_ROOT: ${VCPKG_ROOT}")
  elseif(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
    message(STATUS "Found vcpkg at default location: C:/vcpkg")
  endif()
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EIGEN_TEST_CXX11
    ON
    CACHE BOOL "" FORCE)
set(EIGEN_CXX_STANDARD
    17
    CACHE STRING "C++ standard for Eigen" FORCE)

if(NOT SKBUILD)
  message(
    WARNING "This CMakeLists.txt file should not be used directly.\n"
            "Use 'pip install ${CMAKE_SOURCE_DIR}' to install this as a Python "
            "package.")
endif()

add_subdirectory(src/lib)
