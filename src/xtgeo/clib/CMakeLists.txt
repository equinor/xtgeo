message(STATUS "XTGeo library...")

include_directories(${CMAKE_CURRENT_LIST_DIR}/xtg)

if (MSVC)
  set(XTGFLAGS /Ox)
else()
  set(XTGFLAGS -Wall -Wno-unused-but-set-variable -fPIC)
  set(CXTGEOFLAGS -Wl,--no-undefined)
endif()

set (SRC "${CMAKE_CURRENT_LIST_DIR}/xtg")

# todo: replace globbing with unique list, as globbing is bad practice
FILE(GLOB SOURCES ${SRC}/*.c)

add_library(xtg
  STATIC
  ${SOURCES}
  )

target_compile_options(xtg PRIVATE ${XTGFLAGS})

# ======================================================================================
# Find Python and SWIG
# ======================================================================================

LIST(APPEND CMAKE_PROGRAM_PATH "${PROJECT_SOURCE_DIR}/scripts/_tmp_swig/bin")


# These policies are related to SWIG python libraries naming
if (POLICY CMP0078)  # cmake 3.13+
  # UseSWIG generates standard target names
  cmake_policy(SET CMP0078 NEW)
endif()

if (POLICY CMP0086)  # cmake 3.14+
  # The NEW behavior is to pass -module option to SWIG compiler
  # if SWIG_MODULE_NAME is specified (not sure if this is needed).
  cmake_policy(SET CMP0086 NEW)
endif()

find_package(SWIG 3.0.1 REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs REQUIRED)

message(STATUS "XTGeo Python include path: ${PYTHON_INCLUDE_PATH}")
message(STATUS "XTGeo Python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "XTGeo Python executable ok: ${PYTHON_EXECUTABLE}")

# ======================================================================================
# Looking for numpy; need to avoid that weird old system numpy is used (aka RHEL6)
# THIS WAS A PAIN TO DEBUG
#
# Kudos https://github.com/Eyescale/CMake/blob/master/FindNumPy.cmake
# via https://github.com/DeepLearnPhysics/larcv3/blob/src/CMakeLists.txt
# ======================================================================================

if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
    set(__numpy_out 1)
  endif()
endif()

if (PYTHON_EXECUTABLE)
  # Find out the include path
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; \
             print(numpy.get_include(), end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_path)
  # And the version
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "from __future__ import print_function\ntry: import numpy; \
             print(numpy.__version__, end='')\nexcept:pass\n"
            OUTPUT_VARIABLE __numpy_version)
    message(STATUS "Detected NumPy: ${__numpy_path}; version ${__numpy_version}")
elseif(__numpy_out)
  message(ERROR "Python executable not found.")
endif(PYTHON_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(PYTHON_NUMPY_INCLUDE_DIR)

message(STATUS "XTGeo Numpy include path: ${PYTHON_NUMPY_INCLUDE_DIR}")

# ======================================================================================
# Swigging and linking
# ======================================================================================

swig_add_library(cxtgeo LANGUAGE python SOURCES cxtgeo.i)
message(STATUS "Swigging")

target_include_directories(cxtgeo PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${PYTHON_NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_LIST_DIR}
  )

target_compile_options(cxtgeo PUBLIC ${CXTGEOFLAGS})

swig_link_libraries(cxtgeo xtg)
