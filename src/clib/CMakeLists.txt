message(STATUS "XTGeo library...")

include_directories(${CMAKE_CURRENT_LIST_DIR}/xtg)

find_package(Threads)


if (MSVC)
  set(XTGFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
  set(CXTGEOFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
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

# in case required SWIG is missing or too old and we are within Equinor, need a path to
# a SWIG version that works (RHEL7 have SWIG 2 by default which is too old)
if("$ENV{HOSTNAME}" MATCHES "statoil" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  LIST(APPEND CMAKE_PROGRAM_PATH "/prog/res/opt/rhel7/swig_4.0.2/bin")
  message(STATUS "Using Swig from /prog/res")
endif()

# These policies are related to SWIG python libraries naming
if (CMAKE_VERSION VERSION_GREATER 3.12)
  if (POLICY CMP0078)
    cmake_policy(SET CMP0078 OLD)
  endif()

  if (POLICY CMP0086)
    cmake_policy(SET CMP0086 OLD)
  endif()
else()
  set(UseSWIG_TARGET_NAME_PREFERENCE LEGACY)
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

if (POLICY CMP0053)
  cmake_policy(SET CMP0053 OLD)
endif()

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
    COMMAND @PYTHON_EXECUTABLE@ -c
    "try: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
    OUTPUT_VARIABLE __numpy_path)
  # And the version
  execute_process(
    COMMAND @PYTHON_EXECUTABLE@ -c
            "try: import numpy; print(numpy.__version__, end='')\nexcept:pass\n"
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
# Swigging and linking, with backward compatibility to older cmake versions
# ======================================================================================
message(STATUS "Swigging")

set(LIBRARYNAME cxtgeo)

if (CMAKE_VERSION VERSION_GREATER 3.7)
  swig_add_library(${LIBRARYNAME} LANGUAGE python SOURCES cxtgeo.i)
else()
  swig_add_module(${LIBRARYNAME} python cxtgeo.i)
endif()

set(SWIGTARGET ${SWIG_MODULE_${LIBRARYNAME}_REAL_NAME})

target_include_directories(${SWIGTARGET} PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${PYTHON_NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_LIST_DIR}
  )
target_compile_options(${SWIGTARGET} PUBLIC ${CXTGEOFLAGS})

swig_link_libraries(${LIBRARYNAME} xtg ${PTHREAD_LIBRARY})

python_extension_module(${SWIGTARGET})

set(CXTGEO src/xtgeo/cxtgeo)

install(TARGETS ${SWIGTARGET} LIBRARY DESTINATION ${CXTGEO})
