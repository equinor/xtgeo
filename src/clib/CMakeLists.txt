include_directories(${CMAKE_CURRENT_LIST_DIR}/xtg)
set(SRC "${CMAKE_CURRENT_LIST_DIR}/xtg")

# TODO: replace globbing with unique list, as globbing is bad practice
FILE(GLOB SOURCES ${SRC}/*.c)
add_library(xtg STATIC ${SOURCES})

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(SWIG 3.0.1 COMPONENTS REQUIRED)
include(UseSWIG)

# CMP0053: Simplify variable reference and escape sequence evaluation.
# The OLD behavior for this policy is to honor the legacy behavior for
# variable references and escape sequences.
if (POLICY CMP0053)
  cmake_policy(SET CMP0053 OLD)
endif()

# ======================================================================================
# Looking for numpy; need to avoid that weird old system numpy is used (aka RHEL6)
# THIS WAS A PAIN TO DEBUG
#
# Kudos https://github.com/Eyescale/CMake/blob/master/FindNumPy.cmake
# via https://github.com/DeepLearnPhysics/larcv3/blob/src/CMakeLists.txt
# ======================================================================================

if(NOT Python_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
    set(__numpy_out 1)
  endif()
endif()

if (Python_EXECUTABLE)
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
endif(Python_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${__numpy_path}" "${PYTHON_INCLUDE_PATH}" NO_DEFAULT_PATH)

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(PYTHON_NUMPY_INCLUDE_DIR)

message(STATUS "XTGeo Python include path: ${PYTHON_INCLUDE_PATH}")
message(STATUS "XTGeo Python include dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "XTGeo Python executable  : ${PYTHON_EXECUTABLE}")
message(STATUS "XTGeo numpy include path : ${PYTHON_NUMPY_INCLUDE_DIR}")

# ======================================================================================
# Find swig
# ======================================================================================

# in case required SWIG is missing or too old and we are within Equinor, need a path to
# a SWIG version that works (RHEL7 have SWIG 2 by default which is too old)
if("$ENV{HOSTNAME}" MATCHES "statoil" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  LIST(APPEND CMAKE_PROGRAM_PATH "/prog/res/opt/rhel7/swig_4.0.2/bin")
  message(STATUS "Using Swig from /prog/res")
endif()

# ======================================================================================
# Adjust policies
# ======================================================================================

# CMP0074: find_package() uses <PackageName>_ROOT variables.
# The OLD behavior for this policy is to ignore <PackageName>_ROOT variables.
# The NEW behavior for this policy is to use <PackageName>_ROOT variables.
cmake_policy(SET CMP0074 NEW)

# CMP0078: UseSWIG generates standard target names.
# The OLD behavior for this policy relies on UseSWIG_TARGET_NAME_PREFERENCE
# variable that can be used to specify an explicit preference.
# The value may be one of:
#   LEGACY: legacy strategy is applied. Variable SWIG_MODULE_<name>_REAL_NAME
#     must be used to get real target name. This is the default if not specified.
#   STANDARD: target name matches specified name.
if (POLICY CMP0078)
  cmake_policy(SET CMP0078 OLD)
endif()

# CMP0086: UseSWIG honors SWIG_MODULE_NAME via -module flag.
# The OLD behavior for this policy is to never pass -module option.
# The NEW behavior is to pass -module option to SWIG compiler if
#   SWIG_MODULE_NAME is specified.
if (POLICY CMP0086)
  cmake_policy(SET CMP0086 OLD)
endif()

# ======================================================================================
# Compile swig bindings
# ======================================================================================

message(STATUS "Compiling swig bindings")

if (MSVC)
  set(XTGFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
  set(CXTGEOFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
else()
  set(XTGFLAGS -Wall -Wno-unused-but-set-variable -fPIC)
  set(CXTGEOFLAGS -Wl,--no-undefined)
endif()
target_compile_options(xtg PRIVATE ${XTGFLAGS})

set(PYTHON_MODULE cxtgeo)
swig_add_library(
  ${PYTHON_MODULE}
  LANGUAGE python
  OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}"
  SOURCES cxtgeo.i
)
set(SWIG_MODULE ${SWIG_MODULE_${PYTHON_MODULE}_REAL_NAME})

target_include_directories(
  ${SWIG_MODULE} PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${PYTHON_NUMPY_INCLUDE_DIR}
  ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${SWIG_MODULE} PUBLIC ${CXTGEOFLAGS})
target_link_libraries(${SWIG_MODULE} xtg Python::Module)

# scikit-build-core docs recommend this
if(WIN32)
    set_property(TARGET ${SWIG_MODULE} PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
    set_property(
      TARGET ${SWIG_MODULE}
      PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Root installation directory is set in pyproject.toml
# SWIG_MODULE == _cxtgeo, so this installs to
# xtgeo.cxtgeo._cxtgeo
install(TARGETS ${SWIG_MODULE} LIBRARY DESTINATION ${PYTHON_MODULE})
