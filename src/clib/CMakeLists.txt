include_directories(${CMAKE_CURRENT_LIST_DIR}/xtg)
set(SRC "${CMAKE_CURRENT_LIST_DIR}/xtg")

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(SWIG 3.0.1 COMPONENTS REQUIRED)
include(UseSWIG)
find_package(pybind11 REQUIRED)

# TODO: replace globbing with unique list, as globbing is bad practice
FILE(GLOB SOURCES ${SRC}/*.c)
add_library(xtg STATIC ${SOURCES})

message(STATUS "Compiling swig bindings")

if(MSVC)
  set(XTGFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
  set(CXTGEOFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
else()
  set(XTGFLAGS -Wall -Wno-unused-but-set-variable -fPIC)
  set(CXTGEOFLAGS -Wl,--no-undefined)
endif()

set(PYBIND_SOURCES "${SRC}/grdcp3d_cellvol.cpp")
pybind11_add_module(_internal ${PYBIND_SOURCES} ${SOURCES})

# This will end up set as the `_cxtgeo` module, located in
# `wheel.install-dir` from pyproject.toml
set(SWIG_TARGET cxtgeo)

swig_add_library(
  ${SWIG_TARGET} LANGUAGE python
  OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}"
  SOURCES cxtgeo.i)

target_include_directories(
  ${SWIG_TARGET} PUBLIC
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(xtg PRIVATE ${XTGFLAGS})
target_compile_options(${SWIG_TARGET} PUBLIC ${CXTGEOFLAGS})
target_link_libraries(
  ${SWIG_TARGET} xtg
  ${Python_LIBRARIES} Python::Module Python::NumPy)

message(STATUS "XTGeo Python executable  : ${Python_EXECUTABLE}")
message(STATUS "XTGeo Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "XTGeo numpy include path : ${Python_NumPy_INCLUDE_DIRS}")

# scikit-build-core docs recommend this
if(WIN32)
  set_property(TARGET ${SWIG_TARGET} PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
  set_property(
    TARGET ${SWIG_TARGET}
    PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Installation directory is set in pyproject.toml
install(TARGETS ${SWIG_TARGET} _internal LIBRARY DESTINATION .)
