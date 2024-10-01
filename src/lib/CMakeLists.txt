include_directories(include)
set(SRC "${CMAKE_CURRENT_LIST_DIR}/src")

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)
find_package(
  SWIG 3.0.1
  COMPONENTS
  REQUIRED)
include(UseSWIG)
find_package(pybind11 REQUIRED)

message(STATUS "Python executable     : ${Python_EXECUTABLE}")
message(STATUS "Python include dirs   : ${Python_INCLUDE_DIRS}")
message(STATUS "numpy include path    : ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "pybind11 include path : ${pybind11_INCLUDE_DIRS}")

# TODO: replace globbing with unique list, as globbing is bad practice
file(GLOB SOURCES ${SRC}/*.c)
add_library(xtgeo STATIC ${SOURCES})

if(MSVC)
  set(XTGFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
  set(CXTGEOFLAGS /Ox /wd4996 /wd4267 /wd4244 /wd4305)
else()
  set(XTGFLAGS -Wall -Wno-unused-but-set-variable -Wno-undef -fPIC)
  set(CXTGEOFLAGS -Wall -Wno-unused-but-set-variable -Wno-undef)
endif()

pybind11_add_module(
  _internal
  "${SRC}/init.cpp"
  "${SRC}/common/geometry/interpolate.cpp"
  "${SRC}/common/geometry/polygons.cpp"
  "${SRC}/common/geometry/quadrilateral.cpp"
  "${SRC}/common/geometry/volumes.cpp"
  "${SRC}/grid3d/cell.cpp"
  "${SRC}/grid3d/grid.cpp"
  "${SRC}/cube/cube.cpp"
  "${SRC}/regsurf/sample_grid3d.cpp"
  "${SRC}/regsurf/utilities.cpp"
  ${SOURCES})
target_include_directories(_internal PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

message(STATUS "Compiling swig bindings")

# This will end up set as the `_cxtgeo` module, located in `wheel.install-dir`
# from pyproject.toml
set(SWIG_TARGET cxtgeo)

swig_add_library(
  ${SWIG_TARGET}
  LANGUAGE python OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}"
  SOURCES cxtgeo.i)

target_include_directories(
  ${SWIG_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
                         ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_compile_options(xtgeo PRIVATE ${XTGFLAGS})
target_compile_options(${SWIG_TARGET} PUBLIC ${CXTGEOFLAGS})

target_link_libraries(${SWIG_TARGET} PRIVATE xtgeo ${Python_LIBRARIES}
                                             Python::Module Python::NumPy)

# scikit-build-core docs recommend this
if(WIN32)
  set_property(TARGET ${SWIG_TARGET} PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
  set_property(TARGET ${SWIG_TARGET}
               PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Installation directory is set in pyproject.toml
install(TARGETS ${SWIG_TARGET} _internal LIBRARY DESTINATION .)
