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
find_package(OpenMP)

message(STATUS "Python executable     : ${Python_EXECUTABLE}")
message(STATUS "Python include dirs   : ${Python_INCLUDE_DIRS}")
message(STATUS "numpy include path    : ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "pybind11 include path : ${pybind11_INCLUDE_DIRS}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

find_package(fmt CONFIG QUIET)
if(fmt_FOUND)
  message(STATUS "Found fmt ${fmt_VERSION}")
else()
  message(STATUS "fmt not found, using FetchContent")
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0)
  set(FMT_MASTER_PROJECT OFF)
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(fmt)
endif()

find_package(Eigen3 3.4 CONFIG QUIET)
if(Eigen3_FOUND)
  message(STATUS "Found Eigen3 ${Eigen3_VERSION}")
else()
  message(STATUS "Eigen3 not found, using FetchContent")

  set(EIGEN_BUILD_DOC OFF)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  set(EIGEN_BUILD_FORTRAN
      OFF
      CACHE BOOL "Disable Fortran features in Eigen" FORCE)
  set(EIGEN_TEST_FORTRAN
      OFF
      CACHE BOOL "Disable Fortran tests in Eigen" FORCE)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
  FetchContent_MakeAvailable(eigen)
endif()

if(MSVC)
  set(XTGFLAGS
      /Ox
      /wd4996
      /wd4267
      /wd4244
      /wd4305
      /MP
      /bigobj
      /Zc:__cplusplus
      /d2ReducedOptimizeHugeFunctions
      /MT)
  set(CXTGEOFLAGS
      /Ox
      /wd4996
      /wd4267
      /wd4244
      /wd4305
      /MP
      /bigobj
      /d2ReducedOptimizeHugeFunctions
      /MT)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
else()
  set(XTGFLAGS -Wall -Wno-unused-but-set-variable -Wno-undef -fPIC)
  set(CXTGEOFLAGS -Wall -Wno-unused-but-set-variable -Wno-undef -fPIC)
endif()

# TODO: replace globbing with unique list, as globbing is bad practice
file(GLOB SOURCES ${SRC}/*.c)

pybind11_add_module(
  _internal
  "${SRC}/init.cpp"
  "${SRC}/common/geometry/hexahedron_eval.cpp"
  "${SRC}/common/geometry/hexahedron_point_in_raycasting.cpp"
  "${SRC}/common/geometry/hexahedron_point_in_usingplanes.cpp"
  "${SRC}/common/geometry/hexahedron_point_in_isoparametric.cpp"
  "${SRC}/common/geometry/hexahedron_point_in_tetrahedrons_legacy.cpp"
  "${SRC}/common/geometry/hexahedron_point_in_tetrahedrons_by_scheme.cpp"
  "${SRC}/common/geometry/interpolate.cpp"
  "${SRC}/common/geometry/polygons.cpp"
  "${SRC}/common/geometry/quadrilateral.cpp"
  "${SRC}/common/geometry/tetrahedron.cpp"
  "${SRC}/common/geometry/volumes.cpp"
  "${SRC}/common/logging/logging.cpp"
  "${SRC}/grid3d/cell.cpp"
  "${SRC}/grid3d/cell_xyz.cpp"
  "${SRC}/grid3d/grid.cpp"
  "${SRC}/grid3d/grid_fence_extract.cpp"
  "${SRC}/grid3d/grid_xyz_in.cpp"
  "${SRC}/grid3d/grid_reorganize.cpp"
  "${SRC}/grid3d/grid_surf_oper.cpp"
  "${SRC}/grid3d/zcornsv_utilities.cpp"
  "${SRC}/cube/cube.cpp"
  "${SRC}/regsurf/sample_grid3d.cpp"
  "${SRC}/regsurf/utilities.cpp"
  ${SOURCES})
target_include_directories(_internal PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_internal PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_internal PRIVATE XTGEO_USE_OPENMP)
endif()

target_compile_options(_internal PRIVATE ${XTGFLAGS})
target_link_libraries(_internal PRIVATE fmt::fmt-header-only Eigen3::Eigen)

message(STATUS "Compiling swig bindings")

add_library(xtgeo STATIC ${SOURCES})
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
  set_property(TARGET _internal PROPERTY SUFFIX ".${Python_SOABI}.pyd")
else()
  set_property(TARGET ${SWIG_TARGET}
               PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()

# Installation directory is set in pyproject.toml
install(TARGETS ${SWIG_TARGET} _internal LIBRARY DESTINATION .)
