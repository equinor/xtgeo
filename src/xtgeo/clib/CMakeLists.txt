message(STATUS "<1> Add library...")

include_directories(${CMAKE_CURRENT_LIST_DIR}/xtg)

if (MSVC)
  set (CMAKE_C_FLAGS "/Ox")
else()
  set (CMAKE_C_FLAGS "-Wall -O2 -fPIC -Wno-unused-but-set-variable")
endif()

set (SRC "${CMAKE_CURRENT_LIST_DIR}/xtg")

# todo: replace globbing with unique list, as globbing is bad practice
FILE(GLOB SOURCES ${SRC}/*.c)

add_library(xtg
  STATIC
  ${SOURCES}
  )

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG 3.0.1 REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH}
  ${CMAKE_CURRENT_LIST_DIR}
  ${NumPy_INCLUDE_DIRS}
  )

swig_add_library(cxtgeo LANGUAGE python SOURCES cxtgeo.i)
swig_link_libraries(cxtgeo xtg)
