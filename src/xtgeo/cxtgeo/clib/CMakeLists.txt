# Input to CMAKE
# Author: JRIV
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
set(CMAKE_BUILD_PATH "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

project(cxtgeo C)

enable_testing()
add_subdirectory(src)
#add_subdirectory(lib)
add_subdirectory(ctests)
add_subdirectory(tap)


install(FILES
	      "${CMAKE_BUILD_PATH}/src/libcxtgeo.a"
	      DESTINATION ${CMAKE_SOURCE_DIR}/lib)


add_test(NAME test_x COMMAND test_x)
add_test(NAME test_messages COMMAND test_messages)
add_test(NAME test_point_in_cell_profiling COMMAND test_point_in_cell_profiling)
add_test(NAME test_grd3d_onelayer COMMAND test_grd3d_onelayer)
add_test(NAME test_surf COMMAND test_surf)
add_test(NAME test_roffpropgrid COMMAND test_roffpropgrid)
add_test(NAME test_roffgrid COMMAND test_roffgrid)
add_test(NAME test_eclgridprop COMMAND test_eclgridprop)
add_test(NAME test_cubic_interpolation COMMAND test_cubic_interpolation)
add_test(NAME test_ecl_scan COMMAND test_ecl_scan)
add_test(NAME test_ecl_readrecords COMMAND test_ecl_readrecords)
add_test(NAME test_roff_scan COMMAND test_roff_scan)
# add_test(NAME test_cube COMMAND test_cube)
