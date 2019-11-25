add_executable (test_x test_x.c)
target_link_libraries (test_x cxtgeo tap m)

add_executable (test_messages test_messages.c)
target_link_libraries (test_messages cxtgeo tap m)

add_executable (test_point_in_cell_profiling test_point_in_cell_profiling.c )
target_link_libraries (test_point_in_cell_profiling cxtgeo tap m)

add_executable (test_grd3d_onelayer test_grd3d_onelayer.c )
target_link_libraries (test_grd3d_onelayer cxtgeo tap m)

add_executable (test_surf test_surf.c )
target_link_libraries (test_surf cxtgeo tap m)

add_executable (test_roffpropgrid test_roffpropgrid.c )
target_link_libraries (test_roffpropgrid cxtgeo tap m)

add_executable (test_roffgrid test_roffgrid.c )
target_link_libraries (test_roffgrid cxtgeo tap m)

add_executable (test_eclgridprop test_eclgridprop.c )
target_link_libraries (test_eclgridprop cxtgeo tap m)

add_executable (test_cubic_interpolation test_cubic_interpolation.c )
target_link_libraries (test_cubic_interpolation cxtgeo tap m)

add_executable (test_ecl_scan test_ecl_scan.c )
target_link_libraries (test_ecl_scan cxtgeo tap m)

add_executable (test_ecl_readrecords test_ecl_readrecords.c )
target_link_libraries (test_ecl_readrecords cxtgeo tap m)

add_executable (test_roff_scan test_roff_scan.c )
target_link_libraries (test_roff_scan cxtgeo tap m)

# add_executable (test_cube test_cube.c )
# target_link_libraries (test_cube cxtgeo tap m)
