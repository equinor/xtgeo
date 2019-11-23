use 5.008;
use ExtUtils::MakeMaker;


my @cfiles = qw(
		C_wrap.c
		cube_conv_from_grd3d.c
		cube_ij_from_xy.c
		cube_ijk_from_xyz.c
		cube_import_segy.c
		cube_export_rmsregular.c
		cube_import_rmsregular.c
		cube_scan_segy_hdr.c
		cube_value_ijk.c
		cube_vertical_val_list.c
		cube_xy_from_ij.c
		grd3d_add_zcells.c
		grd3d_adj_dzcorn.c
		grd3d_adj_z_from_map.c
		grd3d_adj_z_from_mapv2.c
		grd3d_adj_z_from_mapv3.c
		grd3d_adj_z_from_zlog.c
		grd3d_calc_abase.c
		grd3d_calc_cell_dip.c
		grd3d_calc_dz.c
		grd3d_calc_sum_dz.c
		grd3d_calc_z.c
		grd3d_cellpoint.c
		grd3d_cell_faultthrows.c
		grd3d_convert_hybrid.c
		grd3d_convert_hybrid2.c
		grd3d_collapse_inact.c
		grd3d_copy_prop_int.c
		grd3d_copy_prop_float.c
		grd3d_corners.c
		grd3d_count_active.c
		grd3d_count_prop_int.c
		grd3d_diff_zcorn.c
		grd3d_export_ecl_grid.c
		grd3d_export_ecl_prop.c
		grd3d_export_ecl_pstart.c
		grd3d_export_flgrs.c
		grd3d_export_grdecl.c
		grd3d_export_grdeclprop.c
		grd3d_export_roff_grid.c
		grd3d_export_roff_end.c
		grd3d_export_roff_prop.c
		grd3d_export_roff_pstart.c
		grd3d_fault_marks.c
		grd3d_flip_depth.c
		grd3d_frac_prop_within_ii.c
		grd3d_ifthen_prop_intint.c
		grd3d_geometrics.c
		grd3d_getcell_by_prop.c
		grd3d_import_ecl_egrid.c
		grd3d_import_ecl_grid.c
		grd3d_import_ecl_prop.c
		grd3d_import_grdecl.c
		grd3d_import_grdeclpar.c
		grd3d_import_roff_grid.c
		grd3d_import_roff_prop.c
		grd3d_inact_by_dz.c
		grd3d_inact_outs_pol.c
		grd3d_interp_prop_verti.c
		grd3d_interp_prop_vertii.c
		grd3d_list_ecl_kw.c
		grd3d_make_active.c
		grd3d_make_z_consistent.c
		grd3d_merge_grids.c
		grd3d_midpoint.c
		grd3d_point_in_cell.c
		grd3d_pol_ftrace.c
		grd3d_prop_infill1_int.c
		grd3d_print_cellinfo.c
		grd3d_pzcorns.c
		grd3d_remap_prop_g2g.c
		grd3d_scan_ecl_egrid_hd.c
		grd3d_scan_ecl_grid_hd.c
		grd3d_scan_roff_bingrid.c
		grd3d_scan_roff_binpar.c
		grd3d_set_active.c
		grd3d_set_dz_cell.c
		grd3d_set_propval_int.c
		grd3d_set_propval_float.c
		grd3d_set_prop_by_pol.c
		grd3d_set_prop_in_pol.c
                grd3d_split_prop_by_pol.c
                grd3d_strip_array.c
		grd3d_zdominance_int.c
		map_chk_point_between.c
		map_export_arcflt.c
		map_export_storm_binary.c
		map_export_surfer_ascii.c
		map_export_ev_ascii.c
		map_get_corners_xy.c
		map_get_z_from_xy.c
		map_import_arcflt.c
		map_import_irap_ascii.c
		map_import_storm_binary.c
		map_import_surfer_ascii.c
		map_median_filter.c
		map_merge_map.c
		map_operation_map.c
		map_operation_value.c
		map_pol_mask.c
		map_sample_grd3d_prop.c
		map_set_value.c
		map_simple_stats.c
		map_slice_cube.c
		map_slice_grd3d.c
		map_tilt.c
		map_wiener_from_grd3d.c
		polys_chk_point_inside.c
		pol_chk_point_inside.c
		pol_close.c
		pol_extend.c
		pol_import_irap.c
		pol_refine.c
		pol_set_entry.c
		pox_copy_pox.c
		pox_export_irap_ascii.c
		pox_import_irap.c
		pox_import_rms.c
		pox_operation_pox.c
		pox_operation_scalar.c
		pox_zsum.c
		u_eightletter.c
		u_read_ecl_bin_record.c
		u_read_ecl_asc_record.c
		u_wri_ecl_bin_record.c
		xtg_messages.c
		xtg_msg_handler.c
		xtg_verbose.c
		x_chk_point_in_cell.c		
		x_cmp_sort.c
		x_conv_vformats.c
		x_diff_angle.c
		x_general.c
		x_ib2ijk.c
		x_ijk2ib.c
		x_interp_map_nodes.c
		x_mapaxes.c
		x_nint.c
		x_regular_geom.c
		x_statistics.c
		x_stretch_points.c
		x_stretch_vector.c
		x_swap_check.c
		x_swap_endian.c
		x_vector_extrapol.c
		x_vector_info.c
		x_vector_linint.c
		);

# object files
my @objs=();
foreach (@cfiles) {
    s/\.c$/\.o/g;
    push @objs, $_;
}

# header files
my @hdrs=glob("libx*.h");

my $instbase="";
my $instlibs="";
if (exists $ENV{'SDP_BINDIST'}) {
    $instbase="$ENV{'SDP_BINDIST'}";
    $instlibs="$ENV{'SDP_BINDIST'}/lib64/perl5";
}

WriteMakefile(
	      'NAME'		=> 'CXTGeo::C',
	      #'VERSION_FROM'	=> 'cxtgeo_version.pl',      # finds $VERSION
	      'AUTHOR'          => 'Jan C. Rivenaes (jriv@statoil.com)',
	      'ABSTRACT'        => 'CXTGeo - C library eXtended Tool for Geoscientists',
	      'LIBS'		=> ['-lm'],         # e.g., '-lm'
	      'OBJECT'		=> "@objs",         # link all the C files too
	      'INSTALL_BASE'    => "$instbase",
	      );

sub  MY::postamble {
    $postamble ="C_wrap.c : C.i @hdrs\n";
    $postamble.="\techo 'Running SWIG'\n";
    $postamble.="\tswig -perl5 C.i\n";
}
