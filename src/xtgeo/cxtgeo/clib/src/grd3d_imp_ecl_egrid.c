/*
 *******************************************************************************
 *
 * Import ECL EGRID (version 2)
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_imp_ecl_egrid.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Import a grid on eclipse EGRID format. This routine requires that an
 *    earlier scanning of the file is done, so that grid dimensions and
 *    the byte positions of the relevant records are known. These records are:
 *    MAPAXES, COORD, ZCORN, ACTNUM. Only binary format is supported.
 *
 * ARGUMENTS:
 *    points_v       i     a [9] matrix with X Y Z of 3 points
 *    nvector        o     a [4] vector with A B C D
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 *******************************************************************************
 */



int grd3d_imp_ecl_egrid (
                         FILE *fc,
                         int nx,
                         int ny,
                         int nz,
                         long bpos_mapaxes,
                         long bpos_coord,
                         long bpos_zcorn,
                         long bpos_actnum,
                         double *p_coord_v,
                         double *p_zcorn_v,
                         int *p_actnum_v,
                         int debug
                         )
{

    /* locals */
    int *idum = NULL;
    float *fdum = NULL;
    double *ddum = NULL;

    double xma1, yma1, xma2, yma2, xma3, yma3, cx, cy, cz;

    float *tmp_mapaxes, *tmp_coord, *tmp_zcorn;
    int nxyz, nmapaxes, ncoord, nzcorn;
    int *tmp_actnum, nactnum;
    int ib = 0, ibb, ibz, kz, jy, ix, ic;

    char sbn[24] = "grd3d_imp_ecl_egrid";

    /*
     * ========================================================================
     * INITIAL TASKS
     * ========================================================================
     */
    xtgverbose(debug);

    nxyz = nx * ny * nz;
    nmapaxes = 6;
    ncoord = (nx + 1) * (ny + 1) * 2 * 3;
    nzcorn = nx * ny *nz * 8;

    tmp_mapaxes = calloc(nmapaxes, sizeof(float));
    tmp_coord = calloc(ncoord, sizeof(float));
    tmp_zcorn = calloc(nzcorn, sizeof(float));
    tmp_actnum = calloc(nxyz, sizeof(int));

    /*=========================================================================
     * Read MAPAXES, which is present if bpos_mapaxes > 0
     * MAPAXES format is 6 numbers:
     * xcoord_endof_yaxis ycoord_endof_yaxis xcoord_origin ycoord_origin
     * xcoord_endof_xaxis ycoord_endof_xaxis
     * MAPAXES is a donkey ear growth fertilizer :-)
     */
    if (bpos_mapaxes > 0) {
        grd3d_read_eclrecord(fc, bpos_mapaxes, 2, idum, 0, tmp_mapaxes,
                             nmapaxes, ddum, 0, debug);
        xma1 = tmp_mapaxes[0];
        yma1 = tmp_mapaxes[1];
        xma2 = tmp_mapaxes[2];
        yma2 = tmp_mapaxes[3];
        xma3 = tmp_mapaxes[4];
        yma3 = tmp_mapaxes[5];
        xtg_speak(sbn, 2, "Mapaxes 0: %f", tmp_mapaxes[0]);
    }


    /*=========================================================================
     * Read COORD
     */
    grd3d_read_eclrecord(fc, bpos_coord, 2, idum, 0, tmp_coord, ncoord, ddum,
                         0, debug);

    /* convert from MAPAXES, if present */
    xtg_speak(sbn, 2, "Conversion of COORD...");

    for (ib = 0; ib < ncoord; ib=ib+3) {
        cx = tmp_coord[ib];
        cy = tmp_coord[ib+1];
        cz = tmp_coord[ib+2];
        x_mapaxes(bpos_mapaxes, &cx, &cy, xma1, yma1, xma2, yma2,
                  xma3, yma3, 0.0, 0.0, 0.0, 0.0, 2, debug);
        p_coord_v[ib] = cx;
        p_coord_v[ib+1] = cy;
        p_coord_v[ib+2] = cz;
    }
    xtg_speak(sbn, 2, "Conversion... DONE");

    /*=========================================================================
     * Read ZCORN
     */
    xtg_speak(sbn, 2, "Read ZCORN...");
    grd3d_read_eclrecord(fc, bpos_zcorn, 2, idum, 0, tmp_zcorn, nzcorn, ddum,
                         0, debug);

    xtg_speak(sbn, 2, "Read ZCORN... DONE");
    /*
     * ZCORN: Eclipse has 8 corners pr cell, while XTGeo format
     * use 4 corners (top of cell) except for last layer where also base is
     * used, i.e. for NZ+1 cell. This may cause problems if GAPS in GRDECL
     * format (like BRILLD test case)
     */

    xtg_speak(sbn, 2, "Transform ZCORN...");

    grd3d_zcorn_convert(nx, ny, nz, tmp_zcorn, p_zcorn_v, 0, debug);

    xtg_speak(sbn, 2, "Transform ZCORN... DONE");


    /*=========================================================================
     * Read ACTNUM directly
     */
    grd3d_read_eclrecord(fc, bpos_actnum, 1, p_actnum_v, nxyz, fdum, 0, ddum,
                         0, debug);

    int nact = 0;
    for (ib = 0; ib < nxyz; ib++) {
        if (p_actnum_v[ib] == 1) nact ++;
    }

    free(tmp_mapaxes);
    free(tmp_coord);
    free(tmp_zcorn);
    free(tmp_actnum);


    return nact;  // shall be number of active cells
}









/*     /\* initialise *\/ */
/*     mamode=0; */
/*     x1=0.0; y1=0.0; x2=0.0; y2=0.0; x3=0.0; y3=0.0; */
/*     ix=1;jy=1;kz=1; */

/*     while (ios == 0) { */

/* 	if (mode == 0) { */
/* 	    xtg_speak(s,4,"Reading binary record..."); */
/* 	    ios=u_read_ecl_bin_record ( */
/* 				       cname, */
/* 				       ctype, */
/* 				       &reclen, */
/* 				       max_alloc_int, */
/* 				       max_alloc_float, */
/* 				       max_alloc_double, */
/* 				       max_alloc_char, */
/* 				       max_alloc_logi, */
/* 				       tmp_int_v, */
/* 				       tmp_float_v, */
/* 				       tmp_double_v, */
/* 				       tmp_string_v, */
/* 				       tmp_logi_v, */
/* 				       fc, */
/* 				       debug */
/* 				       ); */
/* 	} */
/* 	else{ */
/* 	    ios=u_read_ecl_asc_record ( */
/* 				       cname, */
/* 				       ctype, */
/* 				       &reclen, */
/* 				       tmp_int_v, */
/* 				       tmp_float_v, */
/* 				       tmp_double_v, */
/* 				       tmp_string_v, */
/* 				       tmp_logi_v, */
/* 				       fc, */
/* 				       debug */
/* 				       ); */
/* 	} */


/* 	if (ios != 0) break; */
/* 	xtg_speak(s,3,"Record read for file type %d",mode); */


/* 	if (strcmp(cname,"ACTNUM  ")==0) { */
/* 	    xtg_speak(s,2,"Reading ACTNUM values..."); */
/* 	    for (ib=0;ib<nxyz;ib++) { */
/* 		actnum_v[ib]=tmp_int_v[ib]; */
/* 		if (actnum_v[ib]==1) ia++; */
/* 	    } */

/* 	} */

/* 	/\* */
/* 	 * MAPAXES format is 6 numbers: */
/* 	 * xcoord_endof_yaxis ycoord_endof_yaxis xcoord_origin ycoord_origin */
/* 	 * xcoord_endof_xaxis ycoord_endof_xaxis */
/* 	 * ie */
/* 	 * x1 y1 x2 y2 x3 y3 (point 1 is on the new Y axis, point 2 in origo, point 3 on new X axis) */
/* 	 *\/ */


/* 	else if (strcmp(cname,"MAPAXES ")==0) { */

/* 	    xtg_speak(s,2,"Reading MAPAXES values..."); */
/* 	    x1=tmp_float_v[0]; */
/* 	    y1=tmp_float_v[1]; */
/* 	    x2=tmp_float_v[2]; */
/* 	    y2=tmp_float_v[3]; */
/* 	    x3=tmp_float_v[4]; */
/* 	    y3=tmp_float_v[5]; */

/* 	    mamode=1; */
/* 	} */


/* 	else if (strcmp(cname,"COORD   ")==0) { */
/* 	    xtg_speak(s,2,"Reading COORD..., RECLEN is %d",reclen); */


/* 	    for (nn=0; nn<reclen; nn++) { */


/* 		p_coord_v[nn]=tmp_float_v[nn]; */

/* 		xtg_speak(s,4,"Coordinate is %f",p_coord_v[nn]); */

/* 		/\* find xmin/xmax/ymin/ymax: *\/ */
/* 		if (ix==1) { */
/* 		    if (p_coord_v[nn]<xmin) xmin = p_coord_v[nn]; */
/* 		    if (p_coord_v[nn]>xmax) xmax = p_coord_v[nn]; */
/* 		    ix=0; */
/* 		    jy=1; */
/* 		    kz=0; */
/* 		} */
/* 		else if (jy==1) { */
/* 		    if (p_coord_v[nn]<ymin) ymin = p_coord_v[nn]; */
/* 		    if (p_coord_v[nn]>ymax) ymax = p_coord_v[nn]; */
/* 		    ix=0; */
/* 		    jy=0; */
/* 		    kz=1; */
/* 		} */
/* 		else{ */
/* 		    ix=1; */
/* 		    jy=0; */
/* 		    kz=0; */
/* 		} */
/* 	    } */

/* 	} */

/* 	else if (strcmp(cname,"ZCORN   ")==0) { */
/* 	    xtg_speak(s,2,"Reading ZCORN..."); */

/* 	     *\/ */

/* 	    ib=0; */
/* 	    kzread=0; */
/* 	    kk=0; */
/* 	    ic=0; */
/* 	    for (k=1; k<=2*nz; k++) { */

/* 		xtg_speak(s,4,"ZCORN reading %f",tmp_float_v[ic]); */


/* 		if (kzread==0) { */
/* 		    kzread=1; */
/* 		} */
/* 		else{ */
/* 		    kzread=0; */
/* 		} */
/* 		if (k==2*nz && kzread==0) kzread=1; */
/* 		if (kzread==1) { */
/* 		    kk+=1; */
/* 		    xtg_speak(s,2,"Reading layer: %d", kk); */
/* 		} */
/* 		for (j=1; j<=ny; j++) { */
/* 		    /\* "left" cell margin *\/ */
/* 		    for (i=1; i<=nx; i++) { */
/* 			fvalue1=tmp_float_v[ic++]; */
/* 			fvalue2=tmp_float_v[ic++]; */

/* 			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0); */
/* 			if (kzread==1) { */
/* 			    p_zcorn_v[4*ib+1*1-1] = fvalue1; */
/* 			    p_zcorn_v[4*ib+1*2-1] = fvalue2; */
/* 			} */
/* 		    } */
/* 		    /\* "right" cell margin *\/ */
/* 		    for (i=1; i<=nx; i++) { */
/* 			fvalue1=tmp_float_v[ic++]; */
/* 			fvalue2=tmp_float_v[ic++]; */

/* 			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0); */
/* 			if (kzread==1) { */
/* 			    p_zcorn_v[4*ib+1*3-1] = fvalue1; */
/* 			    p_zcorn_v[4*ib+1*4-1] = fvalue2; */
/* 			} */
/* 		    } */
/* 		} */
/* 	    } */
/* 	} */

/* 	else { */
/* 	    xtg_speak(s,2,"Reading (and skipping) record: %s\n", cname); */
/* 	} */

/*     } */

/*     *num_active=ia+1; */
/*     xtg_speak(s,2,"Number of active ACTIVE cells: %d %d", *num_active, ia); */
/*     xtg_speak(s,2,"Number of ACTIVE cells: %d %d", *num_active, ia); */

/*     xtg_speak(s,3,"(1)... XMIN XMAX YMIN YMAX: %10.2f %10.2f %10.2f %10.2f",xmin,xmax,ymin,ymax); */

/*     /\* get min and max of geometry (for use in MAPAXES Tor Barkve method) *\/ */
/*     /\* grd3d_minmax_geom(nx,ny,nz,p_coord_v,p_zcorn_v,actnum_v,  *\/ */
/*     /\* 		      &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, 0, debug); *\/ */

/*     xtg_speak(s,3,"(2)... XMIN XMAX YMIN YMAX: %10.2f %10.2f %10.2f %10.2f",xmin,xmax,ymin,ymax); */


/*     } */


/*     /\* free allocated space *\/ */
/*     xtg_speak(s,2,"Freeing tmp pointers"); */
/*     xtg_speak(s,3,"Freeing tmp_int_v ..."); */
/*     free(tmp_int_v); */

/*     xtg_speak(s,3,"Freeing tmp_float_v ..."); */
/*     free(tmp_float_v); */
/*     xtg_speak(s,3,"Freeing tmp_double_v ..."); */
/*     free(tmp_double_v); */


/*     xtg_speak(s,3,"Freeing tmp_string_v ..."); */
/*     for (i=0; i<nxyz; i++) { */
/*         xtg_speak(s,3,"Freeing tmp_string_v elem..."); */
/*         if (tmp_string_v[i]) free(tmp_string_v[i]); */
/*     } */
/*     if (tmp_string_v) free(tmp_string_v); */

/*     xtg_speak(s,3,"Freeing tmp_logi_v ..."); */
/*     free(tmp_logi_v); */
/*     xtg_speak(s,2,"Leaving routine ..."); */
/* } */
